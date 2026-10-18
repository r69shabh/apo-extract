"""Offline end-to-end dry run + crash/resume test of the SR-PBO optimizer.

Uses the real hiring/resume dataset (5 PDFs + golds) with a directive-aware fake
completer: it drops ``skills`` from its prediction unless the rendered prompt
contains the ``$.skills`` directive. The optimizer must learn that directive from
training failures and improve validation F1 without any API key.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from apo_extract.config import (
    AppConfig,
    BudgetConfig,
    DatasetConfig,
    LlmConfig,
    LLMsConfig,
    OptimizerConfig,
    ScoringConfig,
    SeedPromptConfig,
    SplitPolicyConfig,
)
from apo_extract.dataset.loader import load_dataset
from apo_extract.dataset.splitter import split_documents
from apo_extract.llm.client import LLMClient
from apo_extract.observability.store import RunStore
from apo_extract.optimizer.pareto_beam import Optimizer
from apo_extract.optimizer.proposer import Proposer
from apo_extract.predictor.extractor import Extractor

from conftest import make_scorer

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "extract-bench"
SCHEMA_DIR = DATASET_DIR / "dataset" / "hiring" / "resume"

class DirectiveAwareCompleter:
    """Drops `skills`/`languages` unless their directives appear in the system prompt."""

    MISSING = ("skills", "languages")

    def __init__(self, golds: dict[str, dict]):
        self.golds = golds
        self.calls = 0

    async def __call__(self, request: dict):
        self.calls += 1
        system = request["messages"][0]["content"]
        prediction = copy.deepcopy(self.golds[request["pdf_id"]])
        for field in self.MISSING:
            if f"$.{field}" not in system:
                prediction.pop(field, None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(prediction)))],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=300),
            model=request["model"],
        )


def build_app_config(run_id: str, *, max_iterations: int, enable_rollback: bool = False) -> AppConfig:
    return AppConfig(
        run_id=run_id,
        seed=42,
        dataset=DatasetConfig(
            name="hiring/resume",
            schema_path=str(SCHEMA_DIR / "resume-schema.json"),
            documents_glob=str(SCHEMA_DIR / "pdf+gold" / "*.pdf"),
            gold_dir=str(SCHEMA_DIR / "pdf+gold"),
            split_policy=SplitPolicyConfig(seed=42),
        ),
        scoring=ScoringConfig(),
        llms=LLMsConfig(extractor=LlmConfig(model="fake-model")),
        seed_prompt=SeedPromptConfig(),
        budget=BudgetConfig(max_iterations=max_iterations, max_cost_usd=100.0, max_total_tokens=10_000_000),
        optimizer=OptimizerConfig(
            beam_width=3,
            candidates_per_iteration=2,
            stall_patience=4,
            enable_subtree_rollback=enable_rollback,
        ),
    )


def build_optimizer(config: AppConfig, store: RunStore, completer, run_id: str) -> Optimizer:
    bundle = load_dataset(config.dataset)
    split = split_documents([d.doc_id for d in bundle.documents], config.dataset.split_policy)
    docs = {d.doc_id: d for d in bundle.documents}
    golds = {d.doc_id: d.gold for d in bundle.documents}
    client = LLMClient(
        config.llms.extractor, role="extractor", store=store, run_id=run_id, completer=completer
    )
    extractor = Extractor(client, store, pdf_handler=None)
    proposer = Proposer(
        bundle.schema,
        client=None,
        store=store,
        run_id=run_id,
        duplicate_similarity=config.optimizer.duplicate_similarity,
    )
    return Optimizer(
        config,
        store,
        extractor=extractor,
        scorer=make_scorer(),
        proposer=proposer,
        schema=bundle.schema,
        splits={"train": split.train, "val": split.val, "test": split.test},
        docs=docs,
        seed_text="Extract all fields from the attached resume into JSON matching the schema.",
        run_id=run_id,
    )


class TestDryRun:
    def test_optimizer_learns_directive_offline(self, tmp_path):
        config = build_app_config("dryrun", max_iterations=3)
        store = RunStore(tmp_path / "dryrun")
        bundle = load_dataset(config.dataset)
        golds = {d.doc_id: d.gold for d in bundle.documents}
        completer = DirectiveAwareCompleter(golds)
        optimizer = build_optimizer(config, store, completer, "dryrun")

        summary = optimizer.run()

        assert summary["best_candidate_id"] != "root"
        assert summary["val_f1"] > 0.99
        assert summary["test_f1"] > 0.99
        best = json.loads((store.run_dir / "final" / "best_artifact.json").read_text())
        assert "$.skills" in best["subtree_directives"]
        assert best["subtree_directives"]["$.skills"].strip()

        iterations = store.list_iterations("dryrun")
        assert any(row["decision"] == "ACCEPTED" for row in iterations)
        assert all(row["hypothesis"] for row in iterations if row["decision"] == "ACCEPTED")

    def test_resume_after_keyboard_interrupt(self, tmp_path):
        run_dir = tmp_path / "resumable"
        config = build_app_config("resumable", max_iterations=3)
        bundle = load_dataset(config.dataset)
        golds = {d.doc_id: d.gold for d in bundle.documents}

        seen_pairs: set[tuple[str, str]] = set()
        interrupt_after = 17  # mid iteration 2, ~half its train predictions cached

        class InterruptingCompleter(DirectiveAwareCompleter):
            async def __call__(self, request):
                if len(seen_pairs) >= interrupt_after:
                    raise KeyboardInterrupt("simulated crash mid-run")
                seen_pairs.add((request["prompt_hash"], request["pdf_id"]))
                return await super().__call__(request)

        store1 = RunStore(run_dir)
        optimizer1 = build_optimizer(config, store1, InterruptingCompleter(golds), "resumable")
        with pytest.raises(KeyboardInterrupt):
            optimizer1.run()
        cached_count = store1.conn.execute(
            "SELECT COUNT(*) AS c FROM predictions_cache"
        ).fetchone()
        assert cached_count["c"] >= 4
        store1.close()

        store2 = RunStore(run_dir)
        run_row = store2.get_run("resumable")
        assert run_row["current_iteration"] >= 1

        class NoRepeatCompleter(DirectiveAwareCompleter):
            async def __call__(self, request):
                pair = (request["prompt_hash"], request["pdf_id"])
                assert pair not in seen_pairs, f"cache not reused for {pair}"
                seen_pairs.add(pair)
                return await super().__call__(request)

        completer2 = NoRepeatCompleter(golds)
        optimizer2 = build_optimizer(config, store2, completer2, "resumable")
        summary = optimizer2.run(resume=True)

        assert summary["best_candidate_id"] != "root"
        assert summary["val_f1"] > 0.99
        rows = store2.list_iterations("resumable")
        assert {row["iteration"] for row in rows} >= {1, 2}
        store2.close()
