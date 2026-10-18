"""Subtree-rollback pass: reverts regressed subtrees to parent directives."""

import asyncio
import json

from apo_extract.config import (
    AppConfig,
    BudgetConfig,
    DatasetConfig,
    LlmConfig,
    LLMsConfig,
    OptimizerConfig,
    SeedPromptConfig,
)
from apo_extract.observability.store import RunStore
from apo_extract.optimizer.pareto_beam import Optimizer, SplitEval
from apo_extract.optimizer.safeguards import ArchiveEntry, regressed_subtrees
from apo_extract.predictor.artifact import PromptArtifact

SCHEMA = {"type": "object", "properties": {"skills": {}, "education": {}}}


def make_config(**optimizer_overrides) -> AppConfig:
    defaults = dict(enable_subtree_rollback=True, stall_patience=9)
    defaults.update(optimizer_overrides)
    return AppConfig(
        run_id="rb",
        dataset=DatasetConfig(
            name="x", schema_path="x.json", documents_glob="*.pdf", gold_dir="."
        ),
        llms=LLMsConfig(extractor=LlmConfig(model="fake")),
        seed_prompt=SeedPromptConfig(path="configs/prompts/seed_generic_extraction.txt"),
        budget=BudgetConfig(max_iterations=3),
        optimizer=OptimizerConfig(**defaults),
    )


def make_optimizer(tmp_path, config) -> Optimizer:
    store = RunStore(tmp_path / "rb")
    return Optimizer(
        config,
        store,
        extractor=None,
        scorer=None,
        proposer=None,
        schema=SCHEMA,
        splits={"train": ["a"], "val": ["a"], "test": ["a"]},
        docs={},
        seed_text="seed",
        run_id="rb",
    )


def test_regressed_subtrees_threshold():
    old = {"$.education": {"f1": 1.0}, "$.skills": {"f1": 0.5}}
    new = {"$.education": {"f1": 0.2}, "$.skills": {"f1": 0.9}}
    assert regressed_subtrees(old, new, 0.05) == ["$.education"]
    assert regressed_subtrees(old, new, 0.95) == []


def test_rollback_pass_reverts_and_accepts(tmp_path):
    config = make_config()
    optimizer = make_optimizer(tmp_path, config)
    optimizer.store.init_run("rb", "hash")

    root = PromptArtifact.init_from_schema(SCHEMA, "seed")
    candidate = root.copy("c1_0")
    candidate.subtree_directives["$.skills"] = "Always look in side projects."
    candidate.subtree_directives["$.education"] = "Skip education entirely."
    optimizer.store.save_candidate("root", "rb", 0, root.to_dict(), root.hash())
    optimizer.store.save_candidate(
        "c1_0", "rb", 1, candidate.to_dict(), candidate.hash(),
        "skills fix + education harm", "$.skills",
    )

    optimizer.front = [
        ArchiveEntry("root", 0.60, 0.70, root,
                     {"$.skills": {"f1": 0.2}, "$.education": {"f1": 1.0}}, "", ""),
        ArchiveEntry("c1_0", 0.75, 0.65, candidate,
                     {"$.skills": {"f1": 0.9}, "$.education": {"f1": 0.1}}, "delta", "$.skills"),
    ]
    optimizer._incumbent_cache["root"] = SplitEval(
        f1=0.60, aggregate_f1=0.70, json_valid_rate=1.0, per_doc={},
        merged_subtrees={"$.skills": {"f1": 0.2}, "$.education": {"f1": 1.0}},
    )

    async def fake_eval_split(artifact, split, iteration, tag):
        assert artifact.subtree_directives["$.education"] == ""
        assert artifact.subtree_directives["$.skills"] == "Always look in side projects."
        return SplitEval(
            f1=0.80, aggregate_f1=0.78, json_valid_rate=1.0,
            per_doc={}, merged_subtrees={"$.skills": {"f1": 0.9}, "$.education": {"f1": 0.9}},
        )

    optimizer._eval_split = fake_eval_split
    asyncio.run(optimizer._rollback_pass(root))

    rows = [r for r in optimizer.store.list_iterations("rb") if r["candidate_id"] == "rollback_best"]
    assert len(rows) == 1
    assert rows[0]["mutation_type"] == "subtree_rollback"
    assert rows[0]["target_subtree"] == "$.education"
    assert rows[0]["decision"] == "ACCEPTED"
    repaired_json = optimizer.store.load_candidate("rollback_best", "rb")
    repaired = PromptArtifact.from_dict(repaired_json)
    assert repaired.subtree_directives["$.education"] == ""
    assert repaired.subtree_directives["$.skills"] == "Always look in side projects."


def test_rollback_pass_skips_when_no_regression(tmp_path):
    config = make_config()
    optimizer = make_optimizer(tmp_path, config)
    optimizer.store.init_run("rb", "hash")

    root = PromptArtifact.init_from_schema(SCHEMA, "seed")
    candidate = root.copy("c1_0")
    optimizer.store.save_candidate("root", "rb", 0, root.to_dict(), root.hash())
    optimizer.front = [
        ArchiveEntry("root", 0.60, 0.70, root,
                     {"$.skills": {"f1": 0.2}, "$.education": {"f1": 1.0}}, "", ""),
        ArchiveEntry("c1_0", 0.75, 0.80, candidate,
                     {"$.skills": {"f1": 0.9}, "$.education": {"f1": 0.98}}, "d", "$.skills"),
    ]
    optimizer._incumbent_cache["root"] = SplitEval(
        f1=0.60, aggregate_f1=0.70, json_valid_rate=1.0, per_doc={},
        merged_subtrees={"$.skills": {"f1": 0.2}, "$.education": {"f1": 1.0}},
    )
    asyncio.run(optimizer._rollback_pass(root))
    assert [r for r in optimizer.store.list_iterations("rb") if r["candidate_id"] == "rollback_best"] == []
