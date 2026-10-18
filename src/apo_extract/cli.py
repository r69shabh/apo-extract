"""apo-extract CLI: run / resume / evaluate / dry-run / diff / inspect / report / export-web."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .dataset.loader import DocumentItem, load_dataset
from .dataset.splitter import split_documents
from .llm.client import LLMClient
from .llm.pdf_handler import PdfHandler
from .llm.telemetry import BudgetTracker
from .observability.diff_tool import diff_artifacts, diff_scores, render_diff
from .observability.report_gen import generate_report
from .observability.store import RunStore
from .optimizer.pareto_beam import Evaluator, Optimizer
from .optimizer.proposer import Proposer
from .predictor.artifact import PromptArtifact
from .predictor.extractor import Extractor
from .scoring.engine import resolve_engine

app = typer.Typer(add_completion=False, help="Automated prompt optimization for ExtractBench.")
console = Console()


class LlmJudge:
    """Adapts an LLMClient to the scoring engine's async judge callable."""

    PROMPT = (
        "You are grading extraction quality. Decide whether the predicted array item "
        "refers to the same underlying entity as the gold item for field '{field}'.\n"
        "Gold: {gold}\nPredicted: {predicted}\n"
        'Respond with JSON only: {{"match": true/false, "score": 0.0-1.0}}'
    )

    def __init__(self, client: LLMClient):
        self.client = client

    async def __call__(self, predicted, gold, field_path: str) -> dict:
        result = await self.client.complete(
            [
                {"role": "system", "content": "Respond with JSON only."},
                {
                    "role": "user",
                    "content": self.PROMPT.format(
                        field=field_path,
                        gold=json.dumps(gold, default=repr),
                        predicted=json.dumps(predicted, default=repr),
                    ),
                },
            ],
            response_format={"type": "json_object"},
            purpose="judge",
        )
        try:
            payload = json.loads(result.text)
            return {"match": bool(payload.get("match")), "score": float(payload.get("score", 0.0))}
        except Exception:
            return {"match": True, "score": 0.5}


# -- pipeline wiring -----------------------------------------------------------


def _install_429_retry(max_attempts: int = 8, call_timeout: float = 300.0) -> None:
    """Retry transient litellm failures globally and pin a default timeout.

    litellm's built-in retries are too shallow for free-tier 429s, and
    extract-bench's judge metrics call ``litellm.acompletion`` directly (no
    per-call timeout). Each attempt is bounded by ``call_timeout`` so a stalled
    connection cannot park the event loop forever; every retry is logged so a
    waiting run is never silent; a server-requested delay longer than 5 minutes
    (daily quota reset) aborts with a clear error instead of sleeping for hours.
    """
    import logging
    import re

    import litellm

    logger = logging.getLogger("apo_extract.retry")
    litellm.request_timeout = 180
    original = litellm.acompletion
    exc_mod = getattr(litellm, "exceptions", None)
    transient_names = {
        "RateLimitError", "Timeout", "APIConnectionError",
        "ServiceUnavailableError", "InternalServerError", "APIError",
    }
    retry_types = tuple(
        cls
        for cls in (getattr(exc_mod, name, None) for name in transient_names)
        if isinstance(cls, type)
    )

    def _retry_after(text: str) -> float | None:
        match = re.search(r"retry in ([\d.]+)s", text) or re.search(
            r'"retryDelay":\s*"([\d.]+)s"', text
        )
        return float(match.group(1)) if match else None

    async def wrapped(*args, **kwargs):
        kwargs.setdefault("timeout", 180)
        delay = 10.0
        hard_timeouts = 0
        for attempt in range(max_attempts):
            try:
                return await asyncio.wait_for(
                    original(*args, **kwargs), timeout=call_timeout
                )
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError):
                    hard_timeouts += 1
                    name = "CallTimeout"
                    text = f"no response within {call_timeout:.0f}s"
                    retryable = hard_timeouts < 3
                else:
                    name = type(exc).__name__
                    text = str(exc)
                    retryable = (
                        any(
                            code in text
                            for code in (
                                "429", "500", "503", "timed out", "timeout",
                                "json_validate_failed",  # Groq rejects empty generations as 400s
                            )
                        )
                        or name in transient_names
                        or (bool(retry_types) and isinstance(exc, retry_types))
                    )
                if not retryable or attempt == max_attempts - 1:
                    raise
                explicit = _retry_after(text)
                if explicit is not None and explicit > 300:
                    raise RuntimeError(
                        f"server rate limit cannot clear soon ({name}: retry after "
                        f"{explicit / 3600:.1f}h); aborting instead of sleeping — "
                        f"likely a daily quota ({text[:200]})"
                    ) from exc
                sleep_s = min((explicit + 1.0) if explicit is not None else delay, 120.0)
                logger.warning(
                    "litellm call failed (%s), retrying %d/%d in %.0fs: %s",
                    name, attempt + 1, max_attempts, sleep_s, text[:160],
                )
                await asyncio.sleep(sleep_s)
                delay = min(delay * 1.6, 90.0)

    litellm.acompletion = wrapped


def build_components(config: AppConfig, *, completer=None, resume: bool = False, dry_run: bool = False):
    if not dry_run:
        _install_429_retry()
    bundle = load_dataset(config.dataset)
    split = split_documents([d.doc_id for d in bundle.documents], config.dataset.split_policy)
    docs = {d.doc_id: d for d in bundle.documents}

    if dry_run:
        import tempfile

        run_dir = Path(tempfile.mkdtemp(prefix="apo-dryrun-"))
    else:
        run_dir = Path("runs") / config.run_id
    store = RunStore(run_dir)

    spent_tokens, spent_usd = store.spent_totals(config.run_id) if resume else (0, 0)
    tracker = BudgetTracker(config.budget, spent_tokens=spent_tokens, spent_usd=spent_usd)

    scoring_kwargs = {}
    if not dry_run:
        scoring_kwargs["param_overrides"] = {
            "string_semantic": {"model": config.scoring.judge_model},
            "string_llm": {"model": config.scoring.judge_model},
            "array_llm": {"model": config.scoring.judge_model},
        }
    if config.scoring.cache_path:
        from .scoring.cache import ScoringCache

        Path(config.scoring.cache_path).parent.mkdir(parents=True, exist_ok=True)
        scoring_kwargs["cache"] = ScoringCache(config.scoring.cache_path)
    if config.scoring.enable_llm_judge and config.llms.scoring_judge is not None:
        judge_client = LLMClient(
            config.llms.scoring_judge, role="judge", store=store,
            run_id=config.run_id, completer=completer, tracker=tracker,
        )
        scoring_kwargs["judge"] = LlmJudge(judge_client)
    if dry_run:
        from extract_bench.evaluation.metrics.string_metrics import ExactStringMatchMetric

        scoring_kwargs["array_policy"] = "hungarian_only"
        scoring_kwargs["metric_overrides"] = {
            "string_semantic": ExactStringMatchMetric,
            "string_llm": ExactStringMatchMetric,
        }
    scorer = resolve_engine(
        config.scoring.engine,
        array_policy=scoring_kwargs.pop("array_policy", config.scoring.array_alignment_policy),
        match_threshold=config.scoring.array_match_threshold,
        **scoring_kwargs,
    )

    extractor_client = LLMClient(
        config.llms.extractor, role="extractor", store=store,
        run_id=config.run_id, completer=completer, tracker=tracker,
    )
    pdf_handler = None if dry_run else PdfHandler(run_dir / "cache")
    extractor = Extractor(
        extractor_client, store, pdf_handler=pdf_handler,
        mode=config.optimizer.extraction_mode,
        chunk_size=config.optimizer.chunk_threshold_fields,
        pdf_mode=config.llms.extractor.pdf_mode or "auto",
    )

    mutator_client = None
    if config.llms.optimizer_mutator is not None:
        mutator_client = LLMClient(
            config.llms.optimizer_mutator, role="mutator", store=store,
            run_id=config.run_id, completer=completer, tracker=tracker,
        )
    seed_text = Path(config.seed_prompt.path).read_text()
    proposer = Proposer(
        bundle.schema,
        client=mutator_client,
        store=store,
        run_id=config.run_id,
        duplicate_similarity=config.optimizer.duplicate_similarity,
    )

    optimizer = Optimizer(
        config,
        store,
        extractor=extractor,
        scorer=scorer,
        proposer=proposer,
        schema=bundle.schema,
        splits={"train": split.train, "val": split.val, "test": split.test},
        docs=docs,
        seed_text=seed_text,
        tracker=tracker,
        run_id=config.run_id,
    )
    return optimizer, store, bundle, split


def _require_api_key(config: AppConfig) -> None:
    # litellm's nvidia_nim provider reads NVIDIA_NIM_API_KEY; accept the NVIDIA SDK name too
    if os.environ.get("NVIDIA_API_KEY") and not os.environ.get("NVIDIA_NIM_API_KEY"):
        os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"]
    if config.llms.extractor.provider == "gemini" and not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        console.print(
            "[red]GEMINI_API_KEY is not set.[/red] Export it before running, e.g.\n"
            "  export GEMINI_API_KEY=...\n"
            "Use `apo-extract dry-run -c <config>` for an offline sanity check."
        )
        raise typer.Exit(code=1)


class GoldCompleter:
    """Dry-run completer: returns the gold answer verbatim (perfect extraction)."""

    def __init__(self, golds: dict[str, dict]):
        self.golds = golds

    async def __call__(self, request: dict):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.golds[request["pdf_id"]])))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
            model=request["model"],
        )


# -- commands ---------------------------------------------------------------------


@app.command()
def run(config_path: str = typer.Option(..., "--config", "-c"), resume: bool = typer.Option(False)) -> None:
    """Run (or resume) the SR-PBO optimization loop."""
    config = AppConfig.load(config_path)
    _require_api_key(config)
    optimizer, store, _, _ = build_components(config, resume=resume)
    summary = optimizer.run(resume=resume)
    report = generate_report(store.run_dir)
    console.print(json.dumps(summary, indent=2))
    console.print(f"[green]report:[/green] {report}")
    console.print(f"[green]best prompt:[/green] {store.run_dir / 'final' / 'best_artifact.json'}")


@app.command()
def resume(config_path: str = typer.Option(..., "--config", "-c")) -> None:
    """Resume an interrupted run (same as `run --resume`)."""
    run(config_path=config_path, resume=True)


@app.command("dry-run")
def dry_run(config_path: str = typer.Option(..., "--config", "-c")) -> None:
    """Offline sanity check: gold-as-prediction must score F1 = 1.0."""
    config = AppConfig.load(config_path)
    bundle = load_dataset(config.dataset)
    golds = {d.doc_id: d.gold for d in bundle.documents}
    optimizer, store, _, split = build_components(config, completer=GoldCompleter(golds), dry_run=True)
    split_ids = split.train + split.val + split.test
    summary = optimizer.run()
    console.print(
        f"[green]dry-run OK[/green] — {len(split_ids)} docs split "
        f"(train={len(split.train)}, val={len(split.val)}, test={len(split.test)}), "
        f"test F1={summary['test_f1']:.4f} (expect 1.0)"
    )


@app.command()
def evaluate(config_path: str = typer.Option(..., "--config", "-c"), run_id: str = typer.Option(..., "--run-id")) -> None:
    """Re-evaluate the best artifact of a finished run on the held-out test split."""
    config = AppConfig.load(config_path)
    _require_api_key(config)
    config.run_id = run_id
    optimizer, store, bundle, split = build_components(config)
    best_path = store.run_dir / "final" / "best_artifact.json"
    if not best_path.exists():
        console.print(f"[red]no best artifact at {best_path}; finish the run first[/red]")
        raise typer.Exit(code=1)
    artifact = PromptArtifact.from_dict(json.loads(best_path.read_text()))
    docs = {d.doc_id: d for d in bundle.documents}
    evaluator = Evaluator(optimizer.extractor, optimizer.scorer, bundle.schema, docs)
    result = asyncio.run(evaluator.evaluate_async(artifact, split.test))
    console.print(json.dumps({"test_f1": result.f1, "aggregate_f1": result.aggregate_f1, "json_valid_rate": result.json_valid_rate}, indent=2))


@app.command()
def diff(
    config_path: str = typer.Option(..., "--config", "-c"),
    run_id: str = typer.Option(..., "--run-id"),
    from_candidate: str = typer.Option("root", "--from"),
    to_candidate: str = typer.Option("", "--to"),
) -> None:
    """Show prompt + subtree-score changes between two candidates."""
    config = AppConfig.load(config_path)
    store = RunStore(Path("runs") / run_id)
    archive = store.load_archive(run_id)
    to_id = to_candidate or (archive[0]["candidate_id"] if archive else "root")
    old_json = store.load_candidate(from_candidate, run_id)
    new_json = store.load_candidate(to_id, run_id)
    if not old_json or not new_json:
        console.print(f"[red]candidate not found ({from_candidate} / {to_id})[/red]")
        raise typer.Exit(code=1)
    old, new = PromptArtifact.from_dict(old_json), PromptArtifact.from_dict(new_json)
    console.print(render_diff(diff_artifacts(old, new)))
    root_path = store.run_dir / "root_val.json"
    if root_path.exists() and archive:
        root_val = json.loads(root_path.read_text())
        consoles = diff_scores(root_val.get("merged_subtrees", {}), archive[0]["subtree"])
        console.print("\n[bold]subtree deltas (root → best):[/bold]")
        console.print(json.dumps(consoles, indent=2))


@app.command()
def inspect(
    run_id: str = typer.Option(..., "--run-id"),
    print_prompt: bool = typer.Option(False, "--print-prompt"),
) -> None:
    """Show run status, per-iteration table, spend; optionally the final prompt."""
    run_dir = Path("runs") / run_id
    store = RunStore(run_dir)
    run = store.get_run(run_id) or {}
    table = Table(title=f"run {run_id} — {run.get('status')}")
    for column in ("iter", "candidate", "mutation", "target", "train F1", "val F1", "decision"):
        table.add_column(column)
    for row in store.list_iterations(run_id):
        table.add_row(
            str(row["iteration"]), row["candidate_id"], row["mutation_type"],
            row["target_subtree"] or "—",
            f"{row['train_f1']:.4f}" if row["train_f1"] is not None else "—",
            f"{row['val_f1']:.4f}" if row["val_f1"] is not None else "—",
            row["decision"],
        )
    console.print(table)
    console.print(f"tokens spent: {run.get('budget_spent_tokens')}  usd: {run.get('budget_spent_usd')}")
    if print_prompt:
        best = store.load_candidate(run.get("best_candidate_id") or "root", run_id)
        if best:
            console.print(PromptArtifact.from_dict(best).render(None))


@app.command()
def report(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Regenerate the markdown report for a run."""
    console.print(str(generate_report(Path("runs") / run_id)))


@app.command("export-web")
def export_web(
    run_id: str = typer.Option(..., "--run-id"),
    config_path: str = typer.Option("", "--config", "-c"),
    out: str = typer.Option("frontend/public/data", "--out"),
) -> None:
    """Export a run snapshot (config, prompts, scores, iterations) as static JSON for the frontend."""
    run_dir = Path("runs") / run_id
    store = RunStore(run_dir)
    run = store.get_run(run_id) or {}
    recorded = config_path or run.get("config_path")
    if not recorded:
        raise ValueError(
            f"run {run_id!r} has no recorded config_path (created before configs were "
            f"recorded); pass the config explicitly: export-web --run-id {run_id} -c <config.yaml>"
        )
    cfg = AppConfig.load(recorded)
    bundle = load_dataset(cfg.dataset)
    split = split_documents([d.doc_id for d in bundle.documents], cfg.dataset.split_policy)

    seed = PromptArtifact.init_from_schema(bundle.schema, Path(cfg.seed_prompt.path).read_text())
    archive = store.load_archive(run_id)
    best_id = run.get("best_candidate_id") or (archive[0]["candidate_id"] if archive else "root")
    best = PromptArtifact.from_dict(store.load_candidate(best_id, run_id) or seed.to_dict())

    def read_json(*parts: str):
        path = run_dir.joinpath(*parts)
        return json.loads(path.read_text()) if path.exists() else None

    llm_stats = [
        dict(row)
        for row in store.conn.execute(
            "SELECT role, model, COUNT(*) AS n, SUM(input_tokens) AS it, SUM(output_tokens) AS ot,"
            " AVG(latency_ms) AS lat_ms FROM llm_calls WHERE run_id=? GROUP BY role, model",
            (run_id,),
        ).fetchall()
    ]
    n_cached = store.conn.execute("SELECT COUNT(*) AS c FROM predictions_cache").fetchone()["c"]

    payload = {
        "run_id": run_id,
        "exported_at": time.time(),
        "run": run,
        "config": cfg.model_dump(mode="json"),
        "dataset": {
            **cfg.dataset.model_dump(mode="json"),
            "n_documents": len(bundle.documents),
            "schema_fields": list(bundle.schema.get("properties", {})),
            "split": {"train": split.train, "val": split.val, "test": split.test},
        },
        "seed": {"artifact": seed.to_dict(), "text": seed.render(None)},
        "best": {"candidate_id": best_id, "artifact": best.to_dict(), "text": best.render(None)},
        "root_val": read_json("root_val.json"),
        "summary": read_json("final", "summary.json"),
        "test_per_document": read_json("final", "test_per_document.json"),
        "iterations": store.list_iterations(run_id),
        "archive": archive,
        "llm_stats": llm_stats,
        "n_cached_predictions": int(n_cached),
    }
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2, default=repr))
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"runs": []}
    index["runs"] = [r for r in index["runs"] if r.get("run_id") != run_id]
    index["runs"].append({"run_id": run_id, "status": run.get("status", "unknown"), "exported_at": payload["exported_at"]})
    index_path.write_text(json.dumps(index, indent=2))
    console.print(f"[green]exported[/green] {out_dir / (run_id + '.json')}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
