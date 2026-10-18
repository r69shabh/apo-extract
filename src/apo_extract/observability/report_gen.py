"""Markdown run report: iteration table, Pareto front, subtree deltas, spend."""

from __future__ import annotations

import json
from pathlib import Path

from ..predictor.artifact import PromptArtifact
from .diff_tool import diff_artifacts, diff_scores, render_diff
from .store import RunStore


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    store = RunStore(run_dir)
    run_id = _infer_run_id(store)
    run = store.get_run(run_id) or {}
    summary_path = run_dir / "final" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    best_path = run_dir / "final" / "best_artifact.json"
    best_artifact = PromptArtifact.from_dict(json.loads(best_path.read_text())) if best_path.exists() else None

    lines = [f"# Run report — {run_id}", ""]
    spent_tokens = summary.get("spent_tokens", run.get("budget_spent_tokens")) or 0
    lines += [
        "## Status",
        f"- status: {run.get('status')}",
        f"- best candidate: {summary.get('best_candidate_id')}",
        f"- iterations executed: {run.get('current_iteration')}",
        f"- tokens spent: {int(spent_tokens):,}",
        f"- usd spent: {summary.get('spent_usd', run.get('budget_spent_usd'))}",
        "",
    ]

    if summary:
        lines += [
            "## Held-out test metrics (best artifact)",
            f"- micro F1: {summary.get('test_f1')}",
            f"- macro subtree F1: {summary.get('test_aggregate_f1')}",
            f"- JSON-valid rate: {summary.get('test_json_valid_rate')}",
            "",
        ]

    lines += ["## Iterations", "", "| iter | candidate | mutation | target | train F1 | val F1 | decision |", "|---|---|---|---|---|---|---|"]
    for row in store.list_iterations(run_id):
        train = f"{row['train_f1']:.4f}" if row["train_f1"] is not None else "—"
        val = f"{row['val_f1']:.4f}" if row["val_f1"] is not None else "—"
        lines.append(
            f"| {row['iteration']} | {row['candidate_id']} | {row['mutation_type']} | "
            f"{row['target_subtree'] or '—'} | {train} | {val} | {row['decision']} |"
        )
    lines.append("")

    lines += ["## Accepted hypotheses", ""]
    for row in store.list_iterations(run_id):
        if row["decision"] == "ACCEPTED":
            lines.append(f"- **{row['candidate_id']}** ({row['mutation_type']} → {row['target_subtree']}): {row['hypothesis']}")
    lines.append("")

    archive = store.load_archive(run_id)
    lines += ["## Pareto front", ""]
    for entry in archive:
        lines.append(
            f"- {entry['candidate_id']}: val F1 {entry['val_f1']:.4f}, aggregate {entry['aggregate_f1']:.4f}"
        )
    lines.append("")

    root_val_path = run_dir / "root_val.json"
    if root_val_path.exists() and archive:
        root_val = json.loads(root_val_path.read_text())
        best_entry = archive[0]
        deltas = diff_scores(root_val.get("merged_subtrees", {}), best_entry.get("subtree", {}))
        lines += ["## Subtree deltas (root → best, validation)", "", "| subtree | before | after | delta |", "|---|---|---|---|"]
        for path, stat in deltas.items():
            lines.append(
                f"| {path} | {stat['old_f1']} | {stat['new_f1']} | {stat['delta']:+.4f} |"
            )
        lines.append("")

        root_artifact_path = store.load_candidate("root", run_id)
        if best_artifact is not None and root_artifact_path:
            changes = diff_artifacts(PromptArtifact.from_dict(root_artifact_path), best_artifact)
            lines += ["## Prompt changes (root → best)", "", "```diff", render_diff(changes), "```", ""]

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    store.close()
    return report_path


def _infer_run_id(store: RunStore) -> str:
    row = store.conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return row["run_id"] if row else "unknown"
