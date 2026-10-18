"""Report generation + diff tooling against a synthetic run dir."""

import json

from apo_extract.observability.diff_tool import diff_artifacts, diff_scores, render_diff
from apo_extract.observability.report_gen import generate_report
from apo_extract.observability.store import IterationRecord, RunStore
from apo_extract.predictor.artifact import PromptArtifact

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}}


def test_artifact_diff_detects_changes():
    old = PromptArtifact.init_from_schema(SCHEMA, "seed")
    new = old.copy("c1")
    new.subtree_directives["$.name"] = "Copy the name verbatim."
    new.verification_checklist = "- Recheck nulls."
    changes = diff_artifacts(old, new)
    assert any(c["section"] == "$.name" for c in changes["changed"])
    assert any(c["section"] == "verification_checklist" for c in changes["changed"])
    assert "Copy the name verbatim." in render_diff(changes)


def test_score_diff_reports_deltas():
    deltas = diff_scores(
        {"$.name": {"f1": 0.5}, "$.amount": {"f1": 1.0}},
        {"$.name": {"f1": 0.9}, "$.amount": {"f1": 1.0}},
    )
    assert deltas["$.name"]["delta"] == 0.4
    assert deltas["$.amount"]["delta"] == 0.0


def test_generate_report(tmp_path):
    store = RunStore(tmp_path / "run")
    store.init_run("r1", "hash")
    root = PromptArtifact.init_from_schema(SCHEMA, "seed")
    best = root.copy("c1_0")
    best.subtree_directives["$.name"] = "Copy verbatim."
    store.save_candidate("root", "r1", 0, root.to_dict(), root.hash())
    store.save_candidate("c1_0", "r1", 1, best.to_dict(), best.hash(), "Copy verbatim.", "$.name")
    store.add_to_archive("r1", "c1_0", 0.9, 0.8, {"$.name": {"f1": 0.9}})
    store.log_iteration(
        "r1",
        IterationRecord(
            iteration=1, candidate_id="c1_0", parent_id="root",
            mutation_type="directive_mismatch", target_subtree="$.name",
            hypothesis="Copy names verbatim.", train_f1=0.85, val_f1=0.9,
            decision="ACCEPTED", decision_reason="ok",
        ),
    )
    (store.run_dir / "root_val.json").write_text(json.dumps({"merged_subtrees": {"$.name": {"f1": 0.4}}}))
    final = store.run_dir / "final"
    final.mkdir()
    (final / "summary.json").write_text(
        json.dumps(
            {
                "best_candidate_id": "c1_0", "test_f1": 0.95, "test_aggregate_f1": 0.9,
                "test_json_valid_rate": 1.0, "spent_tokens": 1234, "spent_usd": 0.01,
            }
        )
    )
    (final / "best_artifact.json").write_text(json.dumps(best.to_dict()))
    store.close()

    report_path = generate_report(tmp_path / "run")
    text = report_path.read_text()
    assert "# Run report" in text
    assert "c1_0" in text
    assert "Copy names verbatim." in text
    assert "| $.name | 0.4 | 0.9 | +0.5000 |" in text
    assert "test_f1" not in text  # summary rendered as readable lines, not raw json
    assert "0.95" in text
