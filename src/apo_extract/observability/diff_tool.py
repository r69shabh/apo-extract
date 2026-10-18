"""Artifact/score diffing for iteration summaries, ``apo-extract diff`` and reports."""

from __future__ import annotations

import difflib

from ..predictor.artifact import PromptArtifact


def diff_artifacts(old: PromptArtifact, new: PromptArtifact) -> dict:
    changes: dict[str, list[dict]] = {"added": [], "removed": [], "changed": []}
    old_sections = {
        "global_preamble": old.global_preamble,
        "formatting_and_null_policy": old.formatting_and_null_policy,
        "verification_checklist": old.verification_checklist,
    }
    new_sections = {
        "global_preamble": new.global_preamble,
        "formatting_and_null_policy": new.formatting_and_null_policy,
        "verification_checklist": new.verification_checklist,
    }
    for key in old_sections:
        if old_sections[key] != new_sections[key]:
            changes["changed"].append(
                {"section": key, "old": old_sections[key], "new": new_sections[key]}
            )

    old_keys, new_keys = set(old.subtree_directives), set(new.subtree_directives)
    for key in sorted(new_keys - old_keys):
        if new.subtree_directives[key].strip():
            changes["added"].append({"section": key, "text": new.subtree_directives[key]})
    for key in sorted(old_keys - new_keys):
        if old.subtree_directives[key].strip():
            changes["removed"].append({"section": key, "text": old.subtree_directives[key]})
    for key in sorted(old_keys & new_keys):
        if old.subtree_directives[key] != new.subtree_directives[key]:
            changes["changed"].append(
                {
                    "section": key,
                    "old": old.subtree_directives[key],
                    "new": new.subtree_directives[key],
                }
            )
    return changes


def diff_scores(old_subtrees: dict, new_subtrees: dict) -> dict:
    """Per-subtree F1 deltas between two merged subtree-score dicts."""
    deltas = {}
    for path in sorted(set(old_subtrees) | set(new_subtrees)):
        if path.count(".") != 1:
            continue
        old_f1 = float(old_subtrees.get(path, {}).get("f1", 0.0))
        new_f1 = float(new_subtrees.get(path, {}).get("f1", 0.0))
        deltas[path] = {
            "old_f1": round(old_f1, 4),
            "new_f1": round(new_f1, 4),
            "delta": round(new_f1 - old_f1, 4),
        }
    return deltas


def render_diff(changes: dict) -> str:
    lines: list[str] = []
    for section in ("added", "removed", "changed"):
        for change in changes[section]:
            if section == "changed":
                diff = difflib.unified_diff(
                    str(change["old"]).splitlines(),
                    str(change["new"]).splitlines(),
                    lineterm="",
                    n=1,
                )
                lines.append(f"### {change['section']}")
                lines.extend(line for line in diff if not line.startswith("---") and not line.startswith("+++"))
            else:
                lines.append(f"### {section.upper()}: {change['section']}")
                lines.append(change["text"])
    return "\n".join(lines) if lines else "(no changes)"
