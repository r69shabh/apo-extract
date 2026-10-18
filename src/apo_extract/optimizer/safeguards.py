"""Safeguards: duplicate suppression, Pareto archive, staged gates, budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..predictor.artifact import PromptArtifact


def jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_duplicate(candidate_text: str, reference_texts: list[str], threshold: float) -> bool:
    return any(jaccard(candidate_text, ref) >= threshold for ref in reference_texts)


def is_duplicate_for_target(
    target: str, delta: str, references: list[tuple[str, str]], threshold: float
) -> bool:
    """True if the same edit was already applied to the same target section."""
    return any(
        ref_target == target and jaccard(delta, ref_delta) >= threshold
        for ref_target, ref_delta in references
    )


def stage1_gate(train_f1: float, incumbent_train_f1: float, max_regression: float) -> bool:
    """Gate A: reject candidates that regress training more than allowed."""
    return train_f1 >= incumbent_train_f1 - max_regression


def regressed_subtrees(
    old_subtrees: dict[str, dict], new_subtrees: dict[str, dict], threshold: float
) -> list[str]:
    out = []
    for path, old_bucket in old_subtrees.items():
        if path.count(".") != 1:
            continue
        new_f1 = float(new_subtrees.get(path, {}).get("f1", 0.0))
        old_f1 = float(old_bucket.get("f1", 0.0))
        if old_f1 - new_f1 > threshold:
            out.append(path)
    return out


@dataclass
class ArchiveEntry:
    candidate_id: str
    val_f1: float
    aggregate_f1: float
    artifact: PromptArtifact
    subtree_scores: dict = field(default_factory=dict)
    delta_text: str = ""
    target_subtree: str = ""


def dominates(a: ArchiveEntry, b: ArchiveEntry, eps: float = 1e-9) -> bool:
    return (
        a.val_f1 >= b.val_f1 + eps
        and a.aggregate_f1 >= b.aggregate_f1 + eps
        and (a.val_f1 > b.val_f1 + eps or a.aggregate_f1 > b.aggregate_f1 + eps)
    )


def add_to_front(front: list[ArchiveEntry], entry: ArchiveEntry) -> list[ArchiveEntry]:
    """Insert unless dominated; prune entries the newcomer dominates."""
    for existing in front:
        if existing.candidate_id == entry.candidate_id:
            return front
        if dominates(existing, entry):
            return front
    kept = [e for e in front if not dominates(entry, e)]
    kept.append(entry)
    kept.sort(key=lambda e: (-e.val_f1, -e.aggregate_f1))
    return kept


def selection_key(entry: ArchiveEntry, tolerance: float) -> tuple:
    """Discretize val_f1 into tolerance-wide bands; aggregate breaks ties."""
    band = max(tolerance, 1e-6)
    return (round(entry.val_f1 / band), entry.aggregate_f1, -entry.artifact.estimate_tokens())


def best_selection(front: list[ArchiveEntry], tolerance: float) -> ArchiveEntry:
    return max(front, key=lambda e: selection_key(e, tolerance))
