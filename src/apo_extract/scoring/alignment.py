"""Array alignment policy: Hungarian bipartite matching + optional LLM refinement.

Tier 1 is fully deterministic: a similarity matrix over (predicted, gold) items
is built recursively from leaf metrics, then solved with the Hungarian algorithm.
Tier 2 (optional) asks a cached LLM judge to refine ambiguous pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .cache import ScoringCache, cache_key

DEFAULT_MATCH_THRESHOLD = 0.5
DEFAULT_AMBIGUOUS_LOW = 0.30
DEFAULT_AMBIGUOUS_HIGH = 0.85


@dataclass
class AlignmentResult:
    matched: list[tuple[int, int, float]] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    spurious: list[int] = field(default_factory=list)
    soft_f1: float = 0.0
    hard_f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


def hungarian_assignment(similarity: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-weight bipartite matching via scipy's Hungarian implementation."""
    if not similarity or not similarity[0]:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(similarity, dtype=float)
    rows, cols = linear_sum_assignment(-matrix)
    return list(zip(rows.tolist(), cols.tolist()))


def classify_alignment(
    similarity: list[list[float]],
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> AlignmentResult:
    """Turn a similarity matrix into matched/missed/spurious P/R/F1."""
    m = len(similarity)
    n = len(similarity[0]) if m else 0
    result = AlignmentResult()
    if m == 0 and n == 0:
        result.soft_f1 = result.hard_f1 = result.precision = result.recall = 1.0
        return result

    matched_pairs = []
    for i, j in hungarian_assignment(similarity):
        score = float(similarity[i][j])
        if score >= match_threshold:
            matched_pairs.append((i, j, score))
        else:
            result.missed.append(j)
            result.spurious.append(i)

    matched_rows = {i for i, _, _ in matched_pairs}
    matched_cols = {j for _, j, _ in matched_pairs}
    result.matched = matched_pairs
    result.missed.extend(j for j in range(n) if j not in matched_cols)
    result.spurious.extend(i for i in range(m) if i not in matched_rows)

    soft_tp = sum(s for _, _, s in matched_pairs)
    hard_tp = float(len(matched_pairs))
    fp = float(len(result.spurious))
    fn = float(len(result.missed))

    result.precision = soft_tp / (soft_tp + fp) if soft_tp + fp > 0 else (1.0 if fp == 0 else 0.0)
    denom_r = soft_tp + fn
    result.recall = soft_tp / denom_r if denom_r > 0 else 1.0
    result.soft_f1 = (
        2 * result.precision * result.recall / (result.precision + result.recall)
        if result.precision + result.recall > 0
        else 0.0
    )

    hard_p = hard_tp / (hard_tp + fp) if hard_tp + fp > 0 else (1.0 if fp == 0 else 0.0)
    hard_r = hard_tp / (hard_tp + fn) if hard_tp + fn > 0 else 1.0
    result.hard_f1 = 2 * hard_p * hard_r / (hard_p + hard_r) if hard_p + hard_r > 0 else 0.0
    return result


async def refine_ambiguous_pairs(
    alignment: AlignmentResult,
    similarity: list[list[float]],
    predicted: Sequence[Any],
    gold: Sequence[Any],
    *,
    judge: Any,
    field_path: str,
    cache: ScoringCache | None,
    low: float = DEFAULT_AMBIGUOUS_LOW,
    high: float = DEFAULT_AMBIGUOUS_HIGH,
) -> None:
    """Tier 2: let the LLM judge confirm/reject ambiguous matched pairs in place.

    ``judge`` is an async callable ``(predicted_item, gold_item, field_path) -> dict``
    returning ``{"match": bool, "score": float}``. Results are disk-cached.
    """
    kept: list[tuple[int, int, float]] = []
    demoted: list[tuple[int, int]] = []
    for i, j, score in alignment.matched:
        if not (low < score < high):
            kept.append((i, j, score))
            continue
        key = cache_key("array_judge", predicted[i], gold[j], field_path)
        verdict = cache.get(key) if cache else None
        if verdict is None:
            try:
                verdict = await judge(predicted[i], gold[j], field_path)
            except Exception:
                verdict = {"match": True, "score": score}
            if cache:
                cache.set(key, verdict)
        if verdict.get("match"):
            kept.append((i, j, float(verdict.get("score", score))))
        else:
            demoted.append((i, j))

    if not demoted:
        return

    demoted_rows = {i for i, _ in demoted}
    demoted_cols = {j for _, j in demoted}
    alignment.matched = kept
    alignment.missed.extend(demoted_cols)
    alignment.spurious.extend(demoted_rows)

    soft_tp = sum(s for _, _, s in kept)
    fp = float(len(alignment.spurious))
    fn = float(len(alignment.missed))
    alignment.precision = soft_tp / (soft_tp + fp) if soft_tp + fp > 0 else (1.0 if fp == 0 else 0.0)
    denom_r = soft_tp + fn
    alignment.recall = soft_tp / denom_r if denom_r > 0 else 1.0
    alignment.soft_f1 = (
        2 * alignment.precision * alignment.recall / (alignment.precision + alignment.recall)
        if alignment.precision + alignment.recall > 0
        else 0.0
    )
    hard_tp = float(len(kept))
    hard_p = hard_tp / (hard_tp + fp) if hard_tp + fp > 0 else (1.0 if fp == 0 else 0.0)
    hard_r = hard_tp / (hard_tp + fn) if hard_tp + fn > 0 else 1.0
    alignment.hard_f1 = 2 * hard_p * hard_r / (hard_p + hard_r) if hard_p + hard_r > 0 else 0.0
