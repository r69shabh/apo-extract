"""Metric layer: extract-bench registry + caching + Hungarian array alignment.

Per-leaf metric *semantics* come from the vendored extract-bench package
(compliance by construction). On top of that this module adds:
  - disk caching for stochastic LLM-judge metrics (keyed on values + path),
  - a deterministic Hungarian-matching array metric (``array_llm`` policy
    ``hungarian_plus_llm_cache``) with optional cached LLM refinement of
    ambiguous pairs and per-item breakdowns for diagnostics.
"""

from __future__ import annotations

from contextvars import ContextVar
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Awaitable, Callable

from extract_bench.evaluation.evaluation_config import MetricConfig
from extract_bench.evaluation.metric_registry import MetricRegistry
from extract_bench.evaluation.metrics.array_metrics import ArrayLlmJudgeMetric
from extract_bench.evaluation.metrics.base_metric import BaseMetric, MetricContext, MetricResult
from extract_bench.evaluation.metrics.boolean_metrics import BooleanExactMatchMetric
from extract_bench.evaluation.metrics.llm_metrics import LlmJudgeMetric
from extract_bench.evaluation.metrics.metric_utils import apply_missing_null_policy
from extract_bench.evaluation.metrics.number_metrics import (
    ExactNumberMatchMetric,
    IntegerExactMatchMetric,
    NumberToleranceMetric,
)
from extract_bench.evaluation.metrics.policy_metric import PolicyAwareMetric
from extract_bench.evaluation.metrics.string_metrics import (
    CaseInsensitiveStringMatchMetric,
    ExactStringMatchMetric,
    NormalizedLevenshteinSimilarityMetric,
    StringSemanticMetric,
    UrlNormalizedStringMatchMetric,
)
from extract_bench.evaluation.schema_value_instantiator import MISSING

from .alignment import (
    classify_alignment,
    refine_ambiguous_pairs,
)
from .cache import ScoringCache, cache_key
from .schema_walker import (
    anyof_branch_specs,
    item_leaf_specs,
    resolve_anyof_branch,
)

PATH_CONTEXT: ContextVar[str] = ContextVar("metric_field_path", default="$")

_SEMANTIC_METRIC_IDS = {"string_semantic", "string_llm"}


class CachedMetric:
    """Wraps an LLM-judge metric with a deterministic disk cache."""

    def __init__(self, inner: BaseMetric, cache: ScoringCache):
        self.inner = inner
        self.cache = cache
        self.metric_id = inner.metric_id
        self.recurse_into_children = getattr(inner, "recurse_into_children", True)

    async def evaluate(self, node: MetricContext, config: MetricConfig | None = None) -> MetricResult:
        key = cache_key(
            "metric",
            self.metric_id,
            PATH_CONTEXT.get(),
            node.get_gold_value(),
            node.get_extracted_value(),
            config.params if config else None,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return MetricResult(**cached)
        result = await self.inner.evaluate(node, config)
        self.cache.set(
            key,
            {
                "metric_id": result.metric_id,
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
            },
        )
        return result


def _to_number(value: Any) -> float | None:
    if value is None or value is MISSING:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _normalize_url(url: str) -> str:
    import re

    s = url.strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    return s.split("#")[0]


_MAX_SEQUENCE_MATCH_CHARS = 128


@lru_cache(maxsize=50_000)
def _ratio(a: str, b: str) -> float:
    # SequenceMatcher is quadratic in string length and the alignment matrix
    # compares every (pred, gold) item pair, so keep it to short strings and
    # use a linear word-set Jaccard above the cap.
    if len(a) > _MAX_SEQUENCE_MATCH_CHARS or len(b) > _MAX_SEQUENCE_MATCH_CHARS:
        import re

        words_a = set(re.findall(r"\w+", a.lower()))
        words_b = set(re.findall(r"\w+", b.lower()))
        if not words_a and not words_b:
            return 1.0
        return len(words_a & words_b) / len(words_a | words_b)
    return SequenceMatcher(None, a, b).ratio()


def fast_leaf_similarity(metric_id: str, params: dict | None, gold: Any, pred: Any) -> float:
    """Deterministic similarity used to build the Hungarian matrix.

    Semantic/LLM metrics fall back to normalized Levenshtein similarity so
    alignment never requires a model call; refinement happens in Tier 2.
    """
    if gold is MISSING or pred is MISSING:
        return 1.0 if gold is pred else 0.0
    if gold is None or pred is None:
        return 1.0 if gold is pred else 0.0

    params = params or {}
    if metric_id == "anyof_typed":
        branch = resolve_anyof_branch(params.get("branches", []), gold, pred)
        if branch is None:
            return 1.0 if gold == pred else 0.0
        return fast_leaf_similarity(branch["metric_id"], branch["params"], gold, pred)
    if metric_id == "string_exact":
        return 1.0 if str(gold) == str(pred) else 0.0
    if metric_id == "string_case_insensitive":
        return 1.0 if str(gold).casefold() == str(pred).casefold() else 0.0
    if metric_id == "string_url":
        return 1.0 if _normalize_url(str(gold)) == _normalize_url(str(pred)) else 0.0
    if metric_id == "string_fuzzy" or metric_id in _SEMANTIC_METRIC_IDS:
        case_sensitive = bool(params.get("case_sensitive", False))
        a, b = str(gold), str(pred)
        if not case_sensitive:
            a, b = a.lower(), b.lower()
        return _ratio(a, b)
    if metric_id in ("number_exact", "integer_exact"):
        g, p = _to_number(gold), _to_number(pred)
        if g is None or p is None:
            return 1.0 if g is p else 0.0
        return 1.0 if g == p else 0.0
    if metric_id == "number_tolerance":
        g, p = _to_number(gold), _to_number(pred)
        if g is None or p is None:
            return 1.0 if g is p else 0.0
        tolerance = float(params.get("tolerance", 1e-6)) or 1e-12
        delta = abs(g - p)
        return 1.0 if delta <= tolerance else max(0.0, 1.0 - delta / tolerance)
    if metric_id == "boolean_exact":
        if isinstance(gold, bool) and isinstance(pred, bool):
            return 1.0 if gold == pred else 0.0
        return 1.0 if str(gold).casefold() == str(pred).casefold() else 0.0
    return 1.0 if str(gold) == str(pred) else 0.0


def values_at(item: Any, keys: list[str]) -> list[Any]:
    """Resolve a leaf key chain ("*" expands list indices) to a flat value list."""
    values = [item]
    for key in keys:
        next_values: list[Any] = []
        for value in values:
            if key == "*":
                if isinstance(value, list):
                    next_values.extend(value)
                else:
                    next_values.append(MISSING)
            elif isinstance(value, dict):
                next_values.append(value.get(key, MISSING))
            else:
                next_values.append(MISSING)
        values = next_values
    return values


def _pair_value_similarity(metric_id: str, params: dict | None, gold_values: list[Any], pred_values: list[Any]) -> float:
    """Best-pairing mean similarity for (possibly multi-valued) leaves."""
    if not gold_values and not pred_values:
        return 1.0
    if not gold_values or not pred_values:
        return 0.0
    pairs = sorted(
        (
            (fast_leaf_similarity(metric_id, params, g, p), gi, pi)
            for gi, g in enumerate(gold_values)
            for pi, p in enumerate(pred_values)
        ),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    total = 0.0
    for score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        total += score
    return total / max(len(gold_values), len(pred_values))


def item_similarity(pred_item: Any, gold_item: Any, specs: list[tuple[str, str, dict | None, list[str]]]) -> float:
    if not specs:
        return 1.0 if pred_item == gold_item else 0.0
    scores = [
        _pair_value_similarity(metric_id, params, values_at(gold_item, keys), values_at(pred_item, keys))
        for _, metric_id, params, keys in specs
    ]
    return sum(scores) / len(scores)


PairScorer = Callable[[Any, Any, Any, str], Awaitable[list[dict]]]
ArrayJudge = Callable[[Any, Any, str], Awaitable[dict]]


class HungarianArrayMetric(PolicyAwareMetric, BaseMetric):
    """Deterministic array evaluation with optional LLM refinement (Tier 1 + 2)."""

    metric_id = "array_llm"
    recurse_into_children = False

    def __init__(
        self,
        *,
        pair_scorer: PairScorer | None = None,
        judge: ArrayJudge | None = None,
        cache: ScoringCache | None = None,
        match_threshold: float = 0.5,
    ):
        self.pair_scorer = pair_scorer
        self.judge = judge
        self.cache = cache
        self.match_threshold = match_threshold

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        gold_list = gold if isinstance(gold, list) else [gold]
        pred_list = extracted if isinstance(extracted, list) else [extracted]
        path = PATH_CONTEXT.get()
        specs = item_leaf_specs(node, path)

        similarity = [
            [item_similarity(pred, gold_item, specs) for gold_item in gold_list]
            for pred in pred_list
        ]
        alignment = classify_alignment(similarity, match_threshold=self.match_threshold)

        if self.judge is not None:
            await refine_ambiguous_pairs(
                alignment,
                similarity,
                pred_list,
                gold_list,
                judge=self.judge,
                field_path=path,
                cache=self.cache,
            )

        item_scores: list[dict] = []
        if self.pair_scorer is not None:
            for i, j, sim in alignment.matched:
                leaves = await self.pair_scorer(pred_list[i], gold_list[j], node, path)
                item_scores.append(
                    {"pred_index": i, "gold_index": j, "similarity": round(sim, 6), "leaves": leaves}
                )

        params = config.params if config and config.params else {}
        pass_threshold = float(params.get("pass_threshold", 0.7))
        details = {
            "policy": "hungarian+llm" if self.judge is not None else "hungarian",
            "alignment": {
                "matched": len(alignment.matched),
                "missed": len(alignment.missed),
                "spurious": len(alignment.spurious),
                "soft_f1": round(alignment.soft_f1, 6),
                "hard_f1": round(alignment.hard_f1, 6),
                "precision": round(alignment.precision, 6),
                "recall": round(alignment.recall, 6),
            },
            "matched_pairs": [
                {"pred_index": i, "gold_index": j, "similarity": round(s, 6)}
                for i, j, s in alignment.matched
            ],
            "missed_gold_indices": list(alignment.missed),
            "spurious_pred_indices": list(alignment.spurious),
            "item_scores": item_scores,
            "gold_count": len(gold_list),
            "pred_count": len(pred_list),
        }
        return MetricResult(
            metric_id=self.metric_id,
            score=alignment.soft_f1,
            passed=alignment.soft_f1 >= pass_threshold,
            details=details,
        )


class AnyOfTypedMetric(PolicyAwareMetric, BaseMetric):
    """Evaluates a config-less anyOf by delegating to the value-type-matching branch."""

    metric_id = "anyof_typed"
    recurse_into_children = False

    def __init__(self, registry: MetricRegistry):
        self.registry = registry

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        branches: list[dict]
        params = config.params if config and config.params else None
        if params and params.get("branches"):
            branches = params["branches"]
        else:
            branches = anyof_branch_specs(node)
        branch = resolve_anyof_branch(branches, gold, extracted)
        if branch is None:
            passed = gold == extracted
            return MetricResult(metric_id=self.metric_id, score=1.0 if passed else 0.0, passed=passed)

        child = self._matching_child(node, branch, gold, extracted)
        if child is None:
            score = fast_leaf_similarity(branch["metric_id"], branch["params"], gold, extracted)
            return MetricResult(metric_id=self.metric_id, score=score, passed=score == 1.0)

        child.set_gold_value(gold)
        child.set_extracted_value(extracted)
        sub_metric = self.registry.create_metric(branch["metric_id"])
        sub_config = MetricConfig(metric_id=branch["metric_id"], params=branch["params"])
        result = await sub_metric.evaluate(child, sub_config)
        return MetricResult(
            metric_id=self.metric_id,
            score=result.score,
            passed=result.passed,
            details={"branch": branch["type"], "delegated_to": branch["metric_id"]},
        )

    def _matching_child(self, node: MetricContext, branch: dict, gold: Any, extracted: Any) -> Any:
        for child in getattr(node, "any_of", []) or []:
            if child.get_type() == branch["type"]:
                return child
        return None


_LEAF_METRIC_CLASSES = (
    ExactStringMatchMetric,
    CaseInsensitiveStringMatchMetric,
    UrlNormalizedStringMatchMetric,
    NormalizedLevenshteinSimilarityMetric,
    StringSemanticMetric,
    ExactNumberMatchMetric,
    NumberToleranceMetric,
    IntegerExactMatchMetric,
    BooleanExactMatchMetric,
    LlmJudgeMetric,
    ArrayLlmJudgeMetric,
)


def build_registry(
    *,
    array_policy: str = "hungarian_plus_llm_cache",
    cache: ScoringCache | None = None,
    pair_scorer: PairScorer | None = None,
    judge: ArrayJudge | None = None,
    match_threshold: float = 0.5,
    metric_overrides: dict[str, Any] | None = None,
) -> MetricRegistry:
    registry = MetricRegistry()
    for cls in _LEAF_METRIC_CLASSES:
        registry.register_metric(cls)
    registry.register_metric_factory("anyof_typed", lambda: AnyOfTypedMetric(registry))

    for metric_id, factory in (metric_overrides or {}).items():
        registry.register_metric_factory(metric_id, factory, override=True)

    if array_policy != "extract_bench_default" and "array_llm" not in (metric_overrides or {}):
        registry.register_metric_factory(
            "array_llm",
            lambda: HungarianArrayMetric(
                pair_scorer=pair_scorer,
                judge=judge,
                cache=cache,
                match_threshold=match_threshold,
            ),
            override=True,
        )

    if cache is not None:
        for mid, cls in (("string_semantic", StringSemanticMetric), ("string_llm", LlmJudgeMetric)):
            if mid in (metric_overrides or {}):
                continue
            registry.register_metric_factory(
                mid, lambda cls=cls: CachedMetric(cls(), cache), override=True
            )
    return registry
