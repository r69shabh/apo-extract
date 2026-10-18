"""Scoring engine: per-leaf/subtree P/R/F1 over extract-bench-compliant metrics.

Independent of ``optimizer`` and ``predictor`` (zero imports from them).
Public interface: ``score(predicted, expected, schema) -> ScoreResult``.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from extract_bench.evaluation.metric_registry import MetricRegistry
from extract_bench.evaluation.metrics.base_metric import MetricResult
from extract_bench.evaluation.schema_value_instantiator import MISSING
from extract_bench.infra.nodes import ArraySchema

from .cache import ScoringCache, cache_key
from .metrics import PATH_CONTEXT, build_registry
from .schema_walker import SchemaWalker, WalkNode


def _json_safe(value: Any) -> Any:
    return None if value is MISSING else value


def _aggregate(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else (1.0 if fp == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else (1.0 if fn == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


@dataclass
class NodeScore:
    path: str
    metric_id: str
    score: float
    passed: bool | None
    tp: float
    fp: float
    fn: float
    gold: Any = None
    extracted: Any = None
    reason: str | None = None
    error: str | None = None
    is_item_leaf: bool = False

    def to_dict(self) -> dict:
        return {k: _json_safe(v) for k, v in asdict(self).items()}


@dataclass
class ScoreResult:
    micro_f1: float = 0.0
    macro_subtree_f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    mean_metric_score: float = 0.0
    n_scored_nodes: int = 0
    json_valid: bool = True
    nodes: dict[str, dict] = field(default_factory=dict)
    subtree: dict[str, dict] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreResult":
        return cls(**data)


class ExtractBenchCompliantScorer:
    """Primary scoring engine ("extract_bench_compliant")."""

    name = "extract_bench_compliant"

    def __init__(
        self,
        *,
        cache: ScoringCache | None = None,
        array_policy: str = "hungarian_plus_llm_cache",
        judge: Callable[..., Awaitable[dict]] | None = None,
        match_threshold: float = 0.5,
        param_overrides: dict[str, dict] | None = None,
        metric_overrides: dict[str, Callable[[], Any]] | None = None,
        cache_documents: bool = True,
    ):
        self.cache = cache
        self.array_policy = array_policy
        self.match_threshold = match_threshold
        self.param_overrides = param_overrides or {}
        self.cache_documents = cache_documents
        self.registry: MetricRegistry = build_registry(
            array_policy=array_policy,
            cache=cache,
            pair_scorer=self._score_pair,
            judge=judge,
            match_threshold=match_threshold,
            metric_overrides=metric_overrides,
        )

    # -- public API ---------------------------------------------------------

    def score(self, predicted: Any, expected: Any, schema: dict) -> ScoreResult:
        return asyncio.run(self.score_async(predicted, expected, schema))

    async def score_async(self, predicted: Any, expected: Any, schema: dict) -> ScoreResult:
        key = None
        if self.cache is not None and self.cache_documents:
            key = cache_key(
                "doc_score", predicted, expected, schema, self.array_policy, self.match_threshold
            )
            cached = self.cache.get(key)
            if cached is not None:
                return ScoreResult.from_dict(cached)
        result = await self._score_document(predicted, expected, schema)
        if key is not None:
            self.cache.set(key, result.to_dict())
        return result

    def aggregate(self, results: list[ScoreResult]) -> ScoreResult:
        """Aggregate per-document results (micro-averaged P/R/F1 over all nodes)."""
        if not results:
            return ScoreResult()
        if len(results) == 1:
            return results[0]
        combined = ScoreResult(json_valid=all(r.json_valid for r in results))
        for r in results:
            combined.tp += r.tp
            combined.fp += r.fp
            combined.fn += r.fn
            combined.errors.extend(r.errors)
            for path, node in r.nodes.items():
                agg = combined.nodes.setdefault(
                    path,
                    {
                        "path": path,
                        "metric_id": node.get("metric_id"),
                        "tp": 0.0,
                        "fp": 0.0,
                        "fn": 0.0,
                        "score_sum": 0.0,
                        "count": 0,
                        "is_item_leaf": node.get("is_item_leaf", False),
                    },
                )
                for k in ("tp", "fp", "fn"):
                    agg[k] += node.get(k, 0.0) or 0.0
                agg["score_sum"] += node.get("score") or 0.0
                agg["count"] += 1
        combined.precision, combined.recall, combined.micro_f1 = _aggregate(
            combined.tp, combined.fp, combined.fn
        )
        self._rebuild_subtrees(combined)
        scored = [n for n in combined.nodes.values() if n.get("count")]
        combined.mean_metric_score = (
            sum(n["score_sum"] for n in scored) / sum(n["count"] for n in scored) if scored else 0.0
        )
        combined.n_scored_nodes = sum(n.get("count", 0) for n in scored)
        return combined

    # -- document scoring ---------------------------------------------------

    async def _score_document(self, predicted: Any, expected: Any, schema: dict) -> ScoreResult:
        if not isinstance(predicted, dict) and predicted is not None:
            predicted = None
        if not isinstance(expected, dict):
            raise ValueError("gold value must be a JSON object")

        walker = SchemaWalker(schema)
        result = ScoreResult(json_valid=predicted is not None)
        walk_nodes = list(walker.walk(expected, predicted))

        tasks: list[tuple[WalkNode, Any, Awaitable[MetricResult]]] = []
        for wn in walk_nodes:
            for metric_config in self._metric_configs(wn):
                tasks.append((wn, metric_config, self._run_metric(wn, metric_config)))
        raw_results = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)

        primary: list[NodeScore] = []
        item_leaf_buckets: dict[str, dict] = {}
        for (wn, metric_config, _), raw in zip(tasks, raw_results):
            ns = self._attribute(wn, metric_config.metric_id, raw)
            primary.append(ns)
            result.nodes[ns.path] = ns.to_dict()
            if ns.metric_id == "array_llm":
                for leaf in self._item_leaves_from_result(raw):
                    self._merge_leaf(item_leaf_buckets, leaf)

        for path, bucket in item_leaf_buckets.items():
            _, _, f1 = _aggregate(bucket["tp"], bucket["fp"], bucket["fn"])
            ns = NodeScore(
                path=path,
                metric_id=bucket["metric_id"],
                score=f1,
                passed=f1 >= 0.999,
                tp=bucket["tp"],
                fp=bucket["fp"],
                fn=bucket["fn"],
                gold=bucket.get("gold"),
                extracted=bucket.get("extracted"),
                reason=bucket.get("reason"),
                error=bucket.get("error"),
                is_item_leaf=True,
            )
            result.nodes[path] = ns.to_dict()

        for ns in primary:
            result.tp += ns.tp
            result.fp += ns.fp
            result.fn += ns.fn
            if ns.fp > 0 or ns.fn > 0 or ns.error:
                kind = "error" if ns.error else ("spurious" if ns.fp > 0 and ns.fn == 0 else ("omission" if ns.fn > 0 and ns.fp == 0 else "mismatch"))
                result.errors.append(
                    {
                        "path": ns.path,
                        "metric_id": ns.metric_id,
                        "kind": kind,
                        "gold": _json_safe(ns.gold),
                        "extracted": _json_safe(ns.extracted),
                        "reason": ns.reason,
                        "error": ns.error,
                        "score": ns.score,
                        "fp": ns.fp,
                        "fn": ns.fn,
                    }
                )
        for bucket in item_leaf_buckets.values():
            result.tp += bucket["tp"]
            result.fp += bucket["fp"]
            result.fn += bucket["fn"]
            if bucket["fp"] > 0 or bucket["fn"] > 0:
                result.errors.append(
                    {
                        "path": bucket["path"],
                        "metric_id": bucket["metric_id"],
                        "kind": "array_item_mismatch",
                        "gold": _json_safe(bucket.get("gold")),
                        "extracted": _json_safe(bucket.get("extracted")),
                        "reason": bucket.get("reason"),
                        "error": bucket.get("error"),
                        "score": None,
                        "fp": bucket["fp"],
                        "fn": bucket["fn"],
                    }
                )

        result.precision, result.recall, result.micro_f1 = _aggregate(result.tp, result.fp, result.fn)
        self._rebuild_subtrees(result)
        result.mean_metric_score = (
            sum(ns.score for ns in primary) / len(primary) if primary else 0.0
        )
        result.n_scored_nodes = len(primary)
        return result

    # -- helpers ------------------------------------------------------------

    def _metric_configs(self, wn: WalkNode) -> list[Any]:
        from extract_bench.evaluation.evaluation_config import MetricConfig
        from extract_bench.evaluation.schema_config_helpers import get_evaluation_config
        from extract_bench.infra.nodes import AnyOfSchema

        from .schema_walker import anyof_branch_specs

        config = get_evaluation_config(wn.node)
        if not config.metrics and isinstance(wn.node, AnyOfSchema):
            branches = anyof_branch_specs(wn.node)
            if branches:
                return [MetricConfig(metric_id="anyof_typed", params={"branches": branches})]
            return []

        metric_configs = []
        for mc in config.metrics:
            if not self.registry.has_metric(mc.metric_id):
                continue
            if mc.params is None and mc.metric_id in self.param_overrides:
                mc = MetricConfig(
                    metric_id=mc.metric_id,
                    weight=mc.weight,
                    params=self.param_overrides[mc.metric_id],
                )
            metric_configs.append(mc)
        return metric_configs

    async def _run_metric(self, wn: WalkNode, metric_config: Any) -> MetricResult:
        token = PATH_CONTEXT.set(wn.path)
        try:
            metric = self.registry.create_metric(metric_config.metric_id)
            return await metric.evaluate(wn.node, metric_config)
        finally:
            PATH_CONTEXT.reset(token)

    def _attribute(self, wn: WalkNode, metric_id: str, raw: Any) -> NodeScore:
        base = dict(
            path=wn.path,
            metric_id=metric_id,
            gold=_json_safe(wn.gold),
            extracted=_json_safe(wn.extracted),
        )
        if isinstance(raw, Exception):
            tp = fp = fn = 0.0
            if not wn.gold_absent:
                fn = 1.0
            elif not wn.extracted_absent:
                fp = 1.0
            if wn.gold_absent and wn.extracted_absent:
                fp = fn = 0.0
            return NodeScore(**base, score=0.0, passed=False, tp=tp, fp=fp, fn=fn, error=f"{type(raw).__name__}: {raw}")

        result: MetricResult = raw
        details = result.details or {}
        reason = details.get("reason")
        score = float(result.score)

        if wn.gold_absent and wn.extracted_absent:
            return NodeScore(**base, score=score, passed=result.passed, tp=0.0, fp=0.0, fn=0.0, reason=reason)
        if wn.gold_absent:
            return NodeScore(**base, score=score, passed=result.passed, tp=0.0, fp=1.0, fn=0.0, reason=reason)
        if wn.extracted_absent:
            return NodeScore(**base, score=score, passed=result.passed, tp=0.0, fp=0.0, fn=1.0, reason=reason)

        if metric_id == "array_llm":
            alignment = details.get("alignment")
            if details.get("item_scores"):
                # Matched-pair content is accounted per item leaf; the array node
                # itself only carries missed (FN) and spurious (FP) items.
                return NodeScore(
                    **base,
                    score=score,
                    passed=result.passed,
                    tp=0.0,
                    fp=float(alignment.get("spurious", 0)) if alignment else 0.0,
                    fn=float(alignment.get("missed", 0)) if alignment else 0.0,
                    reason=reason,
                )
            if alignment:
                tp = float(sum(p["similarity"] for p in details.get("matched_pairs", [])))
                return NodeScore(
                    **base,
                    score=score,
                    passed=result.passed,
                    tp=tp,
                    fp=float(alignment.get("spurious", 0)),
                    fn=float(alignment.get("missed", 0)),
                    reason=reason,
                )

        return NodeScore(**base, score=score, passed=result.passed, tp=score, fp=0.0, fn=1.0 - score, reason=reason)

    @staticmethod
    def _item_leaves_from_result(raw: Any) -> list[dict]:
        if isinstance(raw, Exception):
            return []
        details = getattr(raw, "details", None) or {}
        leaves: list[dict] = []
        for item in details.get("item_scores", []):
            leaves.extend(item.get("leaves", []))
        return leaves

    @staticmethod
    def _merge_leaf(buckets: dict[str, dict], leaf: dict) -> None:
        path = leaf["path"]
        bucket = buckets.setdefault(
            path,
            {
                "path": path,
                "metric_id": leaf.get("metric_id"),
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "gold": leaf.get("gold"),
                "extracted": leaf.get("extracted"),
                "reason": leaf.get("reason"),
                "error": leaf.get("error"),
            },
        )
        bucket["tp"] += leaf.get("tp") or 0.0
        bucket["fp"] += leaf.get("fp") or 0.0
        bucket["fn"] += leaf.get("fn") or 0.0
        if bucket["fp"] == 0.0 and bucket["fn"] == 0.0:
            bucket["gold"] = leaf.get("gold")
            bucket["extracted"] = leaf.get("extracted")

    def _rebuild_subtrees(self, result: ScoreResult) -> None:
        subtree: dict[str, dict] = {}
        for path, node in result.nodes.items():
            for prefix in self._path_prefixes(path):
                bucket = subtree.setdefault(
                    prefix, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "score_sum": 0.0, "count": 0}
                )
                for k in ("tp", "fp", "fn"):
                    bucket[k] += node.get(k) or 0.0
                bucket["score_sum"] += node.get("score") or 0.0
                bucket["count"] += 1
        for bucket in subtree.values():
            p, r, f1 = _aggregate(bucket["tp"], bucket["fp"], bucket["fn"])
            bucket["precision"] = round(p, 6)
            bucket["recall"] = round(r, 6)
            bucket["f1"] = round(f1, 6)
            bucket["mean_score"] = (
                round(bucket["score_sum"] / bucket["count"], 6) if bucket["count"] else 0.0
            )
        result.subtree = subtree
        top_children = [
            p for p in subtree if p.startswith("$.") and p.count(".") == 1 and "[*]" not in p
        ]
        if top_children:
            result.macro_subtree_f1 = sum(subtree[p]["f1"] for p in top_children) / len(top_children)
        else:
            result.macro_subtree_f1 = result.micro_f1

    @staticmethod
    def _path_prefixes(path: str) -> list[str]:
        prefixes = ["$"]
        acc = "$"
        for part in path.split(".")[1:]:
            if part.endswith("[*]"):
                acc += "." + part[:-3]
                prefixes.append(acc)
                acc += "[*]"
                prefixes.append(acc)
            else:
                acc += "." + part
                prefixes.append(acc)
        return prefixes

    async def _score_pair(self, pred_item: Any, gold_item: Any, node: Any, base_path: str) -> list[dict]:
        """Score one matched array item via its item sub-schema (recursive)."""
        items_node = getattr(node, "items", None)
        if items_node is None:
            score = 1.0 if pred_item == gold_item else 0.0
            return [
                {
                    "path": f"{base_path}[*]",
                    "metric_id": "exact_item",
                    "score": score,
                    "passed": score == 1.0,
                    "tp": score,
                    "fp": 0.0,
                    "fn": 1.0 - score,
                    "gold": _json_safe(gold_item),
                    "extracted": _json_safe(pred_item),
                    "reason": None,
                    "error": None,
                }
            ]
        # Pairs are scored strictly sequentially and every walk rebinds all values
        # before metrics run, so binding directly onto the shared item subtree is safe.
        subtree = items_node
        walker = SchemaWalker(tree=subtree)
        item_path = f"{base_path}[*]"
        walk_nodes = list(walker.walk_from(subtree, item_path, gold_item, pred_item))
        tasks: list[tuple[WalkNode, Any, Awaitable[MetricResult]]] = []
        for wn in walk_nodes:
            for metric_config in self._metric_configs(wn):
                tasks.append((wn, metric_config, self._run_metric(wn, metric_config)))
        raw_results = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)
        leaves: list[dict] = []
        for (wn, metric_config, _), raw in zip(tasks, raw_results):
            ns = self._attribute(wn, metric_config.metric_id, raw)
            if ns.metric_id == "array_llm":
                nested = self._item_leaves_from_result(raw)
                leaves.extend(nested)
                if not nested or ns.fp > 0 or ns.fn > 0:
                    leaves.append(ns.to_dict())
                continue
            leaves.append(ns.to_dict())
        return leaves


def resolve_engine(spec: str, **kwargs: Any) -> Any:
    """Resolve ``scoring.engine`` config: builtin name or ``module:callable``."""
    if spec in ("extract_bench_compliant", "", None):
        return ExtractBenchCompliantScorer(**kwargs)
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"scoring.engine '{spec}' must be 'module.submodule:callable'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory(**kwargs)
