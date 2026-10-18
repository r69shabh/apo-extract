"""Phase 1 diagnosis: cluster failures by (subtree, error type) and pick targets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..scoring.engine import ScoreResult

ERROR_TYPES = ("invalid_json", "omission", "spurious", "mismatch")

DOCUMENT_TARGET = "$.__document__"


def top_level_key(path: str) -> str:
    """'$.contact[*].email' -> '$.contact'."""
    clean = path.lstrip("$").lstrip(".")
    key = clean.split("[")[0].split(".")[0]
    return f"$.{key}" if key else DOCUMENT_TARGET


@dataclass
class FailureCluster:
    subtree: str
    error_type: str
    count: int
    docs: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    weight: float = 0.0


def cluster_failures(per_doc: dict[str, ScoreResult], *, min_examples: int = 3) -> list[FailureCluster]:
    buckets: dict[tuple[str, str], FailureCluster] = {}
    subtree_f1: dict[str, list[float]] = defaultdict(list)

    for doc_id, result in per_doc.items():
        for path, bucket in result.subtree.items():
            if path.count(".") == 1:
                subtree_f1[path].append(float(bucket.get("f1", 0.0)))

        doc_errors = list(result.errors)
        if not result.json_valid:
            doc_errors.append({"kind": "error", "path": "$", "reason": "invalid JSON output"})
        for err in doc_errors:
            path = err.get("path") or "$"
            subtree = top_level_key(path) if path != "$" else DOCUMENT_TARGET
            if not result.json_valid:
                error_type = "invalid_json"
            else:
                error_type = err.get("kind", "mismatch")
                if error_type == "error":
                    error_type = "mismatch"
            cluster = buckets.setdefault(
                (subtree, error_type),
                FailureCluster(subtree=subtree, error_type=error_type, count=0),
            )
            cluster.count += 1
            if doc_id not in cluster.docs:
                cluster.docs.append(doc_id)
            if len(cluster.examples) < min_examples:
                cluster.examples.append(
                    {
                        "doc_id": doc_id,
                        "path": path,
                        "gold": err.get("gold"),
                        "extracted": err.get("extracted"),
                        "reason": err.get("reason"),
                    }
                )

    clusters = list(buckets.values())
    for cluster in clusters:
        scores = subtree_f1.get(cluster.subtree, [0.0])
        subtree_score = sum(scores) / len(scores)
        cluster.weight = cluster.count * (1.0 + (1.0 - subtree_score))
    clusters.sort(key=lambda c: (-c.weight, c.subtree, c.error_type))
    return clusters


def choose_target(clusters: list[FailureCluster], exclude: set[str] | None = None) -> FailureCluster | None:
    excluded = exclude or set()
    for cluster in clusters:
        if cluster.subtree not in excluded:
            return cluster
    return None
