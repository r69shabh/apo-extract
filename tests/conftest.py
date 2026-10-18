import pytest

from extract_bench.evaluation.metrics.string_metrics import ExactStringMatchMetric

from apo_extract.scoring.engine import ExtractBenchCompliantScorer


def make_scorer(**kwargs) -> ExtractBenchCompliantScorer:
    overrides = {
        "string_semantic": ExactStringMatchMetric,
        "string_llm": ExactStringMatchMetric,
    }
    overrides.update(kwargs.pop("metric_overrides", {}))
    kwargs.setdefault("array_policy", "hungarian_only")
    kwargs.setdefault("cache_documents", False)
    return ExtractBenchCompliantScorer(metric_overrides=overrides, **kwargs)


@pytest.fixture
def scorer() -> ExtractBenchCompliantScorer:
    return make_scorer()
