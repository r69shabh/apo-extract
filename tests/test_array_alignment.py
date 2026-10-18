"""Array alignment: Hungarian matching, matched/missed/spurious P/R/F1, anyOf branches."""

import pytest

from apo_extract.scoring.alignment import classify_alignment
from conftest import make_scorer


def array_schema(item_props: dict | None = None) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "evaluation_config": "array_llm",
                "items": {
                    "type": "object",
                    "properties": item_props
                    or {
                        "name": {"type": "string", "evaluation_config": "string_exact"},
                        "count": {"type": "integer", "evaluation_config": "integer_exact"},
                    },
                },
            }
        },
    }


class TestClassifyAlignment:
    def test_perfect_matching(self):
        result = classify_alignment([[1.0, 0.0], [0.0, 1.0]])
        assert result.soft_f1 == 1.0
        assert result.matched == [(0, 0, 1.0), (1, 1, 1.0)]

    def test_hungarian_is_order_invariant(self):
        a = classify_alignment([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = classify_alignment([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        assert sorted(s for _, _, s in a.matched) == sorted(s for _, _, s in b.matched)
        assert (a.missed, a.spurious) == (b.missed, b.spurious)

    def test_missed_gold_item_reduces_recall(self):
        result = classify_alignment([[1.0, 0.1]])
        assert len(result.missed) == 1 and result.spurious == []
        assert result.recall < 1.0 and result.precision == 1.0

    def test_spurious_pred_item_reduces_precision(self):
        result = classify_alignment([[1.0], [0.1]])
        assert len(result.spurious) == 1 and result.missed == []
        assert result.precision < 1.0 and result.recall == 1.0

    def test_match_threshold_boundary(self):
        assert classify_alignment([[0.5]]).matched  # >= threshold counts as matched
        assert not classify_alignment([[0.49]]).matched

    def test_empty_arrays_are_perfect(self):
        assert classify_alignment([]).soft_f1 == 1.0


class TestHungarianArrayMetric:
    def test_reordered_items_match(self):
        scorer = make_scorer()
        schema = array_schema()
        gold = {"items": [{"name": "a", "count": 1}, {"name": "b", "count": 2}]}
        pred = {"items": [{"name": "b", "count": 2}, {"name": "a", "count": 1}]}
        result = scorer.score(pred, gold, schema)
        assert result.micro_f1 == 1.0

    def test_missing_item_is_missed(self):
        scorer = make_scorer()
        schema = array_schema()
        gold = {"items": [{"name": "a", "count": 1}, {"name": "b", "count": 2}]}
        pred = {"items": [{"name": "a", "count": 1}]}
        result = scorer.score(pred, gold, schema)
        assert result.fn == 1.0
        assert result.recall < 1.0

    def test_extra_item_is_spurious(self):
        scorer = make_scorer()
        schema = array_schema()
        gold = {"items": [{"name": "a", "count": 1}]}
        pred = {"items": [{"name": "a", "count": 1}, {"name": "zzz", "count": 99}]}
        result = scorer.score(pred, gold, schema)
        assert result.fp == 1.0
        assert result.precision < 1.0

    def test_wrong_field_inside_matched_item(self):
        scorer = make_scorer()
        schema = array_schema()
        gold = {"items": [{"name": "a", "count": 1}, {"name": "b", "count": 2}]}
        pred = {"items": [{"name": "a", "count": 7}, {"name": "b", "count": 2}]}
        result = scorer.score(pred, gold, schema)
        leaves = {p: n for p, n in result.nodes.items() if n["is_item_leaf"]}
        assert leaves["$.items[*].count"]["tp"] == 1.0
        assert leaves["$.items[*].count"]["fn"] == 1.0
        assert leaves["$.items[*].name"]["tp"] == 2.0
        assert 0.0 < result.micro_f1 < 1.0

    def test_anyof_typed_branch_resolution(self):
        scorer = make_scorer()
        schema = array_schema(
            {
                "value": {
                    "anyOf": [
                        {"type": "string", "evaluation_config": "string_exact"},
                        {"type": "integer", "evaluation_config": "integer_exact"},
                    ]
                },
            }
        )
        gold = {"items": [{"value": "text"}, {"value": 42}]}
        pred = {"items": [{"value": "text"}, {"value": 42}]}
        result = scorer.score(pred, gold, schema)
        assert result.micro_f1 == 1.0

        pred_bad = {"items": [{"value": "text"}, {"value": 41}]}
        result_bad = scorer.score(pred_bad, gold, schema)
        assert result_bad.micro_f1 < 1.0

    def test_fuzzy_similarity_controls_matching(self):
        scorer = make_scorer()
        schema = array_schema({"name": {"type": "string", "evaluation_config": "string_fuzzy"}})
        gold = {"items": [{"name": "abcdef"}]}
        pred = {"items": [{"name": "abcxyz"}]}  # SequenceMatcher ratio 0.5 -> matched at threshold
        result = scorer.score(pred, gold, schema)
        node = result.nodes["$.items[*].name"]
        assert node["tp"] > 0.0

    def test_empty_array_gold_and_pred_is_true_negative(self):
        scorer = make_scorer()
        schema = array_schema()
        result = scorer.score({"items": []}, {"items": []}, schema)
        assert (result.tp, result.fp, result.fn) == (0.0, 0.0, 0.0)
        assert result.micro_f1 == 1.0

    def test_gold_items_pred_missing_key_is_omission(self):
        scorer = make_scorer()
        schema = array_schema()
        gold = {"items": [{"name": "a", "count": 1}]}
        pred = {"items": [{"name": "a"}]}
        result = scorer.score(pred, gold, schema)
        leaves = {p: n for p, n in result.nodes.items() if n["is_item_leaf"]}
        assert leaves["$.items[*].count"]["fn"] == 1.0


class TestLLMRefinement:
    @pytest.mark.asyncio
    async def test_judge_refines_ambiguous_pairs_and_is_cached(self, tmp_path):
        from apo_extract.scoring.cache import ScoringCache
        from apo_extract.scoring.engine import ExtractBenchCompliantScorer

        cache = ScoringCache(tmp_path / "cache.sqlite")
        calls = []

        async def judge(pred_item, gold_item, field_path):
            calls.append((pred_item, gold_item))
            return {"match": False, "score": 0.0}

        scorer = ExtractBenchCompliantScorer(
            cache=cache,
            array_policy="hungarian_plus_llm_cache",
            judge=judge,
            cache_documents=False,
            metric_overrides={"string_semantic": lambda: _fuzzy(), "string_llm": lambda: _fuzzy()},
        )
        schema = array_schema({"name": {"type": "string", "evaluation_config": "string_semantic"}})
        gold = {"items": [{"name": "abcdef"}]}
        pred = {"items": [{"name": "abcxyz"}]}

        result = await scorer.score_async(pred, gold, schema)
        assert result.fn >= 1.0  # similarity 0.5 is ambiguous; judge demotes to missed
        n_calls = len(calls)
        assert n_calls == 1

        result2 = await scorer.score_async(pred, gold, schema)
        assert len(calls) == n_calls  # second run hit the disk cache


def _fuzzy():
    from extract_bench.evaluation.metrics.string_metrics import NormalizedLevenshteinSimilarityMetric

    return NormalizedLevenshteinSimilarityMetric()
