"""Subtree P/R/F1 roll-ups, macro vs micro aggregation, TN conventions."""

import pytest

from conftest import make_scorer

SCHEMA = {
    "type": "object",
    "properties": {
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "evaluation_config": "string_exact"},
                "phone": {"type": "string", "evaluation_config": "string_exact"},
            },
        },
        "education": {
            "type": "array",
            "evaluation_config": "array_llm",
            "items": {
                "type": "object",
                "properties": {
                    "school": {"type": "string", "evaluation_config": "string_exact"},
                    "degree": {"type": "string", "evaluation_config": "string_exact"},
                },
            },
        },
        "notes": {"type": "string", "evaluation_config": "string_exact"},
    },
}

GOLD = {
    "contact": {"email": "e@x.com", "phone": "555"},
    "education": [{"school": "s1", "degree": "d1"}, {"school": "s2", "degree": "d2"}],
    "notes": "n",
}

PRED = {
    "contact": {"email": "e@x.com"},
    "education": [
        {"school": "s1", "degree": "d1"},
        {"school": "s2", "degree": "d2"},
        {"school": "sX", "degree": "dX"},
    ],
}


class TestSubtreeAggregation:
    def test_nested_subtree_rollup(self):
        scorer = make_scorer()
        result = scorer.score(PRED, GOLD, SCHEMA)

        assert result.subtree["$.contact"]["f1"] == pytest.approx(2 / 3, abs=1e-3)
        assert result.subtree["$.education"]["tp"] == 4.0
        assert result.subtree["$.education"]["fp"] == 1.0
        assert result.subtree["$.education"]["f1"] == pytest.approx(0.8889, abs=1e-3)
        assert result.subtree["$.notes"]["f1"] == 0.0

    def test_micro_vs_macro(self):
        scorer = make_scorer()
        result = scorer.score(PRED, GOLD, SCHEMA)

        assert result.tp == 5.0 and result.fp == 1.0 and result.fn == 2.0
        assert result.micro_f1 == pytest.approx(2 * (5 / 6) * (5 / 7) / ((5 / 6) + (5 / 7)), abs=1e-3)
        expected_macro = (
            result.subtree["$.contact"]["f1"]
            + result.subtree["$.education"]["f1"]
            + result.subtree["$.notes"]["f1"]
        ) / 3
        assert result.macro_subtree_f1 == pytest.approx(expected_macro, abs=1e-3)

    def test_root_equals_sum_of_children(self):
        scorer = make_scorer()
        result = scorer.score(PRED, GOLD, SCHEMA)
        root = result.subtree["$"]
        assert root["tp"] == 5.0 and root["fp"] == 1.0 and root["fn"] == 2.0

    def test_perfect_document_has_empty_denominators_as_one(self):
        scorer = make_scorer()
        result = scorer.score(GOLD, GOLD, SCHEMA)
        assert result.micro_f1 == 1.0
        assert result.macro_subtree_f1 == 1.0
        for bucket in result.subtree.values():
            assert bucket["f1"] == 1.0

    def test_true_negatives_excluded_from_denominators(self):
        scorer = make_scorer()
        schema = {"type": "object", "properties": {"a": {"type": "string", "evaluation_config": "string_exact"}}}
        result = scorer.score({"a": None}, {"a": None}, schema)
        assert result.subtree["$.a"]["tp"] == 0.0
        assert result.subtree["$.a"]["f1"] == 1.0  # convention: no evidence of error

    def test_aggregate_across_documents(self):
        scorer = make_scorer()
        r1 = scorer.score(GOLD, GOLD, SCHEMA)
        r2 = scorer.score(PRED, GOLD, SCHEMA)
        combined = scorer.aggregate([r1, r2])
        assert combined.tp == 5.0 + r1.tp
        assert combined.fp == 1.0
        assert combined.fn == 2.0
        assert combined.subtree["$.contact"]["count"] == 2  # distinct paths: email, phone
