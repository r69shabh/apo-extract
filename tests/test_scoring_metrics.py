"""Per-field metric semantics + null/missing policy (compliance with extract-bench)."""

import glob
import json
import os

import pytest

from apo_extract.scoring.engine import ExtractBenchCompliantScorer
from conftest import make_scorer

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "extract-bench", "dataset")


def flat_schema(prop: dict) -> dict:
    return {"type": "object", "properties": {"value": prop}}


def score_leaf(scorer: ExtractBenchCompliantScorer, prop: dict, gold, pred):
    return scorer.score({"value": pred}, {"value": gold}, flat_schema(prop))


class TestStringMetrics:
    def test_string_exact(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        assert score_leaf(scorer, prop, "Hello", "Hello").nodes["$.value"]["score"] == 1.0
        assert score_leaf(scorer, prop, "Hello", "hello").nodes["$.value"]["score"] == 0.0

    def test_string_case_insensitive(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_case_insensitive"}
        assert score_leaf(scorer, prop, "Hello", "hELLo").nodes["$.value"]["score"] == 1.0

    def test_uri_format_upgrades_string_exact_to_url(self, scorer):
        prop = {"type": "string", "format": "uri", "evaluation_config": "string_exact"}
        result = score_leaf(scorer, prop, "https://example.com/foo/", "http://www.example.com/foo")
        assert result.nodes["$.value"]["score"] == 1.0
        result = score_leaf(scorer, prop, "https://example.com/foo", "https://example.com/bar")
        assert result.nodes["$.value"]["score"] == 0.0


class TestNumericMetrics:
    def test_integer_exact_coerces_numeric_strings(self, scorer):
        prop = {"type": "integer", "evaluation_config": "integer_exact"}
        assert score_leaf(scorer, prop, 100, "100").nodes["$.value"]["score"] == 1.0
        assert score_leaf(scorer, prop, 100, 101).nodes["$.value"]["score"] == 0.0

    def test_number_tolerance_boundary_and_continuity(self, scorer):
        prop = {
            "type": "number",
            "evaluation_config": {"metrics": [{"metric_id": "number_tolerance", "params": {"tolerance": 0.001}}]},
        }
        assert score_leaf(scorer, prop, 1.0, 1.0009).nodes["$.value"]["score"] == 1.0
        assert score_leaf(scorer, prop, 1.0, 1.001).nodes["$.value"]["score"] == 1.0  # exactly at tolerance
        assert score_leaf(scorer, prop, 1.0, 1.002).nodes["$.value"]["score"] == 0.0

    def test_boolean_exact_strict(self, scorer):
        prop = {"type": "boolean", "evaluation_config": "boolean_exact"}
        assert score_leaf(scorer, prop, True, True).nodes["$.value"]["score"] == 1.0
        assert score_leaf(scorer, prop, True, "true").nodes["$.value"]["score"] == 0.0


class TestNullPolicy:
    def test_both_null_is_true_negative_excluded(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        result = scorer.score({"value": None}, {"value": None}, flat_schema(prop))
        assert (result.tp, result.fp, result.fn) == (0.0, 0.0, 0.0)

    def test_null_gold_value_pred_is_spurious(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        result = scorer.score({"value": "hallucinated"}, {"value": None}, flat_schema(prop))
        assert result.fp == 1.0 and result.precision == 0.0

    def test_value_gold_null_pred_is_missed(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        result = scorer.score({"value": None}, {"value": "expected"}, flat_schema(prop))
        assert result.fn == 1.0 and result.recall == 0.0

    def test_missing_key_equals_omission(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        result = scorer.score({}, {"value": "expected"}, flat_schema(prop))
        assert result.fn == 1.0 and result.recall == 0.0

    def test_invalid_prediction_scores_all_misses(self, scorer):
        prop = {"type": "string", "evaluation_config": "string_exact"}
        result = scorer.score(None, {"value": "expected"}, flat_schema(prop))
        assert result.json_valid is False
        assert result.recall == 0.0
        assert result.micro_f1 == 0.0


class TestGoldAgainstGoldAllSchemas:
    @pytest.mark.parametrize("schema_path", sorted(glob.glob(os.path.join(DATASET, "*", "*", "*-schema.json"))))
    def test_gold_vs_gold_is_perfect(self, scorer, schema_path):
        schema = json.load(open(schema_path))
        schema = schema.get("schema_definition", schema)
        gold_dir = os.path.join(os.path.dirname(schema_path), "pdf+gold")
        for gold_path in sorted(glob.glob(os.path.join(gold_dir, "*.gold.json"))):
            gold = json.load(open(gold_path))
            result = scorer.score(gold, gold, schema)
            assert result.micro_f1 == 1.0, gold_path
            assert result.macro_subtree_f1 == 1.0, gold_path
            assert result.errors == [], gold_path
