"""Deterministic reproducibility, disjointness, and small-N allocation of splits."""

import glob
import os

import pytest

from apo_extract.config import SplitPolicyConfig
from apo_extract.dataset.splitter import split_documents

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "extract-bench", "dataset")


def real_doc_ids() -> dict[str, list[str]]:
    ids = {}
    for schema_path in sorted(glob.glob(os.path.join(DATASET, "*", "*", "*-schema.json"))):
        name = os.path.dirname(schema_path).split("dataset/")[1]
        pdfs = sorted(glob.glob(os.path.join(os.path.dirname(schema_path), "pdf+gold", "*.pdf")))
        ids[name] = [os.path.splitext(os.path.basename(p))[0] for p in pdfs]
    return ids


POLICY = SplitPolicyConfig(seed=42)


class TestSplitter:
    def test_reproducible_with_same_seed(self):
        ids = [f"doc_{i:02d}" for i in range(10)]
        a = split_documents(ids, POLICY)
        b = split_documents(ids, POLICY)
        assert a == b

    def test_disjoint_and_complete(self):
        for name, ids in real_doc_ids().items():
            split = split_documents(ids, POLICY)
            all_ids = split.all_ids()
            assert sorted(all_ids) == sorted(ids), name
            assert len(set(all_ids)) == len(all_ids), name

    def test_blueprint_small_n_allocation(self):
        # ratios 0.30/0.40/0.30 with the max() floors from the spec
        expected = {
            5: (2, 2, 1),
            6: (3, 2, 1),
            7: (3, 2, 2),
            10: (3, 4, 3),
        }
        for n, (n_train, n_val, n_test) in expected.items():
            ids = [f"doc_{i:02d}" for i in range(n)]
            split = split_documents(ids, POLICY)
            assert (len(split.train), len(split.val), len(split.test)) == (n_train, n_val, n_test), n

    def test_real_schemas_have_nonempty_splits(self):
        for name, ids in real_doc_ids().items():
            split = split_documents(ids, POLICY)
            assert len(split.train) >= 1, name
            assert len(split.val) >= 2, name
            assert len(split.test) >= 1, name

    def test_rejects_tiny_input(self):
        with pytest.raises(ValueError):
            split_documents(["a", "b"], POLICY)
