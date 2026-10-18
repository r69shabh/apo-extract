"""Deterministic seed-based train/val/test splitter for small-N schemas.

Policy (see README): sort doc ids lexicographically, shuffle with
``random.Random(seed)``, then allocate
    n_test = max(1, floor(test_ratio * N))
    n_val  = max(2, floor(val_ratio * N))
    n_train = N - n_val - n_test (>= 1)
shrinking val/test in that priority order if a schema is too small.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..config import SplitPolicyConfig


@dataclass
class DatasetSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def all_ids(self) -> list[str]:
        return self.train + self.val + self.test


def split_documents(doc_ids: list[str], policy: SplitPolicyConfig) -> DatasetSplit:
    if len(doc_ids) < 3:
        raise ValueError(f"need at least 3 documents to split, got {len(doc_ids)}")

    ordered = sorted(doc_ids)
    shuffled = list(ordered)
    random.Random(policy.seed).shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, math.floor(policy.test_ratio * n))
    n_val = max(2, math.floor(policy.val_ratio * n))
    while n_test + n_val >= n:
        if n_val > 1 and n_val >= n_test:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return DatasetSplit(train=sorted(train), val=sorted(val), test=sorted(test))
