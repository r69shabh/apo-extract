"""Resume replay: a crashed attempt must not re-roll slots that were already recorded."""

import asyncio

from apo_extract.observability.store import IterationRecord
from apo_extract.optimizer.pareto_beam import SplitEval
from apo_extract.optimizer.safeguards import ArchiveEntry
from apo_extract.predictor.artifact import PromptArtifact
from tests.test_rollback import SCHEMA, make_config, make_optimizer


def _record(store, candidate_id, iteration, decision):
    store.log_iteration(
        "rb",
        IterationRecord(
            iteration=iteration,
            candidate_id=candidate_id,
            parent_id="root",
            mutation_type="llm_directive_edit",
            target_subtree="$.skills",
            hypothesis="h",
            train_f1=0.9,
            val_f1=0.9,
            decision=decision,
        ),
    )


def _setup(tmp_path):
    config = make_config(stall_patience=9)
    optimizer = make_optimizer(tmp_path, config)
    optimizer.store.init_run("rb", "hash")
    root = PromptArtifact.init_from_schema(SCHEMA, "seed")
    optimizer.store.save_candidate("root", "rb", 0, root.to_dict(), root.hash())
    optimizer.front = [ArchiveEntry("root", 0.6, 0.7, root, {}, "", "")]
    return optimizer, root


def test_replay_skips_fully_recorded_iteration(tmp_path):
    optimizer, root = _setup(tmp_path)
    recorded = root.copy("c1_0")
    optimizer.store.save_candidate("c1_0", "rb", 1, recorded.to_dict(), recorded.hash())
    _record(optimizer.store, "c1_0", 1, "ACCEPTED")
    recorded1 = root.copy("c1_1")
    optimizer.store.save_candidate("c1_1", "rb", 1, recorded1.to_dict(), recorded1.hash())
    _record(optimizer.store, "c1_1", 1, "REJECTED_STAGE1")

    class ExplodingProposer:
        async def propose_async(self, *args, **kwargs):
            raise AssertionError("must not propose when every slot is recorded")

    optimizer.proposer = ExplodingProposer()
    assert asyncio.run(optimizer._run_iteration(1, root)) is True
    kept = optimizer.store.load_candidate("c1_1", "rb")
    assert kept == recorded1.to_dict()


def test_replay_fills_only_missing_slots(tmp_path):
    optimizer, root = _setup(tmp_path)
    recorded = root.copy("c1_0")
    optimizer.store.save_candidate("c1_0", "rb", 1, recorded.to_dict(), recorded.hash())
    _record(optimizer.store, "c1_0", 1, "ACCEPTED")

    seen = []

    class RecordingProposer:
        async def propose_async(self, incumbent, clusters, reference_deltas, candidate_ids,
                                other_artifacts=None, iteration=0):
            seen.append(list(candidate_ids))
            return []

    optimizer.proposer = RecordingProposer()

    async def fake_eval_split(artifact, split, iteration, tag):
        return SplitEval(f1=0.5, aggregate_f1=0.5, json_valid_rate=1.0, per_doc={}, merged_subtrees={})

    optimizer._eval_split = fake_eval_split
    assert asyncio.run(optimizer._run_iteration(1, root)) is False
    assert seen == [["c1_1"]]
    assert optimizer.store.load_candidate("c1_0", "rb") == recorded.to_dict()
