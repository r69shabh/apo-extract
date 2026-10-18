"""SR-PBO optimizer: staged filtering, Pareto archive with beam, rollback.

References:
  ParetoBanditOptimizer.cs (stage-1 gate, discretized primary metric, beam width)
  PromptBreederOptimizer.cs (tabu list, mutation/crossover, Pareto archive,
  subtask rollback); adapted from evolutionary prompt optimization (Fernando et
  al. 2024) for the ExtractBench small-N regime.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AppConfig
from ..dataset.loader import DocumentItem
from ..observability.store import IterationRecord, RunStore
from ..predictor.artifact import PromptArtifact
from ..scoring.engine import ScoreResult
from .diagnostic import FailureCluster, cluster_failures
from .proposer import Proposer
from .safeguards import (
    ArchiveEntry,
    add_to_front,
    best_selection,
    regressed_subtrees,
    stage1_gate,
)


@dataclass
class SplitEval:
    f1: float
    aggregate_f1: float
    json_valid_rate: float
    per_doc: dict = field(default_factory=dict)
    merged_subtrees: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "f1": self.f1,
            "aggregate_f1": self.aggregate_f1,
            "json_valid_rate": self.json_valid_rate,
            "per_doc": {k: v.to_dict() for k, v in self.per_doc.items()},
            "merged_subtrees": self.merged_subtrees,
        }


class Evaluator:
    """Runs the extractor + scorer over a split; per-doc scores are cached in memory."""

    def __init__(
        self,
        extractor,
        scorer,
        schema: dict,
        docs: dict[str, DocumentItem],
        *,
        iteration: int | None = None,
    ):
        self.extractor = extractor
        self.scorer = scorer
        self.schema = schema
        self.docs = docs
        self.iteration = iteration
        self._score_cache: dict[tuple[str, str], ScoreResult] = {}

    async def evaluate_async(self, artifact: PromptArtifact, doc_ids: list[str]) -> SplitEval:
        per_doc: dict[str, ScoreResult] = {}
        for doc_id in doc_ids:
            item = self.docs[doc_id]
            key = (artifact.hash(), doc_id)
            if key in self._score_cache:
                per_doc[doc_id] = self._score_cache[key]
                continue
            prediction = await self.extractor.extract(artifact, item, self.schema, self.iteration)
            result = await self.scorer.score_async(
                prediction.data if prediction.json_valid else {}, item.gold, self.schema
            )
            self._score_cache[key] = result
            per_doc[doc_id] = result

        f1 = sum(r.micro_f1 for r in per_doc.values()) / len(per_doc)
        aggregate = sum(r.macro_subtree_f1 for r in per_doc.values()) / len(per_doc)
        json_valid = sum(1 for r in per_doc.values() if r.json_valid) / len(per_doc)
        merged = self.scorer.aggregate(list(per_doc.values()))
        return SplitEval(
            f1=f1,
            aggregate_f1=aggregate,
            json_valid_rate=json_valid,
            per_doc=per_doc,
            merged_subtrees=merged.subtree,
        )

    def evaluate(self, artifact: PromptArtifact, doc_ids: list[str]) -> SplitEval:
        return asyncio.run(self.evaluate_async(artifact, doc_ids))


class Optimizer:
    def __init__(
        self,
        config: AppConfig,
        store: RunStore,
        *,
        extractor,
        scorer,
        proposer: Proposer,
        schema: dict,
        splits: dict[str, list[str]],
        docs: dict[str, DocumentItem],
        seed_text: str,
        tracker=None,
        run_id: str | None = None,
    ):
        self.config = config
        self.store = store
        self.extractor = extractor
        self.scorer = scorer
        self.proposer = proposer
        self.schema = schema
        self.splits = splits
        self.docs = docs
        self.seed_text = seed_text
        self.tracker = tracker
        self.run_id = run_id or config.run_id
        self.front: list[ArchiveEntry] = []
        self._incumbent_cache: dict[str, SplitEval] = {}
        self._split_cache: dict = {}

    # -- public API ------------------------------------------------------------

    def run(self, resume: bool = False) -> dict:
        return asyncio.run(self.run_async(resume=resume))

    async def run_async(self, resume: bool = False) -> dict:
        root = PromptArtifact.init_from_schema(self.schema, self.seed_text, artifact_id="root")
        await self._load_state(root, resume)

        start_iteration = self._start_iteration(resume)
        stall = int((self.store.get_run(self.run_id) or {}).get("stall") or 0)

        for iteration in range(start_iteration, self.config.budget.max_iterations + 1):
            if self.tracker is not None:
                self.tracker.check()
            improved = await self._run_iteration(iteration, root)
            stall = 0 if improved else stall + 1
            self.store.update_run(
                self.run_id, current_iteration=iteration, stall=stall,
                budget_spent_tokens=getattr(self.tracker, "spent_tokens", 0),
                budget_spent_usd=getattr(self.tracker, "spent_usd", 0.0),
            )
            if stall >= self.config.optimizer.stall_patience:
                break

        if self.config.optimizer.enable_subtree_rollback:
            await self._rollback_pass(root)

        return await self._finalize(root)

    # -- state ----------------------------------------------------------------

    async def _load_state(self, root: PromptArtifact, resume: bool) -> None:
        self.store.init_run(
            self.run_id,
            self.config.config_hash(),
            getattr(self.config, "config_path", ""),
        )
        stored_root = self.store.load_candidate("root", self.run_id)
        if stored_root is None:
            self.store.save_candidate("root", self.run_id, 0, root.to_dict(), root.hash())

        eval_root = await self._evaluate_cached(root)
        root_subtrees = eval_root.merged_subtrees
        root_entry = ArchiveEntry(
            candidate_id="root",
            val_f1=eval_root.f1,
            aggregate_f1=eval_root.aggregate_f1,
            artifact=root,
            subtree_scores=root_subtrees,
        )

        for row in self.store.load_archive(self.run_id):
            artifact_json = self.store.load_candidate(row["candidate_id"], self.run_id)
            if artifact_json is None:
                continue
            delta_text, target_subtree = self.store.load_candidate_delta(
                row["candidate_id"], self.run_id
            )
            self.front = add_to_front(
                self.front,
                ArchiveEntry(
                    candidate_id=row["candidate_id"],
                    val_f1=float(row["val_f1"]),
                    aggregate_f1=float(row["aggregate_f1"]),
                    artifact=PromptArtifact.from_dict(artifact_json),
                    subtree_scores=row["subtree"],
                    delta_text=delta_text,
                    target_subtree=target_subtree,
                ),
            )
        if not any(e.candidate_id == "root" for e in self.front):
            self.front = add_to_front(self.front, root_entry)
        self._incumbent_cache["root"] = eval_root
        (self.store.run_dir / "root_val.json").write_text(json.dumps(eval_root.to_dict(), indent=2, default=repr))

    def _start_iteration(self, resume: bool) -> int:
        if not resume:
            return 1
        run = self.store.get_run(self.run_id) or {}
        return int(run.get("current_iteration") or 0) + 1

    async def _evaluate_cached(self, artifact: PromptArtifact) -> SplitEval:
        cached = self._incumbent_cache.get(artifact.artifact_id)
        if cached is not None:
            return cached
        result = await self._eval_split(artifact, "val", 0, artifact.artifact_id)
        self._incumbent_cache[artifact.artifact_id] = result
        return result

    # -- one iteration -----------------------------------------------------------

    async def _run_iteration(self, iteration: int, root: PromptArtifact) -> bool:
        candidate_ids: list[str] = []
        for i in range(self.config.optimizer.candidates_per_iteration):
            candidate_id = f"c{iteration}_{i}"
            if self.store.load_candidate(candidate_id, self.run_id) is None:
                candidate_ids.append(candidate_id)  # slot was never proposed (or crashed pre-save)
        if not candidate_ids:
            rows = [
                r for r in self.store.list_iterations(self.run_id)
                if int(r["iteration"]) == iteration
            ]
            return any(r["decision"] == "ACCEPTED" for r in rows)

        incumbent_entry = best_selection(self.front, self.config.optimizer.pareto_aggregate_tolerance)
        incumbent = incumbent_entry.artifact

        train_eval = await self._eval_split(incumbent, "train", iteration, incumbent.artifact_id)
        clusters = cluster_failures(train_eval.per_doc)
        reference_deltas = [
            (e.target_subtree, e.delta_text) for e in self.front if e.delta_text
        ]
        other_artifacts = [
            e.artifact for e in self.front if e.candidate_id != incumbent_entry.candidate_id
        ]
        mutations = await self.proposer.propose_async(
            incumbent,
            clusters,
            reference_deltas,
            candidate_ids,
            other_artifacts=other_artifacts,
            iteration=iteration,
        )

        iteration_dir = self.store.iteration_dir(iteration)
        improved = False
        for mutation in mutations:
            candidate_id = mutation.artifact.artifact_id
            self.store.save_candidate(
                candidate_id, self.run_id, iteration, mutation.artifact.to_dict(),
                mutation.artifact.hash(), mutation.delta_text, mutation.target_subtree,
            )
            record = IterationRecord(
                iteration=iteration,
                candidate_id=candidate_id,
                parent_id=incumbent.artifact_id,
                mutation_type=mutation.mutation_type,
                target_subtree=mutation.target_subtree,
                hypothesis=mutation.hypothesis,
                train_f1=None,
                val_f1=None,
            )

            train_candidate = await self._eval_split(mutation.artifact, "train", iteration, candidate_id)
            record.train_f1 = train_candidate.f1
            (iteration_dir / f"{candidate_id}_train.json").write_text(
                json.dumps(train_candidate.to_dict(), indent=2)
            )

            if not stage1_gate(
                train_candidate.f1,
                train_eval.f1,
                self.config.optimizer.stage1_max_train_regression,
            ):
                record.decision = "REJECTED_STAGE1"
                record.decision_reason = (
                    f"train F1 {train_candidate.f1:.4f} < incumbent {train_eval.f1:.4f} "
                    f"- {self.config.optimizer.stage1_max_train_regression}"
                )
                self.store.log_iteration(self.run_id, record)
                continue

            val_candidate = await self._eval_split(mutation.artifact, "val", iteration, candidate_id)
            record.val_f1 = val_candidate.f1
            record.subtree_scores = val_candidate.merged_subtrees
            (iteration_dir / f"{candidate_id}_val.json").write_text(
                json.dumps(val_candidate.to_dict(), indent=2)
            )

            entry = ArchiveEntry(
                candidate_id=candidate_id,
                val_f1=val_candidate.f1,
                aggregate_f1=val_candidate.aggregate_f1,
                artifact=mutation.artifact,
                subtree_scores=val_candidate.merged_subtrees,
            )
            before = len(self.front)
            new_front = add_to_front(self.front, entry)
            accepted = any(e.candidate_id == candidate_id for e in new_front)
            dominated_removed = len(new_front) < before + 1
            self.front = new_front

            if accepted:
                self.store.add_to_archive(
                    self.run_id, candidate_id, val_candidate.f1,
                    val_candidate.aggregate_f1, val_candidate.merged_subtrees,
                )
                record.decision = "ACCEPTED"
                record.decision_reason = (
                    f"val F1 {val_candidate.f1:.4f} vs incumbent {incumbent_entry.val_f1:.4f}; "
                    f"aggregate {val_candidate.aggregate_f1:.4f}"
                )
                improved = improved or val_candidate.f1 > incumbent_entry.val_f1
            else:
                record.decision = "REJECTED_DOMINATED"
                record.decision_reason = (
                    f"val F1 {val_candidate.f1:.4f} dominated on (val, aggregate) "
                    f"by existing front entries"
                )
            if dominated_removed and accepted:
                self._sync_archive_from_front()
            self.store.log_iteration(self.run_id, record)

        if not mutations:
            self.store.add_tabu(self.run_id, None, f"iteration {iteration}: no viable mutations")
        return improved

    async def _eval_split(self, artifact: PromptArtifact, split: str, iteration: int, tag: str) -> SplitEval:
        key = (artifact.hash(), tag, split)
        if key in self._split_cache:
            return self._split_cache[key]
        evaluator = Evaluator(self.extractor, self.scorer, self.schema, self.docs, iteration=iteration)
        result = await evaluator.evaluate_async(artifact, self.splits[split])
        self._split_cache[key] = result
        return result

    # lazily attached in __init__ via property trick to keep signature small
    def _sync_archive_from_front(self) -> None:
        keep = {e.candidate_id for e in self.front}
        for row in self.store.load_archive(self.run_id):
            if row["candidate_id"] not in keep:
                self.store.remove_from_archive(self.run_id, row["candidate_id"])

    # -- rollback ------------------------------------------------------------------

    async def _rollback_pass(self, root: PromptArtifact) -> None:
        if not self.front:
            return
        best = best_selection(self.front, self.config.optimizer.pareto_aggregate_tolerance)
        if best.candidate_id == "root":
            return
        root_eval = self._incumbent_cache.get("root")
        if root_eval is None:
            root_eval = await self._eval_split(root, "val", 0, "root")
        regressed = regressed_subtrees(
            root_eval.merged_subtrees,
            best.subtree_scores,
            self.config.optimizer.rollback_improve_threshold,
        )
        if not regressed:
            return
        repaired = best.artifact.copy(new_id="rollback_best")
        for path in regressed:
            parent_json = (
                self.store.load_candidate(best.artifact.parent_ids[-1], self.run_id)
                if best.artifact.parent_ids
                else None
            )
            parent = PromptArtifact.from_dict(parent_json) if parent_json else root
            if path in parent.subtree_directives:
                repaired.subtree_directives[path] = parent.subtree_directives[path]
        if repaired.hash() == best.artifact.hash():
            return

        rollback_iteration = self.config.budget.max_iterations + 1
        eval_result = await self._eval_split(repaired, "val", rollback_iteration, "rollback_best")
        record = IterationRecord(
            iteration=rollback_iteration,
            candidate_id="rollback_best",
            parent_id=best.candidate_id,
            mutation_type="subtree_rollback",
            target_subtree=",".join(regressed),
            hypothesis=(
                f"Revert regressed subtrees {regressed} to parent directives while keeping "
                f"gains elsewhere"
            ),
            train_f1=None,
            val_f1=eval_result.f1,
            subtree_scores=eval_result.merged_subtrees,
        )
        entry = ArchiveEntry(
            candidate_id="rollback_best",
            val_f1=eval_result.f1,
            aggregate_f1=eval_result.aggregate_f1,
            artifact=repaired,
            subtree_scores=eval_result.merged_subtrees,
            delta_text="subtree_rollback",
            target_subtree=",".join(regressed),
        )
        before_best_key = self._selection_repr(best)
        new_front = add_to_front(self.front, entry)
        self.store.save_candidate(
            "rollback_best", self.run_id, rollback_iteration, repaired.to_dict(),
            repaired.hash(), "subtree_rollback", ",".join(regressed),
        )
        if any(e.candidate_id == "rollback_best" for e in new_front):
            self.front = new_front
            self.store.add_to_archive(
                self.run_id, "rollback_best", eval_result.f1,
                eval_result.aggregate_f1, eval_result.merged_subtrees,
            )
            new_best = best_selection(self.front, self.config.optimizer.pareto_aggregate_tolerance)
            record.decision = "ACCEPTED" if self._selection_repr(new_best) > before_best_key else "REJECTED_NO_GAIN"
            record.decision_reason = "subtree-local rollback sweep"
            self._sync_archive_from_front()
        else:
            record.decision = "REJECTED_DOMINATED"
            record.decision_reason = "rollback variant dominated by existing front"
        self.store.log_iteration(self.run_id, record)

    def _selection_repr(self, entry: ArchiveEntry) -> tuple:
        return selection_key_tuple(entry, self.config.optimizer.pareto_aggregate_tolerance)

    # -- finalize ---------------------------------------------------------------------

    async def _finalize(self, root: PromptArtifact) -> dict:
        best = best_selection(self.front, self.config.optimizer.pareto_aggregate_tolerance)
        final_dir = self.store.run_dir / "final"
        final_dir.mkdir(exist_ok=True)

        test_eval = await self._eval_split(best.artifact, "test", self.config.budget.max_iterations + 2, "test")
        self.store.update_run(
            self.run_id,
            status="completed",
            best_candidate_id=best.candidate_id,
            budget_spent_tokens=getattr(self.tracker, "spent_tokens", 0),
            budget_spent_usd=getattr(self.tracker, "spent_usd", 0.0),
        )

        summary = {
            "run_id": self.run_id,
            "best_candidate_id": best.candidate_id,
            "test_f1": test_eval.f1,
            "test_aggregate_f1": test_eval.aggregate_f1,
            "test_json_valid_rate": test_eval.json_valid_rate,
            "val_f1": best.val_f1,
            "pareto_front": [
                {"candidate_id": e.candidate_id, "val_f1": e.val_f1, "aggregate_f1": e.aggregate_f1}
                for e in self.front
            ],
            "spent_tokens": getattr(self.tracker, "spent_tokens", 0),
            "spent_usd": getattr(self.tracker, "spent_usd", 0.0),
        }
        (final_dir / "best_artifact.json").write_text(json.dumps(best.artifact.to_dict(), indent=2))
        (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (final_dir / "test_per_document.json").write_text(
            json.dumps(test_eval.to_dict(), indent=2, default=repr)
        )
        return summary


def selection_key_tuple(entry: ArchiveEntry, tolerance: float) -> tuple:
    from .safeguards import selection_key

    return selection_key(entry, tolerance)
