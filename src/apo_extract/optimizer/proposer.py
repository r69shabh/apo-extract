"""Phase 2 proposal: LLM-assisted edits with a deterministic offline fallback,
plus crossover of complementary fixes.

Deduplication operates on each mutation's *delta text* (only what the edit adds),
never on the whole prompt: siblings of the same incumbent share ~95% of their
text, so whole-prompt similarity would reject every useful incremental edit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..predictor.artifact import PromptArtifact
from .diagnostic import DOCUMENT_TARGET, FailureCluster
from .safeguards import is_duplicate, is_duplicate_for_target

_TEMPLATES: dict[str, str] = {
    "invalid_json": (
        "Return only a single valid JSON object with no markdown fences, comments, or "
        "trailing text. If a field is unknown, use null rather than omitting the key."
    ),
    "omission": (
        "Documents in this dataset frequently state this information in less obvious "
        "places (headers, footnotes, exhibits, or amended clauses). Search the entire "
        "document before deciding a value is absent, and only then return null."
    ),
    "spurious": (
        "Never invent values. If the document does not explicitly state a value for "
        "this field, return null \u2014 do not infer or copy values from similar fields."
    ),
    "mismatch": (
        "Copy values exactly as written in the document: same wording, same units, same "
        "digit grouping. Do not normalize, round, translate, or expand abbreviations."
    ),
    "array_item_mismatch": (
        "For array fields, extract one entry per real item and copy each entry's values "
        "exactly as written. Do not merge two items into one, split one item into several, "
        "reorder, or drop entries that do not match an example."
    ),
}

_CHECKLIST_ITEMS = {
    "omission": "Re-scan the full document once for any field left null before finalizing.",
    "spurious": "Verify every non-null value appears verbatim in the document text.",
    "mismatch": "Verify numbers and dates match the document character-for-character.",
    "invalid_json": "Verify your final answer parses as a JSON object.",
}

_MUTATION_PROMPT = """You are optimizing an extraction prompt for a fixed JSON schema.

Current prompt sections:
{sections}

Observed failure cluster: {count} errors of type "{error_type}" in subtree "{subtree}".
Representative failures (up to 3):
{examples}

Previously tried hypotheses (do not repeat):
{tabu}

Propose up to {k} minimal, surgical prompt edits that fix this failure class without
breaking other fields. Each edit targets exactly one section: a subtree key like
"{subtree}", "global_preamble", or "verification_checklist".
Edit text must be additive guidance; never delete existing instructions.

Respond with JSON only:
{{"edits": [{{"target": "...", "text": "...", "hypothesis": "..."}}]}}
"""


@dataclass
class ProposedMutation:
    artifact: PromptArtifact
    mutation_type: str
    target_subtree: str
    hypothesis: str
    delta_text: str = ""


class Proposer:
    def __init__(
        self,
        schema: dict,
        *,
        client=None,
        store=None,
        run_id: str = "",
        duplicate_similarity: float = 0.92,
        seed_text: str = "",
    ):
        self.schema = schema
        self.client = client
        self.store = store
        self.run_id = run_id
        self.duplicate_similarity = duplicate_similarity
        self.seed_text = seed_text

    async def propose_async(
        self,
        incumbent: PromptArtifact,
        clusters: list[FailureCluster],
        reference_deltas: list[tuple[str, str]],
        candidate_ids: list[str],
        *,
        other_artifacts: list[PromptArtifact] | None = None,
        iteration: int | None = None,
    ) -> list[ProposedMutation]:
        mutations: list[ProposedMutation] = []

        if self.client is not None and clusters:
            mutations = await self._propose_llm(
                incumbent, clusters, reference_deltas, candidate_ids, iteration
            )

        if len(mutations) < len(candidate_ids):
            fallback = self._propose_templates(
                incumbent, clusters, reference_deltas, candidate_ids
            )
            seen_ids = {m.artifact.hash() for m in mutations}
            mutations.extend(m for m in fallback if m.artifact.hash() not in seen_ids)

        if len(mutations) < len(candidate_ids):
            crossover = self._propose_crossover(
                incumbent,
                other_artifacts or [],
                reference_deltas + [(m.target_subtree, m.delta_text) for m in mutations],
                new_id=candidate_ids[len(mutations)],
            )
            if crossover is not None:
                mutations.append(crossover)

        for mutation in mutations:
            if self.store is not None:
                self.store.add_tabu(self.run_id, mutation.target_subtree, mutation.hypothesis)
        return mutations[: len(candidate_ids)]

    # -- deterministic fallback (always available, no API key needed) --------

    def _propose_templates(
        self,
        incumbent: PromptArtifact,
        clusters: list[FailureCluster],
        reference_deltas: list[tuple[str, str]],
        candidate_ids: list[str],
    ) -> list[ProposedMutation]:
        proposals: list[ProposedMutation] = []
        occupied = list(reference_deltas)
        cluster_index = 0
        for candidate_id in candidate_ids:
            if cluster_index >= len(clusters):
                break
            cluster = clusters[cluster_index]
            cluster_index += 1

            candidate = incumbent.copy(new_id=candidate_id)
            addition = _TEMPLATES.get(cluster.error_type, _TEMPLATES["mismatch"])
            if cluster.error_type == "invalid_json" or cluster.subtree == DOCUMENT_TARGET:
                candidate.global_preamble = f"{candidate.global_preamble.rstrip()}\n\n{addition}"
                mutation_type, target = "preamble_hardening", "global_preamble"
            else:
                key = f"$.{cluster.subtree.lstrip('$').lstrip('.')}"
                existing = candidate.subtree_directives.get(key, "").strip()
                candidate.subtree_directives[key] = f"{existing}\n{addition}".strip()
                mutation_type, target = f"directive_{cluster.error_type}", key

            delta = addition
            checklist_item = _CHECKLIST_ITEMS.get(cluster.error_type, "")
            if checklist_item and checklist_item not in candidate.verification_checklist:
                candidate.verification_checklist = (
                    f"{candidate.verification_checklist.rstrip()}\n- {checklist_item}".strip()
                )
                delta = f"{delta}\n{checklist_item}"

            if is_duplicate_for_target(target, delta, occupied, self.duplicate_similarity):
                continue
            occupied.append((target, delta))
            hypothesis = (
                f"Add {cluster.error_type} guidance for {target} to fix "
                f"{cluster.count} observed failures ({', '.join(cluster.docs[:3])})."
            )
            proposals.append(
                ProposedMutation(
                    artifact=candidate,
                    mutation_type=mutation_type,
                    target_subtree=target,
                    hypothesis=hypothesis,
                    delta_text=delta,
                )
            )
        return proposals

    # -- crossover: merge directives from complementary front entries --------

    def _propose_crossover(
        self,
        incumbent: PromptArtifact,
        other_artifacts: list[PromptArtifact],
        reference_deltas: list[tuple[str, str]],
        new_id: str,
    ) -> ProposedMutation | None:
        merged_texts: list[str] = []
        for other in other_artifacts:
            if other.hash() == incumbent.hash():
                continue
            candidate = incumbent.copy(new_id=new_id)
            found = False
            for path, text in other.subtree_directives.items():
                text = (text or "").strip()
                if not text:
                    continue
                existing = candidate.subtree_directives.get(path, "").strip()
                if text not in existing:
                    candidate.subtree_directives[path] = f"{existing}\n{text}".strip()
                    merged_texts.append(text)
                    found = True
            for item in (other.verification_checklist or "").splitlines():
                item = item.strip().lstrip("- ").strip()
                if item and item not in candidate.verification_checklist:
                    candidate.verification_checklist = (
                        f"{candidate.verification_checklist.rstrip()}\n- {item}".strip()
                    )
                    merged_texts.append(item)
                    found = True
            if found:
                delta = "\n".join(merged_texts)
                if is_duplicate_for_target("merged", delta, reference_deltas, self.duplicate_similarity):
                    return None
                return ProposedMutation(
                    artifact=candidate,
                    mutation_type="crossover",
                    target_subtree="merged",
                    hypothesis="Merge complementary directive fixes from Pareto front entries.",
                    delta_text=delta,
                )
        return None

    # -- LLM-assisted path (used live; falls back on any failure) ------------

    async def _propose_llm(
        self,
        incumbent: PromptArtifact,
        clusters: list[FailureCluster],
        reference_deltas: list[tuple[str, str]],
        candidate_ids: list[str],
        iteration: int | None,
    ) -> list[ProposedMutation]:
        cluster = clusters[0]
        tabu = []
        if self.store is not None:
            tabu = [row["hypothesis"] for row in self.store.load_tabu(self.run_id, limit=10)]
        prompt = _MUTATION_PROMPT.format(
            sections=json.dumps(
                {
                    "global_preamble": incumbent.global_preamble,
                    "subtree_directives": incumbent.subtree_directives,
                    "verification_checklist": incumbent.verification_checklist,
                },
                indent=1,
            ),
            count=cluster.count,
            error_type=cluster.error_type,
            subtree=cluster.subtree,
            examples=json.dumps(cluster.examples, indent=1, default=repr),
            tabu=json.dumps(tabu),
            k=len(candidate_ids),
        )
        try:
            result = await self.client.complete(
                [
                    {"role": "system", "content": "You propose minimal prompt edits. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                iteration=iteration,
                response_format={"type": "json_object"},
                purpose="proposal",
            )
            payload = json.loads(re.sub(r"^```(?:json)?|```$", "", result.text.strip(), flags=re.MULTILINE))
        except Exception:
            return []

        proposals: list[ProposedMutation] = []
        occupied = list(reference_deltas)
        edits = payload.get("edits", []) if isinstance(payload, dict) else []
        for edit, candidate_id in zip(edits, candidate_ids):
            target = str(edit.get("target", "")).strip()
            text = str(edit.get("text", "")).strip()
            hypothesis = str(edit.get("hypothesis", "")).strip()
            if not target or not text or not hypothesis:
                continue
            candidate = incumbent.copy(new_id=candidate_id)
            delta = text
            if target == "global_preamble":
                candidate.global_preamble = f"{candidate.global_preamble.rstrip()}\n\n{text}"
                mutation_type = "llm_preamble_edit"
            elif target == "verification_checklist":
                candidate.verification_checklist = (
                    f"{candidate.verification_checklist.rstrip()}\n- {text}".strip()
                )
                mutation_type = "llm_checklist_edit"
            elif target in candidate.subtree_directives or f"$.{target}" in candidate.subtree_directives:
                key = target if target.startswith("$.") else f"$.{target}"
                existing = candidate.subtree_directives[key].strip()
                candidate.subtree_directives[key] = f"{existing}\n{text}".strip()
                mutation_type = "llm_directive_edit"
            else:
                continue

            if is_duplicate_for_target(target, delta, occupied, self.duplicate_similarity):
                continue
            occupied.append((target, delta))
            proposals.append(
                ProposedMutation(
                    artifact=candidate,
                    mutation_type=mutation_type,
                    target_subtree=target,
                    hypothesis=hypothesis,
                    delta_text=delta,
                )
            )
        return proposals
