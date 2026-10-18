from apo_extract.optimizer.diagnostic import DOCUMENT_TARGET, FailureCluster
from apo_extract.optimizer.proposer import Proposer
from apo_extract.predictor.artifact import PromptArtifact

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "object"}}},
}


def _proposer() -> Proposer:
    return Proposer(SCHEMA, client=None, store=None, run_id="t", duplicate_similarity=0.92)


def test_template_fallback_covers_array_item_mismatch():
    root = PromptArtifact.init_from_schema(SCHEMA, "Extract.", artifact_id="root")
    clusters = [
        FailureCluster(subtree="items", error_type="array_item_mismatch", count=3, docs=["a"]),
    ]
    proposals = _proposer()._propose_templates(root, clusters, [], ["c0"])
    assert len(proposals) == 1
    assert proposals[0].mutation_type == "directive_array_item_mismatch"
    assert "$.items" in proposals[0].artifact.subtree_directives


def test_template_fallback_unknown_type_uses_mismatch_template():
    root = PromptArtifact.init_from_schema(SCHEMA, "Extract.", artifact_id="root")
    clusters = [
        FailureCluster(subtree=DOCUMENT_TARGET, error_type="brand_new_kind", count=1),
    ]
    proposals = _proposer()._propose_templates(root, clusters, [], ["c0"])
    assert len(proposals) == 1
    assert proposals[0].mutation_type == "preamble_hardening"
    assert "Copy values exactly as written" in proposals[0].artifact.global_preamble
