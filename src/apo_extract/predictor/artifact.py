"""PromptArtifact — the versionable unit the optimizer mutates.

Fields (Codified Context / SR-PBO):
  global_preamble            task framing, output contract
  formatting_and_null_policy canonical JSON policy (nulls, numbers, dates)
  subtree_directives         per top-level-property scoped instructions, keyed "$.field"
  verification_checklist     self-check steps appended for the extractor
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class PromptArtifact:
    global_preamble: str
    formatting_and_null_policy: str = ""
    subtree_directives: dict[str, str] = field(default_factory=dict)
    verification_checklist: str = ""
    artifact_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0

    # -- construction ---------------------------------------------------------

    @classmethod
    def init_from_schema(cls, schema: dict, seed_text: str, artifact_id: str = "root") -> "PromptArtifact":
        directives = {}
        for prop in schema.get("properties", {}):
            directives[f"$.{prop}"] = ""
        return cls(
            global_preamble=seed_text.strip(),
            formatting_and_null_policy=DEFAULT_NULL_POLICY,
            subtree_directives=directives,
            artifact_id=artifact_id,
        )

    def copy(self, new_id: str, generation: int | None = None) -> "PromptArtifact":
        return PromptArtifact(
            global_preamble=self.global_preamble,
            formatting_and_null_policy=self.formatting_and_null_policy,
            subtree_directives=dict(self.subtree_directives),
            verification_checklist=self.verification_checklist,
            artifact_id=new_id,
            parent_ids=[self.artifact_id],
            generation=self.generation + 1 if generation is None else generation,
        )

    # -- identity / diffing ----------------------------------------------------

    def content_dict(self) -> dict:
        return {
            "global_preamble": self.global_preamble,
            "formatting_and_null_policy": self.formatting_and_null_policy,
            "subtree_directives": {k: self.subtree_directives[k] for k in sorted(self.subtree_directives)},
            "verification_checklist": self.verification_checklist,
        }

    def hash(self) -> str:
        payload = json.dumps(self.content_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def normalized_text(self) -> str:
        return " ".join(json.dumps(self.content_dict(), sort_keys=True).lower().split())

    def estimate_tokens(self) -> int:
        return len(self.render(None)) // 4

    # -- rendering -------------------------------------------------------------

    def render(self, schema: dict | None, subtree: str | None = None, subtree_names: list[str] | None = None) -> str:
        """Render the full prompt, or only one subtree's directive (chunked extraction)."""
        if subtree is not None:
            directive = self.subtree_directives.get(subtree, "")
            return (
                f"{self.global_preamble}\n\n{self.formatting_and_null_policy}\n\n"
                f"=== FIELD-SPECIFIC GUIDANCE: {subtree} ===\n{directive}".strip()
            )

        sections = [self.global_preamble, self.formatting_and_null_policy]
        if subtree_names is None:
            subtree_names = [k for k in sorted(self.subtree_directives)]
        lines = ["=== FIELD-SPECIFIC GUIDANCE ==="]
        for key in subtree_names:
            directive = self.subtree_directives.get(key, "").strip()
            if directive:
                lines.append(f"### {key}\n{directive}")
        if len(lines) > 1:
            sections.append("\n".join(lines))
        if self.verification_checklist.strip():
            sections.append("=== VERIFICATION CHECKLIST ===\n" + self.verification_checklist.strip())
        return "\n\n".join(s for s in sections if s and s.strip())

    # -- serialization ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            **self.content_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptArtifact":
        return cls(
            global_preamble=data.get("global_preamble", ""),
            formatting_and_null_policy=data.get("formatting_and_null_policy", ""),
            subtree_directives=dict(data.get("subtree_directives", {})),
            verification_checklist=data.get("verification_checklist", ""),
            artifact_id=data.get("artifact_id", ""),
            parent_ids=list(data.get("parent_ids", [])),
            generation=int(data.get("generation", 0)),
        )


DEFAULT_NULL_POLICY = (
    "Output contract:\n"
    "- Return exactly one JSON object and nothing else (no markdown fences, no commentary).\n"
    "- Use null for any field that is not clearly present in the document; never guess.\n"
    "- Numbers: plain JSON numbers without currency symbols or thousands separators.\n"
    "- Dates: copy the document's date string verbatim; do not reformat.\n"
    "- Strings: extract verbatim text; do not paraphrase or summarize.\n"
    "- Arrays: include every element the schema asks for, in document order.\n"
    "- Enums: only the exact allowed values; otherwise null."
)
