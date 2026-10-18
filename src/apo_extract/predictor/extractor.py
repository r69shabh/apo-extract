"""Extractor: renders the artifact, calls the LLM, parses/caches predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..dataset.loader import DocumentItem
from ..llm.client import LLMClient
from ..llm.pdf_handler import PdfHandler, pdf_sha256
from .artifact import PromptArtifact

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class Prediction:
    data: dict
    raw: str
    json_valid: bool
    from_cache: bool = False


def parse_json_output(text: str) -> tuple[dict | None, bool]:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json

        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired, True
    except Exception:
        pass
    return None, False


class Extractor:
    def __init__(
        self,
        client: LLMClient,
        store=None,
        *,
        pdf_handler: PdfHandler | None = None,
        pdf_mode: str = "auto",
        mode: str = "single_call",
        chunk_size: int = 20,
    ):
        self.client = client
        self.store = store
        self.pdf_handler = pdf_handler
        self.pdf_mode = pdf_mode
        self.mode = mode
        self.chunk_size = chunk_size

    async def extract(
        self,
        artifact: PromptArtifact,
        item: DocumentItem,
        schema: dict,
        iteration: int | None = None,
    ) -> Prediction:
        prompt_hash = artifact.hash()
        pdf_hash = pdf_sha256(item.pdf_path)
        if self.store is not None:
            cached = self.store.get_prediction(prompt_hash, pdf_hash, self.client.model_string)
            if cached is not None:
                return Prediction(
                    data=cached.prediction or {},
                    raw=cached.raw_response_path or "",
                    json_valid=cached.json_valid,
                    from_cache=True,
                )

        properties = list(schema.get("properties", {}))
        if self.mode == "subtree_chunked" and len(properties) > self.chunk_size:
            prediction = await self._extract_chunked(artifact, item, schema, properties, iteration)
        else:
            prediction = await self._extract_once(artifact, item, schema, prompt_hash, iteration)

        if self.store is not None and not prediction.from_cache:
            self.store.put_prediction(
                prompt_hash,
                pdf_hash,
                self.client.model_string,
                prediction.data,
                prediction.json_valid,
            )
        return prediction

    async def _extract_once(
        self,
        artifact: PromptArtifact,
        item: DocumentItem,
        schema: dict,
        prompt_hash: str,
        iteration: int | None,
    ) -> Prediction:
        attachment = None
        if self.pdf_handler is not None:
            attachment = self.pdf_handler.prepare(item.pdf_path, self.pdf_mode)
        messages = self._messages(artifact.render(schema), schema, item)
        result = await self.client.complete(
            messages,
            attachment=attachment,
            iteration=iteration,
            pdf_id=item.doc_id,
            prompt_hash=prompt_hash,
            response_format={"type": "json_object"},
            purpose="extraction",
        )
        parsed, valid = parse_json_output(result.text)
        return Prediction(data=parsed or {}, raw=result.text, json_valid=valid)

    async def _extract_chunked(
        self,
        artifact: PromptArtifact,
        item: DocumentItem,
        schema: dict,
        properties: list[str],
        iteration: int | None,
    ) -> Prediction:
        attachment = None
        if self.pdf_handler is not None:
            attachment = self.pdf_handler.prepare(item.pdf_path, self.pdf_mode)
        merged: dict = {}
        raws: list[str] = []
        all_valid = True
        for start in range(0, len(properties), self.chunk_size):
            batch = properties[start : start + self.chunk_size]
            sub_schema = {
                "type": "object",
                "properties": {k: schema["properties"][k] for k in batch},
                "required": [k for k in schema.get("required", []) if k in batch],
            }
            system = artifact.render(schema, subtree_names=[f"$.{k}" for k in batch])
            messages = self._messages(system, sub_schema, item)
            result = await self.client.complete(
                messages,
                attachment=attachment,
                iteration=iteration,
                pdf_id=item.doc_id,
                prompt_hash=artifact.hash(),
                response_format={"type": "json_object"},
                purpose="extraction_chunk",
            )
            raws.append(result.text)
            parsed, valid = parse_json_output(result.text)
            all_valid = all_valid and valid
            if parsed:
                for key in batch:
                    if key in parsed:
                        merged[key] = parsed[key]
        return Prediction(data=merged, raw="\n---chunk---\n".join(raws), json_valid=all_valid)

    @staticmethod
    def _messages(system_prompt: str, schema: dict, item: DocumentItem) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Extract structured data from the attached document into JSON that "
                    "conforms to this schema:\n"
                    f"{json.dumps(schema, ensure_ascii=True)}\n\n"
                    "Return only the JSON object."
                ),
            },
        ]
