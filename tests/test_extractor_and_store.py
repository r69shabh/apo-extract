"""Artifact rendering/hashing, mock-completer extraction, and prediction caching."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from apo_extract.config import LlmConfig
from apo_extract.dataset.loader import DocumentItem
from apo_extract.llm.client import LLMClient
from apo_extract.observability.store import RunStore
from apo_extract.predictor.artifact import PromptArtifact
from apo_extract.predictor.extractor import Extractor, parse_json_output

DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "extract-bench", "dataset")

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor_name": {"type": "string"},
        "amount": {"type": "number"},
    },
    "required": ["vendor_name"],
}


def make_item(tmp_path) -> DocumentItem:
    pdf = tmp_path / "doc1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return DocumentItem(doc_id="doc1", pdf_path=str(pdf), gold_path="", gold={})


class FakeCompleter:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def __call__(self, request: dict):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            model=request["model"],
        )


class TestArtifact:
    def test_init_creates_directive_per_top_level_property(self):
        artifact = PromptArtifact.init_from_schema(SCHEMA, "Seed instructions.")
        assert set(artifact.subtree_directives) == {"$.vendor_name", "$.amount"}

    def test_hash_changes_on_edit_and_roundtrips(self):
        a = PromptArtifact.init_from_schema(SCHEMA, "seed")
        original = a.hash()
        a.subtree_directives["$.amount"] = "Extract the total as a plain number."
        assert a.hash() != original
        b = PromptArtifact.from_dict(a.to_dict())
        assert b.hash() == a.hash()
        assert b.artifact_id == a.artifact_id

    def test_render_includes_only_nonempty_directives(self):
        a = PromptArtifact.init_from_schema(SCHEMA, "seed")
        a.subtree_directives["$.amount"] = "Amount directive."
        rendered = a.render(SCHEMA)
        assert "Amount directive." in rendered
        assert "$.vendor_name" not in rendered
        assert "VERIFICATION CHECKLIST" not in rendered

    def test_subtree_render(self):
        a = PromptArtifact.init_from_schema(SCHEMA, "seed")
        a.subtree_directives["$.amount"] = "Amount directive."
        assert "FIELD-SPECIFIC GUIDANCE: $.amount" in a.render(SCHEMA, subtree="$.amount")


class TestParseJson:
    def test_plain_and_fenced(self):
        assert parse_json_output('{"a": 1}') == ({"a": 1}, True)
        assert parse_json_output('```json\n{"a": 1}\n```') == ({"a": 1}, True)

    def test_garbage(self):
        assert parse_json_output("no json here at all")[1] is False


class TestExtractorCaching:
    def test_second_extraction_hits_cache(self, tmp_path):
        store = RunStore(tmp_path / "run")
        store.init_run("r1", "hash")
        client = LLMClient(
            LlmConfig(model="fake"),
            role="extractor",
            store=store,
            run_id="r1",
            completer=FakeCompleter({"vendor_name": "Acme", "amount": 3}),
        )
        extractor = Extractor(client, store, pdf_handler=None)
        artifact = PromptArtifact.init_from_schema(SCHEMA, "seed")
        item = make_item(tmp_path)

        first = asyncio.run(extractor.extract(artifact, item, SCHEMA))
        second = asyncio.run(extractor.extract(artifact, item, SCHEMA))

        assert first.json_valid and second.json_valid
        assert second.from_cache
        assert client.completer.calls == 1
        assert second.data == {"vendor_name": "Acme", "amount": 3}

    def test_cache_survives_store_reopen(self, tmp_path):
        item = make_item(tmp_path)
        artifact = PromptArtifact.init_from_schema(SCHEMA, "seed")
        run_dir = tmp_path / "run"

        store1 = RunStore(run_dir)
        store1.init_run("r1", "hash")
        client1 = LLMClient(
            LlmConfig(model="fake"), role="extractor", store=store1, run_id="r1",
            completer=FakeCompleter({"vendor_name": "Acme"}),
        )
        asyncio.run(Extractor(client1, store1, pdf_handler=None).extract(artifact, item, SCHEMA))
        store1.close()

        store2 = RunStore(run_dir)
        client2 = LLMClient(
            LlmConfig(model="fake"), role="extractor", store=store2, run_id="r1",
            completer=FakeCompleter({"vendor_name": "WRONG"}),
        )
        resumed = asyncio.run(Extractor(client2, store2, pdf_handler=None).extract(artifact, item, SCHEMA))
        assert resumed.from_cache
        assert resumed.data == {"vendor_name": "Acme"}
        assert client2.completer.calls == 0

    def test_llm_calls_logged_with_tokens(self, tmp_path):
        store = RunStore(tmp_path / "run")
        store.init_run("r1", "hash")
        client = LLMClient(
            LlmConfig(model="fake"), role="extractor", store=store, run_id="r1",
            completer=FakeCompleter({"vendor_name": "Acme"}),
        )
        item = make_item(tmp_path)
        artifact = PromptArtifact.init_from_schema(SCHEMA, "seed")
        asyncio.run(Extractor(client, store, pdf_handler=None).extract(artifact, item, SCHEMA))
        tokens, usd = store.spent_totals("r1")
        assert tokens == 120
        rows = store.list_iterations("r1")
        assert rows == []
        assert (tmp_path / "run" / "llm_calls.jsonl").exists()
