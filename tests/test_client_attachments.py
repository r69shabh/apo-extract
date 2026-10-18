"""Regression: attached PDF data URIs must reach litellm intact; raw logs stay shrunk.

The live hiring_resume run failed because the log-size shrinker replaced the messages
handed to litellm, so Gemini received "<data-uri N bytes>" placeholders.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from apo_extract.config import LlmConfig
from apo_extract.llm.client import LLMClient
from apo_extract.llm.pdf_handler import PdfAttachment, PdfHandler
from apo_extract.observability.store import RunStore

DATA_URI = "data:application/pdf;base64,JVBERi0xLjQKJcOkw7zDtsOfCg=="


def _install_fake_litellm(captured: dict):
    fake = types.ModuleType("litellm")

    async def acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2),
            model=kwargs["model"],
        )

    fake.acompletion = acompletion
    fake.cost_per_token = lambda **kwargs: (0.0, 0.0)
    fake.completion_cost = lambda **kwargs: 0.0
    return fake


async def test_inline_pdf_reaches_litellm_and_logs_are_shrunk(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "litellm", _install_fake_litellm(captured))

    store = RunStore(tmp_path / "run")
    store.init_run("r1", "hash")
    client = LLMClient(LlmConfig(model="fake"), role="extractor", store=store, run_id="r1")
    attachment = PdfAttachment(
        kind="inline_pdf", payload=DATA_URI, page_count=1, sha256="x",
        extra={"filename": "doc.pdf"},
    )

    await client.complete(
        [{"role": "user", "content": "extract"}], attachment=attachment, pdf_id="doc1"
    )

    sent = json.dumps(captured["messages"])
    assert DATA_URI in sent, "the live call must carry the real data URI"
    assert "<data-uri" not in sent

    raw_files = list((tmp_path / "run" / "raw").glob("req_*.json"))
    assert len(raw_files) == 1
    logged = raw_files[0].read_text()
    assert "<data-uri" in logged and DATA_URI not in logged


async def test_text_attachment_reaches_litellm(monkeypatch, tmp_path):
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "litellm", _install_fake_litellm(captured))

    store = RunStore(tmp_path / "run")
    store.init_run("r1", "hash")
    client = LLMClient(
        LlmConfig(provider="nvidia_nim", model="z-ai/glm-5.3-flash"),
        role="extractor", store=store, run_id="r1",
    )
    attachment = PdfAttachment(kind="text", payload="DOC TEXT BODY", page_count=1, sha256="x")

    await client.complete(
        [{"role": "user", "content": "extract"}], attachment=attachment, pdf_id="doc1"
    )

    sent = json.dumps(captured["messages"])
    assert "DOC TEXT BODY" in sent
    assert captured["model"] == "nvidia_nim/z-ai/glm-5.3-flash"


def test_prepare_text_mode_extracts_pdf_text(tmp_path):
    pdf_path = Path("data/extract-bench/dataset/hiring/resume/pdf+gold/Resume-Academic01.pdf")
    assert pdf_path.exists(), "repo must be run from its root for dataset-relative tests"

    attachment = PdfHandler(tmp_path / "cache").prepare(pdf_path, mode="text")

    assert attachment.kind == "text"
    assert attachment.page_count > 1
    assert len(attachment.payload) > 500


def test_prepare_images_mode_renders_bounded_jpeg(tmp_path):
    pdf_path = Path("data/extract-bench/dataset/sport/swimming/pdf+gold/ma_2023_sw_M-table1.pdf")
    assert pdf_path.exists(), "repo must be run from its root for dataset-relative tests"

    attachment = PdfHandler(tmp_path / "cache").prepare(pdf_path, mode="images")

    assert attachment.kind == "images"
    assert len(attachment.payload) == attachment.page_count
    assert all(uri.startswith("data:image/jpeg;base64,") for uri in attachment.payload)
    # huge PDFs must stay under OpenRouter's 30MB request cap (see amzn_credit_agreement, 218p)
    assert sum(len(uri) for uri in attachment.payload) < 1_000_000
