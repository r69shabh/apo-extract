"""PDF ingestion: sha256 identity, page count, inline/file-API upload caching,
and page-image rendering fallback (pypdfium2)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LARGE_PDF_BYTES = 15 * 1024 * 1024
GOOGLE_UPLOAD_TTL_SECONDS = 40 * 60 * 60  # Gemini File API keeps files 48h


def pdf_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_count(path: str | Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


@dataclass
class PdfAttachment:
    kind: str  # "inline_pdf" | "google_file" | "images" | "text"
    payload: object  # data-uri str | uploaded file uri str | list[data-uri]
    page_count: int
    sha256: str
    extra: dict = field(default_factory=dict)


class PdfHandler:
    """Caches Gemini File API uploads by sha256 so a PDF uploads at most once per run."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_cache_path = self.cache_dir / "pdf_uploads.json"
        self._uploads: dict = {}
        if self.upload_cache_path.exists():
            try:
                self._uploads = json.loads(self.upload_cache_path.read_text())
            except json.JSONDecodeError:
                self._uploads = {}

    def prepare(self, pdf_path: str | Path, mode: str = "auto") -> PdfAttachment:
        import time

        pdf_path = str(pdf_path)
        sha = pdf_sha256(pdf_path)
        size = os.path.getsize(pdf_path)

        if mode == "images":
            pages = self.render_page_images(pdf_path)
            return PdfAttachment("images", pages, len(pages), sha)

        if mode == "text":
            return PdfAttachment("text", self.extract_text(pdf_path), pdf_page_count(pdf_path), sha)

        if mode == "auto":
            mode = "inline_pdf" if size <= LARGE_PDF_BYTES else "google_file"

        if mode == "google_file":
            entry = self._uploads.get(sha)
            if entry and time.time() - entry["ts"] < GOOGLE_UPLOAD_TTL_SECONDS:
                return PdfAttachment("google_file", entry["uri"], entry["pages"], sha)
            uri, pages = self._upload_google(pdf_path, sha)
            self._uploads[sha] = {"uri": uri, "pages": pages, "ts": time.time()}
            self.upload_cache_path.write_text(json.dumps(self._uploads, indent=2))
            return PdfAttachment("google_file", uri, pages, sha)

        data_uri = self._inline_data_uri(pdf_path)
        return PdfAttachment(
            "inline_pdf",
            data_uri,
            pdf_page_count(pdf_path),
            sha,
            extra={"filename": os.path.basename(pdf_path)},
        )

    def extract_text(self, pdf_path: str | Path) -> str:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return "\n\n".join(
                pdf[index].get_textpage().get_text_range() for index in range(len(pdf))
            )
        finally:
            pdf.close()

    def _inline_data_uri(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as fh:
            return "data:application/pdf;base64," + base64.b64encode(fh.read()).decode()

    def _upload_google(self, pdf_path: str, sha: str) -> tuple[str, int]:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for File API uploads of large PDFs "
                "(or set llms.extractor.pdf_mode: images|inline)"
            )
        client = genai.Client(api_key=api_key)
        uploaded = client.files.upload(file=pdf_path)
        pages = pdf_page_count(pdf_path)
        return uploaded.uri, pages

    def render_page_images(self, pdf_path: str | Path, dpi: int = 88, max_pages: int | None = None) -> list[str]:
        # JPEG, not PNG: OpenRouter rejects >30MB of image content per request (base64 counted)
        # and vision models rescale to ~768px tiles, so dense PNGs buy nothing but bytes.
        import io

        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        data_uris = []
        try:
            count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
            for index in range(count):
                page = pdf[index]
                bitmap = page.render(scale=dpi / 72)
                image = bitmap.to_pil().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=75)
                data_uris.append(
                    "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
                )
        finally:
            pdf.close()
        return data_uris
