"""Dataset loader: (PDF, schema, gold) triplets from the as-published ExtractBench tree.

Reads files exactly as shipped (including the ``schema_definition`` envelope some
schemas use) and never modifies, extends, or re-annotates them.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DatasetConfig


@dataclass
class DocumentItem:
    doc_id: str
    pdf_path: str
    gold_path: str
    gold: dict = field(repr=False)


@dataclass
class DatasetBundle:
    name: str
    schema: dict
    schema_path: str
    documents: list[DocumentItem]

    def by_id(self, doc_id: str) -> DocumentItem:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)


def load_schema(schema_path: str | Path) -> dict:
    schema = json.loads(Path(schema_path).read_text())
    if isinstance(schema, dict) and "schema_definition" in schema:
        schema = schema["schema_definition"]
    return schema


def load_dataset(cfg: DatasetConfig) -> DatasetBundle:
    schema = load_schema(cfg.schema_path)
    pdf_paths = sorted(glob.glob(cfg.documents_glob))
    if not pdf_paths:
        raise FileNotFoundError(f"documents_glob matched no PDFs: {cfg.documents_glob}")

    documents: list[DocumentItem] = []
    for pdf_path in pdf_paths:
        stem = Path(pdf_path).stem
        gold_name = cfg.gold_match_pattern.format(doc_stem=stem)
        gold_path = Path(cfg.gold_dir) / gold_name
        if not gold_path.exists():
            raise FileNotFoundError(f"missing gold file for {pdf_path}: {gold_path}")
        documents.append(
            DocumentItem(
                doc_id=stem,
                pdf_path=str(pdf_path),
                gold_path=str(gold_path),
                gold=json.loads(gold_path.read_text()),
            )
        )
    return DatasetBundle(
        name=cfg.name,
        schema=schema,
        schema_path=str(cfg.schema_path),
        documents=documents,
    )
