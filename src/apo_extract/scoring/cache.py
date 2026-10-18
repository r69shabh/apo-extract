"""Persistent deterministic cache for scoring results (SQLite, WAL mode)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
"""


def canonical_json(value: Any) -> str:
    """Stable JSON encoding; non-JSON values fall back to repr."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=repr
    )


def cache_key(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(canonical_json(p) for p in parts).encode()).hexdigest()


class ScoringCache:
    """Tiny durable key-value cache shared by metrics and LLM judges."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._local = threading.local()
        self._conn: sqlite3.Connection | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get(self, key: str) -> Any | None:
        if self._conn is None:
            return None
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=True, default=repr)),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
