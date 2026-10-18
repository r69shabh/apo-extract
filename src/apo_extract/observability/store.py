"""Crash-safe run persistence: SQLite (WAL) + JSONL + per-iteration artifacts.

Everything needed for ``apo-extract resume`` lives under ``runs/<run_id>/``.
Killing the process at any moment must lose no completed work: predictions are
cached per (prompt_hash, pdf_hash, model) and iterations are committed atomically.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config_hash TEXT,
    config_path TEXT,
    status TEXT DEFAULT 'running',
    budget_spent_tokens INTEGER DEFAULT 0,
    budget_spent_usd REAL DEFAULT 0.0,
    started_at REAL,
    current_iteration INTEGER DEFAULT 0,
    stall INTEGER DEFAULT 0,
    best_candidate_id TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    role TEXT,
    model TEXT,
    prompt_hash TEXT,
    pdf_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    raw_request_path TEXT,
    raw_response_path TEXT,
    timestamp REAL
);
CREATE TABLE IF NOT EXISTS predictions_cache (
    prompt_hash TEXT,
    pdf_hash TEXT,
    extractor_model TEXT,
    prediction_json TEXT,
    json_valid INTEGER,
    raw_response_path TEXT,
    created_at REAL,
    PRIMARY KEY (prompt_hash, pdf_hash, extractor_model)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT,
    run_id TEXT,
    iteration INTEGER,
    artifact_json TEXT,
    prompt_hash TEXT,
    delta_text TEXT DEFAULT '',
    target_subtree TEXT DEFAULT '',
    created_at REAL,
    PRIMARY KEY (candidate_id, run_id)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT,
    iteration INTEGER,
    candidate_id TEXT,
    parent_id TEXT,
    mutation_type TEXT,
    target_subtree TEXT,
    hypothesis TEXT,
    train_f1 REAL,
    val_f1 REAL,
    subtree_scores_json TEXT,
    decision TEXT,
    decision_reason TEXT,
    created_at REAL,
    PRIMARY KEY (run_id, iteration, candidate_id)
);
CREATE TABLE IF NOT EXISTS archive (
    run_id TEXT,
    candidate_id TEXT,
    val_f1 REAL,
    aggregate_f1 REAL,
    subtree_json TEXT,
    created_at REAL,
    PRIMARY KEY (run_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS tabu (
    run_id TEXT,
    subtree TEXT,
    hypothesis TEXT,
    created_at REAL
);
"""


@dataclass
class PredCacheEntry:
    prediction: object
    json_valid: bool
    raw_response_path: str | None = None


@dataclass
class IterationRecord:
    iteration: int
    candidate_id: str
    parent_id: str | None
    mutation_type: str
    target_subtree: str | None
    hypothesis: str
    train_f1: float | None
    val_f1: float | None
    subtree_scores: dict = field(default_factory=dict)
    decision: str = "PENDING"
    decision_reason: str = ""


class RunStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "iterations").mkdir(exist_ok=True)
        (self.run_dir / "raw").mkdir(exist_ok=True)
        self.db_path = self.run_dir / "state.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        for migration in (
            "ALTER TABLE candidates ADD COLUMN delta_text TEXT DEFAULT ''",
            "ALTER TABLE candidates ADD COLUMN target_subtree TEXT DEFAULT ''",
        ):
            try:  # migrate run dirs created before delta tracking
                self.conn.execute(migration)
            except sqlite3.OperationalError:
                pass
        self.conn.commit()
        self.jsonl_path = self.run_dir / "llm_calls.jsonl"

    # -- runs ---------------------------------------------------------------

    def init_run(self, run_id: str, config_hash: str, config_path: str = "") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO runs(run_id, config_hash, config_path, status, started_at) VALUES (?,?,?,'running',?)",
            (run_id, config_hash, config_path, time.time()),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def update_run(self, run_id: str, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE runs SET {sets} WHERE run_id = ?", (*fields.values(), run_id)
        )
        self.conn.commit()

    # -- llm calls ----------------------------------------------------------

    def log_llm_call(
        self,
        *,
        run_id: str,
        role: str,
        model: str,
        iteration: int | None = None,
        prompt_hash: str | None = None,
        pdf_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        raw_request_path: str | None = None,
        raw_response_path: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO llm_calls(run_id, iteration, role, model, prompt_hash, pdf_id,
               input_tokens, output_tokens, cost_usd, latency_ms, raw_request_path,
               raw_response_path, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, iteration, role, model, prompt_hash, pdf_id,
                input_tokens, output_tokens, cost_usd, latency_ms,
                raw_request_path, raw_response_path, time.time(),
            ),
        )
        self.conn.commit()
        record = {
            "timestamp": time.time(), "run_id": run_id, "iteration": iteration,
            "role": role, "model": model, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost_usd, "latency_ms": latency_ms,
        }
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def write_raw(self, name: str, payload: object) -> str:
        path = self.run_dir / "raw" / name
        path.write_text(
            json.dumps(payload, ensure_ascii=True, default=repr, indent=2)
        )
        return str(path)

    def spent_totals(self, run_id: str) -> tuple[int, float]:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens),0) AS tok, COALESCE(SUM(cost_usd),0) AS usd FROM llm_calls WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["tok"]), float(row["usd"])

    # -- prediction cache ---------------------------------------------------

    def get_prediction(self, prompt_hash: str, pdf_hash: str, model: str) -> PredCacheEntry | None:
        row = self.conn.execute(
            "SELECT * FROM predictions_cache WHERE prompt_hash=? AND pdf_hash=? AND extractor_model=?",
            (prompt_hash, pdf_hash, model),
        ).fetchone()
        if not row:
            return None
        return PredCacheEntry(
            prediction=json.loads(row["prediction_json"]),
            json_valid=bool(row["json_valid"]),
            raw_response_path=row["raw_response_path"],
        )

    def put_prediction(
        self,
        prompt_hash: str,
        pdf_hash: str,
        model: str,
        prediction: object,
        json_valid: bool,
        raw_response_path: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO predictions_cache
               (prompt_hash, pdf_hash, extractor_model, prediction_json, json_valid, raw_response_path, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                prompt_hash, pdf_hash, model,
                json.dumps(prediction, ensure_ascii=True, default=repr),
                int(json_valid), raw_response_path, time.time(),
            ),
        )
        self.conn.commit()

    # -- candidates / iterations / archive -----------------------------------

    def save_candidate(
        self,
        candidate_id: str,
        run_id: str,
        iteration: int,
        artifact_json: dict,
        prompt_hash: str,
        delta_text: str = "",
        target_subtree: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates(candidate_id, run_id, iteration, artifact_json, prompt_hash, delta_text, target_subtree, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (candidate_id, run_id, iteration, json.dumps(artifact_json, ensure_ascii=True), prompt_hash, delta_text, target_subtree, time.time()),
        )
        self.conn.commit()

    def load_candidate(self, candidate_id: str, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT artifact_json FROM candidates WHERE candidate_id=? AND run_id=?",
            (candidate_id, run_id),
        ).fetchone()
        return json.loads(row["artifact_json"]) if row else None

    def load_candidate_delta(self, candidate_id: str, run_id: str) -> tuple[str, str]:
        row = self.conn.execute(
            "SELECT delta_text, target_subtree FROM candidates WHERE candidate_id=? AND run_id=?",
            (candidate_id, run_id),
        ).fetchone()
        if not row:
            return "", ""
        return row["delta_text"] or "", row["target_subtree"] or ""

    def log_iteration(self, run_id: str, record: IterationRecord) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO iterations(run_id, iteration, candidate_id, parent_id,
               mutation_type, target_subtree, hypothesis, train_f1, val_f1,
               subtree_scores_json, decision, decision_reason, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, record.iteration, record.candidate_id, record.parent_id,
                record.mutation_type, record.target_subtree, record.hypothesis,
                record.train_f1, record.val_f1,
                json.dumps(record.subtree_scores, ensure_ascii=True),
                record.decision, record.decision_reason, time.time(),
            ),
        )
        self.conn.commit()

    def list_iterations(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id=? ORDER BY iteration, candidate_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_iteration(self, run_id: str, iteration: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM iterations WHERE run_id=? AND iteration=? ORDER BY candidate_id",
            (run_id, iteration),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_to_archive(self, run_id: str, candidate_id: str, val_f1: float, aggregate_f1: float, subtree: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO archive(run_id, candidate_id, val_f1, aggregate_f1, subtree_json, created_at) VALUES (?,?,?,?,?,?)",
            (run_id, candidate_id, val_f1, aggregate_f1, json.dumps(subtree, ensure_ascii=True), time.time()),
        )
        self.conn.commit()

    def remove_from_archive(self, run_id: str, candidate_id: str) -> None:
        self.conn.execute("DELETE FROM archive WHERE run_id=? AND candidate_id=?", (run_id, candidate_id))
        self.conn.commit()

    def load_archive(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM archive WHERE run_id=? ORDER BY aggregate_f1 DESC", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            entry["subtree"] = json.loads(entry.pop("subtree_json"))
            out.append(entry)
        return out

    def add_tabu(self, run_id: str, subtree: str | None, hypothesis: str) -> None:
        self.conn.execute(
            "INSERT INTO tabu(run_id, subtree, hypothesis, created_at) VALUES (?,?,?,?)",
            (run_id, subtree, hypothesis, time.time()),
        )
        self.conn.commit()

    def load_tabu(self, run_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT subtree, hypothesis FROM tabu WHERE run_id=? ORDER BY created_at DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def iteration_dir(self, iteration: int) -> Path:
        path = self.run_dir / "iterations" / f"iter_{iteration:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        self.conn.close()
