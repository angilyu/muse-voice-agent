"""Tiny SQLite call log shared by the MCP server and the LiveKit agent worker."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ACTIVE_STATUSES = ("queued", "dispatched", "dialing", "in_progress")
FINAL_STATUSES = ("completed", "failed", "no_answer")

# Columns added after the first release; created on startup for older databases.
_ADDED_COLUMNS = {
    "provider_call_id": "TEXT",
    "started_at": "REAL",  # when the business picked up (unix seconds)
    "ended_at": "REAL",
    "end_reason": "TEXT",  # e.g. Retell's disconnection_reason: user_hangup, agent_hangup...
    "outcome_source": "TEXT",  # agent | transcript | call_system
    "follow_up_of": "TEXT",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    business_name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    task_json TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT,
    summary TEXT,
    details_json TEXT,
    transcript_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    dry_run INTEGER NOT NULL DEFAULT 0,
    provider_call_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    outcome_source TEXT,
    follow_up_of TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class CallStore:
    def __init__(self, path: Path | str):
        self._memory_conn: sqlite3.Connection | None = None
        if str(path) == ":memory:":
            self.path = Path(":memory:")
            self._memory_conn = sqlite3.connect(":memory:", timeout=10)
            self._memory_conn.row_factory = sqlite3.Row
            self._ensure_schema(self._memory_conn)
            return
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(c)

    @staticmethod
    def _ensure_schema(c: sqlite3.Connection) -> None:
        c.executescript(_SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(calls)")}
        for name, sql_type in _ADDED_COLUMNS.items():
            if name not in cols:
                c.execute(f"ALTER TABLE calls ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            self._memory_conn.commit()
            return
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_call(self, task: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
        call_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO calls (id, kind, business_name, phone_number, task_json, status,"
                " dry_run, follow_up_of, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    task["kind"],
                    task["business_name"],
                    task["phone_number"],
                    json.dumps(task),
                    "queued",
                    int(dry_run),
                    task.get("follow_up_of"),
                    now,
                    now,
                ),
            )
        return self.get_call(call_id)  # type: ignore[return-value]

    def update_call(self, call_id: str, **fields: Any) -> None:
        if "details" in fields:
            fields["details_json"] = json.dumps(fields.pop("details"))
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE calls SET {cols} WHERE id = ?", (*fields.values(), call_id))

    def append_transcript(self, call_id: str, role: str, text: str) -> None:
        with self._conn() as c:
            row = c.execute("SELECT transcript_json FROM calls WHERE id = ?", (call_id,)).fetchone()
            if row is None:
                return
            transcript = json.loads(row["transcript_json"])
            transcript.append({"role": role, "text": text, "t": round(time.time(), 2)})
            c.execute(
                "UPDATE calls SET transcript_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(transcript), time.time(), call_id),
            )

    def set_transcript(self, call_id: str, turns: list[dict[str, Any]]) -> None:
        """Replace the transcript wholesale (Retell sends the full transcript on every update)."""
        self.update_call(call_id, transcript_json=json.dumps(turns))

    def get_call(self, call_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_calls(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM calls ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_active(self) -> int:
        marks = ",".join("?" * len(ACTIVE_STATUSES))
        with self._conn() as c:
            (n,) = c.execute(
                f"SELECT COUNT(*) FROM calls WHERE status IN ({marks})", ACTIVE_STATUSES
            ).fetchone()
        return int(n)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["task"] = json.loads(d.pop("task_json"))
    d["details"] = json.loads(d.pop("details_json")) if d.get("details_json") else None
    d["transcript"] = json.loads(d.pop("transcript_json"))
    d["dry_run"] = bool(d["dry_run"])
    return d
