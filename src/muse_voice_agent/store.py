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
UNENDED_CALL_GRACE_SECONDS = 15 * 60

# Columns added after the first release; created on startup for older databases.
_ADDED_COLUMNS = {
    "provider_call_id": "TEXT",
    "started_at": "REAL",  # when the business picked up (unix seconds)
    "ended_at": "REAL",
    "end_reason": "TEXT",  # e.g. Retell's disconnection_reason: user_hangup, agent_hangup...
    "outcome_source": "TEXT",  # agent | transcript | call_system
    "direction": "TEXT NOT NULL DEFAULT 'outbound'",
    "parent_call_id": "TEXT",
    "origin": "TEXT",  # who placed the call, e.g. "mcp:<client name>"; inbound callbacks inherit it
    "seen_at": "REAL",  # when get_updates delivered this inbound call to the MCP client
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
    direction TEXT NOT NULL DEFAULT 'outbound',
    parent_call_id TEXT,
    origin TEXT,
    seen_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class CallStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(calls)")}
            for name, sql_type in _ADDED_COLUMNS.items():
                if name not in cols:
                    c.execute(f"ALTER TABLE calls ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_call(
        self,
        task: dict[str, Any],
        *,
        dry_run: bool,
        direction: str = "outbound",
        parent_call_id: str | None = None,
        status: str = "queued",
        provider_call_id: str | None = None,
        origin: str | None = None,
    ) -> dict[str, Any]:
        call_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO calls (id, kind, business_name, phone_number, task_json, status,"
                " dry_run, provider_call_id, direction, parent_call_id, origin, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    task["kind"],
                    task["business_name"],
                    task["phone_number"],
                    json.dumps(task),
                    status,
                    int(dry_run),
                    provider_call_id,
                    direction,
                    parent_call_id,
                    origin,
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

    def find_callback_parent(
        self, caller_number: str, *, lookback_seconds: int
    ) -> dict[str, Any] | None:
        """Find the recent outbound call an inbound callback is most likely responding to."""
        from .tasks import normalize_phone

        phone = normalize_phone(caller_number)
        since = time.time() - lookback_seconds
        final_marks = ",".join("?" * len(FINAL_STATUSES))
        with self._conn() as c:
            row = c.execute(
                f"""
                SELECT * FROM calls
                WHERE direction = 'outbound'
                  AND dry_run = 0
                  AND phone_number = ?
                  AND created_at >= ?
                ORDER BY
                  CASE WHEN status NOT IN ({final_marks}) THEN 0 ELSE 1 END,
                  created_at DESC
                LIMIT 1
                """,
                (phone, since, *FINAL_STATUSES),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def list_callbacks(self, parent_call_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM calls WHERE parent_call_id = ? ORDER BY created_at DESC",
                (parent_call_id,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def _unseen_where(self) -> tuple[str, tuple[Any, ...]]:
        # Ended inbound calls not yet delivered: callbacks about calls an MCP client placed, plus
        # messages from callers we couldn't match. record_outcome marks a call completed while it
        # is still live, so wait for Retell's end (or a stale record) before delivering it.
        marks = ",".join("?" * len(FINAL_STATUSES))
        return (
            f"direction = 'inbound' AND seen_at IS NULL AND status IN ({marks})"
            " AND (parent_call_id IS NULL OR origin LIKE 'mcp%')"
            " AND (status != 'completed' OR end_reason IS NOT NULL OR ended_at IS NOT NULL"
            " OR updated_at < ?)",
            (*FINAL_STATUSES, time.time() - UNENDED_CALL_GRACE_SECONDS),
        )

    def list_unseen_inbound(self, limit: int = 20) -> list[dict[str, Any]]:
        where, params = self._unseen_where()
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM calls WHERE {where} ORDER BY created_at ASC LIMIT ?", (*params, limit)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_unseen_inbound(self) -> int:
        where, params = self._unseen_where()
        with self._conn() as c:
            (n,) = c.execute(f"SELECT COUNT(*) FROM calls WHERE {where}", params).fetchone()
        return int(n)

    def mark_seen(self, call_ids: list[str]) -> None:
        if not call_ids:
            return
        marks = ",".join("?" * len(call_ids))
        with self._conn() as c:
            c.execute(f"UPDATE calls SET seen_at = ? WHERE id IN ({marks})", (time.time(), *call_ids))

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
