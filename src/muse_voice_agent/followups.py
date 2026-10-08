"""Helpers for follow-up calls that continue an earlier call thread."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .report import build_report
from .store import FINAL_STATUSES, CallStore

MAX_CHAIN_ITEMS = 3
MAX_CONTEXT_CHARS = 2400


class FollowUpError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_follow_up") -> None:
        super().__init__(message)
        self.code = code


def build_follow_up_context(store: CallStore, follow_up_of: str) -> str:
    """Return a compact, bounded summary of the call thread ending at `follow_up_of`."""
    chain = _load_chain(store, follow_up_of)
    parts = [
        "Previous call context (oldest first; use this as context, do not repeat already-answered "
        "questions unless confirming or changing them):"
    ]
    for record in chain:
        parts.append(_summarize_record(record))
    parts.append(
        "For this follow-up: naturally mention the earlier call when relevant, ask for the person "
        "previously spoken to if named above, and focus on the unanswered questions or next step."
    )
    text = "\n".join(parts)
    if len(text) > MAX_CONTEXT_CHARS:
        return text[: MAX_CONTEXT_CHARS - 120].rstrip() + "\n... (older follow-up context truncated)"
    return text


def resolve_follow_up(
    store: CallStore, follow_up_of: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate `follow_up_of`, inherit business fields, and attach prompt context."""
    previous = store.get_call(follow_up_of)
    if previous is None:
        raise FollowUpError(f"No call with id {follow_up_of}; call_ids come from this server.")
    if previous["status"] not in FINAL_STATUSES:
        raise FollowUpError(
            f"Call {follow_up_of} is still {previous['status']}; wait until it is done before "
            "starting a follow-up.",
            code="follow_up_not_ready",
        )

    resolved = dict(kwargs)
    resolved["follow_up_of"] = follow_up_of
    resolved["business_name"] = resolved.get("business_name") or previous["business_name"]
    resolved["phone_number"] = resolved.get("phone_number") or previous["phone_number"]
    resolved["previous_call_context"] = build_follow_up_context(store, follow_up_of)
    return resolved


def _load_chain(store: CallStore, follow_up_of: str) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_id: str | None = follow_up_of
    while current_id and current_id not in seen and len(chain) < MAX_CHAIN_ITEMS:
        seen.add(current_id)
        record = store.get_call(current_id)
        if record is None:
            break
        chain.append(record)
        current_id = record.get("follow_up_of") or (record.get("task") or {}).get("follow_up_of")
    return list(reversed(chain))


def _summarize_record(record: dict[str, Any]) -> str:
    report = build_report(record)
    details = record.get("details") or {}
    ended = report.get("ended_at") or _iso(record.get("updated_at")) or "time unknown"
    lines = [
        f"- Call {record['id']} to {record['business_name']} ({ended})",
        f"  - Reached: {report.get('reached') or 'unknown'}",
        f"  - Outcome: {record.get('outcome') or record['status']}"
        + (f" — {record['summary']}" if record.get("summary") else ""),
    ]
    contact = details.get("contact_person")
    if contact:
        lines.append(f"  - Person spoken to: {contact}")
    answers = report.get("answers") or []
    if answers:
        compact_answers = "; ".join(
            f"{a['question']}: {a['answer']}" for a in answers[:5] if a.get("answer") != "not answered"
        )
        if compact_answers:
            lines.append(f"  - Learned: {compact_answers}")
    facts = _detail_facts(details)
    if facts:
        lines.append(f"  - Details learned: {facts}")
    commitments = _commitments(record, details, report)
    if commitments:
        lines.append(f"  - Promised/committed: {commitments}")
    if report.get("unanswered_questions"):
        lines.append("  - Unanswered: " + "; ".join(report["unanswered_questions"][:5]))
    if report.get("next_steps"):
        lines.append("  - Next steps: " + "; ".join(report["next_steps"][:3]))
    return "\n".join(lines)


def _commitments(record: dict[str, Any], details: dict[str, Any], report: dict[str, Any]) -> str | None:
    bits: list[str] = []
    if report.get("committed_on_users_behalf") and record.get("summary"):
        bits.append(str(record["summary"]))
    if details.get("reference"):
        bits.append(f"reference {details['reference']}")
    if details.get("follow_up"):
        bits.append(str(details["follow_up"]))
    return "; ".join(bits) if bits else None


def _detail_facts(details: dict[str, Any]) -> str | None:
    labels = {
        "quote": "quote",
        "availability": "availability",
        "confirmed_date": "date",
        "confirmed_time": "time",
        "order_total": "total",
        "pickup_time": "pickup",
        "booked_under": "name",
        "reference": "reference",
    }
    facts = [f"{label}: {details[key]}" for key, label in labels.items() if details.get(key)]
    return "; ".join(facts) if facts else None


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
