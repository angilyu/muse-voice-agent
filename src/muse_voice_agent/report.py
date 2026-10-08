"""Post-call report returned to Muse so it can act on the result in the chat."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Retell disconnection_reason (and the LiveKit agent's equivalents) -> (plain words, ended_by).
_END_REASONS: dict[str, tuple[str, str]] = {
    "user_hangup": ("The business hung up.", "business"),
    "agent_hangup": ("The assistant ended the call.", "assistant"),
    "inactivity": ("The call ended after a long silence.", "timeout"),
    "max_duration_reached": ("The call hit the maximum call length.", "timeout"),
    "dial_no_answer": ("Nobody picked up.", "no_answer"),
    "dial_busy": ("The line was busy.", "no_answer"),
    "user_declined": ("The call was declined.", "business"),
    "registered_call_timeout": ("The call never connected.", "system"),
    "voicemail_reached": ("Reached voicemail.", "assistant"),
    "ivr_reached": ("Reached an automated phone menu.", "assistant"),
    "call_transfer": ("The call was transferred.", "business"),
    "manual_stopped": ("The call was stopped manually.", "system"),
}

_SUCCESS_OUTCOMES = {"booked", "ordered", "quote_received", "info_received", "unavailable", "declined"}

_UNANSWERED = re.compile(
    r"^\W*(not answered|unanswered|unknown|n/?a|none|no answer|unclear|"
    r"(they )?(didn'?t|did not|wouldn'?t|would not|couldn'?t|could not) (say|answer|know|tell)|"
    r"not (asked|provided|mentioned|discussed|given|specified|known))\b",
    re.IGNORECASE,
)

_REQUEST_SKIP = {"kind", "business_name", "phone_number", "previous_call_context"}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_unanswered(answer: str | None) -> bool:
    return not answer or not answer.strip() or bool(_UNANSWERED.match(answer.strip()))


def align_answers(
    questions: list[str], answers: list[dict[str, Any]] | None
) -> tuple[list[dict[str, str]], list[str]]:
    """Pair every requested question with the agent's answer; list the ones left unanswered."""
    answers = [a for a in answers or [] if isinstance(a, dict) and a.get("question")]
    by_question = {_norm(a["question"]): a.get("answer") for a in answers}
    by_index = len(answers) == len(questions)
    aligned: list[dict[str, str]] = []
    unanswered: list[str] = []
    matched: set[str] = set()
    for i, question in enumerate(questions):
        key = _norm(question)
        if key in by_question:
            answer = by_question[key]
            matched.add(key)
        elif by_index:
            answer = answers[i].get("answer")
            matched.add(_norm(answers[i]["question"]))
        else:
            answer = None
        if _is_unanswered(answer):
            unanswered.append(question)
            aligned.append({"question": question, "answer": "not answered"})
        else:
            aligned.append({"question": question, "answer": str(answer).strip()})
    # Extra facts the agent recorded that weren't one of the requested questions.
    for a in answers:
        if _norm(a["question"]) not in matched and not _is_unanswered(a.get("answer")):
            aligned.append({"question": a["question"], "answer": str(a["answer"]).strip()})
    return aligned, unanswered


def _reached(record: dict[str, Any]) -> str:
    reason = record.get("end_reason")
    if record.get("outcome") == "voicemail" or reason == "voicemail_reached":
        return "voicemail"
    if reason == "ivr_reached":
        return "phone_menu"
    if record["status"] == "no_answer":
        return "no_answer"
    if record["status"] == "failed":
        return "not_connected"
    if any(t.get("role") == "user" for t in record.get("transcript") or []):
        return "person"
    if record.get("outcome") in _SUCCESS_OUTCOMES:
        return "person"
    return "unknown"


def _next_steps(
    record: dict[str, Any], details: dict[str, Any], unanswered: list[str], source: str | None
) -> list[str]:
    outcome = record.get("outcome")
    status = record["status"]
    steps: list[str] = []
    when = " ".join(x for x in (details.get("confirmed_date"), details.get("confirmed_time")) if x)
    if outcome == "booked":
        steps.append(
            f"Tell the user it's booked{' for ' + when if when else ''}"
            f"{' under ' + details['booked_under'] if details.get('booked_under') else ''}"
            " and offer to add it to their calendar."
        )
    elif outcome == "ordered":
        extras = [
            f"pickup {details['pickup_time']}" if details.get("pickup_time") else "",
            f"total {details['order_total']}" if details.get("order_total") else "",
        ]
        extra = ", ".join(x for x in extras if x)
        steps.append(
            f"Tell the user the order was placed{' (' + extra + ')' if extra else ''}"
            " and offer a pickup reminder."
        )
    elif outcome == "quote_received":
        steps.append("Share the quote and ask whether the user wants to book the job or compare quotes.")
    elif outcome == "info_received":
        steps.append("Share the answers with the user.")
    elif outcome == "unavailable":
        offered = details.get("availability")
        steps.append(
            "Tell the user their request wasn't available"
            + (f"; the business offered: {offered}. Ask whether to take it." if offered else
               " and offer to try another time or business.")
        )
    elif outcome == "declined":
        steps.append("The business wouldn't help by phone; suggest the user contact them directly.")
    elif outcome == "voicemail":
        steps.append("Reached voicemail; offer to call again later or try another business.")
    elif outcome == "needs_followup":
        steps.append(
            details.get("follow_up")
            or "The call ended without a clear result; review the transcript and offer to call back."
        )
    elif status == "no_answer":
        steps.append("Nobody answered; offer to retry later or try another business.")
    elif status == "failed":
        steps.append("The call couldn't be completed; offer to retry.")
    if details.get("follow_up") and outcome != "needs_followup":
        steps.append(f"Follow-up: {details['follow_up']}")
    if unanswered:
        steps.append("Still unanswered: " + "; ".join(unanswered))
    if source == "transcript":
        steps.append("This result was inferred from the transcript; double-check key details with the user.")
    return steps


def speaker_transcript(transcript: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    labels = {"user": "business", "assistant": "assistant"}
    return [
        {"speaker": labels.get(t.get("role", ""), t.get("role", "")), "text": t.get("text", "")}
        for t in transcript or []
        if t.get("text")
    ]


def build_report(record: dict[str, Any]) -> dict[str, Any]:
    """Everything Muse needs after the call, without having to read the transcript itself."""
    task = record.get("task") or {}
    details = record.get("details") or {}
    questions = list(task.get("questions") or [])
    answers, unanswered = align_answers(questions, details.get("answers"))
    started, ended = record.get("started_at"), record.get("ended_at")
    reason = record.get("end_reason")
    words, ended_by = _END_REASONS.get(reason or "", (None, None))
    if reason and words is None:
        words, ended_by = f"Call ended ({reason.replace('_', ' ')}).", "system"
    source = record.get("outcome_source")
    return {
        "follow_up_of": record.get("follow_up_of") or task.get("follow_up_of"),
        "request": {k: v for k, v in task.items() if k not in _REQUEST_SKIP and v not in (None, [], {})},
        "reached": _reached(record),
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "duration_seconds": round(ended - started) if started and ended and ended >= started else None,
        "end_reason": words,
        "ended_by": ended_by,
        "outcome_source": source,
        "committed_on_users_behalf": record.get("outcome") in ("booked", "ordered"),
        "answers": answers,
        "unanswered_questions": unanswered,
        "next_steps": _next_steps(record, details, unanswered, source),
    }
