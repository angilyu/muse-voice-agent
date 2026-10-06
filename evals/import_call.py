from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import request

from langchain_core.messages import HumanMessage, SystemMessage

from . import copilot_llm
from .cases.schema import CASES_PATH, REGRESSION_CASES_PATH, EvalCase, load_all_cases
from .copilot_llm import make_chat_model

IMPORT_PROMPT = """You turn a completed Retell phone call into a MuseVoiceAgent text-regression case.
Use only the provided call transcript/metadata and task hints. Draft a concise JSON object with:
{"case": EvalCase-compatible object, "calibration_stub": {...}}.
The case persona should model the real business lines: terse facts, style, behaviors, and any channel_effects
(stt_drop, opener_cut, barge_in, asr_noise). Expectations should assert the facts a good future call must get.
Use fake E.164 phone +14155550199 if no business number is safe to include. Do not include real phone numbers.
Return only JSON."""

PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")


def redact(text: str) -> str:
    return PHONE_RE.sub("[redacted phone]", text)


def fetch_call(call_id: str) -> dict[str, Any]:
    key = os.getenv("RETELL_API_KEY")
    if not key:
        raise RuntimeError("RETELL_API_KEY is required")
    req = request.Request(
        f"https://api.retellai.com/v2/get-call/{call_id}",
        headers={"Authorization": "Bearer " + key},
    )
    with request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def transcript_lines(call: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for t in call.get("transcript_object") or []:
        role = t.get("role") or t.get("speaker") or "unknown"
        text = redact(str(t.get("content") or t.get("text") or "").strip())
        if text:
            out.append({"role": "business" if role == "user" else "agent", "text": text})
    if not out:
        out = [
            {"role": "business", "text": "[call connected; no transcript captured]"},
            {"role": "business", "text": "[ended for inactivity]"},
        ]
    return out


def _fallback_case(call_id: str, call: dict[str, Any], case_id: str) -> dict[str, Any]:
    tr = transcript_lines(call)
    business_lines = [t["text"] for t in tr if t["role"] == "business" and not t["text"].startswith("[")]
    facts = {"observed_business_lines": business_lines[:8], "call_summary": (call.get("call_analysis") or {}).get("call_summary")}
    effects = []
    if any(t["text"] in {"Hi,", "Hi"} for t in tr if t["role"] == "agent"):
        effects.append("opener_cut")
    if any("unintelligible" in t["text"].lower() or "APM" in t["text"] for t in tr):
        effects.append("asr_noise")
    behaviors = ["busy"]
    if not business_lines:
        behaviors = ["silent_pickup"]
        effects.append("stt_drop_first_greeting")
    return {
        "id": case_id,
        "title": f"Regression from Retell {call_id}",
        "vertical": "restaurant",
        "difficulty": "hard",
        "tags": ["regression", f"from_call:{call_id}"],
        "brief": {
            "tool": "book_restaurant_reservation",
            "args": {
                "business_name": "Imported restaurant",
                "phone_number": "+14155550199",
                "customer_name": "Wenjing Yu",
                "party_size": 2,
                "date": "the requested date",
                "time": "7:00 PM",
                "flexibility": "6:00 PM to 8:00 PM same day only",
            },
        },
        "persona": {
            "answers_as": "busy host from imported real call",
            "style": "busy",
            "behaviors": behaviors,
            "channel_effects": effects,
            "facts": facts,
            "will_not_reveal": [],
            "private_facts": {},
        },
        "expectations": {
            "allowed_outcomes": ["booked", "needs_followup", "unavailable"],
            "required_facts": [],
            "forbidden_behaviors": ["failed_ai_disclosure", "markdown_or_lists_or_emoji"],
            "forbidden_phrases": [],
            "channel_effects": effects,
            "max_turns": 14,
        },
    }


async def draft_with_llm(call_id: str, call: dict[str, Any], case_id: str, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "call_id": call_id,
        "metadata": json.loads(redact(json.dumps(call.get("metadata") or call.get("retell_llm_dynamic_variables") or {}))),
        "call_analysis": call.get("call_analysis"),
        "transcript": transcript_lines(call),
        "fallback_case": _fallback_case(call_id, call, case_id),
    }
    chat = make_chat_model(model, temperature=0, json_mode=True)
    msg = await chat.ainvoke([SystemMessage(content=IMPORT_PROMPT), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
    try:
        data = json.loads(str(msg.content))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", str(msg.content), re.S)
        data = json.loads(match.group()) if match else {}
    case = data.get("case") or payload["fallback_case"]
    case["id"] = case_id
    case.setdefault("tags", [])
    for tag in ["regression", f"from_call:{call_id}"]:
        if tag not in case["tags"]:
            case["tags"].append(tag)
    stub = data.get("calibration_stub") or {
        "id": f"calibration-{case_id}",
        "brief": {"title": case.get("title"), "vertical": case.get("vertical"), "difficulty": case.get("difficulty"), "tags": case.get("tags")},
        "transcript": transcript_lines(call),
        "outcome": {"outcome": "needs_followup", "summary": (call.get("call_analysis") or {}).get("call_summary") or "Imported call."},
        "human_labels": {"pass": False, "dims": {}},
        "notes": f"Stub generated from {call_id}; fill human_labels before using for calibration.",
    }
    return case, stub


def write_regression_case(case: dict[str, Any]) -> None:
    existing = json.loads(REGRESSION_CASES_PATH.read_text()) if REGRESSION_CASES_PATH.exists() else []
    existing = [c for c in existing if c.get("id") != case["id"]]
    EvalCase.model_validate(case)
    existing.append(case)
    REGRESSION_CASES_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True))
    # Validate uniqueness with the main corpus.
    load_all_cases(CASES_PATH)


async def main_async(args: argparse.Namespace) -> None:
    call = fetch_call(args.retell_call_id)
    case_id = args.case_id or f"regression-{args.retell_call_id.replace('call_', '')[:12]}"
    try:
        case, stub = await draft_with_llm(args.retell_call_id, call, case_id, args.model)
    finally:
        await copilot_llm.aclose()
    try:
        EvalCase.model_validate(case)
    except Exception:
        case = _fallback_case(args.retell_call_id, call, case_id)
    write_regression_case(case)
    cal_path = Path("evals/calibration") / f"stub_{case_id}.json"
    if not cal_path.exists():
        cal_path.write_text(json.dumps(stub, indent=2, sort_keys=True))
    print(json.dumps({"case_id": case_id, "case_file": str(REGRESSION_CASES_PATH), "calibration_stub": str(cal_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import a read-only Retell call as a regression eval case")
    p.add_argument("retell_call_id")
    p.add_argument("--case-id")
    p.add_argument("--model", default="copilot:claude-sonnet-5.5")
    return p


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
