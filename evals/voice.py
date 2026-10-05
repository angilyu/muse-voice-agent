from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from muse_voice_agent.config import get_settings
from muse_voice_agent.dispatcher import start_call
from muse_voice_agent.retell import RetellClient
from muse_voice_agent.store import CallStore

from .cases.schema import select_cases
from .judge import AUDIO_JUDGE_PROMPT_VERSION, judge_audio_recording
from .report import aggregate_results, compare_runs, load_json, utc_run_name, write_json

OUR_RETELL_AGENT_ID = "agent_9d63522dcd7094db78634855a6"


def _role(utt: dict[str, Any]) -> str:
    role = (utt.get("role") or utt.get("speaker") or "").lower()
    if role in {"agent", "assistant", "ai"}:
        return "agent"
    return "user"


def _word_bounds(word: dict[str, Any]) -> tuple[float | None, float | None]:
    start = next((word[k] for k in ("start", "start_time", "start_ms") if k in word), None)
    end = next((word[k] for k in ("end", "end_time", "end_ms") if k in word), None)
    if start is None or end is None:
        return None, None
    start_f = float(start)
    end_f = float(end)
    if start_f > 1000 or end_f > 1000:
        start_f /= 1000.0
        end_f /= 1000.0
    return start_f, end_f


def utterance_span(utt: dict[str, Any]) -> tuple[float | None, float | None]:
    words = utt.get("words") or utt.get("word_timestamps") or []
    starts: list[float] = []
    ends: list[float] = []
    for w in words:
        s, e = _word_bounds(w)
        if s is not None and e is not None:
            starts.append(s)
            ends.append(e)
    if starts and ends:
        return min(starts), max(ends)
    start = next((utt[k] for k in ("start", "start_time", "start_ms") if k in utt), None)
    end = next((utt[k] for k in ("end", "end_time", "end_ms") if k in utt), None)
    if start is not None and end is not None:
        s, e = float(start), float(end)
        if s > 1000 or e > 1000:
            s /= 1000.0
            e /= 1000.0
        return s, e
    return None, None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 3)


def _words(utt: dict[str, Any]) -> int:
    if utt.get("words"):
        return len(utt["words"])
    text = utt.get("content") or utt.get("text") or ""
    return len(re.findall(r"\w+", text))


def compute_voice_metrics(call: dict[str, Any]) -> dict[str, Any]:
    transcript = call.get("transcript_object") or []
    turns: list[dict[str, Any]] = []
    for utt in transcript:
        s, e = utterance_span(utt)
        turns.append(
            {
                "role": _role(utt),
                "text": utt.get("content") or utt.get("text") or "",
                "start": s,
                "end": e,
                "words": _words(utt),
            }
        )
    timed = [t for t in turns if t["start"] is not None and t["end"] is not None]
    timed.sort(key=lambda t: t["start"])

    agent_latencies: list[float] = []
    overlaps = {"agent_interrupts_user": 0, "user_barges_in": 0, "total": 0}
    dead_air: list[float] = []
    for prev, cur in zip(timed, timed[1:]):
        gap = float(cur["start"]) - float(prev["end"])
        if prev["role"] == "user" and cur["role"] == "agent":
            agent_latencies.append(round(gap, 3))
        if gap < -0.05:
            overlaps["total"] += 1
            if prev["role"] == "user" and cur["role"] == "agent":
                overlaps["agent_interrupts_user"] += 1
            elif prev["role"] == "agent" and cur["role"] == "user":
                overlaps["user_barges_in"] += 1
        elif gap > 2.5:
            dead_air.append(round(gap, 3))

    durations = {"agent": 0.0, "user": 0.0}
    words = {"agent": 0, "user": 0}
    for t in timed:
        role = t["role"]
        durations[role] += max(0.0, float(t["end"]) - float(t["start"]))
        words[role] += int(t["words"])
    duration_seconds = None
    if call.get("duration_ms") is not None:
        duration_seconds = round(float(call["duration_ms"]) / 1000.0, 3)
    elif timed:
        duration_seconds = round(max(float(t["end"]) for t in timed) - min(float(t["start"]) for t in timed), 3)
    total_talk = durations["agent"] + durations["user"]

    last_agent = next((t for t in reversed(turns) if t["role"] == "agent"), None)
    clean_goodbye = bool(last_agent and re.search(r"\b(bye|goodbye|thank you|thanks)\b", last_agent["text"], re.I))
    reason = call.get("disconnection_reason") or call.get("call_status")
    latency_obj = call.get("latency") or call.get("latency_stats") or {}

    return {
        "call_id": call.get("call_id"),
        "agent_id": call.get("agent_id"),
        "disconnection_reason": reason,
        "call_status": call.get("call_status"),
        "duration_seconds": duration_seconds,
        "turn_count": len(turns),
        "agent_response_latency_seconds": {
            "p50": _percentile(agent_latencies, 0.5),
            "p90": _percentile(agent_latencies, 0.9),
            "max": round(max(agent_latencies), 3) if agent_latencies else None,
            "values": agent_latencies,
        },
        "time_to_first_agent_seconds": round(float(next((t["start"] for t in timed if t["role"] == "agent"), 0.0)), 3) if timed else None,
        "overlaps": overlaps,
        "dead_air_gaps_seconds": {"count": len(dead_air), "max": round(max(dead_air), 3) if dead_air else None, "values": dead_air},
        "talk_ratio": {
            "agent": round(durations["agent"] / total_talk, 3) if total_talk else None,
            "user": round(durations["user"] / total_talk, 3) if total_talk else None,
        },
        "words_per_minute": {
            role: round(words[role] / durations[role] * 60, 1) if durations[role] > 0 else None
            for role in ("agent", "user")
        },
        "retell_latency_stats": latency_obj,
        "agent_hung_up_cleanly": clean_goodbye and reason in {"agent_hangup", "call_ended", "user_hangup", "ended"},
        "call_analysis": call.get("call_analysis"),
    }


async def _get_calls(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = get_settings()
    async with RetellClient(settings.retell_api_key) as client:
        ids = list(args.retell_call_id or [])
        if args.store_call_id:
            store = CallStore(args.store_db or settings.call_db_path)
            for store_id in args.store_call_id:
                rec = store.get_call(store_id)
                if rec and rec.get("provider_call_id"):
                    ids.append(rec["provider_call_id"])
        calls = [await client.get_call(cid) for cid in ids]
        if args.latest:
            recent = await client.list_calls(limit=max(args.latest * 3, args.latest), agent_id=args.agent_id)
            if args.agent_id:
                recent = [c for c in recent if c.get("agent_id") == args.agent_id]
            calls.extend(recent[: args.latest])
        dedup: dict[str, dict[str, Any]] = {}
        for c in calls:
            cid = c.get("call_id") or c.get("id")
            if cid:
                dedup[cid] = c
        return list(dedup.values())


async def score_calls(args: argparse.Namespace) -> dict[str, Any]:
    calls = await _get_calls(args)
    results: list[dict[str, Any]] = []
    for call in calls:
        metrics = compute_voice_metrics(call)
        result: dict[str, Any] = {
            "case_id": call.get("call_id") or call.get("id"),
            "title": f"Retell call {call.get('call_id') or call.get('id')}",
            "vertical": "voice",
            "difficulty": "real_call",
            "tags": ["voice", "retell"],
            "deterministic": {"passed": True, "issues": []},
            "overall_score": None,
            "voice_metrics": metrics,
        }
        if args.audio_judge:
            recording_url = call.get("recording_url")
            if not recording_url:
                result["audio_judge_error"] = "no recording_url on call"
            else:
                judge = await judge_audio_recording(recording_url=recording_url, transcript=call.get("transcript_object") or [], model=args.audio_judge_model)
                result["audio_judge"] = judge.data
                result["audio_judge_latency_seconds"] = round(judge.latency_seconds, 3)
                result["usage"] = {"audio_judge": judge.usage, "estimated_cost_usd": judge.estimated_cost_usd}
        results.append(result)
    run = {
        "run_id": args.run_id or uuid.uuid4().hex[:10],
        "kind": "voice",
        "prompt_versions": {"audio_judge": AUDIO_JUDGE_PROMPT_VERSION if args.audio_judge else None},
        "selection": {"retell_call_ids": args.retell_call_id, "store_call_ids": args.store_call_id, "latest": args.latest, "agent_id": args.agent_id},
        "aggregate": aggregate_results(results),
        "results": results,
    }
    out = args.out or os.path.join("evals", "results", utc_run_name("voice"))
    path = write_json(out, run)
    print(json.dumps({"out": str(path), "calls": len(results), "aggregate": run["aggregate"]}, indent=2))
    return run


async def live_harness(args: argparse.Namespace) -> None:
    if args.allow_live_calls != "I_UNDERSTAND_THIS_PLACES_REAL_CALLS" or os.getenv("EVALS_ALLOW_LIVE_CALLS") != "1":
        raise SystemExit(
            "Live voice eval is disabled. It places real calls and incurs Retell/Twilio costs. "
            "Set EVALS_ALLOW_LIVE_CALLS=1 and pass --allow-live-calls I_UNDERSTAND_THIS_PLACES_REAL_CALLS."
        )
    if not args.persona_phone_number:
        raise SystemExit("--persona-phone-number is required; it should route to your preconfigured Retell persona agent")
    settings = get_settings()
    store = CallStore(args.store_db or Path("evals/results/live_voice_calls.db"))
    cases = select_cases(args.cases, seed=args.seed, limit=args.limit)
    created = []
    for case in cases:
        task = case.brief.task().model_copy(update={"phone_number": args.persona_phone_number})
        record = await start_call(task, settings, store)
        created.append({"case_id": case.id, "call_id": record["id"], "provider_call_id": record.get("provider_call_id")})
        await asyncio.sleep(args.stagger_seconds)
    out = args.out or os.path.join("evals", "results", utc_run_name("live-voice-started"))
    write_json(out, {"created": created, "note": "Score these with evals.voice score --store-call-id ... after calls finish."})
    print(json.dumps({"out": out, "created": created}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score MuseVoiceAgent real voice calls")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score existing Retell calls")
    score.add_argument("--retell-call-id", action="append")
    score.add_argument("--store-call-id", action="append")
    score.add_argument("--store-db")
    score.add_argument("--latest", type=int, default=0)
    score.add_argument("--agent-id", default=OUR_RETELL_AGENT_ID)
    score.add_argument("--audio-judge", action="store_true")
    score.add_argument("--audio-judge-model", default="gpt-4o-audio-preview")
    score.add_argument("--run-id")
    score.add_argument("--out")

    live = sub.add_parser("live", help="Opt-in live eval harness against your own Retell persona number")
    live.add_argument("--cases", default="tag:smoke")
    live.add_argument("--limit", type=int)
    live.add_argument("--seed", type=int)
    live.add_argument("--persona-phone-number")
    live.add_argument("--store-db")
    live.add_argument("--stagger-seconds", type=float, default=20.0)
    live.add_argument("--allow-live-calls")
    live.add_argument("--out")

    compare = sub.add_parser("compare", help="Compare two eval run JSON files")
    compare.add_argument("base")
    compare.add_argument("candidate")
    compare.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        asyncio.run(score_calls(args))
    elif args.command == "live":
        asyncio.run(live_harness(args))
    elif args.command == "compare":
        diff = compare_runs(load_json(args.base), load_json(args.candidate))
        if args.out:
            write_json(args.out, diff)
        print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()
