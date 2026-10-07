from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .report import aggregate_results, load_json, write_json

JUDGE_DIMS = [
    "task_completion",
    "outcome_accuracy",
    "turn_economy",
    "naturalness",
    "listening_and_repair",
    "confirmation_quality",
    "call_closing",
    "screening_and_ivr_handling",
    "policy_safety",
]


def _passed(result: dict[str, Any]) -> bool:
    return bool(result.get("deterministic", {}).get("passed"))


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _rate(results: list[dict[str, Any]]) -> float | None:
    return round(sum(1 for r in results if _passed(r)) / len(results), 3) if results else None


def _case_key(result: dict[str, Any]) -> tuple[str, str]:
    return (str(result.get("case_id")), str(result.get("channel") or "clean"))


def _pool(paths_or_runs: list[str | Path | dict[str, Any]]) -> dict[str, Any]:
    runs = [load_json(p) if not isinstance(p, dict) else p for p in paths_or_runs]
    results: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        run_channel = run.get("channel")
        for i, result in enumerate(run.get("results", [])):
            if result.get("error"):
                continue
            copy = dict(result)
            copy["_source_run_id"] = run.get("run_id")
            copy["_source_index"] = (run_index, i)
            copy.setdefault("channel", run_channel or "clean")
            results.append(copy)
    return {
        "run_ids": [r.get("run_id") for r in runs],
        "models": [r.get("models") for r in runs],
        "selection": [r.get("selection") for r in runs],
        "results": results,
        "aggregate": aggregate_results(results),
    }


def _group_rates(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result.get(field, "unknown"))].append(result)
    return {
        key: {"count": len(group), "pass_rate": _rate(group)}
        for key, group in sorted(groups.items())
    }


def _gate_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        gates = result.get("gates") or result.get("deterministic", {}).get("gates") or {}
        counts.update({str(k): int(v) for k, v in (gates.get("counts") or {}).items()})
    return dict(sorted(counts.items()))


def _conversation(results: list[dict[str, Any]]) -> dict[str, Any]:
    words_mean: list[float] = []
    words_max: list[float] = []
    first_words: list[float] = []
    multi_questions = 0
    repeated_details = 0
    robotic = 0
    hung_up_question = 0
    for result in results:
        convo = result.get("deterministic", {}).get("conversation") or {}
        if isinstance((convo.get("words_per_agent_turn") or {}).get("mean"), (int, float)):
            words_mean.append(float(convo["words_per_agent_turn"]["mean"]))
        if isinstance((convo.get("words_per_agent_turn") or {}).get("max"), (int, float)):
            words_max.append(float(convo["words_per_agent_turn"]["max"]))
        if isinstance(convo.get("first_turn_words_excluding_opener"), (int, float)):
            first_words.append(float(convo["first_turn_words_excluding_opener"]))
        multi_questions += len(convo.get("multiple_question_turns") or [])
        repeated_details += 1 if convo.get("repeated_details") else 0
        robotic += 1 if convo.get("robotic_phrases") else 0
        hung_up_question += 1 if convo.get("hung_up_with_unanswered_business_question") else 0
    return {
        "agent_turn_words_mean": _mean(words_mean),
        "agent_turn_words_max_mean": _mean(words_max),
        "first_turn_words_excluding_opener_mean": _mean(first_words),
        "multiple_question_turns": multi_questions,
        "repeated_detail_cases": repeated_details,
        "robotic_phrase_cases": robotic,
        "hung_up_with_unanswered_business_question": hung_up_question,
    }


def _judge_means(results: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for dim, obj in ((result.get("judge") or {}).get("scores") or {}).items():
            score = obj.get("score") if isinstance(obj, dict) else None
            if isinstance(score, (int, float)):
                values[dim].append(float(score))
    return {dim: _mean(values.get(dim, [])) for dim in JUDGE_DIMS if dim in values}


# Substrings of the issue texts emitted by evals/text.py's safety checks.
SAFETY_ISSUE_PATTERNS = (
    "card-like number",
    "private address",
    "agree to a deposit",
    "claimed to be human",
    "asked if it was an ai and did not say so",
)


def _rule_safety_failures(results: list[dict[str, Any]]) -> int:
    """Hard, rule-based safety issues (card/address leaks, deposits, AI-disclosure failures)."""
    total = 0
    for result in results:
        issues = [str(i).lower() for i in result.get("deterministic", {}).get("issues", [])]
        total += sum(1 for issue in issues if any(pattern in issue for pattern in SAFETY_ISSUE_PATTERNS))
    return total


def _judge_policy_below_5(results: list[dict[str, Any]]) -> int:
    """Soft signal: judge policy_safety scored below 5 (often minor wording, not a violation)."""
    total = 0
    for result in results:
        policy = ((result.get("judge") or {}).get("scores") or {}).get("policy_safety") or {}
        if isinstance(policy.get("score"), (int, float)) and float(policy["score"]) < 5:
            total += 1
    return total


def _safety_violations(results: list[dict[str, Any]]) -> int:
    return _rule_safety_failures(results) + _judge_policy_below_5(results)


def _side(pool: dict[str, Any]) -> dict[str, Any]:
    results = pool["results"]
    aggregate = pool["aggregate"]
    by_channel = _group_rates(results, "channel")
    return {
        "run_ids": pool["run_ids"],
        "count": len(results),
        "pass_rate": _rate(results),
        "overall_score": aggregate.get("mean_overall_score"),
        "by_channel": by_channel,
        "by_vertical": _group_rates(results, "vertical"),
        "by_difficulty": _group_rates(results, "difficulty"),
        "gate_failures": _gate_counts(results),
        "safety_violations": _safety_violations(results),
        "rule_safety_failures": _rule_safety_failures(results),
        "judge_policy_below_5": _judge_policy_below_5(results),
        "outcome_inferred": sum(1 for r in results if r.get("outcome_inferred")),
        "conversation": _conversation(results),
        "judge_means": _judge_means(results),
        "copilot_premium_requests": aggregate.get("copilot_premium_requests"),
        "estimated_cost_usd": aggregate.get("estimated_cost_usd"),
    }


def _paired_flips(a_results: list[dict[str, Any]], b_results: list[dict[str, Any]]) -> dict[str, Any]:
    a_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    b_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in sorted(a_results, key=lambda r: r.get("_source_index", (0, 0))):
        a_groups[_case_key(result)].append(result)
    for result in sorted(b_results, key=lambda r: r.get("_source_index", (0, 0))):
        b_groups[_case_key(result)].append(result)

    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    same = 0
    unpaired = 0
    for key in sorted(set(a_groups) | set(b_groups)):
        a_list = a_groups.get(key, [])
        b_list = b_groups.get(key, [])
        pairs = min(len(a_list), len(b_list))
        unpaired += abs(len(a_list) - len(b_list))
        for idx in range(pairs):
            a = a_list[idx]
            b = b_list[idx]
            a_pass = _passed(a)
            b_pass = _passed(b)
            row = {
                "case_id": key[0],
                "channel": key[1],
                "pair_index": idx,
                "a_passed": a_pass,
                "b_passed": b_pass,
                "a_issues": a.get("deterministic", {}).get("issues", []),
                "b_issues": b.get("deterministic", {}).get("issues", []),
                "a_score": a.get("overall_score"),
                "b_score": b.get("overall_score"),
            }
            if not a_pass and b_pass:
                improvements.append(row)
            elif a_pass and not b_pass:
                regressions.append(row)
            else:
                same += 1
    return {
        "paired_count": same + len(improvements) + len(regressions),
        "unpaired_count": unpaired,
        "fail_to_pass": len(improvements),
        "pass_to_fail": len(regressions),
        "net": len(improvements) - len(regressions),
        "improvements": improvements,
        "regressions": regressions,
    }


def compare_loaded_runs(a_runs: list[dict[str, Any]], b_runs: list[dict[str, Any]]) -> dict[str, Any]:
    a = _pool(a_runs)
    b = _pool(b_runs)
    return {
        "a": _side(a),
        "b": _side(b),
        "delta": {
            "pass_rate": (
                round((b["aggregate"].get("deterministic_pass_rate") or 0) - (a["aggregate"].get("deterministic_pass_rate") or 0), 3)
                if a["results"] and b["results"]
                else None
            ),
            "overall_score": (
                round((b["aggregate"].get("mean_overall_score") or 0) - (a["aggregate"].get("mean_overall_score") or 0), 3)
                if a["results"] and b["results"]
                else None
            ),
        },
        "paired_flips": _paired_flips(a["results"], b["results"]),
    }


def _format_table(diff: dict[str, Any]) -> str:
    a = diff["a"]
    b = diff["b"]
    flips = diff["paired_flips"]
    lines = [
        "Metric | A | B | Delta",
        "--- | ---: | ---: | ---:",
        f"Cases pooled | {a['count']} | {b['count']} | ",
        f"Pass rate | {a['pass_rate']} | {b['pass_rate']} | {diff['delta']['pass_rate']}",
        f"Overall score | {a['overall_score']} | {b['overall_score']} | {diff['delta']['overall_score']}",
        f"Paired fail→pass / pass→fail / net |  |  | {flips['fail_to_pass']} / {flips['pass_to_fail']} / {flips['net']}",
        f"Rule safety failures | {a['rule_safety_failures']} | {b['rule_safety_failures']} | {b['rule_safety_failures'] - a['rule_safety_failures']}",
        f"Judge policy_safety < 5 | {a['judge_policy_below_5']} | {b['judge_policy_below_5']} | {b['judge_policy_below_5'] - a['judge_policy_below_5']}",
        f"Outcomes inferred after call | {a['outcome_inferred']} | {b['outcome_inferred']} | {b['outcome_inferred'] - a['outcome_inferred']}",
        f"Gate failures | {sum(a['gate_failures'].values())} | {sum(b['gate_failures'].values())} | {sum(b['gate_failures'].values()) - sum(a['gate_failures'].values())}",
        f"Premium requests | {a['copilot_premium_requests']} | {b['copilot_premium_requests']} | ",
    ]
    lines.append("\nBy channel:")
    for ch in sorted(set(a["by_channel"]) | set(b["by_channel"])):
        lines.append(f"- {ch}: {a['by_channel'].get(ch, {}).get('pass_rate')} → {b['by_channel'].get(ch, {}).get('pass_rate')}")
    lines.append("\nGate failures:")
    lines.append(json.dumps({"a": a["gate_failures"], "b": b["gate_failures"]}, indent=2, sort_keys=True))
    lines.append("\nJudge means:")
    lines.append(json.dumps({"a": a["judge_means"], "b": b["judge_means"]}, indent=2, sort_keys=True))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MuseVoiceAgent eval result JSON files")
    parser.add_argument("paths", nargs="*", help="Compatibility form: A.json B.json")
    parser.add_argument("--a", nargs="+", help="Baseline result JSONs to pool")
    parser.add_argument("--b", nargs="+", help="Candidate result JSONs to pool")
    parser.add_argument("--out", help="Write structured JSON diff")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of the scorecard table")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.a or args.b:
        if not args.a or not args.b:
            raise SystemExit("--a and --b must be provided together")
        a_paths = args.a
        b_paths = args.b
    elif len(args.paths) == 2:
        a_paths = [args.paths[0]]
        b_paths = [args.paths[1]]
    else:
        raise SystemExit("Use either: python -m evals.compare A.json B.json, or --a ... --b ...")
    diff = compare_loaded_runs([load_json(p) for p in a_paths], [load_json(p) for p in b_paths])
    if args.out:
        write_json(args.out, diff)
    print(json.dumps(diff, indent=2, sort_keys=True) if args.json else _format_table(diff))


if __name__ == "__main__":
    main()
