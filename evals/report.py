from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def utc_run_name(prefix: str = "text") -> str:
    from datetime import datetime, timezone

    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    errored = [r.get("case_id") for r in results if r.get("error")]
    judge_errors = [r.get("case_id") for r in results if (r.get("judge") or {}).get("error")]
    results = [r for r in results if not r.get("error")]
    total = len(results)
    deterministic_passes = [r.get("deterministic", {}).get("passed", False) for r in results]
    rubric_dims: dict[str, list[float]] = defaultdict(list)
    overall_scores: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []
    premium: list[float] = []
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_vertical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        for tag in r.get("tags", []):
            by_tag[tag].append(r)
        by_vertical[r.get("vertical", "unknown")].append(r)
        by_difficulty[r.get("difficulty", "unknown")].append(r)
        for dim, obj in ((r.get("judge") or {}).get("scores") or {}).items():
            score = obj.get("score") if isinstance(obj, dict) else None
            if isinstance(score, (int, float)):
                rubric_dims[dim].append(float(score))
        if isinstance(r.get("overall_score"), (int, float)):
            overall_scores.append(float(r["overall_score"]))
        latencies.extend(float(x) for x in r.get("latencies_seconds", {}).get("agent", []) if isinstance(x, (int, float)))
        usage = r.get("usage", {})
        if isinstance(usage.get("estimated_cost_usd"), (int, float)):
            costs.append(float(usage["estimated_cost_usd"]))
        if isinstance(usage.get("copilot_premium_requests"), (int, float)):
            premium.append(float(usage["copilot_premium_requests"]))

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "pass_rate": round(sum(1 for r in group if r.get("deterministic", {}).get("passed")) / len(group), 3) if group else None,
            "mean_overall_score": _mean([float(r["overall_score"]) for r in group if isinstance(r.get("overall_score"), (int, float))]),
        }

    return {
        "count": total,
        "errored_cases": errored,
        "judge_errors": judge_errors,
        "deterministic_pass_rate": round(sum(deterministic_passes) / total, 3) if total else None,
        "mean_overall_score": _mean(overall_scores),
        "mean_rubric_scores": {k: _mean(v) for k, v in sorted(rubric_dims.items())},
        "agent_latency_seconds": {
            "mean": _mean(latencies),
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "estimated_cost_usd": round(sum(costs), 4),
        "copilot_premium_requests": round(sum(premium), 2),
        "by_tag": {k: summarize(v) for k, v in sorted(by_tag.items())},
        "by_vertical": {k: summarize(v) for k, v in sorted(by_vertical.items())},
        "by_difficulty": {k: summarize(v) for k, v in sorted(by_difficulty.items())},
    }


def compare_runs(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_by_id = {r["case_id"]: r for r in base.get("results", [])}
    cand_by_id = {r["case_id"]: r for r in candidate.get("results", [])}
    common = sorted(set(base_by_id) & set(cand_by_id))
    changes = []
    for cid in common:
        b = base_by_id[cid]
        c = cand_by_id[cid]
        b_score = b.get("overall_score")
        c_score = c.get("overall_score")
        delta = None
        if isinstance(b_score, (int, float)) and isinstance(c_score, (int, float)):
            delta = round(float(c_score) - float(b_score), 3)
        b_pass = b.get("deterministic", {}).get("passed")
        c_pass = c.get("deterministic", {}).get("passed")
        if delta or b_pass != c_pass:
            changes.append(
                {
                    "case_id": cid,
                    "title": c.get("title", b.get("title")),
                    "base_passed": b_pass,
                    "candidate_passed": c_pass,
                    "base_score": b_score,
                    "candidate_score": c_score,
                    "delta": delta,
                    "base_issues": b.get("deterministic", {}).get("issues", []),
                    "candidate_issues": c.get("deterministic", {}).get("issues", []),
                }
            )
    regressions = [x for x in changes if (x.get("delta") or 0) < -0.25 or (x["base_passed"] and not x["candidate_passed"])]
    improvements = [x for x in changes if (x.get("delta") or 0) > 0.25 or (not x["base_passed"] and x["candidate_passed"])]
    return {
        "base_run_id": base.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "common_cases": len(common),
        "base_aggregate": base.get("aggregate"),
        "candidate_aggregate": candidate.get("aggregate"),
        "regressions": regressions,
        "improvements": improvements,
        "all_changes": changes,
    }
