from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

from . import copilot_llm
from .cases.schema import EvalCase
from .judge import judge_text_case
from .text import public_case

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_JUDGE_MODEL = "copilot:claude-sonnet-5.5"
REQUIRED_DIMS = {
    "task_completion",
    "outcome_accuracy",
    "turn_economy",
    "naturalness",
    "listening_and_repair",
    "confirmation_quality",
    "call_closing",
    "screening_and_ivr_handling",
    "policy_safety",
}


def _as_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("items"), list):
            return raw["items"]
        return [raw]
    raise ValueError("calibration files must contain an object, an object with items, or a list")


def load_calibration_items(path: Path = CALIBRATION_DIR) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.json")):
        if p.name.startswith("stub_"):
            continue
        raw = json.loads(p.read_text())
        for item in _as_list(raw):
            validate_calibration_item(item, source=p)
            items.append(item)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("calibration ids must be unique")
    return items


def validate_calibration_item(item: dict[str, Any], *, source: Path | None = None) -> None:
    prefix = f"{source}: " if source else ""
    required = {"id", "brief", "transcript", "outcome", "human_labels", "notes"}
    missing = required - set(item)
    if missing:
        raise ValueError(f"{prefix}{item.get('id', '<unknown>')} missing {sorted(missing)}")
    if not isinstance(item["transcript"], list) or not item["transcript"]:
        raise ValueError(f"{prefix}{item['id']} transcript must be a non-empty list")
    labels = item["human_labels"]
    if not isinstance(labels.get("pass"), bool):
        raise ValueError(f"{prefix}{item['id']} human_labels.pass must be bool")
    dims = labels.get("dims") or {}
    missing_dims = REQUIRED_DIMS - set(dims)
    if missing_dims:
        raise ValueError(f"{prefix}{item['id']} missing dims {sorted(missing_dims)}")
    for dim, bounds in dims.items():
        if bounds is None:
            continue
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(x, (int, float)) for x in bounds)):
            raise ValueError(f"{prefix}{item['id']} dim {dim} must be [min,max] or null")
        if not (1 <= bounds[0] <= bounds[1] <= 5):
            raise ValueError(f"{prefix}{item['id']} dim {dim} has invalid range {bounds}")


def _case_public(item: dict[str, Any]) -> dict[str, Any]:
    case_info = item.get("case")
    if isinstance(case_info, dict):
        try:
            return public_case(EvalCase.model_validate(case_info))
        except Exception:
            pass
    return {
        "id": item["id"],
        "title": item.get("brief", {}).get("title") or item["id"],
        "vertical": item.get("brief", {}).get("vertical", "unknown"),
        "difficulty": item.get("brief", {}).get("difficulty", "medium"),
        "tags": item.get("brief", {}).get("tags", ["calibration"]),
        "brief": item.get("brief", {}),
        "expectations": item.get("expectations", {}),
    }


def _deterministic_stub(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("deterministic") or {
        "passed": item["human_labels"]["pass"],
        "issues": [] if item["human_labels"]["pass"] else ["human-labeled failure"],
        "gates": {"passed": item["human_labels"]["pass"], "failures": [], "counts": {}},
    }


async def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    items = load_calibration_items(Path(args.path))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            jr = await judge_text_case(
                case_public=_case_public(item),
                transcript=item["transcript"],
                agent_transcript=item.get("agent_transcript"),
                outcome=item.get("outcome"),
                deterministic=_deterministic_stub(item),
                model_name=args.judge_model,
            )
            judge = jr.data
            human = item["human_labels"]
            dim_results = {}
            biases = []
            for dim, bounds in human["dims"].items():
                score_obj = (judge.get("scores") or {}).get(dim) or {}
                score = score_obj.get("score") if isinstance(score_obj, dict) else None
                if bounds is None:
                    in_range = score is None
                elif isinstance(score, (int, float)):
                    in_range = bounds[0] <= float(score) <= bounds[1]
                    biases.append(float(score) - statistics.fmean(bounds))
                else:
                    in_range = False
                dim_results[dim] = {"judge": score, "human": bounds, "in_range": in_range}
            return {
                "id": item["id"],
                "expected_pass": human["pass"],
                "judge_pass": bool(judge.get("pass")),
                "pass_agrees": bool(judge.get("pass")) == human["pass"],
                "dim_results": dim_results,
                "leniency_bias": round(statistics.fmean(biases), 3) if biases else None,
                "judge": judge,
                "usage": jr.usage,
            }

    try:
        results = await asyncio.gather(*(one(i) for i in items))
    finally:
        await copilot_llm.aclose()
    dim_total = sum(len(r["dim_results"]) for r in results)
    dim_in = sum(1 for r in results for d in r["dim_results"].values() if d["in_range"])
    bad_items = [r for r in results if not r["expected_pass"]]
    aggregate = {
        "count": len(results),
        "pass_agreement": round(sum(r["pass_agrees"] for r in results) / len(results), 3) if results else None,
        "bad_items_failed": round(sum(not r["judge_pass"] for r in bad_items) / len(bad_items), 3) if bad_items else None,
        "dim_in_range_rate": round(dim_in / dim_total, 3) if dim_total else None,
        "leniency_bias": round(statistics.fmean([r["leniency_bias"] for r in results if r["leniency_bias"] is not None]), 3) if results else None,
        "disagreements": [r["id"] for r in results if not r["pass_agrees"]],
    }
    run = {"judge_model": args.judge_model, "aggregate": aggregate, "results": results}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(run, indent=2, sort_keys=True))
    print(json.dumps({"aggregate": aggregate, "out": args.out}, indent=2))
    return run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run judge calibration on labeled transcripts")
    p.add_argument("--path", default=str(CALIBRATION_DIR))
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out")
    return p


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run_calibration(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
