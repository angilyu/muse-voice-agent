from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from muse_voice_agent.tasks import AnyTask, parse_task

CASES_PATH = Path(__file__).with_name("bay_area_cases.json")
REGRESSION_CASES_PATH = Path(__file__).with_name("regression_cases.json")
SPLITS_PATH = Path(__file__).with_name("splits.json")

ToolName = Literal["place_call", "book_restaurant_reservation", "request_handyman_quote"]
Difficulty = Literal["easy", "medium", "hard"]


class MCPBrief(BaseModel):
    tool: ToolName
    args: dict[str, Any]

    def task_payload(self) -> dict[str, Any]:
        kind = {
            "place_call": "general",
            "book_restaurant_reservation": "restaurant_reservation",
            "request_handyman_quote": "handyman_quote",
        }[self.tool]
        return {"kind": kind, **self.args}

    def task(self) -> AnyTask:
        return parse_task(self.task_payload())


class BusinessPersona(BaseModel):
    answers_as: str = Field(description="Who answers the phone")
    facts: dict[str, Any] = Field(default_factory=dict)
    behaviors: list[str] = Field(default_factory=list)
    style: Literal["busy", "chatty", "terse"] = "busy"
    channel_effects: list[str] = Field(default_factory=list)
    will_not_reveal: list[str] = Field(default_factory=list)
    private_facts: dict[str, Any] = Field(default_factory=dict)


class RequiredFact(BaseModel):
    name: str
    any_of: list[str]
    where: Literal["outcome", "transcript", "agent", "business", "any"] = "outcome"

    @field_validator("any_of")
    @classmethod
    def _not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("required facts need at least one acceptable phrase")
        return v


class Expectations(BaseModel):
    allowed_outcomes: list[str]
    required_facts: list[RequiredFact] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    channel_effects: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=12, ge=2, le=40)

    @field_validator("allowed_outcomes")
    @classmethod
    def _allowed_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("allowed_outcomes cannot be empty")
        return v


class EvalCase(BaseModel):
    id: str
    title: str
    vertical: str
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    brief: MCPBrief
    persona: BusinessPersona
    expectations: Expectations

    @model_validator(mode="after")
    def _validates_real_task(self) -> "EvalCase":
        self.brief.task()
        return self


def load_all_cases(path: Path = CASES_PATH, *, include_regression: bool = True) -> list[EvalCase]:
    paths = [path]
    if include_regression and path == CASES_PATH and REGRESSION_CASES_PATH.exists():
        paths.append(REGRESSION_CASES_PATH)
    raw: list[dict[str, Any]] = []
    for p in paths:
        raw.extend(json.loads(p.read_text()))
    cases = [EvalCase.model_validate(item) for item in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def load_splits(path: Path = SPLITS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def select_cases(selector: str, *, seed: int | None = None, limit: int | None = None) -> list[EvalCase]:
    cases = load_all_cases()
    by_id = {c.id: c for c in cases}
    selected: list[EvalCase] = []
    parts = [p.strip() for p in (selector or "all").split(",") if p.strip()]
    for part in parts or ["all"]:
        if part == "all":
            selected.extend(cases)
        elif part.startswith("split:"):
            split_name = part[6:]
            splits = load_splits()
            if split_name not in {"dev", "heldout"}:
                raise ValueError(f"unknown eval split: {split_name}")
            selected.extend(by_id[cid] for cid in splits[split_name])
        elif part.startswith("tag:"):
            tag = part[4:]
            selected.extend([c for c in cases if tag in c.tags])
        elif part.startswith("vertical:"):
            vertical = part[9:].lower()
            selected.extend([c for c in cases if c.vertical.lower() == vertical])
        elif part.startswith("difficulty:"):
            difficulty = part[11:]
            selected.extend([c for c in cases if c.difficulty == difficulty])
        else:
            matches = [c for c in cases if c.id == part]
            if not matches:
                matches = [c for c in cases if part in c.tags]
            if not matches:
                raise ValueError(f"unknown eval case selector: {part}")
            selected.extend(matches)
    unique = list({c.id: c for c in selected}.values())
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(unique)
    if limit is not None:
        unique = unique[:limit]
    return unique
