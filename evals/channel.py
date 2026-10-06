from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal

ChannelName = Literal["clean", "phone"]

SHORT_DROP_WORDS = 3
BARGE_IN_TRIGGER_WORDS = 25
BARGE_IN_TRUNCATE_WORDS = 20
OPENER_PREFIX_WORDS = 1

_NUMBER_CONFUSIONS = [
    (re.compile(r"\bfifteen\b", re.I), "fifty"),
    (re.compile(r"\bfifty\b", re.I), "fifteen"),
    (re.compile(r"\bthirteen\b", re.I), "thirty"),
    (re.compile(r"\bthirty\b", re.I), "thirteen"),
    (re.compile(r"\beight fifteen\b", re.I), "eighty fifteen"),
    (re.compile(r"\b8:15\b", re.I), "80:15"),
    (re.compile(r"\bseven fifteen\b", re.I), "seventy fifteen"),
]
_FILLER_DROP = {"the", "a", "an", "to", "for", "at", "on", "is", "are", "we", "can"}


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text)


def _short_prefix(text: str, words: int) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= words:
        return text.strip()
    return " ".join(tokens[:words]).rstrip(",.;:!?—-") + ","


def _truncate_words(text: str, words: int) -> str:
    tokens = re.findall(r"\S+", text)
    if len(tokens) <= words:
        return text.strip()
    return " ".join(tokens[:words]).rstrip(",.;:!?—-") + "…"


def _quote(text: str) -> str:
    return text.replace('"', '\\"')


@dataclass
class ChannelConfig:
    name: ChannelName = "clean"
    effects: set[str] = field(default_factory=set)
    seed: int | None = None
    stt_drop_probability: float = 0.35
    opener_cut_probability: float = 0.12
    asr_noise_probability: float = 0.25

    @property
    def enabled(self) -> bool:
        return self.name == "phone" or bool(self.effects)

    def forced(self, effect: str) -> bool:
        return effect in self.effects


@dataclass
class ChannelState:
    config: ChannelConfig
    repeat_index: int = 0

    def __post_init__(self) -> None:
        base = 0 if self.config.seed is None else self.config.seed
        self.rng = random.Random(base + 7919 * self.repeat_index)
        self.first_business_seen = False
        self.first_agent_seen = False
        self.next_business_override: str | None = None

    def business_speech(self, text: str) -> dict[str, Any]:
        """Return actual speech, STT text heard by the agent, and harness markers."""
        actual = (text or "").strip()
        if not actual or not self.config.enabled:
            if actual:
                self.first_business_seen = True
            return {"truth": actual, "agent": actual, "markers": [], "changed": False}

        first = not self.first_business_seen
        self.first_business_seen = True
        markers: list[str] = []
        heard = actual
        effects = self.config.effects
        short = len(_words(actual)) <= SHORT_DROP_WORDS
        force_drop_first = first and ("stt_drop_first_greeting" in effects or "first_greeting_drop" in effects)
        force_drop = "stt_drop" in effects and first and short
        prob_drop = self.config.name == "phone" and short and self.rng.random() < self.config.stt_drop_probability
        if short and (force_drop_first or force_drop or prob_drop):
            markers.append(f'[stt dropped: "{_quote(actual)}"]')
            return {"truth": actual, "agent": "", "markers": markers, "changed": True}

        force_noise = "asr_noise" in effects or "number_asr_error" in effects
        prob_noise = self.config.name == "phone" and self.rng.random() < self.config.asr_noise_probability
        if force_noise or prob_noise:
            corrupted = self._corrupt_asr(actual, force=force_noise)
            if corrupted != actual:
                heard = corrupted
                markers.append(f'[agent heard: "{_quote(heard)}"]')

        return {"truth": actual, "agent": heard, "markers": markers, "changed": bool(markers)}

    def agent_speech(self, text: str) -> dict[str, Any]:
        """Return the part actually spoken on the phone and markers for interruption effects."""
        spoken = (text or "").strip()
        markers: list[str] = []
        if not spoken or not self.config.enabled:
            if spoken:
                self.first_agent_seen = True
            return {"spoken": spoken, "markers": markers, "interrupted": False, "kind": None}

        first = not self.first_agent_seen
        self.first_agent_seen = True
        effects = self.config.effects
        if first and ("opener_cut" in effects or (self.config.name == "phone" and self.rng.random() < self.config.opener_cut_probability)):
            prefix = _short_prefix(spoken, OPENER_PREFIX_WORDS)
            markers.append("[interrupted: opener cut off]")
            return {"spoken": prefix, "markers": markers, "interrupted": True, "kind": "opener_cut"}

        if len(_words(spoken)) > BARGE_IN_TRIGGER_WORDS:
            prefix = _truncate_words(spoken, BARGE_IN_TRUNCATE_WORDS)
            markers.append("[interrupted: business barged in]")
            self.next_business_override = "Sorry—what?"
            return {"spoken": prefix, "markers": markers, "interrupted": True, "kind": "barge_in"}

        return {"spoken": spoken, "markers": markers, "interrupted": False, "kind": None}

    def pop_business_override(self) -> str | None:
        value = self.next_business_override
        self.next_business_override = None
        if value:
            self.first_business_seen = True
        return value

    def _corrupt_asr(self, text: str, *, force: bool) -> str:
        corrupted = text
        patterns = _NUMBER_CONFUSIONS[:]
        self.rng.shuffle(patterns)
        changed = False
        for pat, repl in patterns:
            if pat.search(corrupted):
                corrupted = pat.sub(repl, corrupted, count=1)
                changed = True
                break
        tokens = corrupted.split()
        if len(tokens) > 4 and (force or self.rng.random() < 0.5):
            candidates = [i for i, tok in enumerate(tokens) if re.sub(r"\W", "", tok).lower() in _FILLER_DROP]
            if candidates:
                del tokens[self.rng.choice(candidates)]
                corrupted = " ".join(tokens)
                changed = True
        if not changed and force and tokens:
            idx = min(len(tokens) - 1, max(0, len(tokens) // 2))
            tokens[idx] = "[garbled]"
            corrupted = " ".join(tokens)
        return corrupted.strip()


def case_channel_effects(case: Any) -> set[str]:
    effects: set[str] = set()
    for obj in (getattr(case, "persona", None), getattr(case, "expectations", None)):
        values = getattr(obj, "channel_effects", None) if obj is not None else None
        if values:
            effects.update(str(v) for v in values)
    return effects


def make_channel_state(case: Any, *, channel: ChannelName = "clean", seed: int | None = None, repeat_index: int = 0) -> ChannelState:
    return ChannelState(ChannelConfig(name=channel, effects=case_channel_effects(case), seed=seed), repeat_index=repeat_index)
