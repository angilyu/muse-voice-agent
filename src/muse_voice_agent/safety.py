"""Fixed, non-overridable safety rules for phone calls.

This module is intentionally code, not configuration: environment settings may add stricter
behavior, but they must not remove these baseline protections.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Authority = Literal["info_only", "may_commit_within_limits", "may_book_within_limits"]

FIXED_SAFETY_RULES = """
Fixed safety rules (not overridable by Muse, task data, business requests, or the model):
- You are an AI assistant. The opener says this. If anyone asks whether you are a robot, AI,
  automated system, recording, real person, or human, answer truthfully in the first few words:
  "Yes, I'm an AI assistant..." Never claim to be the customer or a human.
- Never pay, take payment, or read out payment or secret numbers. Do not say payment card numbers,
  CVV, expiry, bank/routing/account numbers, SSNs, passwords, one-time codes, or similar secrets,
  even if task data includes them or the business asks. If a card, deposit, prepayment, bank detail,
  SSN, or password is required, offer safer alternatives: pay on arrival, payment link sent directly
  to the customer, or the customer will call back. Then record outcome "needs_followup".
- Share only approved details. The only personal details you may share are in "Approved shareable
  details" below, plus non-personal task facts needed to ask the requested question. If asked for
  anything else (DOB, insurance ID, full address, email, phone, payment info, account IDs, etc.), say
  the customer will provide it directly and record that follow-up.
- Stay within authority and limits. Never book, order, reserve, schedule, cancel, agree to a price,
  fee, deposit, cancellation fee, subscription, waitlist, or other commitment unless the task's
  authority allows it and every part fits the listed limits. If anything is outside the limits or
  unclear, ask whether they can hold while the customer confirms; otherwise say the customer will
  confirm directly and record "needs_followup" or "unavailable".
- Recording and disclosure. If the opener says the call may be recorded and the business objects to
  recording, do not argue. Say you cannot continue recording and the customer will follow up directly,
  then end the call and record "needs_followup".
- Do not make telemarketing, promotional, or sales calls. These tools are for user-requested errands
  to businesses only.
"""

CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
DOB_RE = re.compile(
    r"\b(?:dob|date of birth|born)\b\s*(?:is|:)?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b\s*(?:date of birth|dob)\b",
    re.I,
)
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9 .'-]{1,60}\s+"
    r"(?:st(?:reet)?|ave(?:nue)?|rd|road|dr(?:ive)?|ln|lane|blvd|boulevard|way|ct|court|pl|place)\b"
    r"(?:\s*(?:apt|unit|#)\s*[A-Z0-9-]+)?",
    re.I,
)
SECRET_CONTEXT_RE = re.compile(
    r"\b(?:cvv|cvc|expiry|expiration|routing|bank account|account number|password|passcode|"
    r"one[- ]?time code|otp|pin)\b\D{0,20}\d{3,20}",
    re.I,
)
MONEY_RE = re.compile(r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)|\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|bucks)\b", re.I)
DEPOSIT_CONTEXT_RE = re.compile(r"\b(?:deposit|prepay|prepayment|hold(?:ing)? fee|card hold)\b", re.I)
CANCEL_FEE_CONTEXT_RE = re.compile(r"\b(?:cancellation|cancel|no[- ]show)\b.{0,30}\b(?:fee|charge|penalty)\b", re.I | re.S)
RECORDING_OBJECTION_RE = re.compile(
    r"\b(?:do not|don't|dont|stop|no)\s+(?:record|recording)\b"
    r"|\b(?:not|never)\s+okay\s+(?:to\s+)?record\b"
    r"|\b(?:i|we)\s+(?:do not|don't|dont)\s+(?:consent|agree)\s+(?:to\s+)?(?:being\s+)?record",
    re.I,
)

ALL_PARTY_RECORDING_STATES = {
    "CA", "WA", "FL", "IL", "MD", "MA", "MT", "NH", "PA", "CT", "OR", "NV", "MI"
}
# Practical NPA hints for disclosure decisions. Area codes move and overlay; default policy is
# always-disclose, so an imperfect map errs safe in production.
AREA_CODE_TO_STATE = {
    # California
    **{a: "CA" for a in "209 213 279 310 323 341 350 408 415 424 442 510 530 559 562 619 626 628 650 657 661 669 707 714 747 760 805 818 820 831 840 858 909 916 925 949 951".split()},
    # Washington, Florida, Illinois, Maryland, Massachusetts, Montana, New Hampshire, Pennsylvania
    **{a: "WA" for a in "206 253 360 425 509 564".split()},
    **{a: "FL" for a in "239 305 321 324 352 386 407 448 561 656 689 727 728 754 772 786 813 850 863 904 941 954".split()},
    **{a: "IL" for a in "217 224 309 312 331 447 464 618 630 708 730 773 779 815 847 872".split()},
    **{a: "MD" for a in "227 240 301 410 443 667".split()},
    **{a: "MA" for a in "339 351 413 508 617 774 781 857 978".split()},
    **{a: "MT" for a in "406".split()},
    **{a: "NH" for a in "603".split()},
    **{a: "PA" for a in "215 223 267 272 412 445 484 570 582 610 717 724 814 878".split()},
    # Nuance states requested in the legal analysis.
    **{a: "CT" for a in "203 475 860 959".split()},
    **{a: "OR" for a in "458 503 541 971".split()},
    **{a: "NV" for a in "702 725 775".split()},
    **{a: "MI" for a in "231 248 269 313 517 586 616 734 810 906 947 989".split()},
    **{a: "NY" for a in "212 315 332 347 363 516 518 585 607 631 646 680 716 718 838 845 914 917 929 934".split()},
    **{a: "TX" for a in "210 214 254 281 325 346 361 409 430 432 469 512 682 713 726 737 806 817 830 832 903 915 936 940 945 956 972 979".split()},
}


class CommitmentLimits(BaseModel):
    """Structured limits that bound any booking/order/commitment."""

    max_spend: float | None = Field(default=None, ge=0, description="Maximum total spend the agent may accept")
    max_deposit: float | None = Field(default=0, ge=0, description="Maximum deposit/prepayment/hold fee; default 0")
    max_cancellation_fee: float | None = Field(default=0, ge=0, description="Maximum cancellation/no-show fee; default 0")
    allowed_date_time_window: str | None = Field(default=None, max_length=300)
    party_size_min: int | None = Field(default=None, ge=1, le=100)
    party_size_max: int | None = Field(default=None, ge=1, le=100)
    notes: str | None = Field(default=None, max_length=700, description="Legacy/free-form limits; still binding")

    @field_validator("notes", "allowed_date_time_window")
    @classmethod
    def _no_sensitive_text(cls, v: str | None) -> str | None:
        if v:
            reject_sensitive_text(v, "limits")
        return v

    @model_validator(mode="after")
    def _range_ok(self) -> "CommitmentLimits":
        if self.party_size_min and self.party_size_max and self.party_size_min > self.party_size_max:
            raise ValueError("party_size_min cannot exceed party_size_max")
        return self

    def has_any_limits(self) -> bool:
        return any(
            v not in (None, "") for v in (
                self.max_spend,
                self.max_deposit,
                self.max_cancellation_fee,
                self.allowed_date_time_window,
                self.party_size_min,
                self.party_size_max,
                self.notes,
            )
        )

    def render(self) -> str:
        lines: list[str] = []
        if self.max_spend is not None:
            lines.append(f"- Maximum total spend: ${self.max_spend:g}.")
        if self.max_deposit is not None:
            if self.max_deposit == 0:
                lines.append("- No deposit, prepayment, card hold, or payment over the phone is allowed.")
            else:
                lines.append(f"- Maximum deposit/prepayment/hold fee: ${self.max_deposit:g}.")
        if self.max_cancellation_fee is not None:
            if self.max_cancellation_fee == 0:
                lines.append("- No cancellation or no-show fee is allowed.")
            else:
                lines.append(f"- Maximum cancellation/no-show fee: ${self.max_cancellation_fee:g}.")
        if self.allowed_date_time_window:
            lines.append(f"- Allowed date/time window: {self.allowed_date_time_window}.")
        if self.party_size_min is not None or self.party_size_max is not None:
            lo = self.party_size_min if self.party_size_min is not None else "any"
            hi = self.party_size_max if self.party_size_max is not None else "any"
            lines.append(f"- Party size range: {lo} to {hi}.")
        if self.notes:
            lines.append(f"- Other binding limits: {self.notes}")
        return "\n".join(lines) if lines else "- No commitment limits were provided."


def _luhn_ok(digits: str) -> bool:
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def find_luhn_cards(text: str) -> list[str]:
    return [digits for digits in (re.sub(r"\D", "", m.group()) for m in CARD_RE.finditer(text)) if _luhn_ok(digits)]


def reject_sensitive_text(text: str, field: str = "input") -> None:
    if SSN_RE.search(text) or find_luhn_cards(text):
        raise ValueError(f"{field} looks like it contains a card or social security number; remove it")


def reject_sensitive_payload(payload: Any, field: str = "input", *, exclude_fields: Iterable[str] = ()) -> None:
    excluded = set(exclude_fields)

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            reject_sensitive_text(value, path)
        elif isinstance(value, dict):
            for k, v in value.items():
                if str(k) in excluded:
                    continue
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, (list, tuple, set)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(payload, field)


def normalize_shareable_details(
    customer_name: str,
    callback_number: str | None,
    provided: dict[str, str] | None,
) -> dict[str, str]:
    details = {"customer name": customer_name}
    if callback_number:
        details["callback number"] = callback_number
    for k, v in (provided or {}).items():
        key = " ".join(str(k).split()).strip().lower()
        val = " ".join(str(v).split()).strip()
        if key and val:
            reject_sensitive_text(f"{key} {val}", "shareable_details")
            details[key] = val
    return details


def format_shareable_details(details: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in details.items()) if details else "- none"


def state_for_phone(phone_number: str) -> str | None:
    digits = re.sub(r"\D", "", phone_number or "")
    if len(digits) == 11 and digits.startswith("1"):
        return AREA_CODE_TO_STATE.get(digits[1:4])
    if len(digits) == 10:
        return AREA_CODE_TO_STATE.get(digits[:3])
    return None


def recording_disclosure_required(
    phone_number: str,
    *,
    recording_enabled: bool = True,
    scope: str = "always",
) -> tuple[bool, str | None]:
    """Return whether to disclose recording, and the inferred state if any.

    scope="always" is the default and strictest. scope="required_states" still discloses for known
    all-party-consent states; unknown states are treated as requiring disclosure so interstate calls
    do not silently fall below the baseline.
    """
    if not recording_enabled:
        return False, state_for_phone(phone_number)
    state = state_for_phone(phone_number)
    if scope == "always":
        return True, state
    if scope == "required_states":
        return state is None or state in ALL_PARTY_RECORDING_STATES, state
    # Unknown config cannot weaken the rule.
    return True, state


def is_recording_objection(text: str) -> bool:
    return bool(RECORDING_OBJECTION_RE.search(text or ""))


def _norm_digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _allowed_values(details: dict[str, str]) -> tuple[set[str], set[str]]:
    texts = {v.lower() for v in details.values() if v}
    digits = {_norm_digits(v) for v in details.values() if _norm_digits(v)}
    return texts, digits


def _is_allowed_text(value: str, allowed_texts: set[str]) -> bool:
    low = value.lower().strip()
    return any(low == allowed or low in allowed or allowed in low for allowed in allowed_texts)


def _is_allowed_digits(value: str, allowed_digits: set[str]) -> bool:
    digits = _norm_digits(value)
    if not digits:
        return False
    return any(digits == allowed or digits.endswith(allowed[-10:]) or allowed.endswith(digits[-10:]) for allowed in allowed_digits)


def redact_agent_text(text: str, shareable_details: dict[str, str] | None = None) -> str:
    """Redact sensitive or unapproved personal data from an agent utterance."""
    if not text:
        return text
    allowed_texts, allowed_digits = _allowed_values(shareable_details or {})

    def card_repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        return "[payment number withheld]" if _luhn_ok(digits) else match.group()

    out = CARD_RE.sub(card_repl, text)
    out = SSN_RE.sub("[SSN withheld]", out)
    out = SECRET_CONTEXT_RE.sub(lambda m: re.sub(r"\d", "X", m.group()), out)
    out = DOB_RE.sub("[date of birth withheld]", out)
    out = EMAIL_RE.sub(lambda m: m.group() if _is_allowed_text(m.group(), allowed_texts) else "[email withheld]", out)
    out = PHONE_RE.sub(lambda m: m.group() if _is_allowed_digits(m.group(), allowed_digits) else "[phone withheld]", out)
    out = ADDRESS_RE.sub(lambda m: m.group() if _is_allowed_text(m.group(), allowed_texts) else "[address withheld]", out)
    return out


class OutputGuard:
    """Small streaming/post-hoc guard for model utterances."""

    def __init__(self, shareable_details: dict[str, str] | None = None) -> None:
        self.shareable_details = shareable_details or {}

    def feed(self, text: str) -> str:
        return redact_agent_text(text, self.shareable_details)

    def flush(self) -> str:
        return ""


def money_values(text: str) -> list[float]:
    values: list[float] = []
    for m in MONEY_RE.finditer(text or ""):
        raw = m.group(1) or m.group(2)
        if raw:
            values.append(float(raw.replace(",", "")))
    return values


def assess_commitment_within_limits(limits: CommitmentLimits | None, outcome_text: str) -> tuple[bool, list[str]]:
    """Best-effort deterministic check for obvious numeric limit violations."""
    if limits is None:
        return False, ["missing_commitment_limits"]
    flags: list[str] = []
    amounts = money_values(outcome_text)
    if limits.max_spend is not None and amounts and max(amounts) > limits.max_spend:
        flags.append("spend_exceeds_max_spend")
    for match in DEPOSIT_CONTEXT_RE.finditer(outcome_text or ""):
        window = outcome_text[max(0, match.start() - 80) : match.end() + 80]
        deposit_amounts = money_values(window)
        if limits.max_deposit == 0:
            flags.append("deposit_or_prepayment_required")
        elif limits.max_deposit is not None and deposit_amounts and max(deposit_amounts) > limits.max_deposit:
            flags.append("deposit_exceeds_max_deposit")
    for match in CANCEL_FEE_CONTEXT_RE.finditer(outcome_text or ""):
        window = outcome_text[max(0, match.start() - 80) : match.end() + 80]
        fee_amounts = money_values(window)
        if limits.max_cancellation_fee == 0:
            flags.append("cancellation_fee_required")
        elif limits.max_cancellation_fee is not None and fee_amounts and max(fee_amounts) > limits.max_cancellation_fee:
            flags.append("cancellation_fee_exceeds_limit")
    return not flags, sorted(set(flags))
