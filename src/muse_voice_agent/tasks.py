"""Call task definitions and the system prompts the phone agent uses for each."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

E164 = re.compile(r"^\+[1-9]\d{6,14}$")


def normalize_phone(raw: str) -> str:
    """Normalize a phone number to E.164. Bare 10-digit numbers are treated as US (+1)."""
    digits = re.sub(r"[^\d+]", "", raw or "")
    if not digits.startswith("+"):
        if len(digits) == 10:
            digits = "+1" + digits
        elif len(digits) == 11 and digits.startswith("1"):
            digits = "+" + digits
        else:
            digits = "+" + digits
    if not E164.match(digits):
        raise ValueError(f"'{raw}' is not a valid phone number (expected E.164, e.g. +14155550123)")
    return digits


_PLACEHOLDER_NAMES = {
    "user", "the user", "customer", "the customer", "client", "me", "myself", "name", "my name",
    "unknown", "n/a", "na", "none", "null", "anonymous", "someone", "somebody", "tbd",
}


class _BaseTask(BaseModel):
    business_name: str = Field(min_length=1, max_length=120)
    phone_number: str = Field(description="Business phone number, E.164 preferred")
    customer_name: str = Field(
        min_length=1, max_length=80, description="Full name of the person the call is made for"
    )
    callback_number: str | None = Field(
        default=None, description="Number the business can call back. Shared only if asked."
    )

    @field_validator("customer_name", mode="before")
    @classmethod
    def _customer_name(cls, v: object) -> object:
        if isinstance(v, str):
            v = " ".join(v.split())
            if not v or v.lower() in _PLACEHOLDER_NAMES:
                raise ValueError(
                    "customer_name must be the real name of the user you're calling for; "
                    "ask the user for their name"
                )
        return v

    @field_validator("phone_number")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalize_phone(v)

    @field_validator("callback_number")
    @classmethod
    def _callback(cls, v: str | None) -> str | None:
        return normalize_phone(v) if v else None


class RestaurantReservation(_BaseTask):
    kind: Literal["restaurant_reservation"] = "restaurant_reservation"
    party_size: int = Field(ge=1, le=30)
    date: str = Field(description="Requested date, e.g. 'Friday Oct 10' or '2026-10-10'")
    time: str = Field(description="Requested time, e.g. '7:30 PM'")
    flexibility: str | None = Field(
        default=None, description="Acceptable alternatives, e.g. 'anytime 6-8:30pm, same day only'"
    )
    special_requests: str | None = Field(default=None, description="Seating, allergies, occasion")


class HandymanQuote(_BaseTask):
    kind: Literal["handyman_quote"] = "handyman_quote"
    job_description: str = Field(min_length=3, max_length=1000)
    location: str = Field(description="City / neighborhood / ZIP. Avoid a full street address.")
    preferred_timing: str | None = Field(default=None, description="e.g. 'weekday mornings next week'")
    budget: str | None = Field(default=None, description="Optional budget to mention if asked")


Authority = Literal["info_only", "may_book_within_limits"]

# Card numbers (13-19 digits, optional spaces/dashes) and US SSNs must never be handed to the agent.
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _reject_sensitive(text: str, field: str) -> None:
    cards = (re.sub(r"\D", "", m.group()) for m in _CARD.finditer(text))
    if _SSN.search(text) or any(_luhn_ok(d) for d in cards):
        raise ValueError(f"{field} looks like it contains a card or social security number; remove it")


class GeneralCall(_BaseTask):
    """Any errand Muse can describe as a brief: availability checks, questions, simple bookings."""

    kind: Literal["general"] = "general"
    goal: str = Field(min_length=3, max_length=1000, description="What the call should accomplish")
    questions: list[str] = Field(
        default_factory=list, max_length=10, description="Specific questions to get answered"
    )
    shareable_details: dict[str, str] = Field(
        default_factory=dict,
        description="Facts the agent may share if relevant (dates, party size, order number...)",
    )
    authority: Authority = Field(
        default="info_only",
        description="info_only: ask and commit to nothing. may_book_within_limits: may book/"
        "reserve/schedule only within `limits`.",
    )
    limits: str | None = Field(
        default=None,
        max_length=500,
        description="Required with may_book_within_limits, e.g. 'Oct 10-12, king room, max $250/night'",
    )

    @field_validator("questions")
    @classmethod
    def _questions(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v if q and q.strip()]
        for q in cleaned:
            if len(q) > 300:
                raise ValueError("each question must be 300 characters or fewer")
            _reject_sensitive(q, "questions")
        return cleaned

    @field_validator("shareable_details")
    @classmethod
    def _details(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 20:
            raise ValueError("at most 20 shareable_details")
        for k, val in v.items():
            if len(k) > 60 or len(val) > 300:
                raise ValueError("shareable_details keys must be <=60 and values <=300 characters")
            _reject_sensitive(f"{k} {val}", "shareable_details")
        return v

    @field_validator("goal", "limits")
    @classmethod
    def _no_sensitive(cls, v: str | None, info) -> str | None:  # noqa: ANN001
        if v:
            _reject_sensitive(v, info.field_name)
        return v

    @model_validator(mode="after")
    def _limits_required(self) -> "GeneralCall":
        if self.authority == "may_book_within_limits" and not (self.limits or "").strip():
            raise ValueError("limits are required when authority is may_book_within_limits")
        return self


AnyTask = RestaurantReservation | HandymanQuote | GeneralCall
CallTask = Annotated[
    Union[RestaurantReservation, HandymanQuote, GeneralCall], Field(discriminator="kind")
]
CALL_TASK_ADAPTER: TypeAdapter[CallTask] = TypeAdapter(CallTask)


def parse_task(data: dict) -> AnyTask:
    return CALL_TASK_ADAPTER.validate_python(data)


_COMMON_RULES = """
How to behave on the phone:
- You are an AI assistant placing a call on behalf of {customer_name}. In your first sentence, say
  you're their assistant calling on their behalf, e.g. "Hi, this is an assistant calling on behalf
  of {customer_name}." You don't need to mention that you're an AI up front.
- If anyone asks whether you're a robot, an AI, automated or a real person, answer honestly that
  you're an AI assistant, then carry on. Never claim or imply that you're human.
- Wait for the other person to speak first; they will usually greet you.
- Speak naturally and briefly: one or two short sentences per turn, no lists, no markdown, no emojis.
- Never invent details about {customer_name}. If asked something you don't know, say you'll have
  {customer_name} follow up, and note it in the outcome.
- Never give out payment card numbers, addresses, or other personal data. You may share the
  customer's name{callback_clause}.
- Never agree to deposits, cancellation fees, or prepayment. If one is required, get the details and
  record the outcome as needs_followup instead of confirming.
- If you reach voicemail or an automated system you can't get through, call record_outcome with
  outcome "voicemail" and then say a short goodbye.
- As soon as you have the answer (success, refusal, or a blocker), call the record_outcome tool
  exactly once with everything you learned, then say a short, polite goodbye. Do not keep talking
  after the goodbye.
"""


def build_system_prompt(task: AnyTask) -> str:
    callback_clause = (
        f" and the callback number {task.callback_number} if they ask for one"
        if task.callback_number
        else ""
    )
    rules = _COMMON_RULES.format(customer_name=task.customer_name, callback_clause=callback_clause)

    if isinstance(task, GeneralCall):
        return _general_goal(task) + "\n" + rules
    if isinstance(task, RestaurantReservation):
        goal = f"""You are calling {task.business_name} to book a table.
Reservation request:
- Name: {task.customer_name}
- Party size: {task.party_size}
- Date: {task.date}
- Time: {task.time}
- Acceptable alternatives: {task.flexibility or "none given; ask the customer to follow up if the time isn't available"}
- Special requests: {task.special_requests or "none"}

Goal: get a confirmed reservation. If the exact time isn't available, accept an alternative only if it
fits the acceptable alternatives above. Confirm the final date, time, party size and the name the
booking is under before ending the call. Use outcome "booked", "unavailable", "declined" or
"needs_followup"."""
    else:
        goal = f"""You are calling {task.business_name} to get a quote for a job.
Job request:
- Customer: {task.customer_name}
- Work needed: {task.job_description}
- Location: {task.location}
- Preferred timing: {task.preferred_timing or "flexible"}
- Budget (only mention if asked): {task.budget or "not specified"}

Goal: find out whether they do this kind of work, a price or price range (or how they price it, e.g.
hourly rate, trip fee, free estimate visit), and their earliest availability. Do NOT book or commit
to anything. Use outcome "quote_received", "declined" (they don't do this work) or "needs_followup"
(e.g. they need a site visit or photos first)."""

    return goal + "\n" + rules


def _general_goal(task: GeneralCall) -> str:
    details = (
        "\n".join(f"- {k}: {v}" for k, v in task.shareable_details.items())
        or "- none beyond the customer's name"
    )
    questions = (
        "\n".join(f"{i}. {q}" for i, q in enumerate(task.questions, 1))
        or "(none listed; get whatever information the goal needs)"
    )
    if task.authority == "may_book_within_limits":
        authority = f"""You MAY book, reserve or schedule, but only if every part of it fits these limits:
{task.limits}
If what they offer falls outside the limits, do not accept it; note the offer and use outcome
"unavailable" or "needs_followup". If you do book, read the final details back to confirm, ask for
a confirmation or reference number, and use outcome "booked"."""
    else:
        authority = """You may NOT book, reserve, order, schedule or agree to anything. You are only gathering
information. If they offer to book or hold something, politely say {customer} will call back to
confirm, and note what they offered.""".format(customer=task.customer_name)

    return f"""You are calling {task.business_name} on behalf of {task.customer_name}.
Goal: {task.goal}

Details you may share if they are relevant or asked for:
{details}

Questions to get answered:
{questions}

Authority:
{authority}

Ask one question at a time. When you record the outcome, put each question and the answer you got
(or "not answered") in `answers`, put any price in `quote`, any date/time availability in
`availability`, and any confirmation number in `reference`. Use outcome "info_received" when you got
the information, "booked" only if you were allowed to book and did, otherwise "unavailable",
"declined" or "needs_followup"."""
