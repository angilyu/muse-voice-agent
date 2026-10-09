"""Call task definitions and the system prompts the phone agent uses for each."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Union

from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .safety import (
    CommitmentLimits,
    FIXED_SAFETY_RULES,
    format_shareable_details,
    normalize_shareable_details,
    recording_disclosure_required,
    reject_sensitive_payload,
    reject_sensitive_text,
)

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


_PLACEHOLDER_ASSISTANT_NAMES = {
    "assistant", "an assistant", "the assistant", "ai", "ai assistant", "an ai assistant", "bot",
    "agent", "none", "null", "n/a", "na", "unknown", "name", "tbd",
}
_ASSISTANT_NAME = re.compile(r"[^\W\d_](?:[^\W\d_]|[' .-]){0,39}")  # letters, spaces, ' . -


class _BaseTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_name: str = Field(min_length=1, max_length=120)
    phone_number: str = Field(description="Business phone number, E.164 preferred")
    customer_name: str = Field(
        min_length=1, max_length=80, description="Full name of the person the call is made for"
    )
    callback_number: str | None = Field(
        default=None, description="Number the business can call back. Shared only if asked."
    )
    assistant_name: str | None = Field(
        default=None,
        max_length=40,
        description="The calling assistant's own name, e.g. 'Eva'. The agent says it in the opener.",
    )
    callback_of: str | None = Field(
        default=None, description="Original outbound call id, for inbound callbacks."
    )
    callback_context: str | None = Field(
        default=None,
        description="Prior-call context to use when answering an inbound callback.",
    )
    shareable_details: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Approved personal details the agent may share if relevant. The server always adds "
            "customer name and callback number; anything else is withheld."
        ),
    )
    recording_disclosure_required: bool = Field(default=True, exclude=True)
    recording_disclosure_state: str | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_sensitive_payload(cls, data: object) -> object:
        reject_sensitive_payload(data, cls.__name__, exclude_fields={"phone_number", "callback_number"})
        return data

    @field_validator("assistant_name", mode="before")
    @classmethod
    def _assistant_name(cls, v: object) -> object:
        if isinstance(v, str):
            v = " ".join(v.split()).strip(" .,")
            if not v or v.lower() in _PLACEHOLDER_ASSISTANT_NAMES:
                return None
            if not _ASSISTANT_NAME.fullmatch(v):
                raise ValueError("assistant_name must be a short name like 'Eva'")
        return v

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

    @field_validator("shareable_details")
    @classmethod
    def _shareable_details(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > 20:
            raise ValueError("at most 20 shareable_details")
        cleaned: dict[str, str] = {}
        for k, val in v.items():
            key = " ".join(str(k).split()).strip().lower()
            value = " ".join(str(val).split()).strip()
            if not key or not value:
                continue
            if len(key) > 60 or len(value) > 300:
                raise ValueError("shareable_details keys must be <=60 and values <=300 characters")
            reject_sensitive_text(f"{key} {value}", "shareable_details")
            cleaned[key] = value
        return cleaned

    @model_validator(mode="after")
    def _fixed_base_safety(self) -> "_BaseTask":
        self.shareable_details = normalize_shareable_details(
            self.customer_name, self.callback_number, self.shareable_details
        )
        required, state = recording_disclosure_required(
            self.phone_number, recording_enabled=True, scope="required_states"
        )
        # Task input may request stricter disclosure, but cannot suppress legally safer disclosure
        # for all-party/unknown states.
        self.recording_disclosure_required = bool(self.recording_disclosure_required or required)
        self.recording_disclosure_state = state
        return self


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


Authority = Literal["info_only", "may_commit_within_limits", "may_book_within_limits"]

class GeneralCall(_BaseTask):
    """Any phone errand Muse can describe as a brief: orders, appointments, questions, bookings."""

    kind: Literal["general"] = "general"
    goal: str = Field(min_length=3, max_length=1000, description="What the call should accomplish")
    questions: list[str] = Field(
        default_factory=list, max_length=10, description="Specific questions to get answered"
    )
    authority: Authority = Field(
        default="info_only",
        description="info_only: ask and commit to nothing. may_commit_within_limits: may book, "
        "order, schedule, reschedule, cancel, or reserve only within `limits`. "
        "may_book_within_limits is a backward-compatible alias.",
    )
    limits: CommitmentLimits | None = Field(
        default=None,
        description=(
            "Structured limits required with may_commit_within_limits/may_book_within_limits. "
            "Legacy strings are accepted as binding notes."
        ),
    )
    max_spend: float | None = Field(default=None, ge=0)
    max_deposit: float | None = Field(default=None, ge=0)
    max_cancellation_fee: float | None = Field(default=None, ge=0)
    allowed_date_time_window: str | None = Field(default=None, max_length=300)
    party_size_min: int | None = Field(default=None, ge=1, le=100)
    party_size_max: int | None = Field(default=None, ge=1, le=100)

    @field_validator("questions")
    @classmethod
    def _questions(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v if q and q.strip()]
        for q in cleaned:
            if len(q) > 300:
                raise ValueError("each question must be 300 characters or fewer")
            reject_sensitive_text(q, "questions")
        return cleaned

    @field_validator("goal", "limits")
    @classmethod
    def _no_sensitive(cls, v: object, info) -> object:  # noqa: ANN001
        if isinstance(v, str):
            reject_sensitive_text(v, info.field_name)
        return v

    @field_validator("limits", mode="before")
    @classmethod
    def _limits(cls, v: object) -> object:
        if isinstance(v, str):
            reject_sensitive_text(v, "limits")
            return {"notes": v}
        return v

    @model_validator(mode="after")
    def _limits_required_and_merged(self) -> "GeneralCall":
        data = (self.limits.model_dump() if self.limits else {}) | {
            k: v
            for k, v in {
                "max_spend": self.max_spend,
                "max_deposit": self.max_deposit,
                "max_cancellation_fee": self.max_cancellation_fee,
                "allowed_date_time_window": self.allowed_date_time_window,
                "party_size_min": self.party_size_min,
                "party_size_max": self.party_size_max,
            }.items()
            if v is not None
        }
        self.limits = CommitmentLimits(**data) if data else None
        if self.authority in {"may_commit_within_limits", "may_book_within_limits"} and not (
            self.limits and self.limits.has_any_limits()
        ):
            raise ValueError(
                "limits are required when authority is may_commit_within_limits "
                "or may_book_within_limits"
            )
        return self


class InboundMessage(BaseModel):
    """An inbound call from a number we cannot match to a recent outbound call."""

    kind: Literal["inbound_message"] = "inbound_message"
    business_name: str = "Unknown inbound caller"
    phone_number: str = Field(description="Inbound caller phone number")
    assistant_name: str | None = Field(default=None, max_length=40)

    @field_validator("phone_number")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalize_phone(v)

    @field_validator("assistant_name", mode="before")
    @classmethod
    def _assistant_name(cls, v: object) -> object:
        if isinstance(v, str):
            v = " ".join(v.split()).strip(" .,")
            if not v or v.lower() in _PLACEHOLDER_ASSISTANT_NAMES:
                return None
            if not _ASSISTANT_NAME.fullmatch(v):
                raise ValueError("assistant_name must be a short name like 'Eva'")
        return v


AnyTask = RestaurantReservation | HandymanQuote | GeneralCall | InboundMessage
CallTask = Annotated[
    Union[RestaurantReservation, HandymanQuote, GeneralCall, InboundMessage],
    Field(discriminator="kind"),
]
CALL_TASK_ADAPTER: TypeAdapter[CallTask] = TypeAdapter(CallTask)


def parse_task(data: dict) -> AnyTask:
    return CALL_TASK_ADAPTER.validate_python(data)


_COMMON_RULES = """
How to talk:
- You are {self_intro} calling on behalf of {customer_name}. Your opener, "{opening_line}", is
  spoken for you automatically the first time you talk, and again when a person picks up after a
  screener, a phone menu or silence. Don't repeat it yourself unless they didn't hear it or ask
  who's calling.
- Sound like a friendly, busy person on the phone, not a script. Keep every turn to one short
  sentence (about 20 words or fewer; two short sentences at most). Ask one thing, then stop and let
  them answer.
- Use contractions and everyday words ("Great", "Got it", "Perfect"). Avoid stiff phrases like "I
  would like to inquire", "Could you please confirm", "I appreciate your assistance", or "Is there
  anything else I can help with".
- Right after the opener, say why you're calling in one short sentence, e.g. "I'd like to book a
  table for two this Friday at 7." Don't mention alternatives, special requests or other details
  until they come up.
- Don't keep repeating details they already have (name, party size, date).
- Speech-to-text makes mistakes. If what they said is garbled, cut off or doesn't make sense, ask
  them to repeat it ("Sorry, what was that?") instead of guessing.
- If they ask you something, answer it first, then carry on. Spell names letter by letter, e.g.
  "W, E, N, J, I, N, G".
- If anyone asks whether you're a robot, an AI, automated or a real person, say honestly that you're
  an AI assistant in the first few words ("Yes, I'm an AI assistant..."), then carry on. If a later
  question mentions AI again, include "AI assistant" in the answer. Never claim or imply that you're
  human.

Screeners, menus, holds and voicemail:
- Call screeners (e.g. Google or iPhone call screening: "say your name and why you're calling",
  "this call is being screened") are recordings that pass your answer on before anyone picks up.
  Answer in ONE sentence with who and why, e.g. "I'm {self_intro} calling for {customer_name} to
  book a table for two this Friday at 7." If your opener was just spoken, only add why ("I'd like to
  book a table for two this Friday at 7."). No apology, no question, no small talk. Then stop and
  wait. When a person picks up, just say why you're calling; the opener is spoken for you.
- If a recording says to press a key ("press 1 to be connected"), call press_digits with that key;
  don't say the digit out loud. For a phone menu, use press_digits for the option that reaches
  reservations, scheduling or a person. If a menu asks you to say an option, say it in a few words.
- If they put you on hold or ask you to wait ("one sec", "hold on", "let me check"), say a quick
  "Sure, no problem." and call wait_on_hold in the same reply. Pick up where you left off when they're
  back.
- If someone other than the right person answers, briefly say why you're calling and ask them to
  help or connect you.
- If you reach voicemail, leave one short message saying who you're calling for, why, and that
  they'll call back, then record outcome "voicemail" and call end_call. If an automated system
  can't get you to a person, record "needs_followup" and call end_call. Don't use "info_received"
  for a menu or recording unless it answered everything requested.

Confirming and wrapping up:
- Before recording a booking, order, appointment, cancellation, reschedule, or other commitment,
  read the key details back once, the way a person would: "So that's two at 7 on Friday under
  Wenjing?" Then wait for them to confirm. Don't recite every field and don't do it twice.
- For food or drink orders, place the order exactly as written. If the brief says to use defaults
  for unspecified options, answer option questions with "the default is fine" or "regular is fine".
  Ask for the total and pickup/ready time, and give the customer's name for pickup. Do not record an
  order as placed until they confirm the order and give either the total, the pickup/ready time, or
  say they cannot provide it.
- Once you have the answer (confirmed, refused or blocked), in ONE reply say a short goodbye like
  "Perfect, thanks so much. Bye!" and call record_outcome with everything you learned, including any
  confirmation number, price or name they gave. Put that goodbye in record_outcome's `say` argument
  (it's spoken for you as soon as you write it). Never ask a question in that reply. If their last
  line asked you something ("Should I put her down?", "Want me to add you to the waitlist?"),
  answer it first in the same reply, e.g. "No need to book yet, Angi will call back. Thanks, bye!"
- If you tell the business the customer will follow up because information is missing or a fee,
  deposit, card, address or prepayment is required, record outcome "needs_followup" in that same
  reply.
- The call stays up after that. If they ask anything else (spell the name, a phone number, repeat a
  detail), answer it. If anything changes, call record_outcome again with the corrected details.
  Call end_call (you may say "Bye!" with it) only once they've said bye or there's nothing left.

Privacy and commitments:
- Never invent details about {customer_name}. If asked something you don't know or are not approved
  to share, say {customer_name} will follow up, and note it in the outcome.
- Approved shareable details:
{shareable_details}
- Do not share any personal detail not on that list.
- For pickup orders, pay-at-pickup is okay when it fits the brief. If the business requires a card
  or prepayment over the phone, do not place the order; record needs_followup.

{fixed_safety_rules}

Today is {today} (Pacific time).
"""


def _today() -> str:
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return f"{now:%A}, {now:%B} {now.day}, {now.year}"


def self_intro(task: AnyTask, *, ai: bool = True) -> str:
    """How the agent names itself: "Eva, an AI assistant" or just "an AI assistant"."""
    role = "an AI assistant" if ai else "an assistant"
    return f"{task.assistant_name}, {role}" if task.assistant_name else role


def opening_line(task: AnyTask) -> str:
    """Fixed first sentence, spoken before the LLM runs so the callee hears us immediately."""
    if isinstance(task, InboundMessage):
        line = f"Hi, this is {self_intro(task, ai=True)}. I can take a message."
    elif getattr(task, "callback_of", None):
        line = f"Hi, this is {self_intro(task, ai=True)} for {task.customer_name}. Thanks for calling back."
    else:
        line = f"Hi, this is {self_intro(task, ai=True)} calling on behalf of {task.customer_name}."
    if getattr(task, "recording_disclosure_required", True):
        line += " This call may be recorded."
    return line


def build_system_prompt(task: AnyTask) -> str:
    if isinstance(task, InboundMessage):
        return _inbound_message_prompt(task)

    rules = _COMMON_RULES.format(
        customer_name=task.customer_name,
        opening_line=opening_line(task),
        self_intro=self_intro(task),
        shareable_details=format_shareable_details(task.shareable_details),
        fixed_safety_rules=FIXED_SAFETY_RULES.strip(),
        today=_today(),
    )

    callback_context = (
        "\nInbound callback context:\n"
        f"- This is an inbound callback to the assistant line for original call {task.callback_of}.\n"
        "- Answer as the assistant for the same customer. Thank them for calling back, connect the "
        "conversation to the original request, and continue or finish that request.\n"
        "- The original authority, limits, privacy rules and no-payment/no-address rules still apply. "
        "Do not expand what you may commit to just because they called back.\n"
        f"- Prior call context and outcome:\n{task.callback_context}\n"
        if getattr(task, "callback_of", None)
        else ""
    )

    if isinstance(task, GeneralCall):
        return callback_context + _general_goal(task) + "\n" + rules
    if isinstance(task, RestaurantReservation):
        goal = f"""You are calling {task.business_name} to book a table.
Reservation request:
- Name: {task.customer_name}
- Party size: {task.party_size}
- Date: {task.date}
- Time: {task.time}
- Acceptable alternatives: {task.flexibility or "none given (only the exact time can be booked)"}
- Special requests: {task.special_requests or "none"}

Goal: get a confirmed reservation. If the exact time isn't available, accept an alternative only if it
fits the acceptable alternatives above. If what they offer doesn't fit, don't just say no: ask once
for something closer, e.g. "Anything closer to 7, like 6:30 or 7:30?". Book it only if it fits the
acceptable alternatives. Otherwise say {task.customer_name} will call back to pick a time and record
"unavailable" with every time they offered in `availability`, e.g. "7 PM full; 5 PM and 8:45 PM
open". Before recording, make sure the date, time, party size and
name are confirmed: if they already said them back, a quick "Perfect" is enough; otherwise read
them back once in one short sentence. If it is a wrong number or not the restaurant, use
"declined" or "needs_followup", not "unavailable". Use outcome "booked", "unavailable", "declined"
or "needs_followup"."""
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

    return callback_context + goal + "\n" + rules


def _inbound_message_prompt(task: InboundMessage) -> str:
    return f"""You are {self_intro(task)} answering an inbound call to an AI assistant phone line.

This caller's number ({task.phone_number}) did not match any recent business this assistant called.

Goal:
- Politely explain this is an AI assistant line.
- Do not claim to know why they are calling.
- Ask for and record their name, best callback number, and reason for calling.
- Never book, order, schedule, cancel, promise, transfer money, share private customer details, or
  commit to anything.
- Once you have a message, say a brief goodbye, call record_outcome with outcome "info_received",
  include the name/number/reason in the summary and answers, then call end_call after goodbye.

How to talk:
- Keep every turn to one short sentence and ask one thing at a time.
- If they ask whether you're a robot, an AI, automated or a real person, say honestly that you're
  an AI assistant in the first few words.
- If they refuse to leave a message or it is clearly a wrong number, record outcome "needs_followup"
  with a compact summary and end the call.

{FIXED_SAFETY_RULES.strip()}

Approved shareable details:
- none

Today is {_today()} (Pacific time).
"""


def _general_goal(task: GeneralCall) -> str:
    details = format_shareable_details(task.shareable_details)
    questions = (
        "\n".join(f"{i}. {q}" for i, q in enumerate(task.questions, 1))
        or "(none listed; get whatever information the goal needs)"
    )
    if task.authority in {"may_commit_within_limits", "may_book_within_limits"}:
        authority = f"""You MAY book, order, reserve, schedule, reschedule, cancel, or otherwise commit, but only if every part of it fits these limits:
{task.limits.render() if task.limits else "- No limits provided; do not commit."}
If what they offer falls outside the limits, don't just say no: ask once whether they have anything
closer (e.g. a nearby time or a similar option). If nothing fits, do not accept it; note every offer
and use outcome "unavailable" or "needs_followup". If the offer fits, commit under {task.customer_name} using only
the shareable details above. For food or drink pickup orders, ordering with pay-at-pickup is allowed;
never give a card number, and use needs_followup if a card or prepayment is required. Ask for a
confirmation or reference number before ending. For orders, also ask for the total and pickup/ready
time. Do not treat missing phone, email, address, or other unprovided details as a blocker unless the
business refuses to hold or confirm without them."""
    else:
        authority = """You may NOT book, reserve, order, schedule or agree to anything. You are only gathering
information. If they offer to book or hold something, politely say {customer} will call back to
confirm, and note what they offered. Use outcome "info_received" when you got the requested
information, even if availability is waitlist-only, full, or not available.""".format(customer=task.customer_name)

    return f"""You are calling {task.business_name} on behalf of {task.customer_name}.
Goal: {task.goal}

Details you may share if they are relevant or asked for:
{details}

Questions to get answered:
{questions}

Authority:
{authority}

Ask one question at a time. Do not ask a new question and say goodbye in the same turn. Before you
record the outcome, make sure every listed question is answered or the business clearly cannot answer
it. When you record the outcome, put each question and the answer you got (or "not answered") in
`answers`, put any price in `quote`, any date/time availability in `availability`, and any
confirmation number in `reference`. Keep outcome facts compact and concrete, e.g. "Thursday 3:45
PM", "$175 flat fee", "out of stock until Monday", or "photo of the damage needed".
Use outcome "info_received" when you got the information, "ordered" when you placed a pickup order,
"booked" when you made another authorized commitment, otherwise "unavailable", "declined" or
"needs_followup"."""
