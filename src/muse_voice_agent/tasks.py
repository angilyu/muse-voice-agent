"""Call task definitions and the system prompts the phone agent uses for each."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Union

from zoneinfo import ZoneInfo

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


_PLACEHOLDER_ASSISTANT_NAMES = {
    "assistant", "an assistant", "the assistant", "ai", "ai assistant", "an ai assistant", "bot",
    "agent", "none", "null", "n/a", "na", "unknown", "name", "tbd",
}
_ASSISTANT_NAME = re.compile(r"[^\W\d_](?:[^\W\d_]|[' .-]){0,39}")  # letters, spaces, ' . -


class _BaseTask(BaseModel):
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
    follow_up_of: str | None = Field(
        default=None,
        max_length=64,
        description="Call id this call follows up on, if any.",
    )
    previous_call_context: str | None = Field(
        default=None,
        max_length=3000,
        description="Compact context from earlier calls in the same follow-up thread.",
    )

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
    """Any phone errand Muse can describe as a brief: orders, appointments, questions, bookings."""

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
        description="info_only: ask and commit to nothing. may_commit_within_limits: may book, "
        "order, schedule, reschedule, cancel, or reserve only within `limits`. "
        "may_book_within_limits is a backward-compatible alias.",
    )
    limits: str | None = Field(
        default=None,
        max_length=500,
        description="Required with may_commit_within_limits/may_book_within_limits, e.g. "
        "'two jasmine milk teas, 25% sugar, less ice, defaults otherwise, pay at pickup'",
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
        if self.authority in {"may_commit_within_limits", "may_book_within_limits"} and not (
            self.limits or ""
        ).strip():
            raise ValueError(
                "limits are required when authority is may_commit_within_limits "
                "or may_book_within_limits"
            )
        return self


AnyTask = RestaurantReservation | HandymanQuote | GeneralCall
CallTask = Annotated[
    Union[RestaurantReservation, HandymanQuote, GeneralCall], Field(discriminator="kind")
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
- Never invent details about {customer_name}. If asked something you don't know, say
  {customer_name} will follow up, and note it in the outcome.
- Never give out payment card numbers, addresses, or other personal data. You may share the
  customer's name{callback_clause}.
- Never agree to deposits, cancellation fees, or prepayment. If one is required, get the details and
  record the outcome as needs_followup instead of confirming.
- For pickup orders, pay-at-pickup is okay when it fits the brief. If the business requires a card
  or prepayment over the phone, do not place the order; record needs_followup.

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
    return f"Hi, this is {self_intro(task, ai=False)} calling on behalf of {task.customer_name}."


def build_system_prompt(task: AnyTask) -> str:
    callback_clause = (
        f" and the callback number {task.callback_number} if they ask for one"
        if task.callback_number
        else ""
    )
    rules = _COMMON_RULES.format(
        customer_name=task.customer_name,
        callback_clause=callback_clause,
        opening_line=opening_line(task),
        self_intro=self_intro(task),
        today=_today(),
    )

    follow_up = _follow_up_prompt(task)

    if isinstance(task, GeneralCall):
        return follow_up + _general_goal(task) + "\n" + rules
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

    return follow_up + goal + "\n" + rules


def _follow_up_prompt(task: AnyTask) -> str:
    if not task.previous_call_context:
        return ""
    return f"""{task.previous_call_context}

Follow-up calling instructions:
- Treat this as a continuation of the earlier call thread, not a cold first call.
- After the fixed opener, briefly say why you are calling back, e.g. "I called earlier about the fence quote and was told to call back after 3."
- Do not re-ask questions already answered above unless you need to confirm, change, or use that answer for the new request.
- If a person was named, ask for them or mention that you spoke with them earlier.

"""


def _general_goal(task: GeneralCall) -> str:
    details = (
        "\n".join(f"- {k}: {v}" for k, v in task.shareable_details.items())
        or "- none beyond the customer's name"
    )
    questions = (
        "\n".join(f"{i}. {q}" for i, q in enumerate(task.questions, 1))
        or "(none listed; get whatever information the goal needs)"
    )
    if task.authority in {"may_commit_within_limits", "may_book_within_limits"}:
        authority = f"""You MAY book, order, reserve, schedule, reschedule, cancel, or otherwise commit, but only if every part of it fits these limits:
{task.limits}
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
