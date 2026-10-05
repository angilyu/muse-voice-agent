"""Call task definitions and the system prompts the phone agent uses for each."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator

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


class _BaseTask(BaseModel):
    business_name: str = Field(min_length=1, max_length=120)
    phone_number: str = Field(description="Business phone number, E.164 preferred")
    customer_name: str = Field(min_length=1, max_length=80, description="Name to book under")
    callback_number: str | None = Field(
        default=None, description="Number the business can call back. Shared only if asked."
    )

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


CallTask = Annotated[Union[RestaurantReservation, HandymanQuote], Field(discriminator="kind")]
CALL_TASK_ADAPTER: TypeAdapter[CallTask] = TypeAdapter(CallTask)


def parse_task(data: dict) -> RestaurantReservation | HandymanQuote:
    return CALL_TASK_ADAPTER.validate_python(data)


_COMMON_RULES = """
How to behave on the phone:
- You are an AI assistant placing a call on behalf of {customer_name}. Say so in your first sentence,
  e.g. "Hi, this is an AI assistant calling on behalf of {customer_name}."
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


def build_system_prompt(task: RestaurantReservation | HandymanQuote) -> str:
    callback_clause = (
        f" and the callback number {task.callback_number} if they ask for one"
        if task.callback_number
        else ""
    )
    rules = _COMMON_RULES.format(customer_name=task.customer_name, callback_clause=callback_clause)

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
