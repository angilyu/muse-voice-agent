"""Recover the call result from the transcript when a call ended before record_outcome.

The agent records the outcome together with its goodbye. If the business hangs up first (e.g. right
after "See you in 15!" while the agent is still asking a last question), the call would otherwise
reach Muse as "ended without a recorded result". After the call, read the transcript once and fill
in the result from what the business actually said.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .graph import CallOutcome, apply_authority
from .tasks import AnyTask

logger = logging.getLogger(__name__)

INFERRED_NOTE = "Result read from the transcript; the call ended before the agent recorded it."

_SYSTEM = """\
An AI assistant phoned a business for a customer. The call ended before the assistant recorded
the result. Read the brief and the transcript and return the result as ONE JSON object.

Use only what the business actually said. Never invent a time, price, name or confirmation.
- "booked" / "ordered": only if the business clearly confirmed the booking or order.
- "quote_received" / "info_received": the questions in the brief were answered.
- "unavailable" / "declined": they can't or won't do it.
- "needs_followup": anything required is missing, unclear, or needs the customer (fees, deposits,
  card, callback, a missing date). Use this when unsure.
- "voicemail": only a voicemail greeting answered.

JSON keys (omit unknown ones): outcome, summary (1-2 sentences for the customer), confirmed_date,
confirmed_time, party_size, booked_under, order_total, pickup_time, quote, availability,
contact_person, follow_up, answers (list of {"question", "answer"} for each question in the
brief; answer "not answered" if they didn't), reference. Return JSON only."""


def has_business_speech(turns: list[dict[str, Any]]) -> bool:
    return any(t.get("role") == "user" and str(t.get("text") or "").strip() for t in turns)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group())


def _default_model() -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    from .config import get_settings, llm_model_init_args

    model_name, model_kwargs = llm_model_init_args(get_settings().llm_model, temperature=0)
    return init_chat_model(model_name, **model_kwargs)


async def infer_outcome(
    task: AnyTask, turns: list[dict[str, Any]], model: BaseChatModel | None = None
) -> CallOutcome | None:
    """Infer a CallOutcome from `turns` ({"role": "assistant"|"user", "text"}), or None.

    Returns None when the business never spoke (nothing to infer) or the model output is unusable.
    """
    if not has_business_speech(turns):
        return None
    lines = "\n".join(
        f"{'Assistant' if t.get('role') == 'assistant' else 'Business'}: {t.get('text', '')}"
        for t in turns
        if str(t.get("text") or "").strip()
    )
    brief = task.model_dump_json(exclude_none=True)
    model = model or _default_model()
    reply = await model.ainvoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=f"Brief:\n{brief}\n\nTranscript:\n{lines}")]
    )
    content = reply.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    try:
        outcome = CallOutcome.model_validate(_extract_json(str(content)))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not infer outcome from transcript: %s", e)
        return None
    outcome = apply_authority(task, outcome)
    follow_up = f"{outcome.follow_up} ({INFERRED_NOTE})" if outcome.follow_up else INFERRED_NOTE
    return outcome.model_copy(update={"follow_up": follow_up})
