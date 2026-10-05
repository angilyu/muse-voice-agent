"""Starts calls: a real outbound call via Retell or LiveKit, or a simulated one (DRY_RUN)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config import Settings
from .store import CallStore
from .tasks import AnyTask, GeneralCall, RestaurantReservation

logger = logging.getLogger("muse_voice_agent.dispatcher")

_background: set[asyncio.Task] = set()


class CallRejected(Exception):
    pass


def _check_allowed(task: AnyTask, settings: Settings, store: CallStore):
    if settings.allowed_dial_prefixes and not any(
        task.phone_number.startswith(p) for p in settings.allowed_dial_prefixes
    ):
        raise CallRejected(
            f"{task.phone_number} is outside ALLOWED_DIAL_PREFIXES {settings.allowed_dial_prefixes}"
        )
    if store.count_active() >= settings.max_concurrent_calls:
        raise CallRejected(
            f"Too many calls in progress (max {settings.max_concurrent_calls}). Try again shortly."
        )


async def start_call(
    task: AnyTask, settings: Settings, store: CallStore
) -> dict[str, Any]:
    _check_allowed(task, settings, store)

    if not settings.dry_run:
        missing = settings.missing_for_live_calls()
        if missing:
            raise CallRejected(
                f"Live calling is not configured; set {', '.join(missing)} in .env or use DRY_RUN=true"
            )

    record = store.create_call(task.model_dump(), dry_run=settings.dry_run)
    call_id = record["id"]

    if settings.dry_run:
        _spawn(_simulate_call(call_id, task, store))
        return store.get_call(call_id)  # type: ignore[return-value]

    if settings.voice_backend == "retell":
        await _start_retell_call(call_id, task, settings, store)
        return store.get_call(call_id)  # type: ignore[return-value]

    from livekit import api

    lkapi = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=settings.agent_name,
                room=f"call-{call_id}",
                metadata=json.dumps({"call_id": call_id}),
            )
        )
        store.update_call(call_id, status="dispatched")
    except Exception as e:  # noqa: BLE001
        logger.exception("agent dispatch failed")
        store.update_call(call_id, status="failed", error=f"dispatch failed: {e}")
    finally:
        await lkapi.aclose()
    return store.get_call(call_id)  # type: ignore[return-value]


def _spawn(coro) -> None:  # noqa: ANN001
    t = asyncio.create_task(coro)
    _background.add(t)
    t.add_done_callback(_background.discard)


async def _start_retell_call(
    call_id: str, task: AnyTask, settings: Settings, store: CallStore
) -> None:
    from . import retell

    try:
        async with retell.client_factory(settings) as client:
            resp = await client.create_phone_call(
                from_number=settings.retell_from_number,
                to_number=task.phone_number,
                agent_id=settings.retell_agent_id,
                metadata={retell.METADATA_KEY: call_id},
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("retell create-phone-call failed")
        store.update_call(call_id, status="failed", error=f"retell create-phone-call failed: {e}")
        return

    retell_call_id = resp["call_id"]
    fields: dict[str, Any] = {"provider_call_id": retell_call_id}
    record = store.get_call(call_id)
    if record and record["status"] == "queued":  # the websocket may already have marked it in_progress
        fields["status"] = "dialing"
    store.update_call(call_id, **fields)
    _spawn(retell.monitor_call(settings, store, call_id, retell_call_id))


async def _simulate_call(
    call_id: str, task: AnyTask, store: CallStore, delay: float = 2.0
) -> None:
    """Fake a call end-to-end so the Muse <-> MCP wiring can be tested without a phone line."""
    await asyncio.sleep(delay / 2)
    store.update_call(call_id, status="in_progress")
    store.append_transcript(call_id, "user", f"Hello, {task.business_name}, how can I help?")
    store.append_transcript(
        call_id,
        "assistant",
        f"Hi, this is an AI assistant calling on behalf of {task.customer_name}. [simulated]",
    )
    await asyncio.sleep(delay / 2)

    if isinstance(task, GeneralCall):
        details = {
            "outcome": "info_received",
            "summary": f"[SIMULATED] {task.business_name} answered: {task.goal}",
            "answers": [
                {"question": q, "answer": "simulated answer"} for q in task.questions
            ],
        }
    elif isinstance(task, RestaurantReservation):
        details = {
            "outcome": "booked",
            "summary": f"[SIMULATED] Table for {task.party_size} at {task.business_name} on "
            f"{task.date} at {task.time} under {task.customer_name}.",
            "confirmed_date": task.date,
            "confirmed_time": task.time,
            "party_size": task.party_size,
            "booked_under": task.customer_name,
        }
    else:
        details = {
            "outcome": "quote_received",
            "summary": f"[SIMULATED] {task.business_name} quoted $150-$250 for "
            f"'{task.job_description}', earliest availability next Tuesday morning.",
            "quote": "$150-$250 (simulated)",
            "availability": "next Tuesday morning (simulated)",
        }
    store.append_transcript(call_id, "assistant", "Thanks so much, goodbye! [simulated]")
    store.update_call(
        call_id,
        status="completed",
        outcome=details["outcome"],
        summary=details["summary"],
        details=details,
    )
