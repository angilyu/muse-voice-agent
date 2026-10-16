"""Retell AI backend.

Retell places the call (over our Twilio SIP trunk) and handles speech-to-text, text-to-speech and
turn-taking. For every turn it asks our server what to say over the "custom LLM" websocket
(https://docs.retellai.com/api-references/llm-websocket), and we answer by running the same
LangGraph graph the LiveKit worker uses.

    Muse -> MCP tool -> dispatcher -> POST /v2/create-phone-call
    Retell <-> wss://<PUBLIC_BASE_URL>/retell/llm/<secret>/<retell_call_id>  (LangGraph per turn)
    monitor_call() polls GET /v2/get-call until the call ends and finalises the record.

The websocket protocol, REST client and filler handling come from ``voice_kit.retell``;
:class:`MuseRetellHandler` adds the call-specific parts (binding to our call record, silent-pickup
and inbound-answer watchers, quick replies after goodbye or on hold).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from starlette.types import ASGIApp
from voice_kit.retell import (
    RetellCallHandler,
    RetellChunk,
    RetellClient,
    RetellError,
    RetellTurn,
    RetellWebsocketRouter,
)
from voice_kit.retell import sync_agent_websocket_url as _sync_agent_websocket_url

from .config import Settings
from .graph import CallControl, CallOutcome, build_call_graph
from .outcome_fallback import has_business_speech, infer_outcome
from .pickup import classify_line, is_note
from .report import build_report
from .store import FINAL_STATUSES, CallStore
from .tasks import InboundMessage, parse_task

__all__ = [
    "METADATA_KEY",
    "MuseRetellHandler",
    "RetellClient",
    "RetellError",
    "build_websocket_router",
    "client_factory",
    "finalize_from_retell",
    "monitor_call",
    "recover_outcome",
    "sync_agent_websocket_url",
]

logger = logging.getLogger("muse_voice_agent.retell")

METADATA_KEY = "muse_call_id"

REMINDER_NUDGE = (
    "[The other person has been silent for a while. If you were waiting on them, briefly check "
    "whether they are still there; otherwise continue.]"
)
SILENT_PICKUP_NOTE = (
    "[The call connected but nobody has spoken for a few seconds. It may be a call-screening "
    "service or a person who didn't hear anything yet. Say why you're calling in one sentence.]"
)
# An incoming call: we answer, so speak first instead of waiting for the caller.
INBOUND_ANSWER_NOTE = (
    "[You just answered this incoming call and the caller hasn't said anything yet. Your opener "
    "was just spoken; give your first reply now.]"
)
# Once we've said goodbye, a few seconds of silence means they're done: hang up.
CLOSING_REMINDER_MS = 4000
_background: set[asyncio.Task] = set()


# Tests swap this out to inject a mock transport.
client_factory: Callable[[Settings], RetellClient] = lambda s: RetellClient(s.retell_api_key)  # noqa: E731


async def sync_agent_websocket_url(settings: Settings) -> str:
    """Point the Retell agent at our current public URL (quick tunnels change on restart)."""
    async with client_factory(settings) as client:
        return await _sync_agent_websocket_url(
            client, settings.retell_agent_id, settings.retell_llm_websocket_url()
        )


# --------------------------------------------------------------------------------------------
# Custom-LLM websocket
# --------------------------------------------------------------------------------------------


def _store_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "assistant" if u.get("role") == "agent" else "user", "text": u.get("content", "")}
        for u in transcript or []
        if (u.get("content") or "").strip()
    ]


def _spawn(coro) -> None:  # noqa: ANN001
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def _timing_fields(call: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, column in (("start_timestamp", "started_at"), ("end_timestamp", "ended_at")):
        if isinstance(call.get(key), (int, float)):
            fields[column] = call[key] / 1000
    if call.get("disconnection_reason"):
        fields["end_reason"] = call["disconnection_reason"]
    return fields


def _inbound_caller_number(call: dict[str, Any], settings: Settings) -> str | None:
    candidates = [
        call.get("from_number"),
        call.get("caller_number"),
        call.get("from"),
        (call.get("phone_call") or {}).get("from_number")
        if isinstance(call.get("phone_call"), dict)
        else None,
    ]
    own_numbers = {n for n in (settings.retell_from_number, call.get("to_number")) if n}
    for number in candidates:
        if isinstance(number, str) and number and number not in own_numbers:
            return number
    return next((n for n in candidates if isinstance(n, str) and n), None)


def _callback_context(parent: dict[str, Any]) -> str:
    task = parent.get("task") or {}
    context: dict[str, Any] = {
        "original_call_id": parent["id"],
        "original_status": parent.get("status"),
        "original_outcome": parent.get("outcome"),
        "original_summary": parent.get("summary"),
        "original_task": {
            k: v
            for k, v in task.items()
            if k not in {"callback_context"} and v not in (None, [], {})
        },
    }
    if parent.get("status") in FINAL_STATUSES:
        context["original_report"] = build_report(parent)
    elif parent.get("details"):
        context["original_details"] = parent["details"]
    transcript = parent.get("transcript") or []
    if transcript:
        context["recent_transcript"] = transcript[-8:]
    return json.dumps(context, ensure_ascii=False, default=str)


class MuseRetellHandler(RetellCallHandler):
    """One Retell phone call: binds it to our call record and answers each turn with the graph."""

    def __init__(
        self,
        retell_call_id: str,
        store: CallStore,
        model: BaseChatModel | None = None,
        settings: Settings | None = None,
        silent_pickup_seconds: float | None = None,
    ) -> None:
        self.retell_call_id = retell_call_id
        self.store = store
        self.model = model
        self.settings = settings
        # Speak first if nobody says anything this long after pickup (None disables).
        self.silent_pickup_seconds = silent_pickup_seconds
        self.control = CallControl()
        self._closing_configured = False
        self._watcher: asyncio.Task | None = None
        self.call_id: str | None = None
        self.graph: Any = None
        self.outcome: CallOutcome | None = None

    async def on_call_details(self, call: dict[str, Any]) -> bool:
        call_id = (call.get("metadata") or {}).get(METADATA_KEY)
        record = self.store.get_call(call_id) if call_id else None
        if record is None and not call_id:
            record = await self._create_inbound_call(call)
            call_id = record["id"] if record else None
        if record is None or record["status"] in FINAL_STATUSES:
            logger.warning(
                "rejecting retell call %s: unknown/finished call %r", self.retell_call_id, call_id
            )
            return False
        self.call_id = call_id
        self.graph = build_call_graph(
            parse_task(record["task"]), self._on_outcome, self.model, control=self.control
        )
        timing = _timing_fields(call)
        self.store.update_call(
            call_id, status="in_progress", provider_call_id=self.retell_call_id, **timing
        )
        logger.info("retell call %s bound to %s", self.retell_call_id, call_id)
        if record.get("direction") == "inbound" and self.settings is not None:
            _spawn(monitor_call(self.settings, self.store, call_id, self.retell_call_id))
            if self.settings.inbound_greet_ms > 0:
                self._watcher = asyncio.create_task(
                    self._answer_inbound(self.settings.inbound_greet_ms / 1000)
                )
        elif self.settings is not None and self.silent_pickup_seconds is not None:
            self._watcher = asyncio.create_task(self._watch_silent_pickup())
        return True

    async def _create_inbound_call(self, call: dict[str, Any]) -> dict[str, Any] | None:
        if self.settings is None:
            return None
        caller = _inbound_caller_number(call, self.settings)
        if not caller:
            logger.warning("inbound retell call %s had no caller number", self.retell_call_id)
            return None
        parent = self.store.find_callback_parent(
            caller,
            lookback_seconds=max(1, self.settings.inbound_callback_lookback_days) * 24 * 60 * 60,
        )
        if parent:
            task = dict(parent["task"])
            task["callback_of"] = parent["id"]
            task["callback_context"] = _callback_context(parent)
            business = parent["business_name"]
            logger.info(
                "inbound retell call %s from %s matched outbound call %s",
                self.retell_call_id,
                caller,
                parent["id"],
            )
        else:
            task = InboundMessage(phone_number=caller).model_dump()
            business = task["business_name"]
            logger.info("inbound retell call %s from unknown caller %s", self.retell_call_id, caller)
        record = self.store.create_call(
            task,
            dry_run=False,
            direction="inbound",
            parent_call_id=parent["id"] if parent else None,
            status="in_progress",
            provider_call_id=self.retell_call_id,
            origin=parent.get("origin") if parent else None,
        )
        if business != record["business_name"]:
            self.store.update_call(record["id"], business_name=business)
            record = self.store.get_call(record["id"]) or record
        return record

    async def on_transcript(self, transcript: list[dict[str, Any]]) -> None:
        if self.call_id:
            self.store.set_transcript(self.call_id, _store_transcript(transcript))

    def _anyone_spoke(self) -> bool:
        transcript = self.session.last_transcript or []
        return (
            self.session.heard_response_request
            or self.control.opener_spoken
            or any((u.get("content") or "").strip() for u in transcript)
        )

    async def _answer_inbound(self, delay: float) -> None:
        """Answer an incoming call like a person picking up: speak first unless the caller did.

        Retell connects us once the inbound call is answered, so there's no ringing to wait out.
        """
        await asyncio.sleep(delay)
        if self._anyone_spoke() or self.session.turn_active:
            return
        self.session.interrupt([HumanMessage(content=INBOUND_ANSWER_NOTE)])

    async def _watch_silent_pickup(self) -> None:
        """Speak first when the line stays silent after pickup.

        Call screeners (and some people) wait for the caller to talk, and Retell's speech-to-text
        can miss a quiet "Hi?", which used to leave both sides silent until the inactivity timeout.
        """
        assert self.settings is not None and self.silent_pickup_seconds is not None
        try:
            async with client_factory(self.settings) as client:
                deadline = time.monotonic() + 120
                while True:
                    if self._anyone_spoke() or time.monotonic() > deadline:
                        return
                    call = await client.get_call(self.retell_call_id)
                    status = call.get("call_status")
                    if status == "ongoing":
                        break
                    if status in ("ended", "error", "not_connected"):
                        return
                    await asyncio.sleep(0.5)
            started = call.get("start_timestamp")
            wait = self.silent_pickup_seconds
            if isinstance(started, (int, float)):
                wait = max(0.0, started / 1000 + wait - time.time())
            await asyncio.sleep(wait)
            if self._anyone_spoke() or self.session.turn_active:
                return
            logger.info("silent pickup on retell call %s; speaking first", self.retell_call_id)
            self.session.interrupt([HumanMessage(content=SILENT_PICKUP_NOTE)])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("silent-pickup watcher failed (retell call %s): %s", self.retell_call_id, e)

    def _on_outcome(self, outcome: CallOutcome) -> None:
        self.outcome = outcome
        logger.info("call outcome: %s", outcome.model_dump_json())
        if self.call_id:
            self.store.update_call(
                self.call_id,
                status="completed",
                outcome=outcome.outcome,
                summary=outcome.summary,
                details=outcome.model_dump(exclude_none=True),
                outcome_source="agent",
            )

    def filler_wait(self, turn: RetellTurn) -> float | None:
        """Seconds to wait before a filler, or None when this turn shouldn't get one."""
        if turn.kind != "response_required":
            return None  # reminders and interrupts answer a note, not a person
        if not self.session.filler_seconds or not self.control.opener_spoken or self.control.closing:
            return None
        last = turn.messages[-1] if turn.messages else None
        if not isinstance(last, HumanMessage) or not isinstance(last.content, str):
            return None
        if is_note(last.content) or classify_line(last.content) != "person":
            return None  # screeners, menus and voicemail don't need to hear us think
        return self.session.filler_seconds

    async def respond(self, turn: RetellTurn) -> AsyncIterator[str | RetellChunk]:
        messages = list(turn.messages)
        if turn.kind == "reminder_required":
            if self.control.closing or self.control.on_hold:
                # Silence after our goodbye means they're done; silence on hold is expected.
                turn.end_call = self.control.closing
                return
            messages.append(HumanMessage(content=REMINDER_NUDGE))
        # The graph clears this as each turn starts; do it now so a hold from the previous turn
        # can't veto this turn's filler before the graph gets going.
        self.control.on_hold = False
        async for chunk in self._graph_chunks(messages, self.filler_wait(turn)):
            if not chunk:
                yield ""  # counts as the first chunk, so no filler goes out
                continue
            if self.control.opener_text and chunk.strip() == self.control.opener_text:
                # The fixed opener (first turn, or again for someone who just picked up);
                # don't let a "Hello?" cut it off.
                yield RetellChunk(chunk, no_interruption_allowed=True)
            else:
                yield chunk
        turn.end_call = self.control.end_requested
        if self.control.pending_digits:
            turn.digit_to_press = self.control.pending_digits
            self.control.pending_digits = None

    async def _graph_chunks(
        self, messages: list[BaseMessage], filler_wait: float | None
    ) -> AsyncIterator[str]:
        """Graph speech chunks; yields one "" if the graph puts us on hold before the filler fires.

        A "Hmm," right after "please hold" would be odd, and the graph only sets ``on_hold`` once
        the turn is running, after ``filler_wait`` has already been decided.
        """
        stream = self.graph.astream({"messages": messages}, stream_mode="custom").__aiter__()
        pending = asyncio.ensure_future(stream.__anext__())
        try:
            if filler_wait is not None:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + filler_wait
                while not pending.done() and loop.time() < deadline:
                    await asyncio.wait({pending}, timeout=min(0.02, deadline - loop.time()))
                    if self.control.on_hold and not pending.done():
                        yield ""
                        break
            while True:
                try:
                    chunk = await pending
                except StopAsyncIteration:
                    return
                pending = asyncio.ensure_future(stream.__anext__())
                if chunk:
                    yield chunk
        finally:
            if not pending.done():
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def after_turn(self, turn: RetellTurn) -> None:
        if self.control.closing and not self._closing_configured:
            self._closing_configured = True
            await self.session.update_agent(
                {"reminder_trigger_ms": CLOSING_REMINDER_MS, "reminder_max_count": 1}
            )

    async def on_close(self) -> None:
        if self._watcher and not self._watcher.done():
            self._watcher.cancel()


def build_websocket_router(
    app: ASGIApp,
    settings: Settings,
    store: CallStore,
    model_factory: Callable[[], BaseChatModel | None] = lambda: None,
) -> RetellWebsocketRouter:
    """Serve Retell's custom-LLM websocket in front of ``app`` (everything else passes through).

    Retell can't send our bearer token, so the websocket is protected by a secret path segment
    plus the requirement that the call's metadata names an active call in our store.
    """
    return RetellWebsocketRouter(
        app,
        secret=settings.retell_ws_secret,
        handler_factory=lambda retell_call_id: MuseRetellHandler(
            retell_call_id,
            store,
            model_factory(),
            settings=settings,
            silent_pickup_seconds=(settings.silent_pickup_ms / 1000) or None,
        ),
        session_kwargs=lambda: {"filler_seconds": (settings.filler_after_ms / 1000) or None},
    )


# --------------------------------------------------------------------------------------------
# Call lifecycle
# --------------------------------------------------------------------------------------------

NO_ANSWER_REASONS = {"dial_no_answer", "dial_busy", "user_declined", "registered_call_timeout"}
INCOMPLETE_REASONS = {
    "user_hangup",
    "agent_hangup",
    "inactivity",
    "max_duration_reached",
    "call_transfer",
    "manual_stopped",
}


def finalize_from_retell(store: CallStore, call_id: str, call: dict[str, Any]) -> None:
    """Apply Retell's final call object to our record (status, transcript) if not already final."""
    if call.get("transcript_object"):
        store.set_transcript(call_id, _store_transcript(call["transcript_object"]))
    timing: dict[str, Any] = {}
    for key, column in (("start_timestamp", "started_at"), ("end_timestamp", "ended_at")):
        if isinstance(call.get(key), (int, float)):
            timing[column] = call[key] / 1000
    if call.get("disconnection_reason"):
        timing["end_reason"] = call["disconnection_reason"]
    store.update_call(call_id, **timing)
    record = store.get_call(call_id)
    if record is None or (record["status"] in FINAL_STATUSES and record["outcome"]):
        return

    reason = call.get("disconnection_reason") or call.get("call_status") or "unknown"
    if reason in NO_ANSWER_REASONS:
        store.update_call(call_id, status="no_answer", error=reason)
    elif reason == "voicemail_reached":
        store.update_call(
            call_id,
            status="completed",
            outcome="voicemail",
            summary="Reached voicemail; no message was left.",
            outcome_source="call_system",
        )
    elif reason == "ivr_reached":
        store.update_call(
            call_id,
            status="completed",
            outcome="needs_followup",
            summary="Reached an automated phone menu and could not get through to a person.",
            outcome_source="call_system",
        )
    elif reason in INCOMPLETE_REASONS:
        store.update_call(
            call_id,
            status="completed",
            outcome="needs_followup",
            summary=f"Call ended without a recorded result ({reason}). See transcript.",
            outcome_source="call_system",
        )
    else:
        store.update_call(call_id, status="failed", error=f"retell: {reason}")


OUTCOME_RECOVERY_TIMEOUT_S = 30.0


async def recover_outcome(
    store: CallStore,
    call_id: str,
    call: dict[str, Any],
    model: BaseChatModel | None = None,
    timeout_s: float = OUTCOME_RECOVERY_TIMEOUT_S,
) -> bool:
    """If a connected call ended before record_outcome, infer the result from the transcript.

    Returns True when an inferred outcome was stored. Any failure leaves the record untouched so
    finalize_from_retell falls back to "needs_followup".
    """
    reason = call.get("disconnection_reason") or ""
    record = store.get_call(call_id)
    if record is None or record["outcome"] or reason not in INCOMPLETE_REASONS:
        return False
    turns = _store_transcript(call.get("transcript_object") or [])
    if not has_business_speech(turns):
        return False
    try:
        task = parse_task(record["task"])
        outcome = await asyncio.wait_for(infer_outcome(task, turns, model=model), timeout_s)
    except Exception as e:  # noqa: BLE001
        logger.warning("outcome recovery failed for %s: %s", call_id, e)
        return False
    if outcome is None:
        return False
    current = store.get_call(call_id)
    if current is None or current["outcome"]:
        return False  # the live session recorded the real result while we were inferring
    logger.info("inferred outcome for %s: %s", call_id, outcome.model_dump_json())
    store.update_call(
        call_id,
        status="completed",
        outcome=outcome.outcome,
        summary=outcome.summary,
        details=outcome.model_dump(exclude_none=True),
        outcome_source="transcript",
    )
    return True


async def monitor_call(
    settings: Settings,
    store: CallStore,
    call_id: str,
    retell_call_id: str,
    poll_seconds: float = 3.0,
) -> None:
    deadline = time.monotonic() + settings.max_call_seconds + 180
    async with client_factory(settings) as client:
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_seconds)
            try:
                call = await client.get_call(retell_call_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("get-call %s failed: %s", retell_call_id, e)
                continue
            if call.get("call_status") in ("ended", "error", "not_connected"):
                await recover_outcome(store, call_id, call)
                finalize_from_retell(store, call_id, call)
                return
    record = store.get_call(call_id)
    if record and record["status"] not in FINAL_STATUSES:
        store.update_call(call_id, status="failed", error="lost track of the Retell call")
