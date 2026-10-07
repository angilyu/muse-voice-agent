"""Retell AI backend.

Retell places the call (over our Twilio SIP trunk) and handles speech-to-text, text-to-speech and
turn-taking. For every turn it asks our server what to say over the "custom LLM" websocket
(https://docs.retellai.com/api-references/llm-websocket), and we answer by running the same
LangGraph graph the LiveKit worker uses.

    Muse -> MCP tool -> dispatcher -> POST /v2/create-phone-call
    Retell <-> wss://<PUBLIC_BASE_URL>/retell/llm/<secret>/<retell_call_id>  (LangGraph per turn)
    monitor_call() polls GET /v2/get-call until the call ends and finalises the record.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import re
import time
from typing import Any, Callable

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import Settings
from .graph import CallControl, CallOutcome, build_call_graph
from .outcome_fallback import has_business_speech, infer_outcome
from .pickup import classify_line, is_note
from .store import FINAL_STATUSES, CallStore
from .tasks import parse_task

logger = logging.getLogger("muse_voice_agent.retell")

API_BASE = "https://api.retellai.com"
WS_PATH_PREFIX = "/retell/llm/"
METADATA_KEY = "muse_call_id"

REMINDER_NUDGE = (
    "[The other person has been silent for a while. If you were waiting on them, briefly check "
    "whether they are still there; otherwise continue.]"
)
SILENT_PICKUP_NOTE = (
    "[The call connected but nobody has spoken for a few seconds. It may be a call-screening "
    "service or a person who didn't hear anything yet. Say why you're calling in one sentence.]"
)
# Once we've said goodbye, a few seconds of silence means they're done: hang up.
CLOSING_REMINDER_MS = 4000
# Said when a reply to a person is slow to start, so the line doesn't go dead while the model thinks.
FILLER = "Hmm, "
_LEADING_FILLER = re.compile(r"^\s*(?:hmm+|um+|uh+)\b[,.!]?\s*", re.I)


class RetellError(Exception):
    pass


class RetellClient:
    """Minimal async client for the Retell REST endpoints we use."""

    def __init__(
        self,
        api_key: str,
        base_url: str = API_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
            transport=transport,
        )

    async def __aenter__(self) -> RetellClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _req(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        resp = await self._http.request(method, path, json=body, params=params)
        if resp.status_code >= 400:
            raise RetellError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json() if resp.content else None

    async def create_phone_call(
        self,
        *,
        from_number: str,
        to_number: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"from_number": from_number, "to_number": to_number}
        if agent_id:
            body["override_agent_id"] = agent_id
        if metadata:
            body["metadata"] = metadata
        return await self._req("POST", "/v2/create-phone-call", body)

    async def get_call(self, retell_call_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/v2/get-call/{retell_call_id}")

    async def list_calls(
        self, *, limit: int = 20, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if agent_id:
            params["agent_id"] = agent_id
        try:
            data = await self._req("GET", "/v2/list-calls", params=params)
        except RetellError:
            body: dict[str, Any] = {"limit": limit}
            if agent_id:
                body["filter_criteria"] = {"agent_id": [agent_id]}
            data = await self._req("POST", "/v2/list-calls", body)
        if isinstance(data, dict):
            for key in ("calls", "data", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
        return data if isinstance(data, list) else []

    async def list_agents(self) -> list[dict[str, Any]]:
        return await self._req("GET", "/list-agents")

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/get-agent/{agent_id}")

    async def create_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/create-agent", body)

    async def update_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("PATCH", f"/update-agent/{agent_id}", body)

    async def list_phone_numbers(self) -> list[dict[str, Any]]:
        return await self._req("GET", "/list-phone-numbers")

    async def import_phone_number(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/import-phone-number", body)

    async def update_phone_number(self, number: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("PATCH", f"/update-phone-number/{number}", body)


# Tests swap this out to inject a mock transport.
client_factory: Callable[[Settings], RetellClient] = lambda s: RetellClient(s.retell_api_key)  # noqa: E731


async def sync_agent_websocket_url(settings: Settings) -> str:
    """Point the Retell agent at our current public URL (quick tunnels change on restart)."""
    url = settings.retell_llm_websocket_url()
    async with client_factory(settings) as client:
        agent = await client.get_agent(settings.retell_agent_id)
        engine = agent.get("response_engine") or {}
        if engine.get("type") == "custom-llm" and engine.get("llm_websocket_url") == url:
            return url
        await client.update_agent(
            settings.retell_agent_id,
            {"response_engine": {"type": "custom-llm", "llm_websocket_url": url}},
        )
    return url


# --------------------------------------------------------------------------------------------
# Custom-LLM websocket
# --------------------------------------------------------------------------------------------


def transcript_to_messages(transcript: list[dict[str, Any]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for utt in transcript or []:
        content = (utt.get("content") or "").strip()
        if not content:
            continue
        if utt.get("role") == "agent":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _store_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "assistant" if u.get("role") == "agent" else "user", "text": u.get("content", "")}
        for u in transcript or []
        if (u.get("content") or "").strip()
    ]


class RetellLLMSession:
    """Handles one Retell custom-LLM websocket connection (one phone call)."""

    def __init__(
        self,
        ws: WebSocket,
        retell_call_id: str,
        store: CallStore,
        model: BaseChatModel | None = None,
        bind_timeout: float = 10.0,
        settings: Settings | None = None,
        silent_pickup_seconds: float | None = None,
        filler_seconds: float | None = None,
    ) -> None:
        self.ws = ws
        self.settings = settings
        # Speak first if nobody says anything this long after pickup (None disables).
        self.silent_pickup_seconds = silent_pickup_seconds
        # Say FILLER if a reply to a person has no words after this long (None disables).
        self.filler_seconds = filler_seconds
        self.control = CallControl()
        self._closing_configured = False
        self._heard_response_request = False
        self._interrupt_id = 0
        self._watcher: asyncio.Task | None = None
        self.retell_call_id = retell_call_id
        self.store = store
        self.model = model
        self.bind_timeout = bind_timeout
        self.call_id: str | None = None
        self.graph: Any = None
        self.outcome: CallOutcome | None = None
        self._bound = asyncio.Event()
        self._turn: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._last_transcript: list[dict[str, Any]] | None = None
        self._closed = False

    async def _close(self, code: int) -> None:
        if not self._closed:
            self._closed = True
            await self.ws.close(code=code)

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def run(self) -> None:
        await self.ws.accept()
        await self._send(
            {"response_type": "config", "config": {"auto_reconnect": True, "call_details": True}}
        )
        # Empty begin message: the business answers first ("Hello, Luigi's").
        await self._send(
            {"response_type": "response", "response_id": 0, "content": "", "content_complete": True}
        )
        try:
            while not self._closed:
                msg = json.loads(await self.ws.receive_text())
                await self._handle(msg)
        except (WebSocketDisconnect, RuntimeError):
            logger.info("retell websocket closed (retell call %s)", self.retell_call_id)
        finally:
            for task in (self._turn, self._watcher):
                if task and not task.done():
                    task.cancel()

    async def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("interaction_type")
        if kind == "ping_pong":
            await self._send({"response_type": "ping_pong", "timestamp": msg.get("timestamp")})
        elif kind == "call_details":
            await self._bind(msg.get("call") or {})
        elif kind == "update_only":
            self._save_transcript(msg.get("transcript"))
        elif kind in ("response_required", "reminder_required"):
            self._save_transcript(msg.get("transcript"))
            if kind == "response_required":
                self._heard_response_request = True
            elif self.control.closing or self.control.on_hold:
                # Silence after our goodbye means they're done; silence on hold is expected.
                await self._send(
                    {
                        "response_type": "response",
                        "response_id": msg["response_id"],
                        "content": "",
                        "content_complete": True,
                        "end_call": self.control.closing,
                    }
                )
                return
            if self._turn and not self._turn.done():
                self._turn.cancel()  # Retell discards responses to older response_ids anyway
            self._turn = asyncio.create_task(self._respond(msg))

    async def _bind(self, call: dict[str, Any]) -> None:
        call_id = (call.get("metadata") or {}).get(METADATA_KEY)
        record = self.store.get_call(call_id) if call_id else None
        if record is None or record["status"] in FINAL_STATUSES:
            logger.warning(
                "rejecting retell call %s: unknown/finished call %r", self.retell_call_id, call_id
            )
            await self._close(1008)
            return
        self.call_id = call_id
        self.graph = build_call_graph(
            parse_task(record["task"]), self._on_outcome, self.model, control=self.control
        )
        self.store.update_call(call_id, status="in_progress", provider_call_id=self.retell_call_id)
        self._bound.set()
        logger.info("retell call %s bound to %s", self.retell_call_id, call_id)
        if self.settings is not None and self.silent_pickup_seconds is not None:
            self._watcher = asyncio.create_task(self._watch_silent_pickup())

    def _anyone_spoke(self) -> bool:
        transcript = self._last_transcript or []
        return (
            self._heard_response_request
            or self.control.opener_spoken
            or any((u.get("content") or "").strip() for u in transcript)
        )

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
            if self._anyone_spoke() or (self._turn and not self._turn.done()):
                return
            logger.info("silent pickup on retell call %s; speaking first", self.retell_call_id)
            self._interrupt_id += 1
            self._turn = asyncio.create_task(self._run_turn(
                [HumanMessage(content=SILENT_PICKUP_NOTE)], interrupt_id=self._interrupt_id
            ))
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
            )

    def _save_transcript(self, transcript: list[dict[str, Any]] | None) -> None:
        if not self.call_id or transcript is None or transcript == self._last_transcript:
            return
        self._last_transcript = transcript
        self.store.set_transcript(self.call_id, _store_transcript(transcript))

    async def _respond(self, msg: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(self._bound.wait(), self.bind_timeout)
        except asyncio.TimeoutError:
            logger.error("no call_details for retell call %s; closing", self.retell_call_id)
            await self._close(1011)
            return

        messages = transcript_to_messages(msg.get("transcript") or [])
        if msg.get("interaction_type") == "reminder_required":
            messages.append(HumanMessage(content=REMINDER_NUDGE))
        await self._run_turn(messages, response_id=msg["response_id"])

    def _filler_wait(self, messages: list[BaseMessage]) -> float | None:
        """Seconds to wait before a filler, or None when this turn shouldn't get one."""
        if not self.filler_seconds or not self.control.opener_spoken or self.control.closing:
            return None
        last = messages[-1] if messages else None
        if not isinstance(last, HumanMessage) or not isinstance(last.content, str):
            return None
        if is_note(last.content) or classify_line(last.content) != "person":
            return None  # screeners, menus and voicemail don't need to hear us think
        return self.filler_seconds

    async def _graph_chunks(self, messages: list[BaseMessage]):
        async for chunk in self.graph.astream({"messages": messages}, stream_mode="custom"):
            if chunk:
                yield chunk

    async def _run_turn(
        self,
        messages: list[BaseMessage],
        *,
        response_id: int | None = None,
        interrupt_id: int | None = None,
    ) -> None:
        """Run one graph turn and stream it as a response (or an agent_interrupt)."""
        if interrupt_id is None:
            base: dict[str, Any] = {"response_type": "response", "response_id": response_id}
        else:
            base = {"response_type": "agent_interrupt", "interrupt_id": interrupt_id}
        filler_wait = self._filler_wait(messages)
        chunks = self._graph_chunks(messages).__aiter__()
        pending: asyncio.Future | None = asyncio.ensure_future(chunks.__anext__())
        filler_sent = False
        try:
            if filler_wait is not None:
                done, _ = await asyncio.wait({pending}, timeout=filler_wait)
                if not done and not self.control.on_hold:
                    filler_sent = True
                    await self._send({**base, "content": FILLER, "content_complete": False})
            first = True
            while pending is not None:
                try:
                    chunk = await pending
                except StopAsyncIteration:
                    pending = None
                    break
                pending = asyncio.ensure_future(chunks.__anext__())
                if filler_sent and first:
                    chunk = _LEADING_FILLER.sub("", chunk)  # no "Hmm, hmm, ..."
                first = False
                if not chunk:
                    continue
                event = {**base, "content": chunk, "content_complete": False}
                if self.control.opener_text and chunk.strip() == self.control.opener_text:
                    # The fixed opener (first turn, or again for someone who just picked up);
                    # don't let a "Hello?" cut it off.
                    event["no_interruption_allowed"] = True
                await self._send(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("graph turn failed (retell call %s)", self.retell_call_id)
            await self._send(
                {**base, "content": "Sorry, could you say that again?", "content_complete": True}
            )
            return
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            with contextlib.suppress(Exception):
                await chunks.aclose()
        final: dict[str, Any] = {
            **base,
            "content": "",
            "content_complete": True,
            "end_call": self.control.end_requested,
        }
        if self.control.pending_digits:
            final["digit_to_press"] = self.control.pending_digits
            self.control.pending_digits = None
        await self._send(final)
        if self.control.closing and not self._closing_configured:
            self._closing_configured = True
            await self._send(
                {
                    "response_type": "update_agent",
                    "agent_config": {
                        "reminder_trigger_ms": CLOSING_REMINDER_MS,
                        "reminder_max_count": 1,
                    },
                }
            )


class RetellWebsocketRouter:
    """ASGI wrapper: serves Retell's custom-LLM websocket, passes everything else to `app`.

    Retell can't send our bearer token, so the websocket is protected by a secret path segment
    plus the requirement that the call's metadata names an active call in our store.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        store: CallStore,
        model_factory: Callable[[], BaseChatModel | None] = lambda: None,
    ) -> None:
        self.app = app
        self.settings = settings
        self.store = store
        self.model_factory = model_factory

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return
        ws = WebSocket(scope, receive, send)
        path: str = scope["path"]
        secret = self.settings.retell_ws_secret
        if not secret or not path.startswith(WS_PATH_PREFIX):
            await ws.close(code=1008)
            return
        supplied, _, retell_call_id = path[len(WS_PATH_PREFIX) :].partition("/")
        if not retell_call_id or not hmac.compare_digest(supplied.encode(), secret.encode()):
            await ws.close(code=1008)
            return
        await RetellLLMSession(
            ws,
            retell_call_id,
            self.store,
            self.model_factory(),
            settings=self.settings,
            silent_pickup_seconds=(self.settings.silent_pickup_ms / 1000) or None,
            filler_seconds=(self.settings.filler_after_ms / 1000) or None,
        ).run()


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
        )
    elif reason == "ivr_reached":
        store.update_call(
            call_id,
            status="completed",
            outcome="needs_followup",
            summary="Reached an automated phone menu and could not get through to a person.",
        )
    elif reason in INCOMPLETE_REASONS:
        store.update_call(
            call_id,
            status="completed",
            outcome="needs_followup",
            summary=f"Call ended without a recorded result ({reason}). See transcript.",
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
