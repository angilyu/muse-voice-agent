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
import hmac
import json
import logging
import time
from typing import Any, Callable

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import Settings
from .graph import CallOutcome, build_call_graph
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
    ) -> None:
        self.ws = ws
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
            if self._turn and not self._turn.done():
                self._turn.cancel()

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
        self.graph = build_call_graph(parse_task(record["task"]), self._on_outcome, self.model)
        self.store.update_call(call_id, status="in_progress", provider_call_id=self.retell_call_id)
        self._bound.set()
        logger.info("retell call %s bound to %s", self.retell_call_id, call_id)

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
        response_id = msg["response_id"]
        try:
            await asyncio.wait_for(self._bound.wait(), self.bind_timeout)
        except asyncio.TimeoutError:
            logger.error("no call_details for retell call %s; closing", self.retell_call_id)
            await self._close(1011)
            return

        messages = transcript_to_messages(msg.get("transcript") or [])
        if msg.get("interaction_type") == "reminder_required":
            messages.append(HumanMessage(content=REMINDER_NUDGE))
        try:
            async for chunk in self.graph.astream({"messages": messages}, stream_mode="custom"):
                if chunk:
                    await self._send(
                        {
                            "response_type": "response",
                            "response_id": response_id,
                            "content": chunk,
                            "content_complete": False,
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("graph turn failed (retell call %s)", self.retell_call_id)
            await self._send(
                {
                    "response_type": "response",
                    "response_id": response_id,
                    "content": "Sorry, could you say that again?",
                    "content_complete": True,
                }
            )
            return
        await self._send(
            {
                "response_type": "response",
                "response_id": response_id,
                "content": "",
                "content_complete": True,
                # Once the result is recorded, hang up after the goodbye has been spoken.
                "end_call": self.outcome is not None,
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
        await RetellLLMSession(ws, retell_call_id, self.store, self.model_factory()).run()


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
                finalize_from_retell(store, call_id, call)
                return
    record = store.get_call(call_id)
    if record and record["status"] not in FINAL_STATUSES:
        store.update_call(call_id, status="failed", error="lost track of the Retell call")
