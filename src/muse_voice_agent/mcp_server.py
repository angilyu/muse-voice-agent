"""MCP server that exposes the phone agent to Muse (or any MCP client) over streamable HTTP.

Run:
    uv run muse-voice-mcp
Then expose it publicly (Muse runs in Meta's cloud), e.g.:
    cloudflared tunnel --url http://127.0.0.1:8765
and give Muse  https://<tunnel-host>/mcp  plus  Authorization: Bearer <MCP_AUTH_TOKEN>.

With VOICE_BACKEND=retell the same server also hosts Retell's custom-LLM websocket at
/retell/llm/<RETELL_WS_SECRET>/<retell_call_id>; set PUBLIC_BASE_URL to the tunnel URL and the
Retell agent is re-pointed at it on startup.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any, Callable

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings, get_settings
from .dispatcher import CallRejected, start_call
from .keepalive import KeepAliveMiddleware
from .report import build_report, speaker_transcript
from .retell import RetellWebsocketRouter, sync_agent_websocket_url
from .store import FINAL_STATUSES, CallStore
from .tasks import Authority, GeneralCall, HandymanQuote, RestaurantReservation

logger = logging.getLogger("muse_voice_agent.mcp")

INSTRUCTIONS = """\
Places real phone calls to businesses on the user's behalf using an AI voice agent.
- Use place_call for ANY phone errand the user approves: placing takeout/pickup food or drink
  orders, scheduling/rescheduling/cancelling appointments, checking order or refill status, asking
  questions, making reservations of any kind, getting quotes, checking availability, store stock,
  hours, pricing, or other business calls.
- book_restaurant_reservation and request_handyman_quote are optional tuned shortcuts for those two
  cases only. For everything else, use place_call rather than refusing because a specialized tool is
  missing.
- place_call defaults to authority="info_only" (the agent commits to nothing). Use
  "may_commit_within_limits" when the user explicitly asked it to book/order/schedule/cancel/etc.,
  and put every limit they gave in `limits` (items, quantities, options, dates/times, price cap,
  "pay at pickup", "defaults for anything else", "no deposit"). "may_book_within_limits" remains a
  backward-compatible alias.
- Pickup orders with pay-at-pickup are allowed when within limits. Never include card numbers, SSNs,
  passwords, or a full home address; if a business requires card prepayment over the phone, the call
  returns needs_followup.
- Only put details the user is comfortable sharing in shareable_details.
- Every call tool returns a call_id immediately; the call itself takes 1-5 minutes. Poll
  get_call_status(call_id) every ~20 seconds until `done` is true.
- When done, get_call_status adds a `report`: answers to each requested question,
  unanswered_questions, whether the agent committed on the user's behalf, how the call ended,
  duration, and suggested next_steps, plus the speaker-labeled transcript. Tell the user the
  summary, key details (date/time, total, quote), any unanswered questions, and offer the next
  steps (e.g. add a booking to their calendar, retry, or call another business).
- Always pass customer_name: the full name of the user you're calling for. The agent introduces
  itself as their assistant. If you don't know the user's name, ask them before calling.
- Always confirm the business, phone number, and the brief with the user before calling.
- Phone numbers should be E.164 (e.g. +14155550123).
"""


def _public_view(
    record: dict[str, Any], include_transcript: bool = False, include_report: bool = False
) -> dict[str, Any]:
    done = record["status"] in FINAL_STATUSES
    view = {
        "call_id": record["id"],
        "kind": record["kind"],
        "business_name": record["business_name"],
        "phone_number": record["phone_number"],
        "status": record["status"],
        "done": done,
        "outcome": record["outcome"],
        "summary": record["summary"],
        "details": record["details"],
        "error": record["error"],
        "simulated": record["dry_run"],
    }
    if include_report and done:
        view["report"] = build_report(record)
    if include_transcript:
        view["transcript"] = speaker_transcript(record["transcript"])
    return view


def build_server(settings: Settings | None = None, store: CallStore | None = None) -> MCPServer:
    settings = settings or get_settings()
    store = store or CallStore(settings.call_db_path)
    mcp = MCPServer(name="muse-voice-agent", instructions=INSTRUCTIONS, version="0.1.0")

    async def _start(task_cls: type, **kwargs: Any) -> dict[str, Any]:
        try:
            task = task_cls(**kwargs)
            record = await start_call(task, settings, store)
        except ValidationError as e:
            return {"error": "invalid_request", "message": str(e)}
        except CallRejected as e:
            return {"error": "rejected", "message": str(e)}
        view = _public_view(record)
        view["next_step"] = "Poll get_call_status with this call_id until done is true."
        return view

    @mcp.tool(
        annotations=ToolAnnotations(
            title="General-purpose phone agent for any business errand",
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    async def place_call(
        business_name: str,
        phone_number: str,
        customer_name: str,
        goal: str,
        questions: list[str] | None = None,
        shareable_details: dict[str, str] | None = None,
        authority: Authority = "info_only",
        limits: str | None = None,
        callback_number: str | None = None,
    ) -> dict[str, Any]:
        """Phone any business for any errand Muse can brief.

        Use this for placing takeout/pickup food or drink orders, scheduling/rescheduling/cancelling
        appointments, checking order/refill/repair status, asking questions, making reservations of
        any kind, getting quotes, checking availability, store stock, hours or pricing. Returns a
        call_id right away.

        Args:
            business_name: Name of the business.
            phone_number: Business phone number, E.164 (e.g. +14155550123).
            customer_name: Required. Full name of the user you're calling for, e.g. "Wenjing Yu".
                The agent introduces itself as their assistant. Ask the user if you don't know it.
            goal: One or two sentences on what the call should accomplish, including exact order
                items and options when ordering.
            questions: Specific questions to get answered (up to 10), answered back in `answers`.
            shareable_details: Facts the agent may share if relevant, e.g. {"dates": "Oct 10-12",
                "guests": "2 adults", "order number": "A1234", "pickup name": "Wenjing"}.
                No card numbers or SSNs.
            authority: "info_only" (default; commit to nothing), "may_commit_within_limits"
                (book/order/schedule/cancel/etc. within limits), or the backward-compatible alias
                "may_book_within_limits".
            limits: Required for may_commit_within_limits/may_book_within_limits, e.g. "two
                jasmine green milk teas, 25% sugar, less ice, defaults otherwise, pickup order,
                pay at pickup, no card over phone" or "Oct 10-12 only, max $250/night, no deposit".
            callback_number: Number the business may call back (shared only if asked).
        """
        return await _start(
            GeneralCall,
            business_name=business_name,
            phone_number=phone_number,
            customer_name=customer_name,
            goal=goal,
            questions=questions or [],
            shareable_details=shareable_details or {},
            authority=authority,
            limits=limits,
            callback_number=callback_number or settings.default_callback_number or None,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Call a restaurant to book a table",
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    async def book_restaurant_reservation(
        restaurant_name: str,
        phone_number: str,
        customer_name: str,
        party_size: int,
        date: str,
        time: str,
        flexibility: str | None = None,
        special_requests: str | None = None,
        callback_number: str | None = None,
    ) -> dict[str, Any]:
        """Phone a restaurant and try to book a table. Returns a call_id right away.

        Args:
            restaurant_name: Name of the restaurant.
            phone_number: Restaurant phone number, E.164 (e.g. +14155550123).
            customer_name: Required. Full name of the user you're calling for, e.g. "Wenjing Yu".
                The agent introduces itself as their assistant. Ask the user if you don't know it.
            party_size: Number of guests.
            date: Requested date, e.g. "Fri Oct 10".
            time: Requested time, e.g. "7:30 PM".
            flexibility: Acceptable alternatives, e.g. "6:30-8:30pm same day". Ask the user and
                pass it when they're flexible: without it the agent can't book a nearby time, so
                if the exact time is taken it only collects the closest times for the user.
            special_requests: Seating preferences, allergies, occasion.
            callback_number: Number the restaurant may call back (shared only if asked).
        """
        return await _start(
            RestaurantReservation,
            business_name=restaurant_name,
            phone_number=phone_number,
            customer_name=customer_name,
            party_size=party_size,
            date=date,
            time=time,
            flexibility=flexibility,
            special_requests=special_requests,
            callback_number=callback_number or settings.default_callback_number or None,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Call a handyman / contractor for a quote",
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    async def request_handyman_quote(
        business_name: str,
        phone_number: str,
        customer_name: str,
        job_description: str,
        location: str,
        preferred_timing: str | None = None,
        budget: str | None = None,
        callback_number: str | None = None,
    ) -> dict[str, Any]:
        """Phone a handyman/contractor and ask for a price quote and availability (does not book).

        Args:
            business_name: Name of the business or person.
            phone_number: Business phone number, E.164 (e.g. +14155550123).
            customer_name: Required. Full name of the user you're calling for, e.g. "Wenjing Yu".
                The agent introduces itself as their assistant. Ask the user if you don't know it.
            job_description: What needs doing, e.g. "replace a leaking kitchen faucet".
            location: City / neighborhood / ZIP (avoid full street address).
            preferred_timing: e.g. "weekday mornings next week".
            budget: Optional budget, only mentioned if asked.
            callback_number: Number the business may call back (shared only if asked).
        """
        return await _start(
            HandymanQuote,
            business_name=business_name,
            phone_number=phone_number,
            customer_name=customer_name,
            job_description=job_description,
            location=location,
            preferred_timing=preferred_timing,
            budget=budget,
            callback_number=callback_number or settings.default_callback_number or None,
        )

    @mcp.tool(annotations=ToolAnnotations(title="Get call status", readOnlyHint=True))
    async def get_call_status(
        call_id: str, include_transcript: bool | None = None
    ) -> dict[str, Any]:
        """Get the status and result of a call started by this server.

        Once `done` is true the response also has a `report` with: request (the brief you sent),
        reached (person/voicemail/phone_menu/no_answer/not_connected), started_at, ended_at,
        duration_seconds, end_reason, ended_by, outcome_source (agent = recorded live,
        transcript = inferred afterwards, call_system = no result recorded),
        committed_on_users_behalf, answers (one per requested question), unanswered_questions,
        and next_steps to offer the user.

        Args:
            call_id: The call_id returned when the call was started.
            include_transcript: Include the speaker-labeled transcript (business/assistant).
                Defaults to only once the call is done.
        """
        record = store.get_call(call_id)
        if record is None:
            return {"error": "not_found", "message": f"No call with id {call_id}"}
        if include_transcript is None:
            include_transcript = record["status"] in FINAL_STATUSES
        return _public_view(record, include_transcript=include_transcript, include_report=True)

    @mcp.tool(annotations=ToolAnnotations(title="List recent calls", readOnlyHint=True))
    async def list_calls(limit: int = 10) -> dict[str, Any]:
        """List the most recent calls (newest first).

        Args:
            limit: Max number of calls to return (1-50).
        """
        limit = max(1, min(limit, 50))
        return {"calls": [_public_view(r) for r in store.list_calls(limit)]}

    return mcp


class BearerAuthMiddleware:
    """Requires `Authorization: Bearer <token>` on every HTTP request except /healthz."""

    def __init__(self, app: ASGIApp, token: str):
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            if scope["path"] == "/healthz":
                await JSONResponse({"ok": True})(scope, receive, send)
                return
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"")
            scheme, _, supplied = auth.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(supplied.strip(), self.token):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(
    settings: Settings | None = None,
    store: CallStore | None = None,
    model_factory: Callable[[], Any] = lambda: None,
) -> ASGIApp:
    settings = settings or get_settings()
    if not settings.mcp_auth_token:
        raise SystemExit("MCP_AUTH_TOKEN must be set in .env (the server is meant to be public).")
    store = store or CallStore(settings.call_db_path)
    mcp = build_server(settings, store)
    # Auth is enforced by the bearer token; Host-header checks would reject tunnel hostnames.
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host=settings.mcp_host,
    )
    # HTTP requests need the bearer token; websockets are only Retell's (secret-path protected).
    app = RetellWebsocketRouter(
        BearerAuthMiddleware(app, settings.mcp_auth_token), settings, store, model_factory
    )
    if settings.keepalive_seconds > 0 and settings.public_base_url:
        app = KeepAliveMiddleware(
            app, f"{settings.public_base_url}/healthz", settings.keepalive_seconds
        )
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    mode = "DRY_RUN (simulated calls)" if settings.dry_run else "LIVE (real phone calls)"
    logger.info("MCP server on http://%s:%s/mcp - %s", settings.mcp_host, settings.mcp_port, mode)
    if not settings.dry_run and settings.voice_backend == "retell":
        missing = settings.missing_for_live_calls()
        if missing:
            logger.warning("Retell not configured (missing %s); calls will be rejected", missing)
        elif not settings.public_base_url:
            logger.warning("PUBLIC_BASE_URL not set; Retell agent websocket URL was not synced")
        else:
            url = asyncio.run(sync_agent_websocket_url(settings))
            logger.info("Retell agent %s -> %s/...", settings.retell_agent_id, url.rsplit("/", 1)[0])
    uvicorn.run(build_app(settings), host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
