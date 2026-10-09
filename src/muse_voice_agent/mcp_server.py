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
from mcp.server.mcpserver import Context, MCPServer
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
from .safety import recording_disclosure_required, reject_sensitive_payload
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
  and pass every limit: structured fields such as max_spend, max_deposit,
  max_cancellation_fee, allowed_date_time_window, party_size_min/party_size_max, plus any other
  binding notes in `limits`. "may_book_within_limits" remains a backward-compatible alias.
- Pickup orders with pay-at-pickup are allowed when within limits. Never include card numbers, SSNs,
  passwords, bank/routing details, CVV/expiry, or other secrets; the server rejects card-like/SSN
  input and the agent will return needs_followup if a business requires card prepayment by phone.
- Only put approved personal details in shareable_details. Default allow-list is customer name plus
  callback number; emails, addresses, DOBs, insurance IDs and similar details are withheld unless
  explicitly passed as shareable_details for this call.
- Every call tool returns a call_id immediately; the call itself takes 1-5 minutes. Poll
  get_call_status(call_id) every ~20 seconds until `done` is true.
- When done, get_call_status adds a `report`: answers to each requested question,
  unanswered_questions, whether the agent committed on the user's behalf, how the call ended,
  duration, and suggested next_steps, plus the speaker-labeled transcript. Tell the user the
  summary, key details (date/time, total, quote), any unanswered questions, and offer the next
  steps (e.g. add a booking to their calendar, retry, or call another business).
- Businesses often call back later (after a voicemail, or "the manager will call you back"). The
  agent answers those calls, links each one to the original call, and keeps it as an unread update.
  Call get_updates to collect them; each update is returned once. Whenever a tool response has
  unread_updates > 0, call get_updates and tell the user what happened.
- After a call ends with voicemail, no_answer or needs_followup, or the business said it would call
  back, offer the user a scheduled task (for example every 2 hours for the next 3 days) that calls
  get_updates and messages them about anything new. This server cannot push to you.
- get_call_status on the original call also lists its callbacks, and list_calls includes inbound
  records with direction="inbound" and callback_of set.
- Always pass customer_name: the full name of the user you're calling for. The agent introduces
  itself as their assistant. If you don't know the user's name, ask them before calling.
- Always pass assistant_name: your own name, the one the user knows you by (e.g. "Eva"). The agent
  opens with "Hi, this is {assistant_name}, an AI assistant calling on behalf of {customer_name}."
- Always confirm the business, phone number, and the brief with the user before calling.
- Phone numbers should be E.164 (e.g. +14155550123).
"""


def _public_view(
    record: dict[str, Any],
    include_transcript: bool = False,
    include_report: bool = False,
    callback_records: list[dict[str, Any]] | None = None,
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
        "direction": record.get("direction") or "outbound",
        "callback_of": record.get("parent_call_id"),
    }
    if include_report and done:
        view["report"] = build_report(record)
    if include_transcript:
        view["transcript"] = speaker_transcript(record["transcript"])
    if callback_records:
        view["callbacks"] = [
            _public_view(r, include_transcript=include_transcript, include_report=include_report)
            for r in callback_records
        ]
    return view


def build_server(settings: Settings | None = None, store: CallStore | None = None) -> MCPServer:
    settings = settings or get_settings()
    store = store or CallStore(settings.call_db_path)
    mcp = MCPServer(name="muse-voice-agent", instructions=INSTRUCTIONS, version="0.1.0")

    def _with_unread(view: dict[str, Any]) -> dict[str, Any]:
        unread = store.count_unseen_inbound()
        if unread:
            view["unread_updates"] = unread
        return view

    async def _start(task_cls: type, ctx: Context | None = None, **kwargs: Any) -> dict[str, Any]:
        try:
            reject_sensitive_payload(kwargs, "tool_input", exclude_fields={"phone_number", "callback_number"})
            disclose, state = recording_disclosure_required(
                kwargs.get("phone_number", ""),
                recording_enabled=settings.call_recording_enabled,
                scope=settings.recording_disclosure_scope,
            )
            kwargs["recording_disclosure_required"] = disclose or bool(
                kwargs.get("recording_disclosure_required")
            )
            kwargs["recording_disclosure_state"] = state
            task = task_cls(**kwargs)
            record = await start_call(task, settings, store)
        except ValidationError as e:
            return {"error": "invalid_request", "message": str(e)}
        except ValueError as e:
            return {"error": "invalid_request", "message": str(e)}
        except CallRejected as e:
            return {"error": "rejected", "message": str(e)}
        store.update_call(record["id"], origin=_origin(ctx))
        view = _public_view(record)
        view["next_step"] = "Poll get_call_status with this call_id until done is true."
        return _with_unread(view)

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
        max_spend: float | None = None,
        max_deposit: float | None = None,
        max_cancellation_fee: float | None = None,
        allowed_date_time_window: str | None = None,
        party_size_min: int | None = None,
        party_size_max: int | None = None,
        callback_number: str | None = None,
        assistant_name: str | None = None,
        ctx: Context | None = None,
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
            shareable_details: Approved personal/details allow-list the agent may share if relevant,
                e.g. {"email": "alex@example.com", "pickup name": "Wenjing", "pet name": "Mochi"}.
                Customer name and callback number are included by default. No card numbers or SSNs.
            authority: "info_only" (default; commit to nothing), "may_commit_within_limits"
                (book/order/schedule/cancel/etc. within limits), or the backward-compatible alias
                "may_book_within_limits".
            limits: Required for may_commit_within_limits/may_book_within_limits, e.g. "two
                jasmine green milk teas, 25% sugar, less ice, defaults otherwise, pickup order,
                pay at pickup, no card over phone" or "Oct 10-12 only, max $250/night, no deposit".
            max_spend: Optional structured maximum total spend the agent may accept.
            max_deposit: Optional structured maximum deposit/prepayment/hold fee. Defaults to no
                deposit when authority permits commitments.
            max_cancellation_fee: Optional structured maximum cancellation/no-show fee. Defaults to
                none when authority permits commitments.
            allowed_date_time_window: Optional structured allowed date/time window.
            party_size_min/party_size_max: Optional structured party size bounds.
            callback_number: The user's number for the business to reach them; the agent may share it.
            assistant_name: Your own name as the user's assistant, e.g. "Eva". The agent says
                "Hi, this is Eva, an AI assistant calling on behalf of {customer_name}."
        """
        return await _start(
            GeneralCall,
            ctx=ctx,
            business_name=business_name,
            phone_number=phone_number,
            customer_name=customer_name,
            goal=goal,
            questions=questions or [],
            shareable_details=shareable_details or {},
            authority=authority,
            limits=limits,
            max_spend=max_spend,
            max_deposit=max_deposit,
            max_cancellation_fee=max_cancellation_fee,
            allowed_date_time_window=allowed_date_time_window,
            party_size_min=party_size_min,
            party_size_max=party_size_max,
            callback_number=callback_number or settings.default_callback_number or None,
            assistant_name=assistant_name,
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
        shareable_details: dict[str, str] | None = None,
        callback_number: str | None = None,
        assistant_name: str | None = None,
        ctx: Context | None = None,
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
            shareable_details: Approved extra personal details the agent may share if relevant.
            callback_number: The user's number for the restaurant to reach them; the agent may share it.
            assistant_name: Your own name as the user's assistant, e.g. "Eva". The agent says
                "Hi, this is Eva, an AI assistant calling on behalf of {customer_name}."
        """
        return await _start(
            RestaurantReservation,
            ctx=ctx,
            business_name=restaurant_name,
            phone_number=phone_number,
            customer_name=customer_name,
            party_size=party_size,
            date=date,
            time=time,
            flexibility=flexibility,
            special_requests=special_requests,
            shareable_details=shareable_details or {},
            callback_number=callback_number or settings.default_callback_number or None,
            assistant_name=assistant_name,
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
        shareable_details: dict[str, str] | None = None,
        callback_number: str | None = None,
        assistant_name: str | None = None,
        ctx: Context | None = None,
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
            shareable_details: Approved extra personal details the agent may share if relevant.
            callback_number: The user's number for the business to reach them; the agent may share it.
            assistant_name: Your own name as the user's assistant, e.g. "Eva". The agent says
                "Hi, this is Eva, an AI assistant calling on behalf of {customer_name}."
        """
        return await _start(
            HandymanQuote,
            ctx=ctx,
            business_name=business_name,
            phone_number=phone_number,
            customer_name=customer_name,
            job_description=job_description,
            location=location,
            preferred_timing=preferred_timing,
            budget=budget,
            shareable_details=shareable_details or {},
            callback_number=callback_number or settings.default_callback_number or None,
            assistant_name=assistant_name,
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
        callbacks = store.list_callbacks(call_id)
        return _with_unread(
            _public_view(
                record,
                include_transcript=include_transcript,
                include_report=True,
                callback_records=callbacks,
            )
        )

    @mcp.tool(annotations=ToolAnnotations(title="List recent calls", readOnlyHint=True))
    async def list_calls(limit: int = 10) -> dict[str, Any]:
        """List the most recent calls (newest first).

        Args:
            limit: Max number of calls to return (1-50).
        """
        limit = max(1, min(limit, 50))
        return _with_unread({"calls": [_public_view(r) for r in store.list_calls(limit)]})

    @mcp.tool(annotations=ToolAnnotations(title="Get callback updates", readOnlyHint=False))
    async def get_updates(limit: int = 20) -> dict[str, Any]:
        """Collect calls businesses made back to the agent that you haven't seen yet.

        Each update is a finished inbound call. kind="callback" means a business called back about
        a call you placed; original_call has that call's id, business, outcome and summary, and
        callback has the new outcome, summary, details and report. kind="message" means an unknown
        caller left a message. Updates are marked seen and not returned again; use get_call_status
        on original_call.call_id to see them later.

        Args:
            limit: Max number of updates to return (1-50). Oldest first.
        """
        limit = max(1, min(limit, 50))
        records = store.list_unseen_inbound(limit)
        updates = []
        for r in records:
            update: dict[str, Any] = {
                "kind": "callback" if r.get("parent_call_id") else "message",
                "callback": _public_view(r, include_report=True),
            }
            parent = store.get_call(r["parent_call_id"]) if r.get("parent_call_id") else None
            if parent:
                update["original_call"] = {
                    "call_id": parent["id"],
                    "business_name": parent["business_name"],
                    "outcome": parent["outcome"],
                    "summary": parent["summary"],
                }
            updates.append(update)
        store.mark_seen([r["id"] for r in records])
        return {"updates": updates, "remaining": store.count_unseen_inbound()}

    return mcp


def _origin(ctx: Context | None) -> str:
    """Tag calls placed through MCP, with the client's name when it sent one (e.g. "mcp:muse")."""
    try:
        info = ctx.session.client_params.clientInfo if ctx else None
    except Exception:  # noqa: BLE001 - no request context or client info (e.g. stateless request)
        info = None
    name = (getattr(info, "name", "") or "").strip()[:60]
    return f"mcp:{name}" if name else "mcp"


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
