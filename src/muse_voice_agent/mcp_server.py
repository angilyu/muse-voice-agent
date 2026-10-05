"""MCP server that exposes the phone agent to Muse (or any MCP client) over streamable HTTP.

Run:
    uv run muse-voice-mcp
Then expose it publicly (Muse runs in Meta's cloud), e.g.:
    cloudflared tunnel --url http://127.0.0.1:8765
and give Muse  https://<tunnel-host>/mcp  plus  Authorization: Bearer <MCP_AUTH_TOKEN>.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import Settings, get_settings
from .dispatcher import CallRejected, start_call
from .store import FINAL_STATUSES, CallStore
from .tasks import HandymanQuote, RestaurantReservation

logger = logging.getLogger("muse_voice_agent.mcp")

INSTRUCTIONS = """\
Places real phone calls to businesses on the user's behalf using an AI voice agent.
- Use book_restaurant_reservation or request_handyman_quote to start a call. They return a call_id
  immediately; the call itself takes 1-5 minutes.
- Poll get_call_status(call_id) every ~20 seconds until `done` is true, then report the summary.
- Always confirm the business, phone number, and request details with the user before calling.
- Phone numbers should be E.164 (e.g. +14155550123).
"""


def _public_view(record: dict[str, Any], include_transcript: bool = False) -> dict[str, Any]:
    view = {
        "call_id": record["id"],
        "kind": record["kind"],
        "business_name": record["business_name"],
        "phone_number": record["phone_number"],
        "status": record["status"],
        "done": record["status"] in FINAL_STATUSES,
        "outcome": record["outcome"],
        "summary": record["summary"],
        "details": record["details"],
        "error": record["error"],
        "simulated": record["dry_run"],
    }
    if include_transcript:
        view["transcript"] = record["transcript"]
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
            title="Call a restaurant to book a table",
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    async def book_restaurant_reservation(
        restaurant_name: str,
        phone_number: str,
        party_size: int,
        date: str,
        time: str,
        customer_name: str | None = None,
        flexibility: str | None = None,
        special_requests: str | None = None,
        callback_number: str | None = None,
    ) -> dict[str, Any]:
        """Phone a restaurant and try to book a table. Returns a call_id right away.

        Args:
            restaurant_name: Name of the restaurant.
            phone_number: Restaurant phone number, E.164 (e.g. +14155550123).
            party_size: Number of guests.
            date: Requested date, e.g. "Fri Oct 10".
            time: Requested time, e.g. "7:30 PM".
            customer_name: Name the reservation is under (defaults to DEFAULT_CUSTOMER_NAME).
            flexibility: Acceptable alternatives, e.g. "6:30-8:30pm same day".
            special_requests: Seating preferences, allergies, occasion.
            callback_number: Number the restaurant may call back (shared only if asked).
        """
        return await _start(
            RestaurantReservation,
            business_name=restaurant_name,
            phone_number=phone_number,
            customer_name=customer_name or settings.default_customer_name or "",
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
        job_description: str,
        location: str,
        customer_name: str | None = None,
        preferred_timing: str | None = None,
        budget: str | None = None,
        callback_number: str | None = None,
    ) -> dict[str, Any]:
        """Phone a handyman/contractor and ask for a price quote and availability (does not book).

        Args:
            business_name: Name of the business or person.
            phone_number: Business phone number, E.164 (e.g. +14155550123).
            job_description: What needs doing, e.g. "replace a leaking kitchen faucet".
            location: City / neighborhood / ZIP (avoid full street address).
            customer_name: Customer name to give (defaults to DEFAULT_CUSTOMER_NAME).
            preferred_timing: e.g. "weekday mornings next week".
            budget: Optional budget, only mentioned if asked.
            callback_number: Number the business may call back (shared only if asked).
        """
        return await _start(
            HandymanQuote,
            business_name=business_name,
            phone_number=phone_number,
            customer_name=customer_name or settings.default_customer_name or "",
            job_description=job_description,
            location=location,
            preferred_timing=preferred_timing,
            budget=budget,
            callback_number=callback_number or settings.default_callback_number or None,
        )

    @mcp.tool(annotations=ToolAnnotations(title="Get call status", readOnlyHint=True))
    async def get_call_status(call_id: str, include_transcript: bool = False) -> dict[str, Any]:
        """Get the status and result of a call started by this server.

        Args:
            call_id: The call_id returned when the call was started.
            include_transcript: Include the turn-by-turn transcript.
        """
        record = store.get_call(call_id)
        if record is None:
            return {"error": "not_found", "message": f"No call with id {call_id}"}
        return _public_view(record, include_transcript=include_transcript)

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


def build_app(settings: Settings | None = None, store: CallStore | None = None) -> ASGIApp:
    settings = settings or get_settings()
    if not settings.mcp_auth_token:
        raise SystemExit("MCP_AUTH_TOKEN must be set in .env (the server is meant to be public).")
    mcp = build_server(settings, store)
    # Auth is enforced by the bearer token; Host-header checks would reject tunnel hostnames.
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host=settings.mcp_host,
    )
    return BearerAuthMiddleware(app, settings.mcp_auth_token)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    mode = "DRY_RUN (simulated calls)" if settings.dry_run else "LIVE (real phone calls)"
    logger.info("MCP server on http://%s:%s/mcp - %s", settings.mcp_host, settings.mcp_port, mode)
    uvicorn.run(build_app(settings), host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
