"""Provision Retell for MuseVoiceAgent: a custom-LLM agent plus the imported Twilio number.

Idempotent: re-running updates what already exists. Reads RETELL_API_KEY, the Twilio credentials and
TWILIO_SIP_USERNAME/PASSWORD from .env (created by setup_twilio_trunk.py), then writes back
RETELL_AGENT_ID, RETELL_FROM_NUMBER, RETELL_WS_SECRET, PUBLIC_BASE_URL and VOICE_BACKEND=retell.

    uv run python scripts/setup_retell.py +1XXXXXXXXXX --public-url https://<tunnel-host>
    uv run python scripts/setup_retell.py +1XXXXXXXXXX --enable-inbound  # also answer callbacks
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import sys
import urllib.parse
import urllib.request

from dotenv import set_key

from muse_voice_agent.config import PROJECT_ROOT, get_settings
from muse_voice_agent.retell import RetellClient, RetellError
from muse_voice_agent.tasks import normalize_phone

ENV_PATH = PROJECT_ROOT / ".env"
AGENT_NAME = "Muse Voice Agent (LangGraph)"
TWILIO_TRUNK_NAME = "muse-voice-agent"
RETELL_SIP_URI = "sip:sip.retellai.com"


def _save(key: str, value: str) -> None:
    set_key(str(ENV_PATH), key, value, quote_mode="never")
    os.environ[key] = value


def _twilio(path: str, data: dict[str, str] | None = None) -> dict:
    sid, token = os.getenv("TWILIO_ACCOUNT_SID", ""), os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        sys.exit("Set TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN or pass --termination-uri")
    req = urllib.request.Request(
        f"https://trunking.twilio.com/v1/{path}",
        data=urllib.parse.urlencode(data).encode() if data is not None else None,
        headers={"Authorization": "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode()},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def _twilio_trunk() -> dict:
    for t in _twilio("Trunks?PageSize=100").get("trunks", []):
        if t.get("friendly_name") == TWILIO_TRUNK_NAME and t.get("domain_name"):
            return t
    sys.exit(f"No Twilio trunk named {TWILIO_TRUNK_NAME!r}; run scripts/setup_twilio_trunk.py first")


def _ensure_twilio_origination() -> None:
    """Route calls to the trunk's numbers into Retell (Twilio needs an origination URI for inbound)."""
    sid = _twilio_trunk()["sid"]
    urls = _twilio(f"Trunks/{sid}/OriginationUrls").get("origination_urls", [])
    if any(u.get("sip_url") == RETELL_SIP_URI and u.get("enabled") for u in urls):
        print(f"Twilio trunk already sends inbound calls to {RETELL_SIP_URI}")
        return
    _twilio(
        f"Trunks/{sid}/OriginationUrls",
        {"SipUrl": RETELL_SIP_URI, "FriendlyName": "retell", "Priority": "10", "Weight": "10", "Enabled": "true"},
    )
    print(f"Twilio trunk now sends inbound calls to {RETELL_SIP_URI}")


async def main_async(
    number: str,
    public_url: str | None,
    termination_uri: str | None,
    *,
    enable_inbound: bool,
) -> None:
    settings = get_settings()
    if not settings.retell_api_key:
        sys.exit("RETELL_API_KEY is not set in .env")

    if not settings.retell_ws_secret:
        _save("RETELL_WS_SECRET", secrets.token_urlsafe(24))
    if public_url:
        _save("PUBLIC_BASE_URL", public_url.rstrip("/"))
    settings = get_settings()
    if not settings.public_base_url:
        sys.exit("Pass --public-url https://<tunnel-host> (or set PUBLIC_BASE_URL)")

    agent_body = {
        "agent_name": AGENT_NAME,
        "response_engine": {
            "type": "custom-llm",
            "llm_websocket_url": settings.retell_llm_websocket_url(),
        },
        "voice_id": settings.retell_voice_id,
        "language": "en-US",
        "max_call_duration_ms": settings.max_call_seconds * 1000,
        # Long enough to sit on hold; silent pickups are handled by SILENT_PICKUP_MS instead.
        "end_call_after_silence_ms": 60_000,
        "ring_duration_ms": 30_000,
        "enable_backchannel": False,
    }

    async with RetellClient(settings.retell_api_key) as client:
        agent_id = settings.retell_agent_id
        if not agent_id:
            agent_id = next(
                (a["agent_id"] for a in await client.list_agents() if a.get("agent_name") == AGENT_NAME),
                "",
            )
        if agent_id:
            await client.update_agent(agent_id, agent_body)
            print(f"Updated Retell agent {agent_id}")
        else:
            agent_id = (await client.create_agent(agent_body))["agent_id"]
            print(f"Created Retell agent {agent_id}")
        _save("RETELL_AGENT_ID", agent_id)

        # Retell removed the single-agent inbound_agent_id/outbound_agent_id fields; use weighted lists.
        outbound = [{"agent_id": agent_id, "weight": 1}]
        inbound = {"inbound_agents": [{"agent_id": agent_id, "weight": 1}]} if enable_inbound else {}
        phone_update: dict[str, object] = {"outbound_agents": outbound, **inbound}
        existing = {p["phone_number"]: p for p in await client.list_phone_numbers()}
        if number in existing:
            await client.update_phone_number(number, phone_update)
            extra = " and inbound callbacks" if enable_inbound else ""
            print(f"Bound {number} (already in Retell) to outbound calls{extra}")
        else:
            username = os.getenv("TWILIO_SIP_USERNAME", "")
            password = os.getenv("TWILIO_SIP_PASSWORD", "")
            if not username or not password:
                sys.exit("TWILIO_SIP_USERNAME/PASSWORD missing; run scripts/setup_twilio_trunk.py")
            await client.import_phone_number(
                {
                    "phone_number": number,
                    "termination_uri": termination_uri or _twilio_trunk()["domain_name"],
                    "sip_trunk_auth_username": username,
                    "sip_trunk_auth_password": password,
                    "outbound_agents": outbound,
                    **inbound,
                    "nickname": "Muse Voice Agent (Twilio)",
                }
            )
            print(f"Imported {number} into Retell via the Twilio trunk")
        if enable_inbound:
            _ensure_twilio_origination()
        _save("RETELL_FROM_NUMBER", number)
        _save("VOICE_BACKEND", "retell")

    print("Done. Restart muse-voice-mcp; it re-syncs the agent URL from PUBLIC_BASE_URL on startup.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("number", help="Caller ID, e.g. +1XXXXXXXXXX (must be on the Twilio trunk)")
    parser.add_argument("--public-url", help="Public https base URL of muse-voice-mcp (tunnel)")
    parser.add_argument("--termination-uri", help="Twilio trunk domain (default: looked up)")
    parser.add_argument(
        "--enable-inbound",
        action="store_true",
        help="Answer callbacks: set the number's Retell inbound_agents and route the Twilio trunk to Retell.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            main_async(
                normalize_phone(args.number),
                args.public_url,
                args.termination_uri,
                enable_inbound=args.enable_inbound,
            )
        )
    except RetellError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
