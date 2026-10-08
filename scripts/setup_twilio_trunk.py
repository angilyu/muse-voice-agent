"""Provision a Twilio Elastic SIP trunk and the matching LiveKit outbound trunk.

Idempotent: re-running reuses what already exists. Reads TWILIO_ACCOUNT_SID,
TWILIO_AUTH_TOKEN and the LIVEKIT_* keys from .env, then writes back
SIP_OUTBOUND_TRUNK_ID (plus the generated SIP credentials) to .env.

    uv run python scripts/setup_twilio_trunk.py +1XXXXXXXXXX
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import string
import sys
import urllib.error
import urllib.parse
import urllib.request

from dotenv import set_key
from livekit import api

from muse_voice_agent.config import PROJECT_ROOT
from muse_voice_agent.tasks import normalize_phone

ENV_PATH = PROJECT_ROOT / ".env"
TRUNK_NAME = "muse-voice-agent"
LIVEKIT_TRUNK_NAME = "muse-voice-agent-twilio"


class Twilio:
    def __init__(self, sid: str, token: str) -> None:
        self.sid = sid
        self._auth = "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode()
        self.core = f"https://api.twilio.com/2010-04-01/Accounts/{sid}"
        self.trunking = "https://trunking.twilio.com/v1"

    def _req(self, method: str, url: str, data: dict | None = None) -> dict:
        body = urllib.parse.urlencode(data).encode() if data else None
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", self._auth)
        if body:
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise SystemExit(f"Twilio {method} {url} failed ({e.code}): {detail}") from None

    def get(self, url: str) -> dict:
        return self._req("GET", url)

    def post(self, url: str, data: dict) -> dict:
        return self._req("POST", url, data)


def _env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise SystemExit(f"{name} is empty in {ENV_PATH}")
    return value


def _save(key: str, value: str) -> None:
    set_key(str(ENV_PATH), key, value, quote_mode="never")
    os.environ[key] = value


def _new_password() -> str:
    # Twilio: >= 12 chars with upper, lower and digit.
    alphabet = string.ascii_letters + string.digits
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(32))
        if any(c.isupper() for c in pw) and any(c.islower() for c in pw) and any(
            c.isdigit() for c in pw
        ):
            return pw


def provision_twilio(tw: Twilio, number: str) -> tuple[str, str, str, str]:
    """Returns (domain, phone_number, sip_username, sip_password)."""
    numbers = tw.get(f"{tw.core}/IncomingPhoneNumbers.json?PhoneNumber={urllib.parse.quote(number)}")
    matches = numbers.get("incoming_phone_numbers", [])
    if not matches:
        raise SystemExit(f"{number} is not a number on this Twilio account")
    phone = matches[0]

    trunks = tw.get(f"{tw.trunking}/Trunks?PageSize=100").get("trunks", [])
    trunk = next((t for t in trunks if t["friendly_name"] == TRUNK_NAME), None)
    if trunk:
        print(f"Twilio trunk exists: {trunk['sid']}")
    else:
        domain = f"muse-voice-{tw.sid[-10:].lower()}.pstn.twilio.com"
        trunk = tw.post(f"{tw.trunking}/Trunks", {"FriendlyName": TRUNK_NAME, "DomainName": domain})
        print(f"Created Twilio trunk: {trunk['sid']}")
    domain = trunk["domain_name"]

    # Credential list used by LiveKit to authenticate outbound INVITEs.
    username = (os.getenv("TWILIO_SIP_USERNAME") or "").strip() or "musevoice"
    password = (os.getenv("TWILIO_SIP_PASSWORD") or "").strip()
    cls = tw.get(f"{tw.core}/SIP/CredentialLists.json?PageSize=100").get("credential_lists", [])
    cl = next((c for c in cls if c["friendly_name"] == TRUNK_NAME), None)
    if not cl:
        cl = tw.post(f"{tw.core}/SIP/CredentialLists.json", {"FriendlyName": TRUNK_NAME})
        print(f"Created credential list: {cl['sid']}")
    creds = tw.get(f"{tw.core}/SIP/CredentialLists/{cl['sid']}/Credentials.json").get(
        "credentials", []
    )
    existing = next((c for c in creds if c["username"] == username), None)
    if existing and password:
        print(f"SIP credential '{username}' exists")
    else:
        password = _new_password()
        if existing:  # password lost locally: rotate it
            tw.post(
                f"{tw.core}/SIP/CredentialLists/{cl['sid']}/Credentials/{existing['sid']}.json",
                {"Password": password},
            )
            print(f"Rotated password for SIP credential '{username}'")
        else:
            tw.post(
                f"{tw.core}/SIP/CredentialLists/{cl['sid']}/Credentials.json",
                {"Username": username, "Password": password},
            )
            print(f"Created SIP credential '{username}'")
        _save("TWILIO_SIP_USERNAME", username)
        _save("TWILIO_SIP_PASSWORD", password)

    linked = tw.get(f"{tw.trunking}/Trunks/{trunk['sid']}/CredentialLists").get(
        "credential_lists", []
    )
    if not any(c["sid"] == cl["sid"] for c in linked):
        tw.post(f"{tw.trunking}/Trunks/{trunk['sid']}/CredentialLists", {"CredentialListSid": cl["sid"]})
        print("Attached credential list to trunk")

    if phone.get("trunk_sid") == trunk["sid"]:
        print(f"{number} already on trunk")
    elif phone.get("trunk_sid"):
        raise SystemExit(f"{number} is attached to another trunk ({phone['trunk_sid']})")
    else:
        tw.post(f"{tw.trunking}/Trunks/{trunk['sid']}/PhoneNumbers", {"PhoneNumberSid": phone["sid"]})
        print(f"Attached {number} to trunk")

    return domain, number, username, password


async def provision_livekit(domain: str, number: str, username: str, password: str) -> str:
    async with api.LiveKitAPI(
        url=_env("LIVEKIT_URL"), api_key=_env("LIVEKIT_API_KEY"), api_secret=_env("LIVEKIT_API_SECRET")
    ) as lk:
        info = api.SIPOutboundTrunkInfo(
            name=LIVEKIT_TRUNK_NAME,
            address=domain,
            numbers=[number],
            auth_username=username,
            auth_password=password,
        )
        existing = await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
        trunk = next((t for t in existing.items if t.name == LIVEKIT_TRUNK_NAME), None)
        if trunk:
            trunk = await lk.sip.update_outbound_trunk(trunk.sip_trunk_id, info)
            print(f"Updated LiveKit outbound trunk: {trunk.sip_trunk_id}")
        else:
            trunk = await lk.sip.create_outbound_trunk(api.CreateSIPOutboundTrunkRequest(trunk=info))
            print(f"Created LiveKit outbound trunk: {trunk.sip_trunk_id}")
        return trunk.sip_trunk_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("number", help="Twilio number to use as caller ID, e.g. +1XXXXXXXXXX")
    args = parser.parse_args()

    number = normalize_phone(args.number)
    tw = Twilio(_env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN"))
    domain, number, username, password = provision_twilio(tw, number)
    trunk_id = asyncio.run(provision_livekit(domain, number, username, password))
    _save("SIP_OUTBOUND_TRUNK_ID", trunk_id)
    print(f"\nSIP_OUTBOUND_TRUNK_ID={trunk_id} saved to .env (caller ID {number}, {domain})")


if __name__ == "__main__":
    sys.exit(main())
