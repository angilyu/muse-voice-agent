"""Minimal Stripe REST client (Checkout, Billing Portal, subscriptions) and webhook verification.

Uses httpx directly rather than the stripe SDK: we only need a handful of form-encoded calls.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

API_BASE = "https://api.stripe.com"


class StripeError(Exception):
    pass


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Encode nested dicts/lists the way Stripe expects: a[b][0][c]=v."""
    out: list[tuple[str, str]] = []
    for key, value in data.items():
        name = f"{prefix}[{key}]" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, dict):
            out.extend(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    out.extend(_flatten(item, f"{name}[{i}]"))
                else:
                    out.append((f"{name}[{i}]", str(item)))
        elif isinstance(value, bool):
            out.append((name, "true" if value else "false"))
        else:
            out.append((name, str(value)))
    return out


class StripeClient:
    def __init__(self, secret_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.secret_key = secret_key
        self.transport = transport

    async def _req(
        self, method: str, path: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.secret_key:
            raise StripeError("Stripe is not configured (STRIPE_SECRET_KEY is empty)")
        async with httpx.AsyncClient(
            base_url=API_BASE, timeout=20, transport=self.transport
        ) as client:
            form = urlencode(_flatten(data)) if data else None
            resp = await client.request(
                method,
                path,
                content=form if form and method != "GET" else None,
                params=form if form and method == "GET" else None,
                headers={"Content-Type": "application/x-www-form-urlencoded"} if form else None,
                auth=(self.secret_key, ""),
            )
        body = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            message = (body.get("error") or {}).get("message") or resp.text[:200]
            raise StripeError(f"Stripe {method} {path} failed ({resp.status_code}): {message}")
        return body

    async def create_checkout_session(
        self,
        *,
        price_id: str,
        account_id: str,
        email: str | None,
        customer_id: str | None,
        success_url: str,
        cancel_url: str,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "client_reference_id": account_id,
            "metadata": {"account_id": account_id},
            "subscription_data": {"metadata": {"account_id": account_id}},
            "allow_promotion_codes": True,
            "success_url": success_url,
            "cancel_url": cancel_url,
        }
        if customer_id:
            data["customer"] = customer_id
        elif email:
            data["customer_email"] = email
        return await self._req("POST", "/v1/checkout/sessions", data)

    async def get_checkout_session(self, session_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/v1/checkout/sessions/{session_id}")

    async def create_portal_session(self, *, customer_id: str, return_url: str) -> dict[str, Any]:
        return await self._req(
            "POST",
            "/v1/billing_portal/sessions",
            {"customer": customer_id, "return_url": return_url},
        )

    async def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/v1/subscriptions/{subscription_id}")

    async def get_price(self, price_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/v1/prices/{price_id}")


def verify_webhook(
    payload: bytes,
    signature_header: str,
    secret: str,
    *,
    tolerance: int = 300,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Check a Stripe-Signature header and return the parsed event, or raise StripeError."""
    if not secret:
        raise StripeError("STRIPE_WEBHOOK_SECRET is not set")
    timestamp = ""
    signatures: list[str] = []
    for part in (signature_header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)
    if not timestamp or not signatures:
        raise StripeError("malformed Stripe-Signature header")
    signed = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in signatures):
        raise StripeError("Stripe signature mismatch")
    try:
        if abs(now() - int(timestamp)) > tolerance:
            raise StripeError("Stripe signature timestamp outside tolerance")
    except ValueError as e:
        raise StripeError("malformed Stripe-Signature timestamp") from e
    return json.loads(payload)


def sign_payload(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """Build a Stripe-Signature header (for tests and local webhook replays)."""
    ts = int(timestamp if timestamp is not None else time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def price_label(price: dict[str, Any]) -> str:
    """'$9.99/month' from a Stripe price object."""
    amount = price.get("unit_amount")
    if amount is None:
        return ""
    currency = (price.get("currency") or "usd").upper()
    symbol = "$" if currency in ("USD", "CAD") else f"{currency} "
    value = f"{amount / 100:.2f}".removesuffix(".00")
    interval = (price.get("recurring") or {}).get("interval")
    count = (price.get("recurring") or {}).get("interval_count") or 1
    if not interval:
        return f"{symbol}{value}"
    per = interval if count == 1 else f"{count} {interval}s"
    return f"{symbol}{value}/{per}"
