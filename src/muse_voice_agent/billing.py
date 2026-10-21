"""Accounts, API keys and call-time entitlement.

Every MCP request carries an API key. The key's account decides whether a new call may start:
new accounts get a one-time free allowance (FREE_TRIAL_MINUTES), after which they need an active
Stripe subscription whose plan (Starter or Plus) includes a number of minutes per billing period. Usage is the talk time
of the account's calls (see CallStore.billed_seconds), so nothing is charged twice and checking a
call's status is never metered. The MCP_AUTH_TOKEN maps to an unmetered admin account.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Plan, Settings
from .store import ADMIN_ACCOUNT_ID, CallStore

KEY_PREFIX = "mva_"
ACTIVE_STATUSES = ("active", "trialing")
COMP_PLAN = "comp"
ADMIN_PLAN = "admin"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    plan TEXT NOT NULL DEFAULT 'free',
    status TEXT NOT NULL DEFAULT 'none',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    price_id TEXT,
    current_period_start REAL,
    current_period_end REAL,
    cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
    grace_until REAL,
    free_seconds INTEGER NOT NULL DEFAULT 0,
    signup_ip TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS accounts_customer ON accounts (stripe_customer_id);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    hint TEXT NOT NULL,
    created_at REAL NOT NULL,
    revoked_at REAL
);
CREATE TABLE IF NOT EXISTS stripe_events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    received_at REAL NOT NULL
);
"""


class BillingError(Exception):
    """A plain-language reason a new call can't start, safe to show the user."""


@dataclass(frozen=True)
class Account:
    id: str
    email: str | None
    plan: str
    status: str
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    price_id: str | None
    current_period_start: float | None
    current_period_end: float | None
    cancel_at_period_end: bool
    grace_until: float | None
    free_seconds: int
    created_at: float

    @property
    def is_admin(self) -> bool:
        return self.plan == ADMIN_PLAN

    def subscription_active(self, now: float | None = None) -> bool:
        now = now or time.time()
        if self.status in ACTIVE_STATUSES:
            return True
        return self.status == "past_due" and bool(self.grace_until and now < self.grace_until)


@dataclass(frozen=True)
class Usage:
    free_used: float
    free_total: int
    paid_used: float
    paid_total: int | None  # None = unlimited; 0 = no subscription

    @property
    def free_left(self) -> float:
        return max(0.0, self.free_total - self.free_used)

    @property
    def paid_left(self) -> float | None:
        if self.paid_total is None:
            return None
        return max(0.0, self.paid_total - self.paid_used)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def minutes(seconds: float) -> str:
    m = seconds / 60
    return f"{m:.0f}" if m >= 10 or m == int(m) else f"{m:.1f}"


def format_date(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %-d, %Y")


class Billing:
    def __init__(self, settings: Settings, store: CallStore):
        self.settings = settings
        self.store = store
        self.path = Path(store.path)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            c.execute(
                "INSERT OR IGNORE INTO accounts (id, plan, status, created_at, updated_at)"
                " VALUES (?, ?, 'none', ?, ?)",
                (ADMIN_ACCOUNT_ID, ADMIN_PLAN, time.time(), time.time()),
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- accounts -------------------------------------------------------------------------

    def get_account(self, account_id: str) -> Account | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return _account(row) if row else None

    def find_by_email(self, email: str) -> Account | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM accounts WHERE email = ?", (normalize_email(email),)
            ).fetchone()
        return _account(row) if row else None

    def find_by_customer(self, customer_id: str) -> Account | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM accounts WHERE stripe_customer_id = ?", (customer_id,)
            ).fetchone()
        return _account(row) if row else None

    def create_account(
        self,
        email: str,
        *,
        plan: str = "free",
        free_seconds: int | None = None,
        signup_ip: str | None = None,
    ) -> Account:
        account_id = "acct_" + uuid.uuid4().hex[:16]
        now = time.time()
        if free_seconds is None:
            free_seconds = max(0, self.settings.free_trial_minutes) * 60
        with self._conn() as c:
            c.execute(
                "INSERT INTO accounts (id, email, plan, free_seconds, signup_ip, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?)",
                (account_id, normalize_email(email), plan, free_seconds, signup_ip, now, now),
            )
        return self.get_account(account_id)  # type: ignore[return-value]

    def update_account(self, account_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE accounts SET {cols} WHERE id = ?", (*fields.values(), account_id))

    def signups_from_ip(self, ip: str, since: float) -> int:
        with self._conn() as c:
            (n,) = c.execute(
                "SELECT COUNT(*) FROM accounts WHERE signup_ip = ? AND created_at >= ?", (ip, since)
            ).fetchone()
        return int(n)

    def list_accounts(self) -> list[Account]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM accounts ORDER BY created_at").fetchall()
        return [_account(r) for r in rows]

    # --- API keys -------------------------------------------------------------------------

    def issue_key(self, account_id: str, *, revoke_existing: bool = True) -> str:
        """Create a new key for the account (returned once; only its hash is stored)."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        now = time.time()
        with self._conn() as c:
            if revoke_existing:
                c.execute(
                    "UPDATE api_keys SET revoked_at = ? WHERE account_id = ? AND revoked_at IS NULL",
                    (now, account_id),
                )
            c.execute(
                "INSERT INTO api_keys (id, account_id, key_hash, hint, created_at)"
                " VALUES (?,?,?,?,?)",
                (uuid.uuid4().hex[:12], account_id, hash_key(key), key[-4:], now),
            )
        return key

    def revoke_keys(self, account_id: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE account_id = ? AND revoked_at IS NULL",
                (time.time(), account_id),
            )
        return cur.rowcount

    def active_key_id(self, account_id: str) -> str | None:
        """Id of the account's current key; account-page sessions are tied to it."""
        if account_id == ADMIN_ACCOUNT_ID:
            return ADMIN_ACCOUNT_ID
        with self._conn() as c:
            row = c.execute(
                "SELECT id FROM api_keys WHERE account_id = ? AND revoked_at IS NULL"
                " ORDER BY created_at DESC LIMIT 1",
                (account_id,),
            ).fetchone()
        return row["id"] if row else None

    def key_hint(self, account_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT hint FROM api_keys WHERE account_id = ? AND revoked_at IS NULL"
                " ORDER BY created_at DESC LIMIT 1",
                (account_id,),
            ).fetchone()
        return row["hint"] if row else None

    def authenticate(self, supplied: str) -> Account | None:
        """Account for a bearer token: the admin token or an unrevoked API key."""
        supplied = supplied.strip()
        if not supplied:
            return None
        admin = self.settings.mcp_auth_token
        if admin and hmac.compare_digest(supplied.encode(), admin.encode()):
            return self.get_account(ADMIN_ACCOUNT_ID)
        with self._conn() as c:
            row = c.execute(
                "SELECT a.* FROM api_keys k JOIN accounts a ON a.id = k.account_id"
                " WHERE k.key_hash = ? AND k.revoked_at IS NULL",
                (hash_key(supplied),),
            ).fetchone()
        return _account(row) if row else None

    # --- entitlement ----------------------------------------------------------------------

    def plan_for(self, account: Account) -> Plan | None:
        """The account's subscription plan. A price we no longer list counts as the cheapest plan."""
        plan = self.settings.plan_for_price(account.price_id)
        if plan is None and account.stripe_subscription_id and self.settings.plans:
            plan = self.settings.plans[0]
        return plan

    def usage(self, account: Account) -> Usage:
        free_used = self.store.billed_seconds(account.id, "free")
        paid_total: int | None = 0
        paid_used = 0.0
        plan = self.plan_for(account)
        if account.subscription_active() and plan is not None:
            paid_total = plan.minutes * 60 if plan.minutes > 0 else None
            paid_used = self.store.billed_seconds(
                account.id, "paid", since=account.current_period_start
            )
        return Usage(free_used, account.free_seconds, paid_used, paid_total)

    def authorize_call(self, account: Account) -> str | None:
        """Pick the allowance a new call draws on, or raise BillingError. None = unmetered."""
        if account.is_admin:
            return None
        usage = self.usage(account)
        if usage.free_left > 0:
            return "free"
        url = self.settings.billing_url
        if account.subscription_active():
            left = usage.paid_left
            if left is None or left > 0:
                return "paid"
            reset = format_date(account.current_period_end)
            when = f" They reset on {reset}." if reset else ""
            plan = self.plan_for(account)
            bigger = [p for p in self.settings.plans if plan and p.minutes > plan.minutes]
            upgrade = (
                f" Upgrade to {bigger[0].name} ({bigger[0].minutes_label.lower()}) at {url}"
                " to keep calling now — your API key stays the same."
                if bigger
                else f" Manage your plan at {url}."
            )
            raise BillingError(
                f"You've used this period's {minutes(usage.paid_total or 0)} included call minutes"
                f"{f' on the {plan.name} plan' if plan else ''}.{when}{upgrade}"
            )
        if account.status != "none" or account.stripe_subscription_id:
            raise BillingError(
                "Your MuseVoiceAgent subscription isn't active. Renew at "
                f"{url} — your API key stays the same."
            )
        if account.plan == COMP_PLAN:
            raise BillingError(
                f"This complimentary account has used its {minutes(account.free_seconds)} minutes"
                f" of calls. Contact {self.settings.support_email} for more."
            )
        raise BillingError(
            f"You've used your {minutes(account.free_seconds)} free minutes of calls. Subscribe at"
            f" {url} to keep placing calls — your API key stays the same."
        )

    def low_balance_notice(self, account: Account, bucket: str | None) -> str | None:
        """A heads-up for the user when this call will likely use up their allowance."""
        if bucket is None:
            return None
        usage = self.usage(account)
        left = usage.free_left if bucket == "free" else usage.paid_left
        if left is None or left > 120:
            return None
        what = "free minutes" if bucket == "free" else "included minutes for this period"
        return (
            f"About {minutes(left)} of your {what} remain, so this may be the last call for now."
            f" More time: {self.settings.billing_url}"
        )

    # --- Stripe state ---------------------------------------------------------------------

    def has_event(self, event_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM stripe_events WHERE id = ?", (event_id,)).fetchone()
        return row is not None

    def record_event(self, event_id: str, event_type: str) -> bool:
        """Remember a webhook event; False if it was already processed."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO stripe_events (id, type, received_at) VALUES (?,?,?)",
                (event_id, event_type, time.time()),
            )
        return cur.rowcount == 1

    def apply_subscription(self, account_id: str, sub: dict[str, Any]) -> None:
        """Copy a Stripe subscription object's entitlement-relevant fields onto the account."""
        account = self.get_account(account_id)
        if account is None:
            return
        status = sub.get("status") or "none"
        current = account.stripe_subscription_id
        replaces = status in ACTIVE_STATUSES and account.status not in ACTIVE_STATUSES
        if current and sub.get("id") != current and not replaces:
            # An older or duplicate subscription mustn't overwrite the one the account uses.
            return
        start, end = subscription_period(sub)
        fields: dict[str, Any] = {
            "status": status,
            "stripe_subscription_id": sub.get("id"),
            "price_id": subscription_price(sub) or account.price_id,
            "cancel_at_period_end": int(bool(sub.get("cancel_at_period_end"))),
        }
        if account.plan not in (ADMIN_PLAN, COMP_PLAN):
            fields["plan"] = "subscription"
        if isinstance(sub.get("customer"), str):
            fields["stripe_customer_id"] = sub["customer"]
        if start:
            fields["current_period_start"] = start
        if end:
            fields["current_period_end"] = end
        if status in ACTIVE_STATUSES:
            fields["grace_until"] = None
        elif status == "past_due" and not account.grace_until:
            fields["grace_until"] = self.grace_deadline()
        self.update_account(account_id, **fields)

    def grace_deadline(self) -> float:
        return time.time() + max(0, self.settings.billing_grace_days) * 86400


def subscription_period(sub: dict[str, Any]) -> tuple[float | None, float | None]:
    """Billing period of a subscription; newer Stripe API versions keep it on the items."""
    start, end = sub.get("current_period_start"), sub.get("current_period_end")
    if not (start and end):
        items = ((sub.get("items") or {}).get("data")) or []
        if items:
            start = start or items[0].get("current_period_start")
            end = end or items[0].get("current_period_end")
    return (float(start) if start else None, float(end) if end else None)


def subscription_price(sub: dict[str, Any]) -> str | None:
    items = ((sub.get("items") or {}).get("data")) or []
    price = (items[0].get("price") if items else None) or sub.get("plan") or {}
    return price.get("id") if isinstance(price, dict) else price


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _account(row: sqlite3.Row) -> Account:
    return Account(
        id=row["id"],
        email=row["email"],
        plan=row["plan"],
        status=row["status"],
        stripe_customer_id=row["stripe_customer_id"],
        stripe_subscription_id=row["stripe_subscription_id"],
        price_id=row["price_id"],
        current_period_start=row["current_period_start"],
        current_period_end=row["current_period_end"],
        cancel_at_period_end=bool(row["cancel_at_period_end"]),
        grace_until=row["grace_until"],
        free_seconds=int(row["free_seconds"]),
        created_at=row["created_at"],
    )
