import json
import time
from dataclasses import replace

import httpx
import pytest
from starlette.testclient import TestClient

from muse_voice_agent import dispatcher
from muse_voice_agent.billing import Billing, BillingError
from muse_voice_agent.billing_web import SESSION_COOKIE, BillingWeb
from muse_voice_agent.config import Settings
from muse_voice_agent.mcp_server import build_app
from muse_voice_agent.store import ADMIN_ACCOUNT_ID, CallStore
from muse_voice_agent.stripe_api import (
    StripeClient,
    StripeError,
    _flatten,
    price_label,
    sign_payload,
    verify_webhook,
)

ADMIN = "admin-token"
WHSEC = "whsec_test"
CALL_ARGS = {
    "business_name": "Luigi's",
    "phone_number": "+14155550123",
    "customer_name": "Angi",
    "goal": "Ask what time they close today.",
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        dry_run=True,
        mcp_auth_token=ADMIN,
        call_db_path=tmp_path / "calls.db",
        public_base_url="https://voice.example",
        stripe_secret_key="sk_test_x",
        stripe_webhook_secret=WHSEC,
        stripe_price_starter="price_starter",
        stripe_price_plus="price_plus",
        free_trial_minutes=10,
        starter_minutes=50,
        plus_minutes=200,
        billing_grace_days=3,
        max_concurrent_calls=5,
        max_concurrent_calls_per_account=2,
        signups_per_ip_per_day=3,
    )


@pytest.fixture
def store(settings) -> CallStore:
    return CallStore(settings.call_db_path)


@pytest.fixture
def billing(settings, store) -> Billing:
    return Billing(settings, store)


@pytest.fixture(autouse=True)
def fast_simulation(monkeypatch):
    original = dispatcher._simulate_call

    async def fast(call_id, task, store, delay=0.02):
        await original(call_id, task, store, delay=delay)

    monkeypatch.setattr(dispatcher, "_simulate_call", fast)


def _talk(store: CallStore, account_id: str, bucket: str, seconds: float, *, ago: float = 0) -> str:
    """Record a finished call with `seconds` of talk time."""
    record = store.create_call(
        {"kind": "general", "business_name": "B", "phone_number": "+14155550100"},
        dry_run=True,
        account_id=account_id,
        billing_bucket=bucket,
    )
    end = time.time() - ago
    store.update_call(record["id"], status="completed", started_at=end - seconds, ended_at=end)
    return record["id"]


class MCP:
    """Speaks raw streamable-HTTP JSON-RPC to the real app, like Muse does."""

    def __init__(self, client: TestClient, token: str):
        self.client = client
        self.token = token
        self.session_id: str | None = None
        self.n = 0

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def post(self, method: str, params: dict | None = None) -> httpx.Response:
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
        return self.client.post("/mcp", headers=self._headers(), json=msg)

    def connect(self) -> "MCP":
        r = self.post(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "muse", "version": "1"},
            },
        )
        assert r.status_code == 200, r.text
        self.session_id = r.headers.get("mcp-session-id")
        self.client.post(
            "/mcp",
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        return self

    def tool(self, name: str, args: dict | None = None) -> dict:
        r = self.post("tools/call", {"name": name, "arguments": args or {}})
        assert r.status_code == 200, r.text
        events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
        result = (events[-1] if events else r.json())["result"]
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])


@pytest.fixture
def http(settings, store):
    with TestClient(build_app(settings, store), base_url="https://voice.example") as client:
        yield client


# --- keys and auth ---------------------------------------------------------------------------


def test_unknown_missing_and_revoked_keys_get_401(http, billing):
    account = billing.create_account("a@example.com")
    old = billing.issue_key(account.id)
    billing.issue_key(account.id)  # rotating revokes the old key
    for headers in ({}, {"Authorization": "Bearer mva_nope"}, {"Authorization": f"Bearer {old}"}):
        r = http.post("/mcp", headers=headers, json={})
        assert r.status_code == 401
        assert "https://voice.example/account" in r.json()["message"]


def test_keys_are_stored_hashed(billing, settings):
    account = billing.create_account("a@example.com")
    key = billing.issue_key(account.id)
    assert key.startswith("mva_") and len(key) > 40
    assert key.encode() not in settings.call_db_path.read_bytes()
    assert billing.authenticate(key).id == account.id
    assert billing.authenticate(ADMIN).id == ADMIN_ACCOUNT_ID
    assert billing.authenticate("") is None


def test_admin_token_still_places_unmetered_calls(http, store):
    mcp = MCP(http, ADMIN).connect()
    started = mcp.tool("place_call", CALL_ARGS)
    assert not started.get("error")
    record = store.get_call(started["call_id"])
    assert record["account_id"] == ADMIN_ACCOUNT_ID
    assert record["billing_bucket"] is None


# --- free trial and subscription gating ------------------------------------------------------


def test_new_key_gets_free_minutes_then_payment_required(http, billing, store):
    account = billing.create_account("a@example.com")
    mcp = MCP(http, billing.issue_key(account.id)).connect()

    started = mcp.tool("place_call", CALL_ARGS)
    assert not started.get("error")
    assert store.get_call(started["call_id"])["billing_bucket"] == "free"

    _talk(store, account.id, "free", 600)
    before = len(store.list_calls(50))
    blocked = mcp.tool("place_call", CALL_ARGS)
    assert blocked["error"] == "payment_required"
    assert "10 free minutes" in blocked["message"]
    assert blocked["billing_url"] == "https://voice.example/account"
    assert len(store.list_calls(50)) == before  # nothing was dialed

    # Past calls stay readable after the allowance runs out.
    status = mcp.tool("get_call_status", {"call_id": started["call_id"]})
    assert status["call_id"] == started["call_id"]
    assert mcp.tool("list_calls")["calls"]


def test_low_balance_notice(http, billing, store):
    account = billing.create_account("a@example.com")
    _talk(store, account.id, "free", 540)
    mcp = MCP(http, billing.issue_key(account.id)).connect()
    started = mcp.tool("place_call", CALL_ARGS)
    assert "1 of your free minutes" in started["billing_notice"]


def test_usage_counts_talk_time_only(billing, store):
    account = billing.create_account("a@example.com")
    unanswered = store.create_call(
        {"kind": "general", "business_name": "B", "phone_number": "+14155550100"},
        dry_run=True, account_id=account.id, billing_bucket="free",
    )
    store.update_call(unanswered["id"], status="no_answer")
    live = store.create_call(
        {"kind": "general", "business_name": "B", "phone_number": "+14155550100"},
        dry_run=True, account_id=account.id, billing_bucket="free",
    )
    store.update_call(live["id"], status="in_progress", started_at=time.time() - 90)
    _talk(store, account.id, "free", 30)
    _talk(store, "acct_other", "free", 500)
    assert 119 < billing.usage(account).free_used < 125


def _subscribe(
    billing, account, *, status="active", period_start=None, grace_until=None, price="price_starter"
):
    start = period_start or time.time() - 86400
    billing.update_account(
        account.id,
        status=status,
        plan="subscription",
        stripe_customer_id="cus_1",
        stripe_subscription_id="sub_1",
        price_id=price,
        current_period_start=start,
        current_period_end=start + 30 * 86400,
        grace_until=grace_until,
    )
    return billing.get_account(account.id)


def test_subscription_unlocks_paid_minutes_until_period_used(billing, store):
    account = billing.create_account("a@example.com")
    _talk(store, account.id, "free", 600)
    with pytest.raises(BillingError):
        billing.authorize_call(account)

    account = _subscribe(billing, account)
    assert billing.authorize_call(account) == "paid"

    _talk(store, account.id, "paid", 50 * 60)
    with pytest.raises(BillingError, match="this period's 50 included call minutes on the Starter plan") as e:
        billing.authorize_call(account)
    assert "Upgrade to Plus (200 call minutes a month)" in str(e.value)

    # Upgrading keeps the period, so the bigger allowance applies right away.
    account = _subscribe(billing, account, price="price_plus")
    assert billing.authorize_call(account) == "paid"
    _talk(store, account.id, "paid", 150 * 60)
    with pytest.raises(BillingError, match="200 included call minutes on the Plus plan") as e:
        billing.authorize_call(account)
    assert "Upgrade" not in str(e.value)

    # Usage from before the new period doesn't count against it.
    account = _subscribe(billing, account, period_start=time.time() + 1)
    assert billing.authorize_call(account) == "paid"


def test_unlimited_plan(settings, store):
    billing = Billing(replace(settings, starter_minutes=0), store)
    account = _subscribe(billing, billing.create_account("a@example.com"), )
    _talk(store, account.id, "free", 600)
    _talk(store, account.id, "paid", 10_000)
    assert billing.authorize_call(account) == "paid"


def test_past_due_grace_then_inactive(billing, store):
    account = billing.create_account("a@example.com")
    _talk(store, account.id, "free", 600)
    account = _subscribe(billing, account, status="past_due", grace_until=time.time() + 3600)
    assert billing.authorize_call(account) == "paid"
    account = _subscribe(billing, account, status="past_due", grace_until=time.time() - 1)
    with pytest.raises(BillingError, match="subscription isn't active"):
        billing.authorize_call(account)
    account = _subscribe(billing, account, status="canceled")
    with pytest.raises(BillingError, match="your API key stays the same"):
        billing.authorize_call(account)


def test_comp_account(billing, store):
    account = billing.create_account("reviewer@example.com", plan="comp", free_seconds=1800)
    _talk(store, account.id, "free", 1700)
    assert billing.authorize_call(account) == "free"
    _talk(store, account.id, "free", 100)
    with pytest.raises(BillingError, match="complimentary account"):
        billing.authorize_call(account)


def test_per_account_concurrency(http, billing, store):
    account = billing.create_account("a@example.com")
    for _ in range(2):
        r = store.create_call(
            {"kind": "general", "business_name": "B", "phone_number": "+14155550100"},
            dry_run=True, account_id=account.id, billing_bucket="free",
        )
        store.update_call(r["id"], status="in_progress")
    mcp = MCP(http, billing.issue_key(account.id)).connect()
    rejected = mcp.tool("place_call", CALL_ARGS)
    assert rejected["error"] == "rejected"
    assert "2 calls in progress" in rejected["message"]


# --- isolation -------------------------------------------------------------------------------


def test_accounts_only_see_their_own_calls(http, billing, store):
    a = billing.create_account("a@example.com")
    b = billing.create_account("b@example.com")
    mcp_a = MCP(http, billing.issue_key(a.id)).connect()
    mcp_b = MCP(http, billing.issue_key(b.id)).connect()
    admin = MCP(http, ADMIN).connect()

    call_a = mcp_a.tool("place_call", CALL_ARGS)["call_id"]
    assert mcp_b.tool("get_call_status", {"call_id": call_a})["error"] == "not_found"
    assert admin.tool("get_call_status", {"call_id": call_a})["error"] == "not_found"
    assert mcp_b.tool("list_calls")["calls"] == []
    assert [c["call_id"] for c in mcp_a.tool("list_calls")["calls"]] == [call_a]

    # A business calling back about A's call reaches A only.
    callback = store.create_call(
        {"kind": "general", "business_name": "Luigi's", "phone_number": "+14155550123"},
        dry_run=False, direction="inbound", parent_call_id=call_a, status="completed",
        origin="mcp:muse", account_id=a.id,
    )
    store.update_call(callback["id"], ended_at=time.time())
    assert mcp_b.tool("get_updates")["updates"] == []
    assert [u["callback"]["call_id"] for u in mcp_a.tool("get_updates")["updates"]] == [
        callback["id"]
    ]


def test_existing_calls_migrate_to_admin(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    CallStore(db)
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO calls (id, kind, business_name, phone_number, task_json, status,"
            " created_at, updated_at) VALUES ('old1','general','B','+1','{}','completed',1,1)"
        )
        c.execute("UPDATE calls SET account_id = NULL")
    assert CallStore(db).get_call("old1")["account_id"] == ADMIN_ACCOUNT_ID


async def test_inbound_callback_inherits_account(settings, store):
    from muse_voice_agent.retell import RetellLLMSession

    parent = store.create_call(
        {"kind": "general", "business_name": "Luigi's", "phone_number": "+14155550123"},
        dry_run=False, account_id="acct_a", billing_bucket="free", origin="mcp:muse",
    )
    store.update_call(parent["id"], status="completed")
    session = RetellLLMSession(None, "rc_in", store, settings=settings)  # type: ignore[arg-type]
    record = await session._create_inbound_call(
        {"from_number": "+14155550123", "to_number": "+16282779475"}
    )
    assert record["parent_call_id"] == parent["id"]
    assert record["account_id"] == "acct_a"
    assert record["billing_bucket"] is None  # callbacks aren't metered
    stranger = await session._create_inbound_call(
        {"from_number": "+14155559999", "to_number": "+16282779475"}
    )
    assert stranger["account_id"] == ADMIN_ACCOUNT_ID



async def test_callback_to_number_shared_by_two_accounts_isnt_attributed(settings, store):
    from muse_voice_agent.retell import RetellLLMSession

    for account_id in ("acct_a", "acct_b"):
        call = store.create_call(
            {"kind": "general", "business_name": "Luigi's", "phone_number": "+14155550123"},
            dry_run=False, account_id=account_id, billing_bucket="free", origin="mcp:muse",
        )
        store.update_call(call["id"], status="completed")
    session = RetellLLMSession(None, "rc_in", store, settings=settings)  # type: ignore[arg-type]
    record = await session._create_inbound_call(
        {"from_number": "+14155550123", "to_number": "+16282779475"}
    )
    assert record["account_id"] == ADMIN_ACCOUNT_ID
    assert not record.get("parent_call_id")


# --- Stripe ----------------------------------------------------------------------------------


def _subscription(status="active", **extra) -> dict:
    now = int(time.time())
    return {
        "id": "sub_1",
        "object": "subscription",
        "customer": "cus_1",
        "status": status,
        "cancel_at_period_end": False,
        "metadata": {},
        "items": {"data": [{
            "current_period_start": now,
            "current_period_end": now + 30 * 86400,
            "price": {"id": "price_starter"},
        }]},
        **extra,
    }


class FakeStripe:
    def __init__(self):
        self.subscription = _subscription()
        self.requests: list[httpx.Request] = []
        self.checkout_session: dict = {}
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            return httpx.Response(500, json={"error": {"message": "down"}})
        path = request.url.path
        if path == "/v1/subscriptions/sub_1":
            return httpx.Response(200, json=self.subscription)
        if path == "/v1/checkout/sessions" and request.method == "POST":
            return httpx.Response(200, json={"id": "cs_1", "url": "https://checkout.stripe.test/cs_1"})
        if path.startswith("/v1/checkout/sessions/"):
            return httpx.Response(200, json=self.checkout_session)
        if path == "/v1/billing_portal/sessions":
            return httpx.Response(200, json={"url": "https://billing.stripe.test/p"})
        if path in ("/v1/prices/price_starter", "/v1/prices/price_plus"):
            amount = 500 if path.endswith("starter") else 2000
            return httpx.Response(
                200, json={"unit_amount": amount, "currency": "usd", "recurring": {"interval": "month"}}
            )
        return httpx.Response(404, json={"error": {"message": "nope"}})

    def client(self) -> StripeClient:
        return StripeClient("sk_test_x", transport=httpx.MockTransport(self.handler))


@pytest.fixture
def fake_stripe() -> FakeStripe:
    return FakeStripe()


@pytest.fixture
def web(settings, billing, fake_stripe):
    app = BillingWeb(settings, billing, fake_stripe.client()).app()
    return TestClient(app, base_url="https://voice.example")


def _event(web, kind: str, obj: dict, event_id: str = "evt_1", secret: str = WHSEC):
    payload = json.dumps({"id": event_id, "type": kind, "data": {"object": obj}}).encode()
    return web.post(
        "/stripe/webhook",
        content=payload,
        headers={"Stripe-Signature": sign_payload(payload, secret)},
    )


def test_webhook_rejects_bad_signature(web):
    r = _event(web, "checkout.session.completed", {}, secret="whsec_wrong")
    assert r.status_code == 400


def test_checkout_completed_activates_subscription(web, billing, store):
    account = billing.create_account("a@example.com")
    _talk(store, account.id, "free", 600)
    r = _event(
        web,
        "checkout.session.completed",
        {"id": "cs_1", "client_reference_id": account.id, "customer": "cus_1", "subscription": "sub_1"},
    )
    assert r.status_code == 200
    account = billing.get_account(account.id)
    assert (account.status, account.stripe_customer_id, account.stripe_subscription_id) == (
        "active", "cus_1", "sub_1",
    )
    assert account.current_period_end and account.current_period_start
    assert billing.authorize_call(account) == "paid"
    # Redelivered events are ignored.
    assert _event(web, "checkout.session.completed", {}).json()["duplicate"] is True


def test_payment_failed_then_deleted(web, billing, fake_stripe, store):
    account = billing.create_account("a@example.com")
    _talk(store, account.id, "free", 600)
    billing.update_account(account.id, stripe_customer_id="cus_1")

    fake_stripe.subscription = _subscription("past_due")
    _event(web, "invoice.payment_failed",
           {"customer": "cus_1", "parent": {"subscription_details": {"subscription": "sub_1"}}},
           event_id="evt_2")
    account = billing.get_account(account.id)
    assert account.status == "past_due"
    assert 2.9 * 86400 < account.grace_until - time.time() <= 3 * 86400
    assert billing.authorize_call(account) == "paid"

    fake_stripe.subscription = _subscription("active")
    _event(web, "invoice.paid", {"customer": "cus_1", "subscription": "sub_1"}, event_id="evt_3")
    account = billing.get_account(account.id)
    assert account.status == "active" and account.grace_until is None

    fake_stripe.subscription = _subscription("canceled")
    _event(web, "customer.subscription.deleted", _subscription("active"), event_id="evt_4")
    account = billing.get_account(account.id)
    assert account.status == "canceled"  # re-read from Stripe, not the (stale) event payload
    with pytest.raises(BillingError, match="isn't active"):
        billing.authorize_call(account)


def test_webhook_asks_stripe_to_retry_on_api_failure(web, billing, fake_stripe):
    account = billing.create_account("a@example.com")
    billing.update_account(account.id, stripe_customer_id="cus_1")
    fake_stripe.down = True
    r = _event(web, "invoice.paid", {"customer": "cus_1", "subscription": "sub_1"}, event_id="evt_9")
    assert r.status_code == 500
    assert not billing.has_event("evt_9")


# --- account pages ---------------------------------------------------------------------------


def test_old_or_second_subscription_doesnt_override_current(billing):
    account = billing.create_account("a@example.com")
    billing.apply_subscription(account.id, _subscription())
    billing.apply_subscription(account.id, _subscription("canceled", id="sub_old"))
    account = billing.get_account(account.id)
    assert (account.stripe_subscription_id, account.status) == ("sub_1", "active")

    # A plan switch on the current subscription changes the allowance.
    plus = _subscription()
    plus["items"]["data"][0]["price"] = {"id": "price_plus"}
    billing.apply_subscription(account.id, plus)
    account = billing.get_account(account.id)
    assert account.price_id == "price_plus" and billing.usage(account).paid_total == 200 * 60

    # Once the current one lapses, a new active subscription takes over.
    billing.apply_subscription(account.id, _subscription("canceled"))
    billing.apply_subscription(account.id, _subscription(id="sub_new"))
    account = billing.get_account(account.id)
    assert (account.stripe_subscription_id, account.status, account.price_id) == (
        "sub_new", "active", "price_starter",
    )


def test_unknown_price_counts_as_cheapest_plan(billing):
    account = billing.create_account("a@example.com")
    sub = _subscription()
    sub["items"]["data"][0]["price"] = {"id": "price_retired"}
    billing.apply_subscription(account.id, sub)
    assert billing.usage(billing.get_account(account.id)).paid_total == 50 * 60


def test_signup_shows_key_once_and_signs_in(web, billing):
    r = web.post("/signup", data={"email": "New@Example.com"})
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    key = r.text.split('id="key">')[1].split("<")[0]
    account = billing.authenticate(key)
    assert account and account.email == "new@example.com" and account.free_seconds == 600
    assert "https://voice.example/mcp" in r.text
    assert SESSION_COOKIE in r.cookies

    page = web.get("/account")
    assert "10 of 10 minutes left" in page.text
    assert "Starter: $5/month" in page.text
    assert key not in page.text
    assert web.post("/signup", data={"email": "new@example.com"}).status_code == 409


def test_signup_limits_and_validation(web, settings):
    assert web.post("/signup", data={"email": "nope"}).status_code == 400
    for i in range(3):
        assert web.post("/signup", data={"email": f"u{i}@example.com"}).status_code == 200
    assert web.post("/signup", data={"email": "u9@example.com"}).status_code == 429


def test_signup_limit_uses_proxy_appended_address(web):
    # Only the last X-Forwarded-For entry comes from our proxy; a forged first entry doesn't help.
    for i in range(4):
        r = web.post(
            "/signup",
            data={"email": f"x{i}@example.com"},
            headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"},
        )
    assert r.status_code == 429
    r = web.post("/signup", data={"email": "y@example.com"}, headers={"X-Forwarded-For": "203.0.113.8"})
    assert r.status_code == 200


def test_signup_closed_without_stripe(settings, billing):
    app = BillingWeb(replace(settings, stripe_secret_key=""), billing).app()
    client = TestClient(app)
    assert client.post("/signup", data={"email": "a@example.com"}).status_code == 503
    assert "Sign-ups open soon" in client.get("/pricing").text


def test_pricing_page_shows_stripe_price(web):
    page = web.get("/pricing").text
    assert "$5/month" in page and "$20/month" in page
    assert "10 free minutes" in page or "10 minutes of calls" in page
    assert "50 call minutes a month" in page and "200 call minutes a month" in page


def test_login_rotate_and_logout(web, billing):
    account = billing.create_account("a@example.com")
    key = billing.issue_key(account.id)
    assert web.post("/account/login", data={"api_key": "mva_bad"}).status_code == 401
    r = web.post("/account/login", data={"api_key": key}, follow_redirects=False)
    assert r.status_code == 303
    r = web.post("/account/rotate-key")
    new_key = r.text.split('id="key">')[1].split("<")[0]
    assert billing.authenticate(key) is None
    assert billing.authenticate(new_key).id == account.id
    web.post("/account/logout")
    assert "Sign in" in web.get("/account").text


def test_rotating_or_revoking_a_key_signs_out_other_browsers(settings, billing, fake_stripe):
    account = billing.create_account("a@example.com")
    key = billing.issue_key(account.id)
    app = BillingWeb(settings, billing, fake_stripe.client()).app()
    laptop = TestClient(app, base_url="https://voice.example")
    phone = TestClient(app, base_url="https://voice.example")
    laptop.post("/account/login", data={"api_key": key})
    phone.post("/account/login", data={"api_key": key})

    r = laptop.post("/account/rotate-key")
    assert r.status_code == 200
    assert "Sign out" in laptop.get("/account").text  # the browser that rotated stays signed in
    assert "Sign in" in phone.get("/account").text

    billing.revoke_keys(account.id)
    assert "Sign in" in laptop.get("/account").text
    r = laptop.post("/account/rotate-key", follow_redirects=False)
    assert r.status_code == 303 and billing.active_key_id(account.id) is None


def test_tampered_session_cookie_is_ignored(web, billing):
    account = billing.create_account("a@example.com")
    web.cookies.set(SESSION_COOKIE, f"{account.id}.9999999999.deadbeef")
    assert "Sign in" in web.get("/account").text


def test_checkout_and_portal_redirect_to_stripe(web, billing, fake_stripe):
    account = billing.create_account("a@example.com")
    web.post("/account/login", data={"api_key": billing.issue_key(account.id)})
    page = web.get("/account").text
    assert "Starter: $5/month for 50 call minutes a month" in page
    assert "Plus: $20/month for 200 call minutes a month" in page
    r = web.post("/account/checkout", data={"plan": "bogus"}, follow_redirects=False)
    assert r.headers["location"] == "/pricing"
    r = web.post("/account/checkout", data={"plan": "plus"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "https://checkout.stripe.test/cs_1"
    sent = dict(httpx.QueryParams(fake_stripe.requests[-1].content.decode()))
    assert sent["mode"] == "subscription"
    assert sent["line_items[0][price]"] == "price_plus"
    assert sent["client_reference_id"] == account.id
    assert sent["customer_email"] == "a@example.com"
    assert sent["success_url"].endswith("session_id={CHECKOUT_SESSION_ID}")

    # Returning from Checkout applies the subscription without waiting for the webhook.
    fake_stripe.checkout_session = {
        "id": "cs_1", "client_reference_id": account.id, "customer": "cus_1", "subscription": "sub_1"
    }
    r = web.get("/account/checkout/return?session_id=cs_1", follow_redirects=False)
    assert r.headers["location"] == "/account?checkout=success"
    assert billing.get_account(account.id).status == "active"
    page = web.get("/account").text
    assert "Change plan, billing or cancel" in page and "/month for" not in page

    r = web.post("/account/portal", follow_redirects=False)
    assert r.headers["location"] == "https://billing.stripe.test/p"
    # A second checkout goes to the portal instead of starting another subscription.
    r = web.post("/account/checkout", data={"plan": "plus"}, follow_redirects=False)
    assert r.headers["location"] == "https://billing.stripe.test/p"


def test_web_pages_skip_bearer_auth_but_mcp_does_not(http):
    assert http.get("/pricing").status_code == 200
    assert http.get("/account").status_code == 200
    assert http.post("/stripe/webhook", content=b"{}").status_code == 400  # signature, not 401
    assert http.post("/mcp", json={}).status_code == 401


# --- stripe_api helpers ----------------------------------------------------------------------


def test_flatten_and_price_label():
    assert _flatten({"a": {"b": [{"c": 1}], "d": True}, "e": None}) == [
        ("a[b][0][c]", "1"),
        ("a[d]", "true"),
    ]
    assert price_label({"unit_amount": 900, "currency": "usd", "recurring": {"interval": "month"}}) == "$9/month"
    assert price_label({"unit_amount": 1999, "currency": "eur", "recurring": {"interval": "year", "interval_count": 2}}) == "EUR 19.99/2 years"


def test_verify_webhook_tolerance():
    payload = b'{"id":"evt"}'
    header = sign_payload(payload, WHSEC, timestamp=1000)
    assert verify_webhook(payload, header, WHSEC, now=lambda: 1100)["id"] == "evt"
    with pytest.raises(StripeError, match="tolerance"):
        verify_webhook(payload, header, WHSEC, now=lambda: 2000)
    with pytest.raises(StripeError):
        verify_webhook(payload, "garbage", WHSEC)


# --- admin CLI -------------------------------------------------------------------------------


def test_cli(settings, billing, monkeypatch, capsys):
    from muse_voice_agent import billing_cli

    monkeypatch.setattr(billing_cli, "get_settings", lambda: settings)
    assert billing_cli.main(["create-comp", "--email", "R@example.com", "--minutes", "45"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    account = billing.authenticate(key)
    assert account.plan == "comp" and account.free_seconds == 2700
    assert billing_cli.main(["create-comp", "--email", "r@example.com", "--minutes", "5"]) == 1

    assert billing_cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "r@example.com" in out and "free 0/45 min" in out and "admin" not in out

    assert billing_cli.main(["issue-key", "--email", "r@example.com"]) == 0
    new_key = capsys.readouterr().out.strip()
    assert billing.authenticate(key) is None and billing.authenticate(new_key)
    assert billing_cli.main(["revoke", "--email", "r@example.com"]) == 0
    assert billing.authenticate(new_key) is None
    assert billing_cli.main(["revoke", "--email", "nobody@example.com"]) == 1
