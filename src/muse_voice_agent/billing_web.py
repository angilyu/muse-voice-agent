"""Account pages (sign up, pricing, API key, subscribe, manage) and the Stripe webhook.

These are served by the MCP server itself, next to /mcp, because they need the billing database.
Sign-in is with the API key; there are no passwords. The static product site links here.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import re
import time
from typing import Any
from urllib.parse import parse_qs

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from .billing import Account, Billing, format_date, minutes
from .config import Plan, Settings
from .stripe_api import StripeClient, StripeError, price_label, verify_webhook

logger = logging.getLogger("muse_voice_agent.billing")

SESSION_COOKIE = "mva_session"
SESSION_DAYS = 30
SITE_URL = "https://angilyu.github.io/muse-voice-agent/"
WEB_PATHS = ("/pricing", "/signup", "/account", "/stripe/webhook")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_web_path(path: str) -> bool:
    return path in WEB_PATHS or path.startswith("/account/")


_CSS = """
:root{--ink:#1d1b2e;--muted:#5b5872;--brand:#4f46e5;--teal:#0d9488;--line:#e6e4f0;--bg:#f7f7fb}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
color:var(--ink);background:var(--bg)}header{background:#fff;border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;justify-content:space-between;padding:14px 20px}
.wrap{max-width:760px;margin:0 auto;padding:0 20px}main.wrap{padding:28px 20px 60px}
a{color:var(--brand)}.brand{font-weight:700;text-decoration:none;color:var(--ink)}
nav a{margin-left:16px;text-decoration:none}h1{font-size:1.8rem;margin:.2em 0 .4em}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0}
.plans{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px}
.price{font-size:1.6rem;font-weight:700}.muted{color:var(--muted)}
input[type=email],input[type=text],input[type=password]{width:100%;padding:11px 12px;border:1px solid
var(--line);border-radius:10px;font:inherit}button,.btn{display:inline-block;background:var(--brand);
color:#fff;border:0;border-radius:10px;padding:11px 18px;font:inherit;font-weight:600;cursor:pointer;
text-decoration:none}button.secondary{background:#fff;color:var(--brand);border:1px solid var(--brand)}
form.inline{display:inline}.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
code.key{display:block;word-break:break-all;background:#f1f0fa;padding:12px;border-radius:10px;
font-size:.95rem}.notice{background:#ecfdf5;border-color:#a7f3d0}.warn{background:#fff7ed;
border-color:#fed7aa}.meter{height:10px;background:#eceaf6;border-radius:6px;overflow:hidden}
.meter span{display:block;height:100%;background:linear-gradient(90deg,var(--brand),var(--teal))}
dt{font-weight:600;margin-top:10px}dd{margin:2px 0 0}
"""


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Muse Voice Agent</title>
<link rel="icon" href="{SITE_URL}icon.svg"><style>{_CSS}</style></head><body>
<header><div class="wrap"><a class="brand" href="{SITE_URL}">Muse Voice Agent</a>
<nav><a href="/pricing">Pricing</a><a href="/account">Account</a><a href="{SITE_URL}tools.html">Docs</a>
</nav></div></header><main class="wrap">{body}</main></body></html>"""
    return HTMLResponse(doc, status_code=status_code, headers={"Cache-Control": "no-store"})


def _e(value: Any) -> str:
    return html.escape(str(value or ""))


class BillingWeb:
    def __init__(self, settings: Settings, billing: Billing, stripe: StripeClient | None = None):
        self.settings = settings
        self.billing = billing
        self.stripe = stripe or StripeClient(settings.stripe_secret_key)
        secret = settings.billing_session_secret or hmac.new(
            settings.mcp_auth_token.encode(), b"account-session", hashlib.sha256
        ).hexdigest()
        self._secret = secret.encode()
        self._prices: dict[str, tuple[float, str]] = {}

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/pricing", self.pricing, methods=["GET"]),
                Route("/signup", self.signup, methods=["POST"]),
                Route("/account", self.account, methods=["GET"]),
                Route("/account/login", self.login, methods=["POST"]),
                Route("/account/logout", self.logout, methods=["POST"]),
                Route("/account/checkout", self.checkout, methods=["POST"]),
                Route("/account/checkout/return", self.checkout_return, methods=["GET"]),
                Route("/account/portal", self.portal, methods=["POST"]),
                Route("/account/rotate-key", self.rotate_key, methods=["POST"]),
                Route("/stripe/webhook", self.webhook, methods=["POST"]),
            ]
        )

    # --- helpers --------------------------------------------------------------------------

    def _base(self, request: Request) -> str:
        return self.settings.public_base_url or str(request.base_url).rstrip("/")

    def _sign(self, value: str) -> str:
        return hmac.new(self._secret, value.encode(), hashlib.sha256).hexdigest()

    def _set_session(self, response: Response, account_id: str, request: Request) -> None:
        # The session names the account's current key, so rotating or revoking it signs out
        # every browser that signed in with the old one.
        key_id = self.billing.active_key_id(account_id) or ""
        expires = int(time.time()) + SESSION_DAYS * 86400
        value = f"{account_id}.{key_id}.{expires}"
        response.set_cookie(
            SESSION_COOKIE,
            f"{value}.{self._sign(value)}",
            max_age=SESSION_DAYS * 86400,
            httponly=True,
            secure=self._base(request).startswith("https://"),
            samesite="lax",
        )

    def _session_account(self, request: Request) -> Account | None:
        raw = request.cookies.get(SESSION_COOKIE, "")
        value, _, sig = raw.rpartition(".")
        parts = value.split(".")
        if len(parts) != 3 or not (parts[0] and parts[1] and parts[2].isdigit() and sig):
            return None
        account_id, key_id, expires = parts
        if not hmac.compare_digest(sig, self._sign(value)) or int(expires) < time.time():
            return None
        if self.billing.active_key_id(account_id) != key_id:
            return None
        return self.billing.get_account(account_id)

    async def _form(self, request: Request) -> dict[str, str]:
        body = (await request.body()).decode("utf-8", "replace")
        return {k: v[0] for k, v in parse_qs(body).items()}

    def _client_ip(self, request: Request) -> str:
        # Proxies append the address they saw, so only the last entry can't be forged.
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[-1].strip()
        return request.client.host if request.client else ""

    async def _price_label(self, plan: Plan) -> str:
        cached = self._prices.get(plan.price_id)
        if cached and time.time() - cached[0] < 600:
            return cached[1]
        label = ""
        if self.settings.stripe_secret_key:
            try:
                label = price_label(await self.stripe.get_price(plan.price_id))
            except StripeError as e:
                logger.warning("couldn't load Stripe price for %s: %s", plan.key, e)
        label = label or plan.price_hint
        self._prices[plan.price_id] = (time.time(), label)
        return label

    # --- pages ----------------------------------------------------------------------------

    async def pricing(self, request: Request) -> Response:
        free = self.settings.free_trial_minutes
        cards = ""
        for plan in self.settings.plans:
            cards += f'''<div class="card"><h2>{_e(plan.name)}</h2>
<div class="price">{_e(await self._price_label(plan))}</div>
<p>{_e(plan.minutes_label)}.</p>
</div>'''
        if self.settings.billing_configured:
            signup = """<form method="post" action="/signup"><label for="email">Email</label>
<input id="email" type="email" name="email" required autocomplete="email" placeholder="you@example.com">
<div class="row"><button type="submit">Create account and get API key</button></div></form>
<p class="muted">Already have a key? <a href="/account">Sign in</a>.</p>"""
        else:
            signup = '<p class="muted">Sign-ups open soon.</p>'
        body = f"""<h1>Pricing</h1>
<p>Start with {free} free minutes of calls. No card needed. Subscribe when you want to keep calling.</p>
<div class="plans">
<div class="card"><h2>Free trial</h2><div class="price">$0</div>
<p>{free} minutes of calls to try it, one time per account.</p></div>
{cards}</div>
<div class="card"><h2>Get started</h2>{signup}</div>
<p class="muted">Minutes count talk time, from when the business answers to hang-up. Calls nobody
answers are free, and so are callbacks from businesses. Checking a call's status or results is
never charged. Unused minutes don't roll over. Switch plans or cancel anytime from your account; a canceled plan stays active until the end of the paid period. Payments are processed by Stripe. See the
<a href="{SITE_URL}terms.html">terms</a> and <a href="{SITE_URL}privacy.html">privacy policy</a>.</p>"""
        return _page("Pricing", body)

    async def signup(self, request: Request) -> Response:
        if not self.settings.billing_configured:
            return _page("Sign up", "<h1>Sign-ups open soon</h1>", 503)
        email = (await self._form(request)).get("email", "").strip()
        if not _EMAIL.match(email) or len(email) > 254:
            return _page("Sign up", "<h1>Check your email</h1><p>That email address doesn't look "
                         'right. <a href="/pricing">Try again</a>.</p>', 400)
        if self.billing.find_by_email(email):
            return _page("Sign up", f"""<h1>You already have an account</h1>
<p><a href="/account">Sign in with your API key</a>. Lost it? Email
<a href="mailto:{_e(self.settings.support_email)}">{_e(self.settings.support_email)}</a> from
{_e(email)} and we'll send a new one.</p>""", 409)
        ip = self._client_ip(request)
        limit = self.settings.signups_per_ip_per_day
        if ip and limit and self.billing.signups_from_ip(ip, time.time() - 86400) >= limit:
            return _page("Sign up", "<h1>Too many sign-ups</h1><p>Please try again tomorrow, or "
                         f"email {_e(self.settings.support_email)}.</p>", 429)
        account = self.billing.create_account(email, signup_ip=ip)
        key = self.billing.issue_key(account.id)
        logger.info("new account %s", account.id)
        response = self._key_page(request, key, "Your account is ready")
        self._set_session(response, account.id, request)
        return response

    def _key_page(self, request: Request, key: str, title: str) -> HTMLResponse:
        base = self._base(request)
        body = f"""<h1>{_e(title)}</h1>
<div class="card notice"><h2>Your API key</h2>
<p>Copy it now. For your security it won't be shown again. You can make a new one from your
account page anytime.</p><code class="key" id="key">{_e(key)}</code>
<div class="row"><button type="button" onclick="navigator.clipboard.writeText(
document.getElementById('key').textContent);this.textContent='Copied'">Copy key</button></div></div>
<div class="card"><h2>Connect it to Muse</h2><dl>
<dt>Server URL</dt><dd><code>{_e(base)}/mcp</code></dd>
<dt>Authentication</dt><dd>API key / bearer token: paste the key above.</dd></dl>
<p class="muted">Muse sends it as <code>Authorization: Bearer &lt;key&gt;</code>.</p></div>
<p><a class="btn" href="/account">Go to your account</a></p>"""
        return _page(title, body)

    async def account(self, request: Request) -> Response:
        account = self._session_account(request)
        if account is None:
            return self._login_page()
        return await self._dashboard(account, request.query_params.get("checkout"))

    def _login_page(self, error: str = "", status_code: int = 200) -> HTMLResponse:
        err = f'<div class="card warn">{_e(error)}</div>' if error else ""
        body = f"""<h1>Your account</h1>{err}
<div class="card"><h2>Sign in</h2><form method="post" action="/account/login">
<label for="key">API key</label><input id="key" type="password" name="api_key" required
autocomplete="off" placeholder="mva_..."><div class="row"><button type="submit">Sign in</button></div>
</form></div><p>New here? <a href="/pricing">Create an account</a> and get
{self.settings.free_trial_minutes} free minutes.</p>"""
        return _page("Account", body, status_code)

    async def _dashboard(self, account: Account, checkout: str | None) -> HTMLResponse:
        if account.is_admin:
            return _page("Account", "<h1>Admin account</h1><p>This is the owner key. Calls are "
                         "unmetered.</p>" + _logout_form())
        usage = self.billing.usage(account)
        url_note = ""
        if checkout == "success":
            url_note = ('<div class="card notice">Thanks! Your subscription is being set up. '
                        "Refresh in a moment if it doesn't show as active yet.</div>")
        elif checkout == "cancelled":
            url_note = '<div class="card warn">Checkout was cancelled. You weren\'t charged.</div>'
        meters = _meter("Free minutes", usage.free_used, usage.free_total)
        active = account.subscription_active()
        plan_obj = self.billing.plan_for(account)
        if active:
            if usage.paid_total is None:
                meters += f"<p>{_e(plan_obj.name if plan_obj else 'Your')} plan: unlimited minutes.</p>"
            else:
                meters += _meter("This period", usage.paid_used, usage.paid_total)
        plan = self._plan_status(account)
        if not active and usage.free_left <= 0:
            if account.plan == "comp":
                url_note += ('<div class="card warn">Your complimentary minutes are used up. Contact '
                             f'<a href="mailto:{_e(self.settings.support_email)}">'
                             f"{_e(self.settings.support_email)}</a> for more.</div>")
            else:
                url_note += ('<div class="card warn">You\'ve used your free minutes. Subscribe to keep '
                             "placing calls. Your API key stays the same.</div>")
        elif active and usage.paid_left == 0:
            url_note += ('<div class="card warn">You\'ve used this period\'s minutes. They reset on '
                         f"{_e(format_date(account.current_period_end))}, or switch to a bigger plan "
                         "under Change plan.</div>")
        actions = []
        if self.settings.billing_configured and not self._has_subscription(account) and (
            account.plan != "comp"
        ):
            for p in self.settings.plans:
                actions.append(
                    '<form class="inline" method="post" action="/account/checkout">'
                    f'<input type="hidden" name="plan" value="{_e(p.key)}">'
                    f"<button>{_e(p.name)}: {_e(await self._price_label(p))} for "
                    f"{_e(p.minutes_label.lower())}</button></form>"
                )
        if account.stripe_customer_id and self.settings.billing_configured:
            label = "Change plan, billing or cancel" if active else "Manage billing"
            actions.append('<form class="inline" method="post" action="/account/portal">'
                           f'<button class="secondary">{label}</button></form>')
        hint = self.billing.key_hint(account.id)
        key_line = f"Active key ends in <code>{_e(hint)}</code>." if hint else "No active key."
        body = f"""<h1>Your account</h1>{url_note}<p class="muted">{_e(account.email)}</p>
<div class="card"><h2>Plan</h2><p>{plan}</p>{meters}<div class="row">{''.join(actions)}</div></div>
<div class="card"><h2>API key</h2><p>{key_line} Making a new key turns off the old one, so update
it in Muse afterwards.</p><form method="post" action="/account/rotate-key"
onsubmit="return confirm('Make a new key? The current key stops working.')">
<button class="secondary">Make a new API key</button></form></div>{_logout_form()}"""
        return _page("Account", body)

    def _has_subscription(self, account: Account) -> bool:
        """A subscription Stripe may still bill; changes go through the portal, not a new checkout."""
        return bool(account.stripe_subscription_id) and account.status not in (
            "canceled", "incomplete_expired", "none",
        )

    def _plan_status(self, account: Account) -> str:
        plan = self.billing.plan_for(account)
        name = _e(plan.name if plan else "Your")
        if account.plan == "comp" and not account.stripe_subscription_id:
            return "Complimentary account."
        if account.status in ("active", "trialing"):
            end = format_date(account.current_period_end)
            if account.cancel_at_period_end:
                return f"{name} plan, cancelled. Active until {end}."
            return f"{name} plan, active. Renews {end}." if end else f"{name} plan, active."
        if account.status == "past_due":
            if account.subscription_active():
                return (f"{name} plan: the last payment failed. Update your card by "
                        f"{format_date(account.grace_until)} to keep calling.")
            return f"{name} plan: payment failed. Update your card to keep calling."
        if account.stripe_subscription_id:
            return f"{name} plan isn't active. Subscribe again to keep calling."
        return "Free trial."

    async def login(self, request: Request) -> Response:
        key = (await self._form(request)).get("api_key", "")
        account = self.billing.authenticate(key)
        if account is None:
            return self._login_page("That key isn't valid. Check it, or make a new one.", 401)
        response = RedirectResponse("/account", status_code=303)
        self._set_session(response, account.id, request)
        return response

    async def logout(self, request: Request) -> Response:
        response = RedirectResponse("/account", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    async def checkout(self, request: Request) -> Response:
        account = self._session_account(request)
        if account is None or account.is_admin:
            return RedirectResponse("/account", status_code=303)
        if not self.settings.billing_configured:
            return _page("Subscribe", "<h1>Subscriptions open soon</h1>", 503)
        if self._has_subscription(account):
            return await self.portal(request)
        plan = self.settings.plan((await self._form(request)).get("plan", ""))
        if plan is None:
            return RedirectResponse("/pricing", status_code=303)
        base = self._base(request)
        try:
            session = await self.stripe.create_checkout_session(
                price_id=plan.price_id,
                account_id=account.id,
                email=account.email,
                customer_id=account.stripe_customer_id,
                success_url=f"{base}/account/checkout/return?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{base}/account?checkout=cancelled",
            )
        except StripeError as e:
            logger.error("checkout session failed for %s: %s", account.id, e)
            return _page("Subscribe", "<h1>Checkout isn't available right now</h1><p>Please try "
                         "again in a few minutes.</p>", 502)
        return RedirectResponse(session["url"], status_code=303)

    async def checkout_return(self, request: Request) -> Response:
        """Apply the subscription right away instead of waiting for the webhook."""
        account = self._session_account(request)
        session_id = request.query_params.get("session_id", "")
        if account and session_id.startswith("cs_"):
            try:
                session = await self.stripe.get_checkout_session(session_id)
                if session.get("client_reference_id") == account.id:
                    await self._sync_checkout(session)
            except StripeError as e:
                logger.warning("checkout return sync failed: %s", e)
        return RedirectResponse("/account?checkout=success", status_code=303)

    async def portal(self, request: Request) -> Response:
        account = self._session_account(request)
        if account is None or not account.stripe_customer_id:
            return RedirectResponse("/account", status_code=303)
        try:
            session = await self.stripe.create_portal_session(
                customer_id=account.stripe_customer_id, return_url=f"{self._base(request)}/account"
            )
        except StripeError as e:
            logger.error("portal session failed for %s: %s", account.id, e)
            return _page("Billing", "<h1>Billing settings aren't available right now</h1>", 502)
        return RedirectResponse(session["url"], status_code=303)

    async def rotate_key(self, request: Request) -> Response:
        account = self._session_account(request)
        if account is None or account.is_admin:
            return RedirectResponse("/account", status_code=303)
        key = self.billing.issue_key(account.id)
        response = self._key_page(request, key, "Your new API key")
        self._set_session(response, account.id, request)
        return response

    # --- Stripe webhook -------------------------------------------------------------------

    async def webhook(self, request: Request) -> Response:
        payload = await request.body()
        try:
            event = verify_webhook(
                payload,
                request.headers.get("stripe-signature", ""),
                self.settings.stripe_webhook_secret,
            )
        except StripeError as e:
            logger.warning("rejected Stripe webhook: %s", e)
            return JSONResponse({"error": "invalid signature"}, status_code=400)
        event_id, kind = event.get("id", ""), event.get("type", "")
        if self.billing.has_event(event_id):
            return JSONResponse({"received": True, "duplicate": True})
        obj = (event.get("data") or {}).get("object") or {}
        try:
            await self.handle_event(kind, obj)
        except StripeError as e:
            logger.error("Stripe webhook %s (%s) failed: %s", event_id, kind, e)
            return JSONResponse({"error": "retry"}, status_code=500)
        self.billing.record_event(event_id, kind)
        return JSONResponse({"received": True})

    async def handle_event(self, kind: str, obj: dict[str, Any]) -> None:
        if kind == "checkout.session.completed":
            await self._sync_checkout(obj)
        elif kind.startswith("customer.subscription."):
            account_id = self._account_for(obj.get("metadata"), obj.get("customer"))
            if account_id:
                # Re-read the subscription so a late, out-of-order event can't apply stale state.
                await self._sync_subscription(account_id, obj["id"])
        elif kind in ("invoice.paid", "invoice.payment_failed"):
            sub_id = _invoice_subscription(obj)
            account_id = self._account_for(None, obj.get("customer"))
            if account_id and sub_id:
                await self._sync_subscription(account_id, sub_id)
                if kind == "invoice.payment_failed":
                    account = self.billing.get_account(account_id)
                    if (
                        account
                        and account.stripe_subscription_id == sub_id
                        and not account.grace_until
                        and account.status != "canceled"
                    ):
                        self.billing.update_account(
                            account_id, grace_until=self.billing.grace_deadline()
                        )

    def _account_for(self, metadata: dict[str, Any] | None, customer: Any) -> str | None:
        account_id = (metadata or {}).get("account_id")
        if account_id and self.billing.get_account(account_id):
            return account_id
        if isinstance(customer, str):
            account = self.billing.find_by_customer(customer)
            if account:
                return account.id
        return None

    async def _sync_checkout(self, session: dict[str, Any]) -> None:
        account_id = session.get("client_reference_id") or (session.get("metadata") or {}).get(
            "account_id"
        )
        if not account_id or self.billing.get_account(account_id) is None:
            logger.warning("checkout session %s has no known account", session.get("id"))
            return
        fields: dict[str, Any] = {}
        if isinstance(session.get("customer"), str):
            fields["stripe_customer_id"] = session["customer"]
        self.billing.update_account(account_id, **fields)
        sub_id = session.get("subscription")
        if isinstance(sub_id, dict):
            sub_id = sub_id.get("id")
        if sub_id:
            await self._sync_subscription(account_id, sub_id)

    async def _sync_subscription(self, account_id: str, subscription_id: str) -> None:
        sub = await self.stripe.get_subscription(subscription_id)
        self.billing.apply_subscription(account_id, sub)
        logger.info("account %s subscription %s -> %s", account_id, sub.get("id"), sub.get("status"))


def _invoice_subscription(invoice: dict[str, Any]) -> str | None:
    sub = invoice.get("subscription")
    if not sub:
        details = ((invoice.get("parent") or {}).get("subscription_details")) or {}
        sub = details.get("subscription")
    if isinstance(sub, dict):
        sub = sub.get("id")
    return sub or None


def _meter(label: str, used: float, total: float) -> str:
    pct = 100 if total <= 0 else min(100, round(100 * used / total))
    left = max(0.0, total - used)
    return (f'<p><strong>{_e(label)}:</strong> {minutes(left)} of {minutes(total)} minutes left</p>'
            f'<div class="meter"><span style="width:{pct}%"></span></div>')


def _logout_form() -> str:
    return ('<form method="post" action="/account/logout"><button class="secondary">Sign out'
            "</button></form>")
