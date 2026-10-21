from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _list(name: str, default: str) -> list[str]:
    return [p.strip() for p in _str(name, default).split(",") if p.strip()]


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    price_id: str
    minutes: int  # included per billing period; 0 = unlimited
    price_hint: str  # shown when the Stripe price can't be loaded

    @property
    def minutes_label(self) -> str:
        return "Unlimited call minutes" if self.minutes <= 0 else f"{self.minutes} call minutes a month"


@dataclass(frozen=True)
class LLMModelSpec:
    model: str
    reasoning_effort: str | None = None


def parse_llm_model_spec(raw: str) -> LLMModelSpec:
    """Parse ``provider:model[@reasoning_effort]`` for production model settings."""
    value = (raw or "").strip()
    if not value:
        raise ValueError("LLM model must not be empty")
    model, sep, effort = value.rpartition("@")
    if not sep:
        return LLMModelSpec(model=value)
    model = model.strip()
    effort = effort.strip()
    if not model or not effort:
        raise ValueError(f"Invalid LLM model spec {raw!r}; expected provider:model[@effort]")
    return LLMModelSpec(model=model, reasoning_effort=effort)


def llm_model_init_args(
    raw: str,
    *,
    temperature: float | None = 0.3,
    service_tier: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the model name and ``init_chat_model`` kwargs for a production model spec.

    Reasoning-effort specs intentionally omit temperature because OpenAI reasoning models reject
    non-default temperature. ``service_tier`` (e.g. "priority") only applies to OpenAI models.
    """
    spec = parse_llm_model_spec(raw)
    kwargs: dict[str, Any] = {}
    if spec.reasoning_effort:
        kwargs["reasoning_effort"] = spec.reasoning_effort
        if spec.model.startswith("openai:"):
            # OpenAI Chat Completions currently rejects reasoning_effort together with tools for
            # gpt-5.4. LangChain's Responses API path supports streaming + tool calling.
            kwargs["use_responses_api"] = True
    elif temperature is not None:
        kwargs["temperature"] = temperature
    tier = (service_tier or "").strip().lower()
    if tier and tier != "default" and spec.model.startswith("openai:"):
        kwargs["service_tier"] = tier
    return spec.model, kwargs


@dataclass(frozen=True)
class Settings:
    # Which voice/telephony stack places live calls: "retell" or "livekit"
    voice_backend: str = field(default_factory=lambda: _str("VOICE_BACKEND", "retell").lower())

    # Retell (telephony + STT/TTS; our LangGraph graph is its custom LLM over a websocket)
    retell_api_key: str = field(default_factory=lambda: _str("RETELL_API_KEY"), repr=False)
    retell_agent_id: str = field(default_factory=lambda: _str("RETELL_AGENT_ID"))
    retell_from_number: str = field(default_factory=lambda: _str("RETELL_FROM_NUMBER"))
    retell_voice_id: str = field(default_factory=lambda: _str("RETELL_VOICE_ID", "cartesia-Cleo"))
    retell_ws_secret: str = field(default_factory=lambda: _str("RETELL_WS_SECRET"), repr=False)
    # Public https:// base URL of this server (e.g. the cloudflared tunnel); Retell connects back to it
    public_base_url: str = field(default_factory=lambda: _str("PUBLIC_BASE_URL").rstrip("/"))

    # LiveKit
    livekit_url: str = field(default_factory=lambda: _str("LIVEKIT_URL"))
    livekit_api_key: str = field(default_factory=lambda: _str("LIVEKIT_API_KEY"))
    livekit_api_secret: str = field(default_factory=lambda: _str("LIVEKIT_API_SECRET"), repr=False)
    sip_outbound_trunk_id: str = field(default_factory=lambda: _str("SIP_OUTBOUND_TRUNK_ID"))
    agent_name: str = field(default_factory=lambda: _str("AGENT_NAME", "muse-voice-agent"))

    # Models
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL", "openai:gpt-5.4@low"))
    # OpenAI processing tier. "default" (standard processing) is used for live calls and evals;
    # "priority" answers a little faster at about 2x the token price and is opt-in only.
    llm_service_tier: str = field(default_factory=lambda: _str("LLM_SERVICE_TIER", "default"))
    stt_model: str = field(default_factory=lambda: _str("STT_MODEL", "assemblyai/universal-3-5-pro"))
    tts_model: str = field(default_factory=lambda: _str("TTS_MODEL", "fishaudio/s2.1-pro"))
    tts_voice: str = field(
        default_factory=lambda: _str("TTS_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a")
    )

    # MCP server
    mcp_host: str = field(default_factory=lambda: _str("MCP_HOST", "127.0.0.1"))
    mcp_port: int = field(default_factory=lambda: _int("MCP_PORT", 8765))
    mcp_auth_token: str = field(default_factory=lambda: _str("MCP_AUTH_TOKEN"), repr=False)
    # >0: ping PUBLIC_BASE_URL/healthz this often so free hosts don't sleep (600 on Render Free)
    keepalive_seconds: int = field(default_factory=lambda: _int("KEEPALIVE_SECONDS", 0))

    # Behaviour / safety
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))
    allowed_dial_prefixes: list[str] = field(
        default_factory=lambda: _list("ALLOWED_DIAL_PREFIXES", "+1")
    )
    # If Retell (or another backend) records calls, disclose it in the opener. "always" is the
    # default; "required_states" still discloses for all-party-consent states or unknown area codes.
    recording_disclosure_scope: str = field(
        default_factory=lambda: _str("RECORDING_DISCLOSURE_SCOPE", "always").lower()
    )
    call_recording_enabled: bool = field(default_factory=lambda: _bool("CALL_RECORDING_ENABLED", True))
    # Retell accepts 1 minute to 2 hours, so the value is clamped to that range.
    max_call_seconds: int = field(
        default_factory=lambda: min(max(_int("MAX_CALL_SECONDS", 7200), 60), 7200)
    )
    max_concurrent_calls: int = field(default_factory=lambda: _int("MAX_CONCURRENT_CALLS", 3))
    # Retell: speak first if the line is silent this long after pickup (0 disables).
    silent_pickup_ms: int = field(default_factory=lambda: _int("SILENT_PICKUP_MS", 3000))
    # Retell inbound: answer an incoming call this long after it connects if the caller hasn't
    # spoken yet (0 waits for the caller to speak first).
    inbound_greet_ms: int = field(default_factory=lambda: _int("INBOUND_GREET_MS", 500))
    # Retell: if a reply to a person has no words yet after this long, say "Hmm," so the line
    # doesn't go dead while the model thinks (0 disables).
    filler_after_ms: int = field(default_factory=lambda: _int("FILLER_AFTER_MS", 1500))
    # Retell inbound: match callbacks to outbound calls from the same number this many days back.
    inbound_callback_lookback_days: int = field(
        default_factory=lambda: _int("INBOUND_CALLBACK_LOOKBACK_DAYS", 14)
    )
    default_customer_name: str = field(default_factory=lambda: _str("DEFAULT_CUSTOMER_NAME"))
    default_callback_number: str = field(default_factory=lambda: _str("DEFAULT_CALLBACK_NUMBER"))

    # Billing: every API key belongs to an account. New accounts get FREE_TRIAL_MINUTES of call
    # time, then need an active Stripe subscription. MCP_AUTH_TOKEN stays an unmetered admin key.
    stripe_secret_key: str = field(default_factory=lambda: _str("STRIPE_SECRET_KEY"), repr=False)
    stripe_webhook_secret: str = field(
        default_factory=lambda: _str("STRIPE_WEBHOOK_SECRET"), repr=False
    )
    free_trial_minutes: int = field(default_factory=lambda: _int("FREE_TRIAL_MINUTES", 10))
    # Monthly plans: a Stripe recurring price id and the call minutes it includes per billing
    # period (0 = unlimited). A plan without a price id isn't offered.
    stripe_price_starter: str = field(default_factory=lambda: _str("STRIPE_PRICE_STARTER"))
    starter_minutes: int = field(default_factory=lambda: _int("STARTER_MINUTES", 50))
    stripe_price_plus: str = field(default_factory=lambda: _str("STRIPE_PRICE_PLUS"))
    plus_minutes: int = field(default_factory=lambda: _int("PLUS_MINUTES", 200))
    billing_grace_days: int = field(default_factory=lambda: _int("BILLING_GRACE_DAYS", 3))
    max_concurrent_calls_per_account: int = field(
        default_factory=lambda: _int("MAX_CONCURRENT_CALLS_PER_ACCOUNT", 2)
    )
    # A metered call is ended politely once its talk time passes the account's remaining minutes
    # (at dial time) by this much. MAX_CALL_SECONDS still caps every call.
    call_overage_minutes: int = field(default_factory=lambda: _int("CALL_OVERAGE_MINUTES", 20))
    signups_per_ip_per_day: int = field(default_factory=lambda: _int("SIGNUPS_PER_IP_PER_DAY", 3))
    # Signs account-page session cookies; derived from MCP_AUTH_TOKEN when unset.
    billing_session_secret: str = field(
        default_factory=lambda: _str("BILLING_SESSION_SECRET"), repr=False
    )
    support_email: str = field(default_factory=lambda: _str("SUPPORT_EMAIL", "angilyu@gmail.com"))
    # Sign-up and sign-in codes are emailed over SMTP (port 587 with STARTTLS, or 465 with TLS).
    # SMTP_HOST=log writes emails to the server log instead; only for local development.
    smtp_host: str = field(default_factory=lambda: _str("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: _int("SMTP_PORT", 587))
    smtp_username: str = field(default_factory=lambda: _str("SMTP_USERNAME"))
    smtp_password: str = field(default_factory=lambda: _str("SMTP_PASSWORD"), repr=False)
    email_from: str = field(default_factory=lambda: _str("EMAIL_FROM"))

    # Storage
    call_db_path: Path = field(
        default_factory=lambda: Path(_str("CALL_DB_PATH", str(PROJECT_ROOT / "data" / "calls.db")))
    )

    def missing_for_live_calls(self) -> list[str]:
        if self.voice_backend == "livekit":
            required = {
                "LIVEKIT_URL": self.livekit_url,
                "LIVEKIT_API_KEY": self.livekit_api_key,
                "LIVEKIT_API_SECRET": self.livekit_api_secret,
                "SIP_OUTBOUND_TRUNK_ID": self.sip_outbound_trunk_id,
            }
        elif self.voice_backend == "retell":
            required = {
                "RETELL_API_KEY": self.retell_api_key,
                "RETELL_AGENT_ID": self.retell_agent_id,
                "RETELL_FROM_NUMBER": self.retell_from_number,
                "RETELL_WS_SECRET": self.retell_ws_secret,
            }
        else:
            return [f"VOICE_BACKEND (unknown value {self.voice_backend!r}; use retell or livekit)"]
        return [k for k, v in required.items() if not v]

    @property
    def plans(self) -> list[Plan]:
        """Subscription plans on offer, cheapest first."""
        plans = [
            Plan("starter", "Starter", self.stripe_price_starter, self.starter_minutes, "$5/month"),
            Plan("plus", "Plus", self.stripe_price_plus, self.plus_minutes, "$20/month"),
        ]
        return [p for p in plans if p.price_id]

    def plan(self, key: str) -> Plan | None:
        return next((p for p in self.plans if p.key == key), None)

    def plan_for_price(self, price_id: str | None) -> Plan | None:
        return next((p for p in self.plans if price_id and p.price_id == price_id), None)

    @property
    def billing_configured(self) -> bool:
        """Sign-ups and checkout need Stripe; without it only existing keys work."""
        return bool(self.stripe_secret_key and self.plans)

    @property
    def email_configured(self) -> bool:
        return bool(self.smtp_host and (self.email_from or self.smtp_host == "log"))

    @property
    def signups_open(self) -> bool:
        """New accounts need Stripe (to pay later) and email (to verify the address)."""
        return self.billing_configured and self.email_configured

    @property
    def billing_url(self) -> str:
        """Where users sign up, subscribe and manage their key."""
        return f"{self.public_base_url}/account" if self.public_base_url else "/account"

    @property
    def pricing_url(self) -> str:
        return f"{self.public_base_url}/pricing" if self.public_base_url else "/pricing"

    def retell_llm_websocket_url(self) -> str:
        """URL Retell dials for the custom LLM; Retell appends /{retell_call_id}."""
        if not self.public_base_url or not self.retell_ws_secret:
            raise ValueError("PUBLIC_BASE_URL and RETELL_WS_SECRET must be set")
        base = self.public_base_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        return f"{base}/retell/llm/{self.retell_ws_secret}"


def get_settings() -> Settings:
    return Settings()
