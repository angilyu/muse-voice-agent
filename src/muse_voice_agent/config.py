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
    # OpenAI processing tier for live calls: "priority" answers faster at a higher token price;
    # "default" (or empty) uses standard processing.
    llm_service_tier: str = field(default_factory=lambda: _str("LLM_SERVICE_TIER", "priority"))
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
    max_call_seconds: int = field(default_factory=lambda: _int("MAX_CALL_SECONDS", 300))
    max_concurrent_calls: int = field(default_factory=lambda: _int("MAX_CONCURRENT_CALLS", 3))
    # Retell: speak first if the line is silent this long after pickup (0 disables).
    silent_pickup_ms: int = field(default_factory=lambda: _int("SILENT_PICKUP_MS", 3000))
    # Retell: if a reply to a person has no words yet after this long, say "Hmm," so the line
    # doesn't go dead while the model thinks (0 disables).
    filler_after_ms: int = field(default_factory=lambda: _int("FILLER_AFTER_MS", 1500))
    default_customer_name: str = field(default_factory=lambda: _str("DEFAULT_CUSTOMER_NAME"))
    default_callback_number: str = field(default_factory=lambda: _str("DEFAULT_CALLBACK_NUMBER"))

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
