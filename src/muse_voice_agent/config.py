from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
class Settings:
    # LiveKit
    livekit_url: str = field(default_factory=lambda: _str("LIVEKIT_URL"))
    livekit_api_key: str = field(default_factory=lambda: _str("LIVEKIT_API_KEY"))
    livekit_api_secret: str = field(default_factory=lambda: _str("LIVEKIT_API_SECRET"))
    sip_outbound_trunk_id: str = field(default_factory=lambda: _str("SIP_OUTBOUND_TRUNK_ID"))
    agent_name: str = field(default_factory=lambda: _str("AGENT_NAME", "muse-voice-agent"))

    # Models
    llm_model: str = field(default_factory=lambda: _str("LLM_MODEL", "openai:gpt-4.1-mini"))
    stt_model: str = field(default_factory=lambda: _str("STT_MODEL", "assemblyai/universal-3-5-pro"))
    tts_model: str = field(default_factory=lambda: _str("TTS_MODEL", "fishaudio/s2.1-pro"))
    tts_voice: str = field(
        default_factory=lambda: _str("TTS_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a")
    )

    # MCP server
    mcp_host: str = field(default_factory=lambda: _str("MCP_HOST", "127.0.0.1"))
    mcp_port: int = field(default_factory=lambda: _int("MCP_PORT", 8765))
    mcp_auth_token: str = field(default_factory=lambda: _str("MCP_AUTH_TOKEN"))

    # Behaviour / safety
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN", True))
    allowed_dial_prefixes: list[str] = field(
        default_factory=lambda: _list("ALLOWED_DIAL_PREFIXES", "+1")
    )
    max_call_seconds: int = field(default_factory=lambda: _int("MAX_CALL_SECONDS", 300))
    max_concurrent_calls: int = field(default_factory=lambda: _int("MAX_CONCURRENT_CALLS", 3))
    default_customer_name: str = field(default_factory=lambda: _str("DEFAULT_CUSTOMER_NAME"))
    default_callback_number: str = field(default_factory=lambda: _str("DEFAULT_CALLBACK_NUMBER"))

    # Storage
    call_db_path: Path = field(
        default_factory=lambda: Path(_str("CALL_DB_PATH", str(PROJECT_ROOT / "data" / "calls.db")))
    )

    def missing_for_live_calls(self) -> list[str]:
        required = {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "SIP_OUTBOUND_TRUNK_ID": self.sip_outbound_trunk_id,
        }
        return [k for k, v in required.items() if not v]


def get_settings() -> Settings:
    return Settings()
