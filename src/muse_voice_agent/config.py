from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from voice_kit.env import env_bool as _bool
from voice_kit.env import env_int as _int
from voice_kit.env import env_list as _list
from voice_kit.env import env_str as _str
from voice_kit.llm import LLMModelSpec, llm_model_init_args, parse_llm_model_spec
from voice_kit.presets import VoicePreset, preset_from_env

__all__ = ["LLMModelSpec", "Settings", "get_settings", "llm_model_init_args", "parse_llm_model_spec"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# Unprefixed names Muse has always read for its voice; they override MUSE_* preset variables.
_LEGACY_VOICE_ENV = {"STT_MODEL": "stt_model", "TTS_MODEL": "tts_model", "TTS_VOICE": "tts_voice"}


def voice_preset_from_env() -> VoicePreset:
    """LiveKit voice preset from ``MUSE_VOICE_PRESET`` (and other ``MUSE_*`` voice-kit variables),
    with the legacy ``STT_MODEL``/``TTS_MODEL``/``TTS_VOICE`` names still honoured."""
    preset = preset_from_env(prefix="MUSE_")
    legacy = {field_name: _str(env_name) for env_name, field_name in _LEGACY_VOICE_ENV.items()}
    legacy = {k: v for k, v in legacy.items() if v}
    return preset.with_overrides(**legacy) if legacy else preset


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
    # LiveKit STT/TTS/turn-taking (voice-kit preset; Retell configures its own voice)
    voice: VoicePreset = field(default_factory=voice_preset_from_env)

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
    max_call_seconds: int = field(default_factory=lambda: _int("MAX_CALL_SECONDS", 300))
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
        from voice_kit.retell import retell_llm_websocket_url

        if not self.public_base_url or not self.retell_ws_secret:
            raise ValueError("PUBLIC_BASE_URL and RETELL_WS_SECRET must be set")
        return retell_llm_websocket_url(self.public_base_url, self.retell_ws_secret)


def get_settings() -> Settings:
    return Settings()
