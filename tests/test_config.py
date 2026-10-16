import pytest

from muse_voice_agent.config import Settings, llm_model_init_args, parse_llm_model_spec


def test_parse_llm_model_spec_with_reasoning_effort():
    spec = parse_llm_model_spec("openai:gpt-5.4@low")

    assert spec.model == "openai:gpt-5.4"
    assert spec.reasoning_effort == "low"


def test_llm_model_init_args_uses_reasoning_effort_without_temperature():
    model, kwargs = llm_model_init_args("openai:gpt-5.4@low", temperature=0.3)

    assert model == "openai:gpt-5.4"
    assert kwargs == {"reasoning_effort": "low", "use_responses_api": True}


def test_llm_model_init_args_keeps_temperature_for_non_reasoning_model():
    model, kwargs = llm_model_init_args("openai:gpt-4.1-mini", temperature=0.3)

    assert model == "openai:gpt-4.1-mini"
    assert kwargs == {"temperature": 0.3}


@pytest.mark.parametrize("value", ["", "openai:gpt-5.4@", "@low"])
def test_parse_llm_model_spec_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_llm_model_spec(value)


def test_llm_model_init_args_service_tier_only_for_openai():
    _, kwargs = llm_model_init_args("openai:gpt-5.4@low", service_tier="priority")
    assert kwargs["service_tier"] == "priority"
    assert "service_tier" not in llm_model_init_args("openai:gpt-5.4@low", service_tier="default")[1]
    assert "service_tier" not in llm_model_init_args("anthropic:claude-x", service_tier="priority")[1]


def test_service_tier_defaults_to_standard_processing(monkeypatch):
    monkeypatch.delenv("LLM_SERVICE_TIER", raising=False)
    tier = Settings().llm_service_tier
    assert tier == "default"
    assert "service_tier" not in llm_model_init_args("openai:gpt-5.4@low", service_tier=tier)[1]


def test_voice_preset_keeps_legacy_env_names(monkeypatch):
    for name in ("MUSE_VOICE_PRESET", "MUSE_STT_MODEL", "MUSE_TTS_MODEL", "MUSE_TTS_VOICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STT_MODEL", "deepgram/nova-3")
    monkeypatch.setenv("TTS_MODEL", "cartesia/sonic-3")
    monkeypatch.setenv("TTS_VOICE", "v1")

    voice = Settings().voice

    assert (voice.stt_model, voice.tts_model, voice.tts_voice) == ("deepgram/nova-3", "cartesia/sonic-3", "v1")


def test_voice_preset_defaults_match_previous_settings(monkeypatch):
    for name in ("MUSE_VOICE_PRESET", "MUSE_STT_MODEL", "MUSE_TTS_MODEL", "MUSE_TTS_VOICE", "STT_MODEL", "TTS_MODEL", "TTS_VOICE"):
        monkeypatch.delenv(name, raising=False)

    voice = Settings().voice

    assert voice.name == "default" and voice.language == "en"
    assert voice.stt_model == "assemblyai/universal-3-5-pro"
    assert voice.tts_model == "fishaudio/s2.1-pro"
    assert voice.tts_voice == "fa4c9eb3dccc4806b382b40d61c6b10a"
