import pytest

from muse_voice_agent.config import llm_model_init_args, parse_llm_model_spec


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
