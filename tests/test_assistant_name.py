import pytest
from pydantic import ValidationError

from muse_voice_agent.graph import CallControl, _line_notes, _status_notes
from muse_voice_agent.tasks import GeneralCall, build_system_prompt, opening_line


def _task(**kw) -> GeneralCall:
    return GeneralCall(
        business_name="Hotel Zed",
        phone_number="+14155550100",
        customer_name="Wenjing Yu",
        goal="Check king room availability",
        **kw,
    )


def test_opener_uses_assistant_name():
    task = _task(assistant_name="  Eva ")
    assert task.assistant_name == "Eva"
    assert opening_line(task) == "Hi, this is Eva, an assistant calling on behalf of Wenjing Yu."
    prompt = build_system_prompt(task)
    assert "You are Eva, an AI assistant calling on behalf of Wenjing Yu." in prompt
    assert "I'm Eva, an AI assistant calling for Wenjing Yu" in prompt


def test_opener_without_name_is_unchanged():
    task = _task()
    assert opening_line(task) == "Hi, this is an assistant calling on behalf of Wenjing Yu."
    assert "You are an AI assistant calling on behalf of Wenjing Yu." in build_system_prompt(task)


@pytest.mark.parametrize("placeholder", ["", "assistant", "AI assistant", "N/A", "unknown"])
def test_placeholder_assistant_names_fall_back_to_generic(placeholder):
    assert _task(assistant_name=placeholder).assistant_name is None


@pytest.mark.parametrize("bad", ["Eva1234", "x" * 41, "Eva! call me", "123"])
def test_invalid_assistant_names_rejected(bad):
    with pytest.raises(ValidationError):
        _task(assistant_name=bad)


def test_screener_and_cut_off_notes_use_assistant_name():
    task = _task(assistant_name="Eva")
    assert "Eva, an AI assistant calling for Wenjing Yu" in _line_notes(
        task, "screener", intro_now=False, reintro=False
    )[0]
    control = CallControl()
    control.opener_spoken = True
    notes = _status_notes(task, control, [])
    assert any("I'm Eva, the assistant for Wenjing Yu." in n for n in notes)
