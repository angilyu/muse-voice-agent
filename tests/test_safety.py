import pytest
from pydantic import ValidationError

from muse_voice_agent.graph import CallOutcome, apply_authority
from muse_voice_agent.report import build_report
from muse_voice_agent.safety import OutputGuard, recording_disclosure_required
from muse_voice_agent.tasks import GeneralCall, RestaurantReservation, build_system_prompt, opening_line


def _general(**overrides):
    data = {
        "business_name": "Spa",
        "phone_number": "+14155550100",
        "customer_name": "Angi",
        "goal": "Book a massage",
    }
    data.update(overrides)
    return GeneralCall(**data)


def test_luhn_and_ssn_input_rejected_anywhere():
    with pytest.raises(ValidationError, match="card"):
        _general(goal="Use card 4111 1111 1111 1111 if needed")
    with pytest.raises(ValidationError, match="social security"):
        _general(shareable_details={"ssn": "123-45-6789"})


def test_output_guard_redacts_unapproved_details_but_allows_allow_list():
    guard = OutputGuard({"customer name": "Angi", "callback number": "+14155550123", "email": "angi@example.com"})
    text = guard.feed(
        "Call +1 (415) 555-0123 or angi@example.com, not +1 650 555 9999, "
        "DOB 01/02/1990, 123 Main St, card 4111 1111 1111 1111."
    )
    assert "+1 (415) 555-0123" in text
    assert "angi@example.com" in text
    assert "[phone withheld]" in text
    assert "[date of birth withheld]" in text
    assert "[address withheld]" in text
    assert "[payment number withheld]" in text


def test_default_shareable_details_and_prompt_allow_list():
    task = _general(callback_number="415-555-0123")
    assert task.shareable_details == {"customer name": "Angi", "callback number": "+14155550123"}
    prompt = build_system_prompt(task)
    assert "Approved shareable details:" in prompt
    assert "customer name: Angi" in prompt
    assert "callback number: +14155550123" in prompt
    assert "Do not share any personal detail not on that list" in prompt


def test_structured_limits_rendered_and_enforced_on_outcome_and_report():
    task = _general(
        authority="may_commit_within_limits",
        max_spend=50,
        limits="Friday afternoon only; pay at pickup",
    )
    prompt = build_system_prompt(task)
    assert "Maximum total spend: $50" in prompt
    assert "Friday afternoon only" in prompt

    outcome = apply_authority(
        task,
        CallOutcome(outcome="ordered", summary="Order total is $75", order_total="$75"),
    )
    assert outcome.outcome == "needs_followup"
    assert outcome.committed_within_limits is False
    assert "spend_exceeds_max_spend" in outcome.safety_flags

    report = build_report(
        {
            "status": "completed",
            "outcome": outcome.outcome,
            "task": task.model_dump(),
            "details": outcome.model_dump(exclude_none=True),
            "transcript": [{"role": "user", "text": "Spa"}],
            "started_at": 1.0,
            "ended_at": 2.0,
            "end_reason": "agent_hangup",
            "outcome_source": "agent",
        }
    )
    assert report["committed_on_users_behalf"] is False
    assert report["commitment_within_limits"] is False
    assert "spend_exceeds_max_spend" in report["safety_flags"]
    assert report["needs_user_action"] is True


def test_opener_always_says_ai_and_recording_cannot_be_disabled_by_task_input():
    task = RestaurantReservation(
        business_name="Luigi's",
        phone_number="+14155550123",
        customer_name="Angi",
        party_size=2,
        date="Friday",
        time="7pm",
        recording_disclosure_required=False,
    )
    assert task.recording_disclosure_required is True
    assert opening_line(task).startswith("Hi, this is an AI assistant calling on behalf of Angi.")
    assert "This call may be recorded." in opening_line(task)


def test_recording_disclosure_state_logic():
    assert recording_disclosure_required("+14155550123", scope="required_states") == (True, "CA")
    assert recording_disclosure_required("+12125550123", scope="required_states") == (False, "NY")
    assert recording_disclosure_required("+12125550123", scope="always") == (True, "NY")
    assert recording_disclosure_required("+12125550123", recording_enabled=False) == (False, "NY")
