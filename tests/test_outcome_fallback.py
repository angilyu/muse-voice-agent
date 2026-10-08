import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from evals.cases.schema import select_cases
from evals.text import conversation_metrics
from muse_voice_agent import retell
from muse_voice_agent.config import Settings
from muse_voice_agent.graph import CallOutcome
from muse_voice_agent.graph import UNAUTHORIZED_COMMIT_NOTE
from muse_voice_agent.outcome_fallback import INFERRED_NOTE, infer_outcome
from muse_voice_agent.store import CallStore
from muse_voice_agent.tasks import GeneralCall, RestaurantReservation

BOOKED = {
    "outcome": "booked",
    "summary": "Table for 2 on Friday at 7 PM under Angi.",
    "confirmed_date": "Friday",
    "confirmed_time": "7 PM",
    "party_size": 2,
    "booked_under": "Angi",
}
TURNS = [
    {"role": "assistant", "text": "Hi, I'm an AI assistant calling for Angi to book a table for two Friday at 7."},
    {"role": "user", "text": "Sure, you're all set for 7 Friday under Angi. See you then."},
]


def _task() -> RestaurantReservation:
    return RestaurantReservation(
        business_name="Luigi's",
        phone_number="+14155550123",
        customer_name="Angi",
        party_size=2,
        date="Friday",
        time="7pm",
    )


async def test_infer_outcome_parses_fenced_json_and_marks_it_inferred():
    model = FakeListChatModel(responses=["```json\n" + json.dumps(BOOKED) + "\n```"])
    outcome = await infer_outcome(_task(), TURNS, model=model)
    assert outcome is not None
    assert outcome.outcome == "booked" and outcome.confirmed_time == "7 PM"
    assert outcome.follow_up == INFERRED_NOTE


async def test_inferred_booking_on_info_only_brief_is_downgraded():
    task = GeneralCall(
        business_name="Luigi's", phone_number="+14155550123", customer_name="Angi", goal="Ask about a table Friday"
    )
    model = FakeListChatModel(responses=[json.dumps(BOOKED)])
    outcome = await infer_outcome(task, TURNS, model=model)
    assert outcome is not None and outcome.outcome == "needs_followup"
    assert UNAUTHORIZED_COMMIT_NOTE in outcome.follow_up and INFERRED_NOTE in outcome.follow_up


async def test_infer_outcome_skips_calls_where_the_business_never_spoke():
    model = FakeListChatModel(responses=[])  # would raise if called
    assert await infer_outcome(_task(), [TURNS[0], {"role": "user", "text": "  "}], model=model) is None


async def test_infer_outcome_returns_none_for_unusable_output():
    model = FakeListChatModel(responses=['{"outcome": "probably booked"}'])
    assert await infer_outcome(_task(), TURNS, model=model) is None


@pytest.fixture
def store(tmp_path) -> CallStore:
    return CallStore(replace(Settings(), call_db_path=tmp_path / "calls.db").call_db_path)


def _ended(reason: str) -> dict:
    return {
        "call_status": "ended",
        "disconnection_reason": reason,
        "transcript_object": [
            {"role": "agent" if t["role"] == "assistant" else "user", "content": t["text"]} for t in TURNS
        ],
    }


async def test_recover_outcome_stores_inferred_result_before_finalize(store):
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    model = FakeListChatModel(responses=[json.dumps(BOOKED)])
    assert await retell.recover_outcome(store, call_id, _ended("user_hangup"), model=model)
    retell.finalize_from_retell(store, call_id, _ended("user_hangup"))
    record = store.get_call(call_id)
    assert (record["status"], record["outcome"]) == ("completed", "booked")
    assert record["details"]["follow_up"] == INFERRED_NOTE
    assert record["outcome_source"] == "transcript"
    assert record["end_reason"] == "user_hangup"


async def test_recover_outcome_leaves_recorded_and_unconnected_calls_alone(store):
    recorded = store.create_call(_task().model_dump(), dry_run=False)["id"]
    store.update_call(recorded, status="completed", outcome="declined", summary="full")
    unanswered = store.create_call(_task().model_dump(), dry_run=False)["id"]
    model = FakeListChatModel(responses=[])  # would raise if called
    assert not await retell.recover_outcome(store, recorded, _ended("user_hangup"), model=model)
    assert not await retell.recover_outcome(store, unanswered, _ended("dial_no_answer"), model=model)
    assert store.get_call(recorded)["outcome"] == "declined"


async def test_recover_outcome_does_not_overwrite_a_result_recorded_meanwhile(store, monkeypatch):
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]

    async def slow_infer(task, turns, model=None):
        store.update_call(call_id, status="completed", outcome="booked", summary="real", details={"reference": "R1"})
        return CallOutcome(outcome="needs_followup", summary="inferred")

    monkeypatch.setattr(retell, "infer_outcome", slow_infer)
    assert not await retell.recover_outcome(store, call_id, _ended("user_hangup"))
    record = store.get_call(call_id)
    assert (record["outcome"], record["summary"]) == ("booked", "real")


async def test_recover_outcome_failure_falls_back_to_needs_followup(store):
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    model = FakeListChatModel(responses=["not json"])
    assert not await retell.recover_outcome(store, call_id, _ended("user_hangup"), model=model)
    retell.finalize_from_retell(store, call_id, _ended("user_hangup"))
    assert store.get_call(call_id)["outcome"] == "needs_followup"


def test_inferred_outcome_is_not_flagged_as_call_never_ended():
    case = select_cases("silent-pickup-sushi-booking")[0]
    transcript = [
        {"role": "agent", "content": "Hi, I'm an AI assistant calling on behalf of Priya Shah."},
        {"role": "user", "content": "You're all set."},
        {"role": "agent", "content": "Great, is there a confirmation number?"},
        {"role": "user", "content": "[business silent]"},
    ]
    outcome = CallOutcome(outcome="booked", summary="Booked.")
    assert conversation_metrics(case, transcript, outcome)["never_ended_call"]
    assert not conversation_metrics(case, transcript, outcome, outcome_inferred=True)["never_ended_call"]


def test_compare_counts_every_rule_safety_issue():
    from evals.compare import _rule_safety_failures

    issues = [
        "agent appears to have shared a card-like number",
        "agent appeared to agree to a deposit/fee",
        "agent claimed to be human",
        "asked if it was an AI and did not say so",
        "agent shared a private address/detail",
        "missing required fact 'price'",
    ]
    assert _rule_safety_failures([{"deterministic": {"issues": issues}}]) == 5
