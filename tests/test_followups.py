import asyncio
from dataclasses import replace

import pytest
from mcp import Client

from muse_voice_agent import dispatcher
from muse_voice_agent.config import Settings
from muse_voice_agent.followups import build_follow_up_context, resolve_follow_up
from muse_voice_agent.mcp_server import build_server
from muse_voice_agent.store import CallStore
from muse_voice_agent.tasks import build_system_prompt, parse_task


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        dry_run=True,
        mcp_auth_token="test-token",
        call_db_path=tmp_path / "calls.db",
        allowed_dial_prefixes=["+1"],
        max_concurrent_calls=3,
    )


@pytest.fixture(autouse=True)
def fast_simulation(monkeypatch):
    original = dispatcher._simulate_call

    async def fast(call_id, task, store, delay=0.05):
        await original(call_id, task, store, delay=delay)

    monkeypatch.setattr(dispatcher, "_simulate_call", fast)


def _data(result) -> dict:
    assert not result.is_error, result
    return result.structured_content


def _completed_prior(store: CallStore) -> dict:
    prior = store.create_call(
        {
            "kind": "general",
            "business_name": "Dana's Fencing",
            "phone_number": "+14155550123",
            "customer_name": "Angi",
            "goal": "Ask for a fence repair quote",
            "questions": ["Can you quote a leaning fence?", "When should I call back?"],
            "shareable_details": {"location": "94110"},
            "authority": "info_only",
            "limits": None,
            "callback_number": None,
            "assistant_name": "Eva",
            "follow_up_of": None,
            "previous_call_context": None,
        },
        dry_run=True,
    )
    store.append_transcript(prior["id"], "user", "Dana speaking.")
    store.update_call(
        prior["id"],
        status="completed",
        outcome="info_received",
        summary="Dana can quote the fence repair and said to call back after 3pm when the owner is in.",
        details={
            "outcome": "info_received",
            "summary": "Dana can quote the fence repair and said to call back after 3pm.",
            "contact_person": "Dana",
            "answers": [
                {"question": "Can you quote a leaning fence?", "answer": "Yes, after photos."},
                {"question": "When should I call back?", "answer": "After 3pm for the owner."},
            ],
            "follow_up": "Call back after 3pm and ask for the owner.",
        },
        outcome_source="agent",
        started_at=10.0,
        ended_at=40.0,
        end_reason="agent_hangup",
    )
    return store.get_call(prior["id"])


async def test_follow_up_validation_errors(settings):
    store = CallStore(":memory:")
    pending = store.create_call(
        {
            "kind": "general",
            "business_name": "Dana's Fencing",
            "phone_number": "+14155550123",
            "customer_name": "Angi",
            "goal": "Ask for a quote",
            "questions": [],
            "shareable_details": {},
            "authority": "info_only",
            "limits": None,
            "callback_number": None,
            "assistant_name": None,
            "follow_up_of": None,
            "previous_call_context": None,
        },
        dry_run=True,
    )
    async with Client(build_server(settings, store=store)) as client:
        missing_business = _data(
            await client.call_tool(
                "place_call",
                {"customer_name": "Angi", "goal": "Check availability"},
            )
        )
        unknown = _data(
            await client.call_tool(
                "place_call",
                {"customer_name": "Angi", "goal": "Call back", "follow_up_of": "missing123"},
            )
        )
        not_ready = _data(
            await client.call_tool(
                "place_call",
                {"customer_name": "Angi", "goal": "Call back", "follow_up_of": pending["id"]},
            )
        )

    assert missing_business["error"] == "invalid_request"
    assert "business_name and phone_number" in missing_business["message"]
    assert unknown["error"] == "invalid_follow_up"
    assert "No call with id missing123" in unknown["message"]
    assert not_ready["error"] == "follow_up_not_ready"


def test_build_follow_up_context_is_compact(settings):
    store = CallStore(":memory:")
    prior = _completed_prior(store)
    context = build_follow_up_context(store, prior["id"])
    inherited = resolve_follow_up(
        store,
        prior["id"],
        {"business_name": None, "phone_number": None, "goal": "Call back"},
    )
    overridden = resolve_follow_up(
        store,
        prior["id"],
        {"business_name": "Dana's New Line", "phone_number": "+14155559999", "goal": "Call back"},
    )

    assert "Previous call context" in context
    assert "Dana" in context
    assert "after 3pm" in context
    assert "Can you quote a leaning fence?: Yes, after photos." in context
    assert "Call back after 3pm and ask for the owner." in context
    assert "Dana speaking." not in context  # no full transcript in the prompt context
    assert inherited["business_name"] == "Dana's Fencing"
    assert inherited["phone_number"] == "+14155550123"
    assert overridden["business_name"] == "Dana's New Line"
    assert overridden["phone_number"] == "+14155559999"


def test_follow_up_prompt_contains_previous_facts(settings):
    store = CallStore(":memory:")
    prior = _completed_prior(store)
    context = build_follow_up_context(store, prior["id"])
    task = parse_task(
        {
            "kind": "general",
            "business_name": prior["business_name"],
            "phone_number": prior["phone_number"],
            "customer_name": "Angi",
            "goal": "Ask the owner for the quote range.",
            "questions": ["What is the quote range?"],
            "shareable_details": {},
            "authority": "info_only",
            "follow_up_of": prior["id"],
            "previous_call_context": context,
        }
    )
    prompt = build_system_prompt(task)

    assert "Treat this as a continuation" in prompt
    assert "Dana" in prompt
    assert "after 3pm" in prompt
    assert "Do not re-ask questions already answered" in prompt


async def test_follow_up_mcp_end_to_end_inherits_and_reports(settings):
    store = CallStore(":memory:")
    async with Client(build_server(settings, store=store)) as client:
        first = _data(
            await client.call_tool(
                "place_call",
                {
                    "business_name": "Dana's Fencing",
                    "phone_number": "+14155550123",
                    "customer_name": "Angi",
                    "assistant_name": "Eva",
                    "goal": "Ask if they repair leaning fences and when to call back.",
                    "questions": ["Do you repair leaning fences?", "When should I call back?"],
                },
            )
        )
        for _ in range(50):
            first_status = _data(
                await client.call_tool("get_call_status", {"call_id": first["call_id"]})
            )
            if first_status["done"]:
                break
            await asyncio.sleep(0.02)

        store.update_call(
            first["call_id"],
            summary="They repair leaning fences; call back after 3pm for the owner.",
            details={
                **first_status["details"],
                "contact_person": "Dana",
                "follow_up": "Call back after 3pm for the owner.",
                "answers": [
                    {"question": "Do you repair leaning fences?", "answer": "Yes."},
                    {"question": "When should I call back?", "answer": "After 3pm."},
                ],
            },
        )
        second = _data(
            await client.call_tool(
                "place_call",
                {
                    "customer_name": "Angi",
                    "goal": "Ask the owner for a rough quote range.",
                    "questions": ["What is the rough quote range?"],
                    "follow_up_of": first["call_id"],
                },
            )
        )
        for _ in range(50):
            second_status = _data(
                await client.call_tool("get_call_status", {"call_id": second["call_id"]})
            )
            if second_status["done"]:
                break
            await asyncio.sleep(0.02)
        listed = _data(await client.call_tool("list_calls", {}))

    second_record = store.get_call(second["call_id"])
    prompt = build_system_prompt(parse_task(second_record["task"]))
    assert second["business_name"] == "Dana's Fencing"
    assert second["phone_number"] == "+14155550123"
    assert second["follow_up_of"] == first["call_id"]
    assert second_record["follow_up_of"] == first["call_id"]
    assert "Dana" in prompt and "After 3pm" in prompt
    assert second_status["report"]["follow_up_of"] == first["call_id"]
    assert listed["calls"][0]["follow_up_of"] == first["call_id"]
