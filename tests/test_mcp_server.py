import asyncio
import sqlite3
from dataclasses import replace

import httpx
import pytest
from mcp import Client

from muse_voice_agent import dispatcher
from muse_voice_agent.config import Settings
from muse_voice_agent.mcp_server import build_app, build_server
from muse_voice_agent.store import CallStore


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


async def test_lists_expected_tools(settings):
    async with Client(build_server(settings)) as client:
        listed = (await client.list_tools()).tools
        tools = {t.name for t in listed}
    assert [t.name for t in listed][:3] == [
        "place_call",
        "book_restaurant_reservation",
        "request_handyman_quote",
    ]
    assert tools == {
        "book_restaurant_reservation",
        "request_handyman_quote",
        "place_call",
        "get_call_status",
        "list_calls",
        "get_updates",
    }


async def test_restaurant_booking_dry_run_end_to_end(settings):
    async with Client(build_server(settings)) as client:
        started = _data(
            await client.call_tool(
                "book_restaurant_reservation",
                {
                    "restaurant_name": "Luigi's",
                    "phone_number": "(415) 555-0123",
                    "customer_name": "Angi",
                    "party_size": 4,
                    "date": "Friday",
                    "time": "7pm",
                },
            )
        )
        assert started["phone_number"] == "+14155550123"
        assert started["done"] is False

        for _ in range(50):
            status = _data(
                await client.call_tool(
                    "get_call_status", {"call_id": started["call_id"], "include_transcript": True}
                )
            )
            if status["done"]:
                break
            await asyncio.sleep(0.02)

    assert status["status"] == "completed"
    assert status["outcome"] == "booked"
    assert status["details"]["party_size"] == 4
    assert status["simulated"] is True
    assert len(status["transcript"]) >= 2
    assert set(status["transcript"][0]) == {"speaker", "text"}
    report = status["report"]
    assert report["request"]["party_size"] == 4
    assert report["committed_on_users_behalf"] is True
    assert report["outcome_source"] == "agent"
    assert report["reached"] == "person"
    assert report["duration_seconds"] is not None
    assert report["ended_by"] == "assistant"
    assert "calendar" in report["next_steps"][0]


async def test_report_and_transcript_only_once_done(settings):
    async with Client(build_server(settings)) as client:
        started = _data(
            await client.call_tool(
                "place_call",
                {
                    "business_name": "Hotel Zed",
                    "phone_number": "+14155550100",
                    "customer_name": "Angi",
                    "goal": "Check rooms",
                },
            )
        )
        early = _data(await client.call_tool("get_call_status", {"call_id": started["call_id"]}))
        assert early["done"] is False
        assert "report" not in early and "transcript" not in early
        for _ in range(50):
            status = _data(
                await client.call_tool(
                    "get_call_status", {"call_id": started["call_id"], "include_transcript": False}
                )
            )
            if status["done"]:
                break
            await asyncio.sleep(0.02)
        listed = _data(await client.call_tool("list_calls", {}))
    assert "report" in status and "transcript" not in status
    assert "report" not in listed["calls"][0]


async def test_handyman_quote_dry_run(settings):
    async with Client(build_server(settings)) as client:
        started = _data(
            await client.call_tool(
                "request_handyman_quote",
                {
                    "business_name": "Bob's Fixit",
                    "phone_number": "+14155550199",
                    "customer_name": "Angi",
                    "job_description": "replace a leaking kitchen faucet",
                    "location": "94110",
                },
            )
        )
        await asyncio.sleep(0.2)
        status = _data(await client.call_tool("get_call_status", {"call_id": started["call_id"]}))
        listed = _data(await client.call_tool("list_calls", {}))
    assert status["outcome"] == "quote_received"
    assert listed["calls"][0]["call_id"] == started["call_id"]


async def test_original_call_status_surfaces_inbound_callback(settings):
    store = CallStore(settings.call_db_path)
    parent = store.create_call(
        {
            "kind": "general",
            "business_name": "Hotel Zed",
            "phone_number": "+14155550100",
            "customer_name": "Angi",
            "goal": "Check rooms",
            "questions": [],
            "shareable_details": {},
            "authority": "info_only",
            "limits": None,
            "callback_number": None,
            "assistant_name": None,
        },
        dry_run=False,
    )
    store.update_call(parent["id"], status="completed", outcome="needs_followup", summary="They will call back.")
    callback = store.create_call(
        {
            **parent["task"],
            "callback_of": parent["id"],
            "callback_context": "prior call context",
        },
        dry_run=False,
        direction="inbound",
        parent_call_id=parent["id"],
        status="completed",
    )
    store.update_call(
        callback["id"],
        outcome="info_received",
        summary="They called back with room availability.",
        outcome_source="agent",
    )

    async with Client(build_server(settings, store)) as client:
        status = _data(await client.call_tool("get_call_status", {"call_id": parent["id"]}))
        listed = _data(await client.call_tool("list_calls", {"limit": 2}))

    assert status["callbacks"][0]["call_id"] == callback["id"]
    assert status["callbacks"][0]["callback_of"] == parent["id"]
    assert status["callbacks"][0]["report"]["outcome_source"] == "agent"
    assert listed["calls"][0]["direction"] == "inbound"
    assert listed["calls"][0]["callback_of"] == parent["id"]


async def test_general_call_dry_run_returns_answers(settings):
    async with Client(build_server(settings)) as client:
        started = _data(
            await client.call_tool(
                "place_call",
                {
                    "business_name": "Hotel Zed",
                    "phone_number": "+14155550100",
                    "customer_name": "Angi",
                    "goal": "Check king room availability for Oct 10-12",
                    "questions": ["Is a king room available?", "Nightly rate incl. tax?"],
                    "shareable_details": {"dates": "Oct 10-12", "guests": "2 adults"},
                },
            )
        )
        assert started["kind"] == "general"
        await asyncio.sleep(0.2)
        status = _data(await client.call_tool("get_call_status", {"call_id": started["call_id"]}))
    assert status["outcome"] == "info_received"
    assert [a["question"] for a in status["details"]["answers"]] == [
        "Is a king room available?",
        "Nightly rate incl. tax?",
    ]
    assert [a["question"] for a in status["report"]["answers"]] == [
        "Is a king room available?",
        "Nightly rate incl. tax?",
    ]
    assert status["report"]["request"]["shareable_details"] == {
        "customer name": "Angi",
        "dates": "Oct 10-12",
        "guests": "2 adults",
    }


async def test_assistant_name_is_spoken_in_opener(settings):
    async with Client(build_server(settings)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        for name in ("place_call", "book_restaurant_reservation", "request_handyman_quote"):
            assert "assistant_name" in tools[name].input_schema["properties"]
        started = _data(
            await client.call_tool(
                "place_call",
                {
                    "business_name": "Hotel Zed",
                    "phone_number": "+14155550100",
                    "customer_name": "Wenjing Yu",
                    "assistant_name": "Eva",
                    "goal": "Check rooms",
                },
            )
        )
        for _ in range(50):
            status = _data(await client.call_tool("get_call_status", {"call_id": started["call_id"]}))
            if status["done"]:
                break
            await asyncio.sleep(0.02)
    assert status["transcript"][1]["text"].startswith(
        "Hi, this is Eva, an AI assistant calling on behalf of Wenjing Yu. This call may be recorded."
    )
    assert status["report"]["request"]["assistant_name"] == "Eva"


async def test_place_call_schema_is_general_purpose(settings):
    async with Client(build_server(settings)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    tool = tools["place_call"]
    description = " ".join((tool.description or "").split()).lower()
    schema = tool.input_schema
    authority = schema["properties"]["authority"]

    assert "general-purpose phone agent" in (tool.annotations.title or "").lower()
    assert "food or drink orders" in description
    assert "scheduling/rescheduling/cancelling" in description
    assert "may_commit_within_limits" in authority["enum"]
    assert "may_book_within_limits" in authority["enum"]
    assert "book/order/schedule/cancel" in description


async def test_general_call_validation(settings):
    base = {
        "business_name": "Spa",
        "phone_number": "+14155550100",
        "customer_name": "Angi",
        "goal": "Book a massage",
    }
    async with Client(build_server(settings)) as client:
        no_limits = _data(
            await client.call_tool("place_call", {**base, "authority": "may_commit_within_limits"})
        )
        alias = _data(
            await client.call_tool(
                "place_call",
                {
                    **base,
                    "authority": "may_book_within_limits",
                    "limits": "today after 4 PM, under $100, no deposit",
                },
            )
        )
        card = _data(
            await client.call_tool(
                "place_call", {**base, "shareable_details": {"card": "4111 1111 1111 1111"}}
            )
        )
        ssn = _data(
            await client.call_tool(
                "place_call", {**base, "goal": "Book a massage for SSN 123-45-6789"}
            )
        )
        tracking = _data(
            await client.call_tool(
                "place_call", {**base, "shareable_details": {"tracking": "1234567890123456"}}
            )
        )
    assert no_limits["error"] == "invalid_request" and "limits" in no_limits["message"]
    assert alias["error"] is None and alias["kind"] == "general"
    assert card["error"] == "invalid_request" and "card" in card["message"]
    assert ssn["error"] == "invalid_request" and "social security" in ssn["message"]
    assert tracking["error"] is None and tracking["kind"] == "general"  # long non-card numbers (order/tracking) are allowed


async def test_rejects_disallowed_country_and_bad_numbers(settings):
    async with Client(build_server(settings)) as client:
        foreign = _data(
            await client.call_tool(
                "book_restaurant_reservation",
                {
                    "restaurant_name": "Le Bistro",
                    "phone_number": "+33142685300",
                    "customer_name": "Angi",
                    "party_size": 2,
                    "date": "Friday",
                    "time": "8pm",
                },
            )
        )
        bad = _data(
            await client.call_tool(
                "book_restaurant_reservation",
                {
                    "restaurant_name": "X",
                    "phone_number": "12",
                    "customer_name": "Angi",
                    "party_size": 2,
                    "date": "Friday",
                    "time": "8pm",
                },
            )
        )
    assert foreign["error"] == "rejected"
    assert bad["error"] == "invalid_request"
    assert CallStore(settings.call_db_path).list_calls() == []


async def test_live_mode_requires_livekit_config(settings):
    live = replace(
        settings, dry_run=False, voice_backend="livekit", livekit_url="", sip_outbound_trunk_id=""
    )
    async with Client(build_server(live)) as client:
        res = _data(
            await client.call_tool(
                "request_handyman_quote",
                {
                    "business_name": "Bob's",
                    "phone_number": "+14155550199",
                    "customer_name": "Angi",
                    "job_description": "fix a door",
                    "location": "94110",
                },
            )
        )
    assert res["error"] == "rejected"
    assert "SIP_OUTBOUND_TRUNK_ID" in res["message"]


async def test_http_app_requires_bearer_token(settings):
    app = build_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tunnel.example") as http:
        assert (await http.get("/healthz")).status_code == 200
        assert (await http.post("/mcp", json={})).status_code == 401
        wrong = await http.post("/mcp", json={}, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401


async def test_customer_name_is_required(settings):
    async with Client(build_server(settings)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        for name in ("book_restaurant_reservation", "request_handyman_quote", "place_call"):
            assert "customer_name" in tools[name].input_schema["required"], name

        base = {"business_name": "Spa", "phone_number": "+14155550100", "goal": "Ask hours"}
        missing = await client.call_tool("place_call", base)
        assert missing.is_error
        for bad in ("   ", "the user", "Unknown"):
            result = _data(await client.call_tool("place_call", {**base, "customer_name": bad}))
            assert result["error"] == "invalid_request" and "ask the user" in result["message"]
        ok = _data(await client.call_tool("place_call", {**base, "customer_name": "  Wenjing   Yu "}))
    assert ok["error"] is None


def _inbound(store: CallStore, *, parent: dict | None, status: str = "completed", **fields) -> dict:
    record = store.create_call(
        {"kind": "inbound_message", "business_name": "Unknown caller", "phone_number": "+14155550999",
         "customer_name": None} if parent is None else {**parent["task"], "callback_of": parent["id"]},
        dry_run=False,
        direction="inbound",
        parent_call_id=parent["id"] if parent else None,
        status=status,
        origin=parent.get("origin") if parent else None,
    )
    store.update_call(record["id"], **fields)
    return store.get_call(record["id"])


async def test_calls_placed_over_mcp_are_tagged_with_client(settings):
    store = CallStore(settings.call_db_path)
    async with Client(build_server(settings, store)) as client:
        started = _data(
            await client.call_tool(
                "place_call",
                {"business_name": "Hotel Zed", "phone_number": "+14155550100", "customer_name": "Angi",
                 "goal": "Check rooms"},
            )
        )
    assert store.get_call(started["call_id"])["origin"].startswith("mcp")


async def test_get_updates_returns_each_finished_callback_once(settings):
    store = CallStore(settings.call_db_path)
    parent = store.create_call(
        {"kind": "general", "business_name": "Hotel Zed", "phone_number": "+14155550100",
         "customer_name": "Angi", "goal": "Check rooms"},
        dry_run=False,
        origin="mcp:muse",
    )
    store.update_call(parent["id"], status="completed", outcome="voicemail", summary="Left a voicemail.")
    callback = _inbound(store, parent=parent, outcome="info_received", summary="Rooms available.",
                        end_reason="user_hangup")
    live = _inbound(store, parent=parent, outcome="info_received", summary="Still talking.")
    message = _inbound(store, parent=None, outcome="needs_followup", summary="Caller left a message.",
                       end_reason="user_hangup")
    stale = _inbound(store, parent=parent, outcome="info_received", summary="Ended without a final event.")
    with sqlite3.connect(settings.call_db_path) as c:
        c.execute("UPDATE calls SET updated_at = 0 WHERE id = ?", (stale["id"],))
    not_from_mcp = store.create_call({**parent["task"]}, dry_run=False)
    _inbound(store, parent=not_from_mcp, outcome="info_received", end_reason="user_hangup")

    async with Client(build_server(settings, store)) as client:
        listed = _data(await client.call_tool("list_calls", {}))
        first = _data(await client.call_tool("get_updates", {}))
        again = _data(await client.call_tool("get_updates", {}))
        listed_after = _data(await client.call_tool("list_calls", {}))

    assert listed["unread_updates"] == 3
    ids = [u["callback"]["call_id"] for u in first["updates"]]
    assert ids == [callback["id"], message["id"], stale["id"]]
    assert live["id"] not in ids
    by_id = {u["callback"]["call_id"]: u for u in first["updates"]}
    assert by_id[callback["id"]]["kind"] == "callback"
    assert by_id[callback["id"]]["original_call"]["summary"] == "Left a voicemail."
    assert by_id[message["id"]]["kind"] == "message"
    assert "original_call" not in by_id[message["id"]]
    assert again == {"updates": [], "remaining": 0}
    assert "unread_updates" not in listed_after


async def test_get_updates_respects_limit(settings):
    store = CallStore(settings.call_db_path)
    for _ in range(3):
        _inbound(store, parent=None, outcome="needs_followup", end_reason="user_hangup")
    async with Client(build_server(settings, store)) as client:
        page = _data(await client.call_tool("get_updates", {"limit": 2}))
        rest = _data(await client.call_tool("get_updates", {"limit": 2}))
    assert (len(page["updates"]), page["remaining"]) == (2, 1)
    assert (len(rest["updates"]), rest["remaining"]) == (1, 0)
