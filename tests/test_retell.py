import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from langchain_core.messages import AIMessage
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_graph import FakeToolModel

from muse_voice_agent import retell
from muse_voice_agent.config import Settings
from muse_voice_agent.dispatcher import start_call
from muse_voice_agent.mcp_server import build_app
from muse_voice_agent.store import CallStore
from muse_voice_agent.tasks import RestaurantReservation

SECRET = "s3cret-path"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        dry_run=False,
        voice_backend="retell",
        mcp_auth_token="test-token",
        call_db_path=tmp_path / "calls.db",
        allowed_dial_prefixes=["+1"],
        retell_api_key="key_test",
        retell_agent_id="agent_test",
        retell_from_number="+16282779475",
        retell_ws_secret=SECRET,
        public_base_url="https://example.trycloudflare.com",
        silent_pickup_ms=0,
    )


def _task() -> RestaurantReservation:
    return RestaurantReservation(
        business_name="Luigi's",
        phone_number="+14155550123",
        customer_name="Angi",
        party_size=2,
        date="Friday",
        time="7pm",
    )


def _collect(ws, response_id: int) -> tuple[str, dict]:
    text = ""
    while True:
        msg = ws.receive_json()
        assert msg["response_type"] == "response"
        assert msg["response_id"] == response_id
        text += msg["content"]
        if msg["content_complete"]:
            return text, msg


def test_websocket_url_is_derived_from_public_base_url(settings):
    assert settings.retell_llm_websocket_url() == (
        f"wss://example.trycloudflare.com/retell/llm/{SECRET}"
    )


def test_custom_llm_websocket_conversation(settings):
    store = CallStore(settings.call_db_path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    tool_call = {
        "name": "record_outcome",
        "args": {"outcome": "booked", "summary": "Table for 2 Friday 7pm.", "party_size": 2},
        "id": "tc_1",
    }
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(content="I'd like to book a table for two."),
                # goodbye and record_outcome in one response: no second model pass
                AIMessage(content="Perfect, thanks so much. Bye!", tool_calls=[tool_call]),
                AIMessage(content="Sure, it's A, N, G, I.", tool_calls=[]),
            ]
        )
    )
    app = build_app(settings, store, model_factory=lambda: model)

    with TestClient(app).websocket_connect(f"/retell/llm/{SECRET}/rc_1") as ws:
        assert ws.receive_json() == {
            "response_type": "config",
            "config": {"auto_reconnect": True, "call_details": True},
        }
        begin = ws.receive_json()
        assert begin["response_id"] == 0 and begin["content"] == "" and begin["content_complete"]

        ws.send_json({"interaction_type": "ping_pong", "timestamp": 123})
        assert ws.receive_json() == {"response_type": "ping_pong", "timestamp": 123}

        ws.send_json(
            {"interaction_type": "call_details", "call": {"metadata": {"muse_call_id": call_id}}}
        )
        transcript = [{"role": "user", "content": "Hello, Luigi's."}]
        ws.send_json(
            {"interaction_type": "response_required", "response_id": 1, "transcript": transcript}
        )
        text, last = _collect(ws, 1)
        assert text.strip() == (
            "Hi, this is an assistant calling on behalf of Angi. I'd like to book a table for two."
        )
        assert last["end_call"] is False
        assert store.get_call(call_id)["status"] == "in_progress"

        transcript += [
            {"role": "agent", "content": text.strip()},
            {"role": "user", "content": "Sure, Friday at 7 for two under Angi, you're all set."},
        ]
        ws.send_json(
            {"interaction_type": "response_required", "response_id": 2, "transcript": transcript}
        )
        text, last = _collect(ws, 2)
        assert text.strip() == "Perfect, thanks so much. Bye!"
        assert last["end_call"] is False  # stay on in case they have a question
        assert ws.receive_json() == {
            "response_type": "update_agent",
            "agent_config": {"reminder_trigger_ms": 4000, "reminder_max_count": 1},
        }
        assert store.get_call(call_id)["outcome"] == "booked"

        transcript += [
            {"role": "agent", "content": text.strip()},
            {"role": "user", "content": "Oh wait, how do you spell the name?"},
        ]
        ws.send_json(
            {"interaction_type": "response_required", "response_id": 3, "transcript": transcript}
        )
        text, last = _collect(ws, 3)
        assert text.strip() == "Sure, it's A, N, G, I."
        assert last["end_call"] is False

        transcript += [
            {"role": "agent", "content": text.strip()},
            {"role": "user", "content": "Got it, thanks, bye!"},
        ]
        ws.send_json(
            {"interaction_type": "response_required", "response_id": 4, "transcript": transcript}
        )
        text, last = _collect(ws, 4)
        assert text.strip() == "Bye!"  # answered without the model
        assert last["end_call"] is True

    record = store.get_call(call_id)
    assert record["status"] == "completed"
    assert record["outcome"] == "booked"
    assert record["provider_call_id"] == "rc_1"
    assert [t["role"] for t in record["transcript"]][:3] == ["user", "assistant", "user"]


def _bound_session(settings, store, model, path="rc_1"):
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    app = build_app(settings, store, model_factory=lambda: model)
    client = TestClient(app)
    return call_id, client, f"/retell/llm/{SECRET}/{path}"


def test_silence_after_goodbye_hangs_up_but_not_on_hold(settings):
    store = CallStore(settings.call_db_path)
    tool_call = {"name": "record_outcome", "args": {"outcome": "booked", "summary": "ok"}, "id": "t"}
    hold = {"name": "wait_on_hold", "args": {}, "id": "h"}
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(content="Book two for Friday at 7?"),
                AIMessage(content="Sure, no problem.", tool_calls=[hold]),
                AIMessage(content="Great, thanks. Bye!", tool_calls=[tool_call]),
            ]
        )
    )
    call_id, client, path = _bound_session(settings, store, model)
    with client.websocket_connect(path) as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {"muse_call_id": call_id}}})
        t = [{"role": "user", "content": "Hello?"}]
        ws.send_json({"interaction_type": "response_required", "response_id": 1, "transcript": t})
        _collect(ws, 1)
        t += [{"role": "agent", "content": "..."}, {"role": "user", "content": "Hold on one sec."}]
        ws.send_json({"interaction_type": "response_required", "response_id": 2, "transcript": t})
        text, last = _collect(ws, 2)
        assert text.strip() == "Sure, no problem." and last["end_call"] is False
        ws.send_json({"interaction_type": "reminder_required", "response_id": 3, "transcript": t})
        text, last = _collect(ws, 3)
        assert text == "" and last["end_call"] is False  # quiet on hold
        t += [{"role": "user", "content": "Okay, you're booked."}]
        ws.send_json({"interaction_type": "response_required", "response_id": 4, "transcript": t})
        _collect(ws, 4)
        assert ws.receive_json()["response_type"] == "update_agent"
        ws.send_json({"interaction_type": "reminder_required", "response_id": 5, "transcript": t})
        text, last = _collect(ws, 5)
        assert text == "" and last["end_call"] is True


def test_press_digits_is_sent_after_speech(settings):
    store = CallStore(settings.call_db_path)
    press = {"name": "press_digits", "args": {"digits": "1"}, "id": "p"}
    model = FakeToolModel(messages=iter([AIMessage(content="Calling for Angi.", tool_calls=[press])]))
    call_id, client, path = _bound_session(settings, store, model)
    with client.websocket_connect(path) as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {"muse_call_id": call_id}}})
        t = [{"role": "user", "content": "To be connected, press 1."}]
        ws.send_json({"interaction_type": "response_required", "response_id": 1, "transcript": t})
        text, last = _collect(ws, 1)
        assert "Calling for Angi." in text
        assert last["digit_to_press"] == "1" and last["end_call"] is False


def test_silent_pickup_speaks_first_with_protected_opener(settings, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"call_id": "rc_1", "call_status": "ongoing"})

    monkeypatch.setattr(
        retell,
        "client_factory",
        lambda s: retell.RetellClient(s.retell_api_key, transport=httpx.MockTransport(handler)),
    )
    settings = replace(settings, silent_pickup_ms=50)
    store = CallStore(settings.call_db_path)
    model = FakeToolModel(messages=iter([AIMessage(content="I'd like to book a table for two.")]))
    call_id, client, path = _bound_session(settings, store, model)
    with client.websocket_connect(path) as ws:
        ws.receive_json(), ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {"muse_call_id": call_id}}})
        events = []
        while True:
            msg = ws.receive_json()
            assert msg["response_type"] == "agent_interrupt" and msg["interrupt_id"] == 1
            events.append(msg)
            if msg["content_complete"]:
                break
    assert events[0]["content"].startswith("Hi, this is an assistant calling on behalf of Angi.")
    assert events[0]["no_interruption_allowed"] is True
    assert "no_interruption_allowed" not in events[1]
    assert "".join(e["content"] for e in events).strip().endswith("book a table for two.")


def test_websocket_rejects_wrong_secret(settings):
    app = build_app(settings, CallStore(settings.call_db_path))
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/retell/llm/wrong/rc_1") as ws:
            ws.receive_json()


def test_websocket_rejects_unknown_call(settings):
    app = build_app(settings, CallStore(settings.call_db_path))
    with TestClient(app).websocket_connect(f"/retell/llm/{SECRET}/rc_1") as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_json({"interaction_type": "call_details", "call": {"metadata": {"muse_call_id": "nope"}}})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


def test_http_still_requires_bearer_token(settings):
    app = build_app(settings, CallStore(settings.call_db_path))
    assert TestClient(app).post("/mcp", json={}).status_code == 401


async def test_dispatcher_places_retell_call_and_finalizes(settings, monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v2/create-phone-call":
            return httpx.Response(201, json={"call_id": "rc_9", "call_status": "registered"})
        if request.url.path == "/v2/get-call/rc_9":
            return httpx.Response(
                200,
                json={
                    "call_id": "rc_9",
                    "call_status": "ended",
                    "disconnection_reason": "dial_no_answer",
                    "transcript_object": [],
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        retell,
        "client_factory",
        lambda s: retell.RetellClient(s.retell_api_key, transport=httpx.MockTransport(handler)),
    )
    original_monitor = retell.monitor_call
    monkeypatch.setattr(
        retell,
        "monitor_call",
        lambda *a, **kw: original_monitor(*a, poll_seconds=0.01, **kw),
    )

    store = CallStore(settings.call_db_path)
    record = await start_call(_task(), settings, store)
    assert record["status"] == "dialing"
    assert record["provider_call_id"] == "rc_9"

    body = json.loads(requests[0].content)
    assert body == {
        "from_number": "+16282779475",
        "to_number": "+14155550123",
        "override_agent_id": "agent_test",
        "metadata": {"muse_call_id": record["id"]},
    }
    assert requests[0].headers["authorization"] == "Bearer key_test"

    for _ in range(200):
        if store.get_call(record["id"])["status"] == "no_answer":
            break
        await asyncio.sleep(0.01)
    final = store.get_call(record["id"])
    assert final["status"] == "no_answer"
    assert final["error"] == "dial_no_answer"


@pytest.mark.parametrize(
    "reason,status,outcome",
    [
        ("user_hangup", "completed", "needs_followup"),
        ("voicemail_reached", "completed", "voicemail"),
        ("dial_busy", "no_answer", None),
        ("error_llm_websocket_open", "failed", None),
    ],
)
def test_finalize_maps_disconnection_reasons(settings, reason, status, outcome):
    store = CallStore(settings.call_db_path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    retell.finalize_from_retell(
        store,
        call_id,
        {
            "call_status": "ended",
            "disconnection_reason": reason,
            "transcript_object": [{"role": "user", "content": "Hello?"}],
        },
    )
    record = store.get_call(call_id)
    assert (record["status"], record["outcome"]) == (status, outcome)
    assert record["transcript"] == [{"role": "user", "text": "Hello?"}]


def test_finalize_records_timing_and_end_reason(settings):
    store = CallStore(settings.call_db_path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    store.update_call(call_id, status="completed", outcome="booked", summary="ok", outcome_source="agent")
    retell.finalize_from_retell(
        store,
        call_id,
        {
            "call_status": "ended",
            "disconnection_reason": "agent_hangup",
            "start_timestamp": 1_700_000_000_000,
            "end_timestamp": 1_700_000_075_500,
        },
    )
    record = store.get_call(call_id)
    assert (record["started_at"], record["ended_at"]) == (1_700_000_000.0, 1_700_000_075.5)
    assert record["end_reason"] == "agent_hangup"
    assert (record["outcome"], record["outcome_source"]) == ("booked", "agent")


def test_finalize_fallback_marks_call_system_source(settings):
    store = CallStore(settings.call_db_path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    retell.finalize_from_retell(
        store, call_id, {"call_status": "ended", "disconnection_reason": "user_hangup"}
    )
    assert store.get_call(call_id)["outcome_source"] == "call_system"


def test_store_migrates_old_database(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute(
            "CREATE TABLE calls (id TEXT PRIMARY KEY, kind TEXT NOT NULL, business_name TEXT NOT NULL,"
            " phone_number TEXT NOT NULL, task_json TEXT NOT NULL, status TEXT NOT NULL,"
            " outcome TEXT, summary TEXT, details_json TEXT, transcript_json TEXT NOT NULL DEFAULT '[]',"
            " error TEXT, dry_run INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,"
            " updated_at REAL NOT NULL)"
        )
    store = CallStore(path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    store.update_call(call_id, provider_call_id="rc_1", started_at=1.0, end_reason="agent_hangup")
    record = store.get_call(call_id)
    assert (record["provider_call_id"], record["started_at"], record["outcome_source"]) == ("rc_1", 1.0, None)


def test_finalize_keeps_recorded_outcome(settings):
    store = CallStore(settings.call_db_path)
    call_id = store.create_call(_task().model_dump(), dry_run=False)["id"]
    store.update_call(call_id, status="completed", outcome="booked", summary="ok")
    retell.finalize_from_retell(
        store, call_id, {"call_status": "ended", "disconnection_reason": "agent_hangup"}
    )
    assert store.get_call(call_id)["outcome"] == "booked"
