from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.generated.session_events import (  # noqa: E402
    AssistantMessageData,
    AssistantMessageDeltaData,
    AssistantMessageToolRequest,
    AssistantUsageData,
    SessionIdleData,
)

from evals import copilot_llm  # noqa: E402
from evals.copilot_llm import ChatCopilot, make_chat_model, parse_model_name, render_messages  # noqa: E402
from evals.judge import normalize_usage  # noqa: E402
from evals.text import DEFAULT_AGENT_MODEL, PROD_AGENT_MODEL, build_parser  # noqa: E402


@dataclass
class _Ev:
    data: Any


class FakeSession:
    def __init__(self, script: list[Any], created: dict[str, Any]):
        self.script = script
        self.created = created
        self.handler = None
        self.sent: list[str] = []
        self.disconnected = False

    def on(self, handler):
        self.handler = handler
        return lambda: None

    async def send(self, prompt: str) -> str:
        self.sent.append(prompt)
        for data in self.script:
            self.handler(_Ev(data))
        return "msg-1"

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeClient:
    def __init__(self, script: list[Any]):
        self.script = script
        self.sessions: list[FakeSession] = []

    async def create_session(self, **kwargs: Any) -> FakeSession:
        s = FakeSession(self.script, kwargs)
        self.sessions.append(s)
        return s


@pytest.fixture(autouse=True)
def _fresh_pools():
    copilot_llm._POOLS.clear()
    yield
    copilot_llm._POOLS.clear()


def _fake(monkeypatch, script: list[Any]) -> FakeClient:
    client = FakeClient(script)

    async def _client():
        return client

    monkeypatch.setattr(copilot_llm, "_client", _client)
    return client


def _usage(cost: float) -> AssistantUsageData:
    return AssistantUsageData(model="m", input_tokens=10, output_tokens=5, cost=cost)


def test_parse_model_name_and_routing():
    assert parse_model_name("copilot:gpt-5.4-mini@low") == ("gpt-5.4-mini", "low")
    assert parse_model_name("copilot:claude-sonnet-5.5") == ("claude-sonnet-5.5", None)
    m = make_chat_model("copilot:gpt-5.4-mini@low")
    assert isinstance(m, ChatCopilot) and m.model == "gpt-5.4-mini" and m.reasoning_effort == "low"


def test_render_messages_single_turn_and_transcript():
    system, prompt = render_messages([SystemMessage(content="sys"), HumanMessage(content="hi")])
    assert (system, prompt) == ("sys", "hi")

    system, prompt = render_messages(
        [
            SystemMessage(content="sys"),
            HumanMessage(content="Nopa, how can I help?"),
            AIMessage(content="Table for 2?", tool_calls=[{"name": "record_outcome", "args": {"outcome": "booked"}, "id": "c1"}]),
            ToolMessage(content="Outcome recorded.", tool_call_id="c1"),
        ]
    )
    assert system == "sys"
    assert "User: Nopa, how can I help?" in prompt
    assert "You: Table for 2?" in prompt
    assert 'You called tool record_outcome with {"outcome": "booked"}' in prompt
    assert "Tool result: Outcome recorded." in prompt


async def test_chat_copilot_streams_text_and_usage(monkeypatch):
    client = _fake(
        monkeypatch,
        [
            AssistantMessageDeltaData(delta_content="Hi, I'm an AI", message_id="a"),
            AssistantMessageDeltaData(delta_content=" assistant.", message_id="a"),
            AssistantMessageDeltaData(delta_content="Table for 2?", message_id="b"),
            AssistantMessageData(content="ignored, already streamed", message_id="a"),
            _usage(0.33),
            SessionIdleData(),
        ],
    )
    msg = await ChatCopilot(model="claude-haiku-4.5").ainvoke([SystemMessage(content="be brief"), HumanMessage(content="hello")])
    assert msg.content == "Hi, I'm an AI assistant. Table for 2?"
    assert msg.usage_metadata["input_tokens"] == 10
    assert normalize_usage(msg)["premium_requests"] == pytest.approx(0.33)
    created = client.sessions[0].created
    assert created["system_message"] == {"mode": "replace", "content": "be brief"}
    assert created["available_tools"] == []  # built-in agent tools disabled
    pool = next(iter(copilot_llm._POOLS.values()))
    assert list(pool.sessions.values()) == [client.sessions[0]]  # kept for the next turn
    await copilot_llm.aclose()
    assert client.sessions[0].disconnected


async def test_chat_copilot_continues_session_with_only_new_messages(monkeypatch):
    client = _fake(
        monkeypatch,
        [AssistantMessageDeltaData(delta_content="Table for 2 at 7?", message_id="a"), _usage(0.0), SessionIdleData()],
    )
    model = ChatCopilot(model="gpt-5.4-mini")
    sys_msg = SystemMessage(content="You are calling a restaurant.")
    first = await model.ainvoke([sys_msg, HumanMessage(content="Nopa, how can I help?")])
    # The caller rebuilds history from spoken text (whitespace may differ) and adds the business reply.
    await model.ainvoke(
        [sys_msg, HumanMessage(content="Nopa, how can I help?"), AIMessage(content=f"  {first.content} "), HumanMessage(content="Sure, name?")]
    )
    assert len(client.sessions) == 1
    assert client.sessions[0].sent == ["Nopa, how can I help?", "Sure, name?"]

    # A conversation the pool hasn't seen falls back to a fresh session with a rendered transcript.
    await model.ainvoke([sys_msg, HumanMessage(content="Hi"), AIMessage(content="Something else"), HumanMessage(content="ok")])
    assert len(client.sessions) == 2
    assert "Conversation so far" in client.sessions[1].sent[0]


async def test_chat_copilot_surfaces_tool_calls(monkeypatch):
    @tool
    def record_outcome(outcome: str) -> str:
        """Record the call outcome."""
        return "ok"

    client = _fake(
        monkeypatch,
        [
            AssistantMessageData(
                content="",
                message_id="a",
                tool_requests=[AssistantMessageToolRequest(name="record_outcome", tool_call_id="t1", arguments={"outcome": "booked"})],
            ),
            _usage(0.0),
            SessionIdleData(),
        ],
    )
    model = ChatCopilot(model="gpt-5.4-mini").bind_tools([record_outcome])
    msg = await model.ainvoke([HumanMessage(content="confirmed, #4471")])
    assert msg.tool_calls == [{"name": "record_outcome", "args": {"outcome": "booked"}, "id": "t1", "type": "tool_call"}]
    created = client.sessions[0].created
    assert created["available_tools"] == ["record_outcome"]
    assert created["tools"][0].is_terminal


def test_cli_defaults_to_copilot_and_latency_uses_prod_model(monkeypatch):
    from evals import text

    captured = {}

    async def fake_run_suite(args):
        captured["agent"] = args.agent_model

    monkeypatch.setattr(text, "run_suite", fake_run_suite)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    text.main([])
    assert captured["agent"] == DEFAULT_AGENT_MODEL
    text.main(["--latency"])
    assert captured["agent"] == PROD_AGENT_MODEL
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-5.4-mini")
    text.main(["--latency"])
    assert captured["agent"] == "openai:gpt-5.4-mini"
    args = build_parser().parse_args([])
    assert args.judge_model.startswith("copilot:") and args.simulator_model.startswith("copilot:")


class FlakyClient(FakeClient):
    """First session fails with a Copilot error; later sessions follow the script."""

    async def create_session(self, **kwargs: Any) -> FakeSession:
        from copilot.generated.session_events import SessionErrorData

        bad = [SessionErrorData(error_type="query", message="Failed to get response")]
        s = FakeSession(bad if not self.sessions else self.script, kwargs)
        self.sessions.append(s)
        return s


async def test_transient_error_retries_on_fresh_session(monkeypatch):
    client = FlakyClient(
        [AssistantMessageData(content="Hi there", message_id="a"), _usage(0.33), SessionIdleData()]
    )

    async def _client():
        return client

    monkeypatch.setattr(copilot_llm, "_client", _client)
    model = ChatCopilot(model="m", retry_backoff=0)
    reply = await model.ainvoke([HumanMessage(content="hello")])
    assert reply.content == "Hi there"
    assert len(client.sessions) == 2 and client.sessions[0].disconnected
    assert len(model.usage_log) == 2


async def test_persistent_error_raises(monkeypatch):
    from copilot.generated.session_events import SessionErrorData

    client = _fake(monkeypatch, [SessionErrorData(error_type="query", message="down")])
    model = ChatCopilot(model="m", retry_backoff=0, max_retries=1)
    with pytest.raises(copilot_llm.CopilotError):
        await model.ainvoke([HumanMessage(content="hello")])
    assert len(client.sessions) == 2
