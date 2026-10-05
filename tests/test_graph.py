from typing import Any

import pytest
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from muse_voice_agent.graph import CallOutcome, build_call_graph
from muse_voice_agent.tasks import RestaurantReservation


class FakeToolModel(GenericFakeChatModel):
    """Fake chat model that streams both text and tool calls."""

    def bind_tools(self, tools: Any, **kwargs: Any):  # type: ignore[override]
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        msg = next(self.messages)
        for word in msg.content.split(" ") if msg.content else []:
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " ", id="fake"))
        if msg.tool_calls:
            chunks = [
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=chunks, id="fake")
            )


def _task() -> RestaurantReservation:
    return RestaurantReservation(
        business_name="Luigi's",
        phone_number="(415) 555-0123",
        customer_name="Angi",
        party_size=2,
        date="Friday",
        time="7pm",
    )


async def _run(graph, messages) -> list[str]:
    spoken: list[str] = []
    async for item in graph.astream({"messages": messages}, stream_mode="custom"):
        spoken.append(item)
    return spoken


@pytest.mark.asyncio
async def test_speaks_text_without_calling_tool():
    outcomes: list[CallOutcome] = []
    model = FakeToolModel(messages=iter([AIMessage(content="Hi, this is an AI assistant.")]))
    graph = build_call_graph(_task(), outcomes.append, model=model)

    spoken = await _run(graph, [HumanMessage(content="Hello, Luigi's.")])

    assert "".join(spoken).strip() == "Hi, this is an AI assistant."
    assert outcomes == []


@pytest.mark.asyncio
async def test_record_outcome_then_goodbye_is_spoken_but_tool_output_is_not():
    outcomes: list[CallOutcome] = []
    tool_call = {
        "name": "record_outcome",
        "args": {
            "outcome": "booked",
            "summary": "Table for 2 Friday 7pm under Angi.",
            "confirmed_time": "7pm",
            "party_size": 2,
        },
        "id": "call_1",
    }
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="Thanks so much, goodbye!"),
            ]
        )
    )
    graph = build_call_graph(_task(), outcomes.append, model=model)

    spoken = await _run(graph, [HumanMessage(content="You're all set for Friday at 7.")])

    text = "".join(spoken)
    assert "goodbye" in text
    assert "Outcome recorded" not in text
    assert len(outcomes) == 1
    assert outcomes[0].outcome == "booked"
    assert outcomes[0].party_size == 2


@pytest.mark.asyncio
async def test_livekit_llm_adapter_streams_only_spoken_text():
    from livekit.agents.llm import ChatContext
    from livekit.plugins import langchain

    outcomes: list[CallOutcome] = []
    tool_call = {
        "name": "record_outcome",
        "args": {"outcome": "unavailable", "summary": "Fully booked Friday."},
        "id": "call_2",
    }
    model = FakeToolModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="Okay, thanks, bye!")]
        )
    )
    adapter = langchain.LLMAdapter(
        graph=build_call_graph(_task(), outcomes.append, model=model), stream_mode="custom"
    )
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Sorry, we're fully booked Friday.")

    spoken = ""
    async with adapter.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                spoken += chunk.delta.content

    assert spoken.strip() == "Okay, thanks, bye!"
    assert outcomes[0].outcome == "unavailable"


async def test_info_only_call_cannot_report_a_booking():
    from muse_voice_agent.tasks import GeneralCall

    task = GeneralCall(
        business_name="Hotel Zed",
        phone_number="+14155550100",
        customer_name="Angi",
        goal="Check king room availability Oct 10-12",
        questions=["Is a king room available?"],
    )
    outcomes: list[CallOutcome] = []
    tool_call = {
        "name": "record_outcome",
        "args": {
            "outcome": "booked",
            "summary": "Booked a king room.",
            "answers": [{"question": "Is a king room available?", "answer": "yes, $210"}],
        },
        "id": "call_1",
    }
    model = FakeToolModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="Bye!")])
    )
    graph = build_call_graph(task, outcomes.append, model=model)
    await _run(graph, [HumanMessage(content="Hotel Zed, how can I help?")])
    assert outcomes[0].outcome == "needs_followup"
    assert "not authorized" in outcomes[0].follow_up
    assert outcomes[0].answers[0].answer == "yes, $210"
