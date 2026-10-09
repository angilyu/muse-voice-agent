import asyncio
import time
from typing import Any

import pytest
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from muse_voice_agent.graph import CallOutcome, _ToolMarkupFilter, build_call_graph
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
    model = FakeToolModel(messages=iter([AIMessage(content="Could I book a table for two?")]))
    graph = build_call_graph(_task(), outcomes.append, model=model)

    spoken = await _run(
        graph,
        [
            HumanMessage(content="Hello, Luigi's."),
            AIMessage(content="Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded."),
            HumanMessage(content="How can I help?"),
        ],
    )

    assert "".join(spoken).strip() == "Could I book a table for two?"
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

    assert spoken.strip() == "Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded. Okay, thanks, bye!"
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


async def test_info_only_call_cannot_report_an_order():
    from muse_voice_agent.tasks import GeneralCall

    task = GeneralCall(
        business_name="HeyTea",
        phone_number="+14155550142",
        customer_name="Wenjing Yu",
        goal="Ask whether jasmine green milk tea is available",
        questions=["Is it available?"],
    )
    outcomes: list[CallOutcome] = []
    tool_call = {
        "name": "record_outcome",
        "args": {
            "outcome": "ordered",
            "summary": "Ordered two jasmine green milk teas.",
            "order_total": "$12.80",
            "pickup_time": "15 minutes",
        },
        "id": "call_order",
    }
    model = FakeToolModel(messages=iter([AIMessage(content="Done, thanks, bye!", tool_calls=[tool_call])]))
    graph = build_call_graph(task, outcomes.append, model=model)
    await _run(graph, [HumanMessage(content="HeyTea.")])
    assert outcomes[0].outcome == "needs_followup"
    assert "not authorized to commit" in outcomes[0].follow_up
    assert outcomes[0].order_total == "$12.80"


@pytest.mark.asyncio
async def test_asked_if_robot_answer_stays_truthful():
    model = FakeToolModel(messages=iter([AIMessage(content="Yes, I'm an AI assistant calling for Angi.")]))
    graph = build_call_graph(_task(), lambda o: None, model=model)
    spoken = "".join(await _run(graph, [HumanMessage(content="Are you a robot or a real person?")]))
    assert "AI assistant" in spoken


def test_general_order_prompt_allows_commitments_and_defaults():
    from muse_voice_agent.tasks import GeneralCall, build_system_prompt

    task = GeneralCall(
        business_name="HeyTea",
        phone_number="+14155550142",
        customer_name="Wenjing Yu",
        goal="Order two jasmine green milk teas.",
        authority="may_commit_within_limits",
        limits="Two jasmine green milk teas, 25% sugar, less ice, defaults otherwise, pay at pickup.",
    )

    prompt = build_system_prompt(task)
    assert "You MAY book, order, reserve, schedule, reschedule, cancel" in prompt
    assert "default is fine" in prompt
    assert "Ask for the total and pickup/ready time" in prompt
    assert "pay-at-pickup" in prompt


def test_raw_streamed_tool_markup_is_filtered_across_chunks():
    f = _ToolMarkupFilter()
    chunks = ['<function=record_outcome ', '={"outcome":"booked"} ', "/> ", "Thanks, goodbye!"]

    spoken = "".join(f.feed(chunk) for chunk in chunks) + f.flush()

    assert "<function=" not in spoken
    assert "record_outcome" not in spoken
    assert spoken.strip() == "Thanks, goodbye!"


class SlowRecordingModel(FakeToolModel):
    """Takes 0.5 s to its first token, like a real LLM over the network."""

    seen: list = []

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        SlowRecordingModel.seen.append(list(messages))
        await asyncio.sleep(0.5)
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            yield chunk


@pytest.mark.asyncio
async def test_first_turn_speaks_fixed_opener_before_the_llm():
    opener = "Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded."
    SlowRecordingModel.seen = []
    model = SlowRecordingModel(messages=iter([AIMessage(content="Could I book a table for two?")]))
    graph = build_call_graph(_task(), lambda o: None, model=model)

    spoken = []
    first_item_after = None
    start = time.perf_counter()
    async for item in graph.astream(
        {"messages": [HumanMessage(content="Hello, Luigi's.")]}, stream_mode="custom"
    ):
        if first_item_after is None:
            first_item_after = time.perf_counter() - start
        spoken.append(item)

    assert spoken[0] == opener + " "
    assert first_item_after < 0.2  # doesn't wait for the model's first token
    assert "".join(spoken).strip() == f"{opener} Could I book a table for two?"
    last = SlowRecordingModel.seen[0][-1]
    assert isinstance(last, AIMessage) and last.content == opener  # the model knows it was said
    assert opener in SlowRecordingModel.seen[0][0].content  # and the system prompt says not to repeat it


@pytest.mark.asyncio
async def test_goodbye_with_outcome_is_one_pass_and_farewell_skips_the_model():
    from muse_voice_agent.graph import CallControl

    first = {"name": "record_outcome", "args": {"outcome": "booked", "summary": "2 at 7"}, "id": "a"}
    fixed = {"name": "record_outcome", "args": {"outcome": "booked", "summary": "2 at 7:30"}, "id": "b"}
    model = FakeToolModel(
        messages=iter(
            [
                AIMessage(content="Perfect, thanks. Bye!", tool_calls=[first]),
                AIMessage(content="Oh, 7:30 works too, thanks!", tool_calls=[fixed]),
            ]
        )
    )
    outcomes: list[CallOutcome] = []
    control = CallControl(opener_spoken=True)
    graph = build_call_graph(_task(), outcomes.append, model=model, control=control)
    history = [
        AIMessage(content="Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded. Table for two Friday at 7?"),
        HumanMessage(content="Sure, you're booked."),
    ]
    assert "".join(await _run(graph, history)).strip() == "Perfect, thanks. Bye!"
    assert control.closing and not control.end_requested and len(outcomes) == 1

    history += [AIMessage(content="Perfect, thanks. Bye!"), HumanMessage(content="Actually it's 7:30.")]
    await _run(graph, history)
    assert outcomes[-1].summary == "2 at 7:30" and not control.end_requested

    history += [AIMessage(content="Oh, 7:30 works too, thanks!"), HumanMessage(content="Okay, bye!")]
    assert "".join(await _run(graph, history)) == "Bye!"  # model iterator is exhausted: not called
    assert control.end_requested


@pytest.mark.asyncio
async def test_cut_off_intro_is_flagged_to_the_model():
    from muse_voice_agent.graph import CallControl

    SlowRecordingModel.seen = []
    model = SlowRecordingModel(messages=iter([AIMessage(content="Sorry, I'm calling for Angi.")]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=CallControl(opener_spoken=True))
    await _run(graph, [AIMessage(content="Hi,"), HumanMessage(content="Sorry, what was that about?")])
    assert "introduction got cut off" in _all_text(SlowRecordingModel.seen[0])


def _all_text(messages) -> str:
    return "\n".join(m.content if isinstance(m.content, str) else str(m.content) for m in messages)


OPENER = "Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded."


@pytest.mark.asyncio
async def test_person_greeting_after_silent_pickup_hears_opener_again():
    from muse_voice_agent.graph import CallControl

    SlowRecordingModel.seen = []
    model = SlowRecordingModel(messages=iter([AIMessage(content="I'd like a table for two Friday at 7.")]))
    control = CallControl(opener_spoken=True)
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    history = [AIMessage(content=OPENER + " I'd like a table for two Friday at 7."), HumanMessage(content="Hello?")]
    spoken = "".join(await _run(graph, history))
    assert spoken.startswith(OPENER)
    assert "table for two" in spoken
    seen = SlowRecordingModel.seen[0]
    assert seen[-2].content == OPENER  # opener already said in this turn
    assert "just picked up" in seen[-1].content and seen[0].content.endswith("(Pacific time).\n")


@pytest.mark.asyncio
async def test_second_hello_from_a_person_does_not_repeat_the_opener():
    from muse_voice_agent.graph import CallControl

    model = FakeToolModel(messages=iter([AIMessage(content="Hi! Table for two Friday at 7?")]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=CallControl(opener_spoken=True))
    history = [
        HumanMessage(content="Luigi's, how can I help?"),
        AIMessage(content=OPENER + " Table for two Friday at 7?"),
        HumanMessage(content="Hello?"),
    ]
    assert not "".join(await _run(graph, history)).startswith(OPENER)


@pytest.mark.asyncio
async def test_screener_flow_answers_waits_quietly_then_reintroduces():
    from muse_voice_agent.graph import CallControl

    SlowRecordingModel.seen = []
    model = SlowRecordingModel(
        messages=iter(
            [AIMessage(content="I'd like to book a table for Friday."), AIMessage(content="Table for two Friday at 7?")]
        )
    )
    control = CallControl()
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    screener = "Hi, the person you're calling is using a screening service. Go ahead and say your name and why you're calling."
    history = [HumanMessage(content=screener)]
    first = "".join(await _run(graph, history))
    assert first.startswith(OPENER) and "call screener" in SlowRecordingModel.seen[0][-1].content

    history += [AIMessage(content=first), HumanMessage(content="Thanks, please stay on the line.")]
    assert "".join(await _run(graph, history)) == ""
    assert control.on_hold and len(SlowRecordingModel.seen) == 1  # no model call

    history += [HumanMessage(content="Hi, this is Luigi's.")]
    third = "".join(await _run(graph, history))
    assert third.startswith(OPENER) and third.strip().endswith("Friday at 7?")


@pytest.mark.asyncio
async def test_phone_menu_skips_opener_then_person_gets_it():
    from muse_voice_agent.graph import CallControl

    press = {"name": "press_digits", "args": {"digits": "2"}, "id": "p"}
    model = FakeToolModel(
        messages=iter([AIMessage(content="", tool_calls=[press]), AIMessage(content="Table for two Friday at 7?")])
    )
    control = CallControl()
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    menu = "Thank you for calling Luigi's. For hours, press 1. For reservations, press 2."
    assert "".join(await _run(graph, [HumanMessage(content=menu)])).strip() == ""
    assert control.pending_digits == "2" and not control.opener_spoken
    spoken = "".join(await _run(graph, [HumanMessage(content=menu), HumanMessage(content="Reservations, this is Tom.")]))
    assert spoken.startswith(OPENER)


@pytest.mark.asyncio
async def test_record_outcome_say_is_spoken_when_model_emits_no_text():
    from muse_voice_agent.graph import CallControl

    call = {
        "name": "record_outcome",
        "args": {"say": "No worries, Angi will call back. Thanks, bye!", "outcome": "needs_followup", "summary": "Only 5 PM"},
        "id": "r",
    }
    outcomes: list[CallOutcome] = []
    control = CallControl(opener_spoken=True)
    model = FakeToolModel(messages=iter([AIMessage(content="", tool_calls=[call])]))  # a 2nd call would raise
    graph = build_call_graph(_task(), outcomes.append, model=model, control=control)
    history = [AIMessage(content=OPENER + " Table for two Friday at 7?"), HumanMessage(content="We only have 5. That okay?")]
    assert "".join(await _run(graph, history)).strip() == "No worries, Angi will call back. Thanks, bye!"
    assert outcomes[0].outcome == "needs_followup" and control.closing


@pytest.mark.asyncio
async def test_record_outcome_say_is_not_repeated_after_text():
    call = {"name": "record_outcome", "args": {"say": "Perfect, bye!", "outcome": "booked", "summary": "x"}, "id": "r"}
    model = FakeToolModel(messages=iter([AIMessage(content="Perfect, thanks. Bye!", tool_calls=[call])]))
    graph = build_call_graph(_task(), lambda o: None, model=model)
    spoken = "".join(await _run(graph, [HumanMessage(content="Luigi's."), AIMessage(content=OPENER), HumanMessage(content="You're booked.")]))
    assert spoken.strip() == "Perfect, thanks. Bye!"


def test_say_streamer_handles_partial_json_and_escapes():
    from muse_voice_agent.graph import _SayStreamer

    streamer = _SayStreamer()
    out = []
    for piece in ['{"sa', 'y": "No wor', 'ries, \\', '"5\\" is too early', '. Bye!", "outcome": "unav']:
        chunk = AIMessageChunk(content="", tool_call_chunks=[{"name": "record_outcome" if not out else None, "args": piece, "id": "r", "index": 0}])
        out.append(streamer.feed(chunk))
    assert "".join(out) == 'No worries, "5" is too early. Bye!'
    assert streamer.done


@pytest.mark.asyncio
async def test_end_call_and_hold_tools_end_the_turn():
    from muse_voice_agent.graph import CallControl

    end = {"name": "end_call", "args": {}, "id": "e"}
    control = CallControl(opener_spoken=True)
    model = FakeToolModel(messages=iter([AIMessage(content="", tool_calls=[end])]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    await _run(graph, [AIMessage(content="Hi, Angi here."), HumanMessage(content="Wrong number.")])
    assert control.end_requested


@pytest.mark.asyncio
async def test_first_goodbye_does_not_hang_up_on_a_person_but_voicemail_does():
    from muse_voice_agent.graph import CallControl

    for outcome, should_end in (("booked", False), ("voicemail", True)):
        calls = [
            {"name": "record_outcome", "args": {"outcome": outcome, "summary": "x"}, "id": "r"},
            {"name": "end_call", "args": {}, "id": "e"},
        ]
        control = CallControl(opener_spoken=True)
        model = FakeToolModel(messages=iter([AIMessage(content="Thanks, bye!", tool_calls=calls)]))
        graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
        await _run(graph, [AIMessage(content="Hi, for Angi."), HumanMessage(content="You're set.")])
        assert control.end_requested is should_end
        assert control.outcome.outcome == outcome


@pytest.mark.asyncio
async def test_punctuation_only_text_is_not_spoken():
    from muse_voice_agent.graph import CallControl

    control = CallControl(opener_spoken=True)
    calls = [{"name": "press_digits", "args": {"digits": "1"}, "id": "p"}]
    model = FakeToolModel(messages=iter([AIMessage(content="...", tool_calls=calls)]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    spoken = []
    async for chunk in graph.astream(
        {"messages": [AIMessage(content="Hi, for Angi."), HumanMessage(content="Press 1.")]},
        stream_mode="custom",
    ):
        spoken.append(str(chunk))
    assert "".join(spoken).strip() == ""
    assert control.pending_digits == "1"


@pytest.mark.asyncio
async def test_does_not_hang_up_while_their_question_is_pending():
    from muse_voice_agent.graph import CallControl, CallOutcome

    for their_line, should_end in (
        ("Want me to add you to the waitlist?", False),
        ("Thanks, bye!", True),
    ):
        control = CallControl(opener_spoken=True, closing=True)
        control.outcome = CallOutcome(outcome="info_received", summary="x")
        calls = [{"name": "end_call", "args": {}, "id": "e"}]
        model = FakeToolModel(messages=iter([AIMessage(content="No thanks. Bye!", tool_calls=calls)]))
        graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
        control.closing = False  # skip the farewell fast path so the model runs
        await _run(graph, [AIMessage(content="Hi, for Angi."), HumanMessage(content=their_line)])
        assert control.end_requested is should_end, their_line


@pytest.mark.asyncio
async def test_silent_end_call_on_question_gets_spoken_fallback():
    from muse_voice_agent.graph import CallControl

    control = CallControl(opener_spoken=True)
    calls = [{"name": "end_call", "args": {}, "id": "e"}]
    model = FakeToolModel(messages=iter([AIMessage(content="", tool_calls=calls)]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=control)
    spoken = await _run(
        graph,
        [AIMessage(content="Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded."), HumanMessage(content="What's a good callback number?")],
    )
    assert "could you say that again" in "".join(spoken)
    assert control.end_requested is False


def test_silent_hangup_fallback_matches_their_last_line():
    from muse_voice_agent.graph import _silent_hangup_fallback

    assert _silent_hangup_fallback("Anything else I can help with?", done=True) == "No, that's all. Thanks, bye!"
    assert "say that again" in _silent_hangup_fallback("Can you spell the name?", done=True)
    assert "say that again" in _silent_hangup_fallback("Okay, what day?", done=False)
    assert _silent_hangup_fallback("Great, you're booked.", done=True) == "Thanks so much. Bye!"


def test_voicemail_note_skips_the_intro_only_when_the_opener_was_just_spoken():
    from muse_voice_agent.graph import _line_notes

    fresh = _line_notes(_task(), "voicemail", intro_now=True, reintro=False)[0]
    later = _line_notes(_task(), "voicemail", intro_now=False, reintro=False)[0]
    assert "don't introduce yourself again" in fresh
    assert "don't introduce yourself again" not in later


@pytest.mark.asyncio
async def test_screener_repair_line_does_not_block_reintroduction_to_the_connected_person():
    from muse_voice_agent.graph import CallControl

    model = FakeToolModel(messages=iter([AIMessage(content="Table for two Friday at 7?")]))
    graph = build_call_graph(_task(), lambda o: None, model=model, control=CallControl(opener_spoken=True))
    history = [
        HumanMessage(content="Hi, who's calling and what's this regarding?"),
        AIMessage(content=OPENER + " I'd like a table…"),
        HumanMessage(content="Sorry—what?"),
        AIMessage(content=OPENER + " I'd like to book a table for Friday."),
        HumanMessage(content="Thanks, please stay on the line while I connect you."),
        HumanMessage(content="Hello?"),
    ]
    assert "".join(await _run(graph, history)).startswith(OPENER)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunked", [False, True])
async def test_model_repeating_the_opener_is_not_spoken_twice(chunked):
    from muse_voice_agent.graph import _OpenerDropper

    if chunked:
        dropper = _OpenerDropper(OPENER)
        parts = [OPENER[:10], OPENER[10:30], OPENER[30:] + " Calling", " about jeans."]
        assert "".join(dropper.feed(p) for p in parts) + dropper.flush() == "Calling about jeans."
        other = _OpenerDropper(OPENER)
        assert other.feed("Hi") == "" and other.feed(" there!") == "Hi there!"
        partial = _OpenerDropper(OPENER)
        assert partial.feed("Hi, this") == "" and partial.flush() == "Hi, this"
        return
    model = FakeToolModel(messages=iter([AIMessage(content=OPENER + " I'm calling about hemming jeans. Bye!")]))
    graph = build_call_graph(_task(), lambda o: None, model=model)
    spoken = "".join(await _run(graph, [HumanMessage(content="You've reached Luigi's. Please leave a message.")]))
    assert spoken.count("on behalf of") == 1 and spoken.strip().endswith("hemming jeans. Bye!")
