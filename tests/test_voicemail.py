import pytest

from langchain_core.messages import HumanMessage

from muse_voice_agent.graph import CallOutcome, build_call_graph
from muse_voice_agent.tasks import (
    GeneralCall,
    RestaurantReservation,
    build_voicemail_message,
    speak_phone_number,
)
from test_graph import FakeToolModel


def test_speak_phone_number_digit_by_digit():
    assert speak_phone_number("+14155550123") == (
        "one, four, one, five, five, five, five, zero, one, two, three"
    )


def test_voicemail_message_uses_assistant_name_and_repeats_callback():
    task = RestaurantReservation(
        business_name="Luigi's",
        phone_number="+14155550100",
        customer_name="Wenjing Yu",
        assistant_name="Eva",
        callback_number="+14155550123",
        party_size=2,
        date="Friday",
        time="7 PM",
    )

    msg = build_voicemail_message(task)

    spoken = speak_phone_number("+14155550123")
    assert msg.startswith("Hi, this is Eva, an AI assistant calling on behalf of Wenjing Yu.")
    assert "table for 2 on Friday at 7 PM" in msg
    assert msg.count(spoken) == 2
    assert msg.endswith("Thank you, goodbye.")
    assert len(msg.split()) <= 75


def test_voicemail_message_without_assistant_name_and_with_general_details():
    task = GeneralCall(
        business_name="North Beach Tailors",
        phone_number="+14155550242",
        customer_name="Angi",
        callback_number="+14155550123",
        goal="Ask if they can hem jeans by Friday",
        questions=["Can jeans be hemmed by Friday?"],
        shareable_details={"item": "jeans", "needed by": "Friday"},
    )

    msg = build_voicemail_message(task)

    assert msg.startswith("Hi, this is an AI assistant calling on behalf of Angi.")
    assert "hem jeans by Friday" in msg
    assert "item: jeans" in msg
    assert len(msg.split()) <= 75


def test_voicemail_message_requires_callback_number():
    task = GeneralCall(
        business_name="North Beach Tailors",
        phone_number="+14155550242",
        customer_name="Angi",
        goal="Ask if they can hem jeans by Friday",
    )
    with pytest.raises(ValueError):
        build_voicemail_message(task)


async def test_graph_leaves_deterministic_voicemail_and_hangs_up():
    task = GeneralCall(
        business_name="North Beach Tailors",
        phone_number="+14155550242",
        customer_name="Angi",
        assistant_name="Eva",
        callback_number="+14155550123",
        goal="Ask if they can hem jeans by Friday",
    )
    outcomes: list[CallOutcome] = []
    graph = build_call_graph(task, outcomes.append, model=FakeToolModel(messages=iter([])))

    spoken = []
    async for item in graph.astream(
        {"messages": [HumanMessage(content="You've reached North Beach Tailors. Leave a message after the tone.")]},
        stream_mode="custom",
    ):
        spoken.append(item)

    assert "".join(spoken) == build_voicemail_message(task)
    assert graph.control.end_requested
    assert outcomes[0].outcome == "voicemail"
    assert outcomes[0].voicemail_message == build_voicemail_message(task)
    assert outcomes[0].callback_number == "+14155550123"


async def test_graph_hangs_up_without_message_for_full_mailbox():
    task = GeneralCall(
        business_name="North Beach Tailors",
        phone_number="+14155550242",
        customer_name="Angi",
        callback_number="+14155550123",
        goal="Ask if they can hem jeans by Friday",
    )
    outcomes: list[CallOutcome] = []
    graph = build_call_graph(task, outcomes.append, model=FakeToolModel(messages=iter([])))

    spoken = []
    async for item in graph.astream(
        {"messages": [HumanMessage(content="Sorry, this mailbox is full and cannot accept new messages.")]},
        stream_mode="custom",
    ):
        spoken.append(item)

    assert spoken == []
    assert graph.control.end_requested
    assert outcomes[0].outcome == "voicemail"
    assert outcomes[0].voicemail_message is None
    assert "could not take a message" in outcomes[0].summary


async def test_graph_hangs_up_without_message_when_disabled():
    task = GeneralCall(
        business_name="North Beach Tailors",
        phone_number="+14155550242",
        customer_name="Angi",
        callback_number="+14155550123",
        goal="Ask if they can hem jeans by Friday",
        leave_voicemail=False,
    )
    outcomes: list[CallOutcome] = []
    graph = build_call_graph(task, outcomes.append, model=FakeToolModel(messages=iter([])))

    spoken = []
    async for item in graph.astream(
        {"messages": [HumanMessage(content="Please leave a message after the beep.")]},
        stream_mode="custom",
    ):
        spoken.append(item)

    assert spoken == []
    assert graph.control.end_requested
    assert outcomes[0].voicemail_message is None
