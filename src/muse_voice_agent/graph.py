"""LangGraph workflow that drives the phone conversation.

The graph is wrapped by livekit.plugins.langchain.LLMAdapter, which calls it once per agent turn with
the conversation so far. The `caller` node streams the model's spoken text through LangGraph's custom
stream writer (so tool calls / tool results are never read aloud), and the `record_outcome` tool
reports the call result back to whoever owns the call (the LiveKit agent worker).
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from .tasks import AnyTask, GeneralCall, build_system_prompt

Outcome = Literal[
    "booked",
    "quote_received",
    "info_received",
    "unavailable",
    "declined",
    "needs_followup",
    "voicemail",
]


class Answer(BaseModel):
    question: str
    answer: str = Field(description='What they said, or "not answered"')


class CallOutcome(BaseModel):
    """Structured result of the phone call."""

    outcome: Outcome = Field(description="Overall result of the call")
    summary: str = Field(description="One or two sentence summary for the customer")
    confirmed_date: str | None = Field(default=None, description="Reservation/appointment date")
    confirmed_time: str | None = Field(default=None, description="Reservation/appointment time")
    party_size: int | None = Field(default=None, description="Confirmed party size")
    booked_under: str | None = Field(default=None, description="Name the booking is under")
    quote: str | None = Field(default=None, description="Price, range, or pricing model quoted")
    availability: str | None = Field(default=None, description="Earliest availability offered")
    contact_person: str | None = Field(default=None, description="Who you spoke with")
    follow_up: str | None = Field(default=None, description="Anything the customer must do next")
    answers: list[Answer] | None = Field(
        default=None, description="Each question you were asked to get answered, with the answer"
    )
    reference: str | None = Field(default=None, description="Confirmation / reference number")


class CallState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content or []
    )


def build_call_graph(
    task: AnyTask,
    on_outcome: Callable[[CallOutcome], None],
    model: BaseChatModel | None = None,
):
    if model is None:
        from langchain.chat_models import init_chat_model

        from .config import get_settings

        model = init_chat_model(get_settings().llm_model, temperature=0.3)

    system_prompt = build_system_prompt(task)
    recorded: dict[str, bool] = {"done": False}

    @tool("record_outcome", args_schema=CallOutcome)
    def record_outcome(**kwargs: Any) -> str:
        """Record the result of this phone call. Call exactly once, right before saying goodbye."""
        if recorded["done"]:
            return "Outcome already recorded. Just say goodbye."
        recorded["done"] = True
        outcome = CallOutcome(**kwargs)
        if (
            isinstance(task, GeneralCall)
            and task.authority == "info_only"
            and outcome.outcome == "booked"
        ):
            # The brief didn't authorize a booking; never report one as done.
            note = "Agent was not authorized to book; confirm with the business yourself."
            outcome = outcome.model_copy(
                update={
                    "outcome": "needs_followup",
                    "follow_up": f"{outcome.follow_up} {note}" if outcome.follow_up else note,
                }
            )
        on_outcome(outcome)
        return "Outcome recorded. Now say a brief, polite goodbye and stop talking."

    tools = [record_outcome]
    llm_with_tools = model.bind_tools(tools)

    async def caller(state: CallState) -> dict[str, list[BaseMessage]]:
        writer = get_stream_writer()
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        full: AIMessageChunk | None = None
        async for chunk in llm_with_tools.astream(messages):
            text = _text(chunk)
            if text:
                writer(text)
            full = chunk if full is None else full + chunk
        if full is None:
            return {"messages": []}
        return {
            "messages": [
                AIMessage(content=full.content, tool_calls=full.tool_calls, id=full.id)
            ]
        }

    builder = StateGraph(CallState)
    builder.add_node("caller", caller)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "caller")
    builder.add_conditional_edges("caller", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "caller")
    return builder.compile()
