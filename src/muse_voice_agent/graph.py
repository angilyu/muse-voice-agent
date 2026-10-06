"""LangGraph workflow that drives the phone conversation.

The graph runs once per agent turn with the conversation so far (LiveKit's LLMAdapter, the Retell
websocket, or the text evals). The `caller` node streams the model's spoken text through LangGraph's
custom stream writer (so tool calls / tool results are never read aloud). Tools report back through a
per-call `CallControl` that the voice backend reads after every turn:

- record_outcome: stores the result (may be called again to correct it); the call stays up.
- end_call / press_digits / wait_on_hold: telephony actions, carried out by the backend.

To save a model round trip, a turn ends right after the tools run whenever the model already spoke in
the same response (e.g. "Perfect, thanks. Bye!" + record_outcome) or used a telephony tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from .tasks import AnyTask, GeneralCall, build_system_prompt, opening_line

_TOOL_MARKER = "<function="
_TOOL_MARKUP = re.compile(r"<function=.*?/>", re.S)

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


@dataclass
class CallControl:
    """Per-call state shared between the graph's tools and the voice backend."""

    opener_spoken: bool = False
    outcome: CallOutcome | None = None
    # The outcome is recorded and we've said goodbye; we're waiting for them to wrap up.
    closing: bool = False
    end_requested: bool = False
    on_hold: bool = False
    # DTMF digits to send after the current turn's speech; the backend clears it once sent.
    pending_digits: str | None = None


TERMINAL_TOOLS = {"end_call", "press_digits", "wait_on_hold"}
_DIGITS = re.compile(r"^[0-9*#]{1,12}$")
_FAREWELL = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all right|great|perfect|awesome|sounds (?:good|great)|cool|got it|"
    r"will do|you'?re all set|yep|yeah|yes|sure|thanks?|"
    r"thank you(?: so much| very much)?|you too|same to you|no problem|bye(?:[- ]bye)?|goodbye|"
    r"have a (?:good|great|nice) (?:one|day|night|evening)|take care|see you|see ya|"
    r"you'?re welcome)[\s,.!]*)+$",
    re.I,
)


def is_farewell(text: str) -> bool:
    """True for a pure sign-off ("Okay, thanks, bye!") with no question or new information."""
    text = text.strip()
    return bool(_FAREWELL.match(text)) and bool(
        re.search(r"\bbye\b|goodbye|take care|have a (?:good|great|nice)|see (?:you|ya)", text, re.I)
    )


class CallState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _text(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content or []
    )


class _ToolMarkupFilter:
    """Drop raw streamed tool-call markup before it reaches TTS."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            start = self._buffer.find(_TOOL_MARKER)
            if start < 0:
                keep = self._partial_marker_len(self._buffer)
                emit, self._buffer = self._buffer[: len(self._buffer) - keep], self._buffer[len(self._buffer) - keep :]
                out.append(emit)
                break
            out.append(self._buffer[:start])
            end = self._buffer.find("/>", start)
            if end < 0:
                self._buffer = self._buffer[start:]
                break
            self._buffer = self._buffer[end + 2 :]
        return "".join(out)

    def flush(self) -> str:
        text = "" if self._buffer.startswith(_TOOL_MARKER) else self._buffer
        self._buffer = ""
        return _TOOL_MARKUP.sub("", text)

    @staticmethod
    def _partial_marker_len(text: str) -> int:
        max_len = min(len(text), len(_TOOL_MARKER) - 1)
        for n in range(max_len, 0, -1):
            if _TOOL_MARKER.startswith(text[-n:]):
                return n
        return 0


def _status_notes(task: AnyTask, control: CallControl, history: list[BaseMessage]) -> str:
    notes: list[str] = []
    if control.outcome is not None:
        o = control.outcome
        notes.append(
            f"You already recorded the outcome ({o.outcome}: {o.summary})"
            + (" and said goodbye." if control.closing else ". Say a short goodbye now.")
            + " Stay on the line until they're done: if they ask anything (spell the name, repeat a "
            "detail, a callback number), answer it briefly. If a detail changes, call record_outcome "
            "again with the corrected details. Once they say bye or there's nothing left, say a quick "
            '"Bye!" and call end_call in the same reply.'
        )
    elif control.opener_spoken:
        first = task.customer_name.split()[0].lower()
        heard = any(
            isinstance(m, AIMessage) and first in _content_text(m).lower() for m in history
        )
        if not heard:
            notes.append(
                "Your introduction got cut off, so they don't know who you are. Start this reply by "
                f'briefly saying who you are, e.g. "Sorry, this is an assistant calling for '
                f'{task.customer_name}."'
            )
    if not notes:
        return ""
    return "\n\nCall status right now:\n- " + "\n- ".join(notes)


def _hold_hangup_until_they_reply(
    msg: AIMessageChunk, their_last_line: str, control: CallControl
) -> AIMessageChunk:
    """Drop end_call when it comes with the first goodbye to a person, or while their last line was a
    question, so they can still answer, say bye or ask something ("how do you spell that?").
    Silence then ends the call. Voicemail hangs up."""
    names = [tc["name"] for tc in msg.tool_calls]
    recorded = [tc for tc in msg.tool_calls if tc["name"] == "record_outcome"]
    if "end_call" not in names:
        return msg
    outcomes = {(tc.get("args") or {}).get("outcome") for tc in recorded}
    if control.outcome is not None:
        outcomes.add(control.outcome.outcome)
    if "voicemail" in outcomes:
        return msg
    asked = "?" in their_last_line and not is_farewell(their_last_line)
    if not recorded and not (asked and _text(msg).strip()):
        return msg
    kept = [tc for tc in msg.tool_calls if tc["name"] != "end_call"]
    return AIMessageChunk(content=msg.content, tool_calls=kept, id=msg.id)


def _content_text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c or [])


def build_call_graph(
    task: AnyTask,
    on_outcome: Callable[[CallOutcome], None],
    model: BaseChatModel | None = None,
    control: CallControl | None = None,
):
    if model is None:
        from langchain.chat_models import init_chat_model

        from .config import get_settings, llm_model_init_args

        model_name, model_kwargs = llm_model_init_args(get_settings().llm_model, temperature=0.3)
        model = init_chat_model(model_name, **model_kwargs)

    control = control if control is not None else CallControl()
    system_prompt = build_system_prompt(task)

    @tool("record_outcome", args_schema=CallOutcome)
    async def record_outcome(**kwargs: Any) -> str:
        """Record the result of this phone call, together with a short goodbye in the same reply.
        Call it again later only if a detail changes. This does not hang up."""
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
        control.outcome = outcome
        on_outcome(outcome)
        return (
            "Recorded. If you haven't said goodbye yet, say a short one now (no questions). Don't "
            "hang up until they're done; answer anything else they ask."
        )

    @tool("end_call")
    async def end_call() -> str:
        """Hang up once your current sentence has been spoken. Use after goodbyes are exchanged,
        or when the call can't go anywhere (wrong number, they hung up, voicemail left)."""
        control.end_requested = True
        return "Hanging up."

    @tool("press_digits")
    async def press_digits(digits: str) -> str:
        """Press keypad digits (0-9, * or #) on a phone menu or call-screening prompt, e.g. "1"."""
        digits = re.sub(r"[\s,-]", "", digits)
        if not _DIGITS.match(digits):
            return "Invalid digits; use only 0-9, * and #."
        control.pending_digits = digits  # models sometimes emit the same call twice
        return f"Pressing {digits}."

    @tool("wait_on_hold")
    async def wait_on_hold() -> str:
        """You've been put on hold or asked to wait; stay quiet until someone speaks again."""
        control.on_hold = True
        return "Waiting quietly."

    tools = [record_outcome, end_call, press_digits, wait_on_hold]
    llm_with_tools = model.bind_tools(tools)

    opener = opening_line(task)

    async def caller(state: CallState) -> dict[str, list[BaseMessage]]:
        writer = get_stream_writer()
        history = state["messages"]
        control.on_hold = False
        last = history[-1] if history else None
        if control.closing and isinstance(last, HumanMessage) and is_farewell(_content_text(last)):
            # They're just signing off; answer in kind without waiting on the model.
            writer("Bye!")
            control.end_requested = True
            return {"messages": [AIMessage(content="Bye!")]}

        # On our first turn, speak the fixed opener before the LLM runs; its latency then overlaps
        # with the opener's playback instead of leaving the callee in silence.
        first_turn = not control.opener_spoken and not any(
            isinstance(m, AIMessage) for m in history
        )
        notes = "" if first_turn else _status_notes(task, control, history)
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt + notes), *history]
        if first_turn:
            writer(opener + " ")
            messages.append(AIMessage(content=opener))
        control.opener_spoken = True
        full: AIMessageChunk | None = None
        tool_filter = _ToolMarkupFilter()
        held: str | None = ""  # don't speak until there's a word, so "..." alone is never voiced

        def speak(text: str) -> None:
            nonlocal held
            if held is None:
                writer(text)
            elif re.search(r"\w", held + text):
                writer(held + text)
                held = None
            else:
                held += text

        async for chunk in llm_with_tools.astream(messages):
            text = tool_filter.feed(_text(chunk))
            if text:
                speak(text)
            full = chunk if full is None else full + chunk
        text = tool_filter.flush()
        if text:
            speak(text)
        if held is not None and full is not None and held.strip():
            full = AIMessageChunk(content="", tool_calls=full.tool_calls, id=full.id)
        if full is not None and full.tool_calls:
            their_last = next(
                (_content_text(m) for m in reversed(history) if isinstance(m, HumanMessage)), ""
            )
            full = _hold_hangup_until_they_reply(full, their_last, control)
        if (
            full is not None
            and any(tc["name"] == "end_call" for tc in full.tool_calls)
            and not _text(full).strip()
        ):
            # Never hang up in silence.
            writer("Bye!")
            full = full + AIMessageChunk(content="Bye!", id=full.id)
        if full is None:
            return {"messages": [AIMessage(content=opener)] if first_turn else []}
        content: Any = full.content
        if first_turn:
            content = f"{opener} {_text(full).strip()}".strip()
        if control.outcome is not None and _text(full).strip() and not full.tool_calls:
            control.closing = True
        return {
            "messages": [AIMessage(content=content, tool_calls=full.tool_calls, id=full.id)]
        }

    def after_tools(state: CallState) -> str:
        ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if ai is None:
            return "caller"
        names = {tc["name"] for tc in ai.tool_calls}
        said = _content_text(ai).strip()
        spoke = bool(said.removeprefix(opener).strip())  # the fixed opener doesn't count
        if "record_outcome" in names and spoke:
            control.closing = True
        if names & TERMINAL_TOOLS or ("record_outcome" in names and spoke):
            return END
        return "caller"

    builder = StateGraph(CallState)
    builder.add_node("caller", caller)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "caller")
    builder.add_conditional_edges("caller", tools_condition, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", after_tools, {"caller": "caller", END: END})
    graph = builder.compile()
    graph.control = control  # type: ignore[attr-defined]
    return graph
