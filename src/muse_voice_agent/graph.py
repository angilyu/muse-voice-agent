"""LangGraph workflow that drives the phone conversation.

The graph runs once per agent turn with the conversation so far (LiveKit's LLMAdapter, the Retell
websocket, or the text evals). The `caller` node streams the model's spoken text through LangGraph's
custom stream writer (so tool calls / tool results are never read aloud). Tools report back through a
per-call `CallControl` that the voice backend reads after every turn:

- record_outcome: stores the result (may be called again to correct it); the call stays up.
- end_call / press_digits / wait_on_hold: telephony actions, carried out by the backend.

To save a model round trip, a turn ends right after the tools run whenever the model already spoke in
the same response (e.g. "Perfect, thanks. Bye!" + record_outcome) or used a telephony tool. The model
often emits record_outcome before any text, so the goodbye is also its first argument (`say`), which
is spoken while the rest of the arguments are still streaming.

Before the model runs, each business line is classified (see `pickup`), so screeners, phone menus and
voicemail are handled without waiting on the model where possible, and a person who picks up after
we've spoken into silence, to a screener or to a menu hears the opener again.
"""

from __future__ import annotations

import json
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
from pydantic import BaseModel, Field, create_model

from .pickup import classify_line, is_greeting, is_note
from .safety import OutputGuard, assess_commitment_within_limits, is_recording_objection, redact_agent_text
from .tasks import (
    AnyTask,
    GeneralCall,
    InboundMessage,
    build_system_prompt,
    opening_line,
    self_intro,
)

_TOOL_MARKER = "<function="
_TOOL_MARKUP = re.compile(r"<function=.*?/>", re.S)

Outcome = Literal[
    "booked",
    "ordered",
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
    order_total: str | None = Field(default=None, description="Total price quoted for an order")
    pickup_time: str | None = Field(default=None, description="Pickup or ready time for an order")
    quote: str | None = Field(default=None, description="Price, range, or pricing model quoted")
    availability: str | None = Field(default=None, description="Earliest availability offered")
    contact_person: str | None = Field(default=None, description="Who you spoke with")
    follow_up: str | None = Field(default=None, description="Anything the customer must do next")
    answers: list[Answer] | None = Field(
        default=None, description="Each question you were asked to get answered, with the answer"
    )
    reference: str | None = Field(default=None, description="Confirmation / reference number")
    committed_within_limits: bool | None = Field(
        default=None,
        description="Server-set safety flag: whether any commitment was within the task limits.",
    )
    safety_flags: list[str] = Field(
        default_factory=list,
        description="Server-set safety flags such as unauthorized_commitment or limit violation.",
    )


SAY_FIELD = "say"
# Same fields as CallOutcome with the spoken goodbye first, so it streams before the rest.
RecordOutcomeArgs = create_model(
    "RecordOutcomeArgs",
    say=(
        str,
        Field(
            default="",
            description=(
                "The short goodbye you say out loud right now, e.g. \"Perfect, thanks so much. Bye!\" "
                "(answer their last question first if they asked one). Always fill this in."
            ),
        ),
    ),
    **{name: (f.annotation, f) for name, f in CallOutcome.model_fields.items()},
)


@dataclass
class CallControl:
    """Per-call state shared between the graph's tools and the voice backend."""

    opener_spoken: bool = False
    # The fixed opener text; the voice backend keeps it from being interrupted.
    opener_text: str = ""
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


def _line_notes(task: AnyTask, kind: str | None, *, intro_now: bool, reintro: bool) -> list[str]:
    """Notes about what's on the line, from the keyword classifier."""
    name = getattr(task, "customer_name", "the caller")
    if reintro:
        return [
            "A person just picked up and didn't hear you before (you were talking to silence, a "
            "screener or a menu). Your opener was just spoken again; now say why you're calling in "
            "one short sentence, as if for the first time. Don't say \"as I said\"."
        ]
    if kind == "screener":
        return [
            "This line is an automated call screener, not a person. Answer it in one sentence: who "
            f"you are ({self_intro(task)} calling for {name}) and why you're calling"
            + (" (your opener just covered who, so only add why)" if intro_now else "")
            + ". No apology and no question. Then stop and wait for a person."
        ]
    if kind == "voicemail":
        return [
            "This is voicemail or an answering machine. Leave one short message: why you're calling "
            f"for {name}, and that {name} will call back"
            + (" (your opener just said who you are, so don't introduce yourself again)" if intro_now else "")
            + ". Then record outcome \"voicemail\" and call end_call in the same reply."
        ]
    if kind == "menu":
        return [
            "This is an automated phone menu, not a person. Don't introduce yourself or chat. Use "
            "press_digits for the option that reaches reservations, scheduling, orders or a person "
            "(or the operator). If it only takes spoken options, say the option in a few words."
        ]
    return []


def _status_notes(
    task: AnyTask, control: CallControl, history: list[BaseMessage], *, intro_now: bool = False
) -> list[str]:
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
    elif control.opener_spoken and not intro_now:
        customer_name = getattr(task, "customer_name", "")
        first = customer_name.split()[0].lower() if customer_name else ""
        heard = any(
            first and isinstance(m, AIMessage) and first in _content_text(m).lower() for m in history
        )
        if customer_name and not heard:
            notes.append(
                "Your introduction got cut off, so they don't know who you are. Start this reply "
                "without apologizing, e.g. \"I'm "
                + (f"{task.assistant_name}, " if task.assistant_name else "")
                + f'the AI assistant for {customer_name}."'
            )
    return notes


def _notes_message(notes: list[str]) -> SystemMessage | None:
    # A trailing message, not part of the system prompt, so the prompt prefix stays cacheable.
    if not notes:
        return None
    return SystemMessage(content="Call status right now:\n- " + "\n- ".join(notes))


_PARTIAL_SAY = re.compile(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)(")?')


class _OpenerDropper:
    """Drops a copy of the opener the model sometimes starts with; it was already spoken."""

    def __init__(self, opener: str) -> None:
        self.opener = opener.strip().lower()
        self.buf = ""
        self.done = False

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buf += text
        head = self.buf.lstrip().lower()
        if len(head) < len(self.opener) and self.opener.startswith(head):
            return ""  # might still be the opener
        self.done = True
        out, self.buf = self.buf, ""
        if head.startswith(self.opener):
            return out.lstrip()[len(self.opener) :].lstrip()
        return out

    def flush(self) -> str:
        self.done = True
        out, self.buf = self.buf, ""
        return out


def _strip_opener(text: str, opener: str) -> str:
    if text.lower().startswith(opener.strip().lower()):
        return text[len(opener.strip()) :].lstrip()
    return text


class _SayStreamer:
    """Pull the `say` argument out of record_outcome's streamed JSON arguments as it arrives."""

    def __init__(self) -> None:
        self._args: dict[int, list[str]] = {}
        self._names: dict[int, str] = {}
        self.spoken = ""
        self.done = False

    def feed(self, chunk: AIMessageChunk) -> str:
        """Return newly available `say` text."""
        for tc in getattr(chunk, "tool_call_chunks", None) or []:
            idx = tc.get("index") or 0
            if tc.get("name"):
                self._names[idx] = tc["name"]
            if tc.get("args"):
                self._args.setdefault(idx, []).append(tc["args"])
        if self.done:
            return ""
        for idx, name in self._names.items():
            if name != "record_outcome":
                continue
            m = _PARTIAL_SAY.search("".join(self._args.get(idx, [])))
            if not m:
                return ""
            raw = m.group(1)
            if m.group(2):
                self.done = True
            else:
                raw = re.sub(r"\\(?:u[0-9a-fA-F]{0,3})?$", "", raw)
            try:
                value = json.loads(f'"{raw}"')
            except ValueError:
                return ""
            new = value[len(self.spoken) :] if value.startswith(self.spoken) else ""
            self.spoken = value if value.startswith(self.spoken) else self.spoken
            return new
        return ""


_ANYTHING_ELSE = re.compile(
    r"\b(?:anything else|something else|is that (?:it|all)|that(?:'s| is) (?:it|all)|all set)\b", re.I
)


def _silent_hangup_fallback(their_last_line: str, *, done: bool) -> str:
    """What to say when the model tried to hang up without words but the call must stay open."""
    if done and _ANYTHING_ELSE.search(their_last_line):
        return "No, that's all. Thanks, bye!"
    if "?" in their_last_line or not done:
        return "Sorry, could you say that again?"
    return "Thanks so much. Bye!"


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
    their_turn_is_open = not is_farewell(their_last_line) and bool(their_last_line.strip())
    asked = "?" in their_last_line and their_turn_is_open
    should_wait_after_outcome = control.outcome is not None and their_turn_is_open
    if not recorded and not asked and not should_wait_after_outcome:
        return msg
    kept = [tc for tc in msg.tool_calls if tc["name"] != "end_call"]
    if not _text(msg).strip():
        fallback = _silent_hangup_fallback(their_last_line, done=control.outcome is not None or bool(recorded))
        return AIMessageChunk(content=fallback, tool_calls=kept, id=msg.id)
    return AIMessageChunk(content=msg.content, tool_calls=kept, id=msg.id)


def _content_text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c or [])


UNAUTHORIZED_COMMIT_NOTE = "Agent was not authorized to commit; confirm with the business yourself."


def apply_authority(task: AnyTask, outcome: CallOutcome) -> CallOutcome:
    """Enforce non-overridable authority/limit rules on recorded outcomes."""
    if outcome.outcome not in {"booked", "ordered"}:
        return outcome.model_copy(update={"committed_within_limits": None})

    flags = list(outcome.safety_flags or [])
    if isinstance(task, InboundMessage):
        flags.append("unauthorized_commitment")
        note = "Unknown inbound caller; no commitment was authorized."
        return outcome.model_copy(
            update={
                "outcome": "needs_followup",
                "follow_up": f"{outcome.follow_up} {note}" if outcome.follow_up else note,
                "committed_within_limits": False,
                "safety_flags": sorted(set(flags)),
            }
        )
    if isinstance(task, GeneralCall) and task.authority == "info_only":
        flags.append("unauthorized_commitment")
        note = UNAUTHORIZED_COMMIT_NOTE
        return outcome.model_copy(
            update={
                "outcome": "needs_followup",
                "follow_up": f"{outcome.follow_up} {note}" if outcome.follow_up else note,
                "committed_within_limits": False,
                "safety_flags": sorted(set(flags)),
            }
        )

    if isinstance(task, GeneralCall):
        text = " ".join(
            str(v)
            for k, v in outcome.model_dump(exclude_none=True).items()
            if k not in {"committed_within_limits", "safety_flags"} and not isinstance(v, list)
        )
        ok, limit_flags = assess_commitment_within_limits(task.limits, text)
        flags.extend(limit_flags)
        if not ok:
            note = "The business required something outside the user's limits; confirm directly."
            return outcome.model_copy(
                update={
                    "outcome": "needs_followup",
                    "follow_up": f"{outcome.follow_up} {note}" if outcome.follow_up else note,
                    "committed_within_limits": False,
                    "safety_flags": sorted(set(flags)),
                }
            )
        return outcome.model_copy(update={"committed_within_limits": True, "safety_flags": sorted(set(flags))})

    if getattr(task, "kind", None) == "restaurant_reservation":
        within = outcome.party_size in (None, getattr(task, "party_size", None))
        if not within:
            flags.append("party_size_outside_limits")
        return outcome.model_copy(update={"committed_within_limits": within, "safety_flags": sorted(set(flags))})

    return outcome.model_copy(update={"committed_within_limits": False, "safety_flags": ["unsupported_commitment_type"]})


def build_call_graph(
    task: AnyTask,
    on_outcome: Callable[[CallOutcome], None],
    model: BaseChatModel | None = None,
    control: CallControl | None = None,
):
    if model is None:
        from langchain.chat_models import init_chat_model

        from .config import get_settings, llm_model_init_args

        settings = get_settings()
        model_name, model_kwargs = llm_model_init_args(
            settings.llm_model, temperature=0.3, service_tier=settings.llm_service_tier
        )
        model = init_chat_model(model_name, **model_kwargs)

    control = control if control is not None else CallControl()
    system_prompt = build_system_prompt(task)

    @tool("record_outcome", args_schema=RecordOutcomeArgs)
    async def record_outcome(**kwargs: Any) -> str:
        """Record the result of this phone call, together with a short goodbye in the same reply.
        Call it again later only if a detail changes. This does not hang up."""
        kwargs.pop(SAY_FIELD, None)
        outcome = apply_authority(task, CallOutcome(**kwargs))
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
    control.opener_text = opener

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

        last_line = _content_text(last) if isinstance(last, HumanMessage) else ""
        kind = classify_line(last_line) if last_line and not is_note(last_line) else None
        if (
            last_line
            and getattr(task, "recording_disclosure_required", True)
            and is_recording_objection(last_line)
            and control.outcome is None
        ):
            text = (
                "I understand. I can't continue if you don't consent to recording, so the "
                "customer will follow up directly. Goodbye."
            )
            outcome = CallOutcome(
                outcome="needs_followup",
                summary="The business objected to call recording, so the assistant ended the call.",
                follow_up="Business objected to recording; customer should follow up directly.",
                committed_within_limits=None,
                safety_flags=["recording_objected"],
            )
            control.outcome = outcome
            control.closing = True
            control.end_requested = True
            on_outcome(outcome)
            writer(redact_agent_text(text, getattr(task, "shareable_details", {})))
            return {"messages": [AIMessage(content=text)]}
        earlier_lines = [
            _content_text(m)
            for m in history[:-1]
            if isinstance(m, HumanMessage) and not is_note(_content_text(m))
        ]
        # Someone has heard us only if the line just before this one was a person; a screener
        # handoff, menu or voicemail in between means whoever speaks now is new to the call.
        person_before = bool(earlier_lines) and classify_line(earlier_lines[-1]) == "person"
        spoke_before = any(isinstance(m, AIMessage) and _content_text(m).strip() for m in history)

        if kind == "screener_wait" and not person_before and control.outcome is None:
            # A screener or forwarding service is connecting us ("please stay on the line"); talking
            # now would only be heard by the recording, so wait quietly for a person.
            control.on_hold = True
            return {"messages": []}

        # Speak the fixed opener before the LLM runs, so its latency overlaps with the opener's
        # playback instead of leaving the callee in silence: on our first turn (unless it's a phone
        # menu), and again when a person first greets us after we spoke to silence, a screener or a
        # menu, since they never heard who we are.
        reintro = (
            control.outcome is None
            and kind == "person"
            and not person_before
            and spoke_before
            and is_greeting(last_line, getattr(task, "business_name", None))
        )
        first_turn = not control.opener_spoken and not spoke_before and kind != "menu"
        intro_now = first_turn or reintro
        notes = _line_notes(task, kind, intro_now=intro_now, reintro=reintro)
        notes += _status_notes(task, control, history, intro_now=intro_now)
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *history]
        if intro_now:
            writer(opener + " ")
            messages.append(AIMessage(content=opener))
            control.opener_spoken = True
        if (note := _notes_message(notes)) is not None:
            messages.append(note)
        full: AIMessageChunk | None = None
        tool_filter = _ToolMarkupFilter()
        say_streamer = _SayStreamer()
        output_guard = OutputGuard(getattr(task, "shareable_details", {}))
        held: str | None = ""  # don't speak until there's a word, so "..." alone is never voiced
        spoke_text = False
        dropper = _OpenerDropper(opener) if intro_now else None

        def speak(text: str) -> None:
            nonlocal held
            if dropper is not None:
                text = dropper.feed(text)
                if not text:
                    return
            if held is None:
                writer(output_guard.feed(text))
            elif re.search(r"\w", held + text):
                writer(output_guard.feed(held + text))
                held = None
            else:
                held += text

        async for chunk in llm_with_tools.astream(messages):
            text = tool_filter.feed(_text(chunk))
            if text and not say_streamer.spoken:
                speak(text)
                spoke_text = spoke_text or bool(text.strip())
            said = say_streamer.feed(chunk)
            if said and not spoke_text:
                speak(said)
            full = chunk if full is None else full + chunk
        text = tool_filter.flush()
        if text and not say_streamer.spoken:
            speak(text)
            spoke_text = spoke_text or bool(text.strip())
        if full is not None and not spoke_text:
            # Providers that don't stream tool arguments: speak the goodbye from the final call.
            final_say = next(
                (
                    str((tc.get("args") or {}).get(SAY_FIELD) or "")
                    for tc in full.tool_calls
                    if tc["name"] == "record_outcome"
                ),
                "",
            ).strip()
            if final_say.startswith(say_streamer.spoken) and final_say != say_streamer.spoken:
                speak(final_say[len(say_streamer.spoken) :])
                say_streamer.spoken = final_say
            if say_streamer.spoken.strip():
                full = AIMessageChunk(
                    content=say_streamer.spoken.strip(), tool_calls=full.tool_calls, id=full.id
                )
        if dropper is not None and (rest := dropper.flush()):
            speak(rest)
        if held is not None and full is not None and held.strip():
            full = AIMessageChunk(content="", tool_calls=full.tool_calls, id=full.id)
        if tail := output_guard.flush():
            writer(tail)
        if full is not None and full.tool_calls:
            their_last = next(
                (_content_text(m) for m in reversed(history) if isinstance(m, HumanMessage)), ""
            )
            before_hold_text = _text(full).strip()
            full = _hold_hangup_until_they_reply(full, their_last, control)
            after_hold_text = _text(full).strip()
            if after_hold_text and not before_hold_text:
                writer(after_hold_text)
        if (
            full is not None
            and any(tc["name"] == "end_call" for tc in full.tool_calls)
            and not _text(full).strip()
        ):
            # Never hang up in silence.
            writer("Bye!")
            full = full + AIMessageChunk(content="Bye!", id=full.id)
        if full is None:
            return {"messages": [AIMessage(content=opener)] if intro_now else []}
        content: Any = full.content
        if intro_now:
            content = f"{opener} {_strip_opener(_text(full).strip(), opener)}".strip()
        if isinstance(content, str):
            content = redact_agent_text(content, getattr(task, "shareable_details", {}))
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
