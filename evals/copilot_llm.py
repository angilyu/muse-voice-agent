"""LangChain chat model backed by GitHub Copilot (via the Copilot SDK), for evals.

Evals make thousands of LLM calls; routing them through a Copilot subscription keeps the OpenAI key
for production traffic. Each conversation gets its own Copilot session with the caller's system prompt (the Copilot agent
prompt is replaced and built-in tools are disabled). When a call extends a conversation this process
has already answered, only the new messages are sent to that session, so the model sees real
multi-turn history; otherwise the history is rendered as a transcript in a fresh session. Tool calls
are surfaced as normal LangChain ``tool_calls`` so the production LangGraph graph runs unchanged.

Model names look like ``copilot:<model>`` or ``copilot:<model>@<reasoning_effort>``.
Auth: ``COPILOT_GITHUB_TOKEN`` if set, otherwise the Copilot CLI's logged-in user.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

PREFIX = "copilot:"


def parse_model_name(name: str) -> tuple[str, str | None]:
    """``copilot:gpt-5.4-mini@low`` -> ("gpt-5.4-mini", "low")."""
    if name.startswith(PREFIX):
        name = name[len(PREFIX) :]
    model, _, effort = name.partition("@")
    return model, effort or None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def render_messages(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """Split LangChain messages into (system prompt, user prompt).

    A lone user message is sent verbatim; multi-turn history becomes a labelled transcript, since each
    Copilot call is a fresh session.
    """
    system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(rest) == 1 and isinstance(rest[0], HumanMessage):
        return system, _text(rest[0].content)
    lines = _render_lines(rest)
    prompt = (
        "Conversation so far (oldest first):\n"
        + "\n".join(lines)
        + "\n\nContinue as 'You': reply with only your next message. "
        "Do not repeat earlier lines or add speaker labels."
    )
    return system, prompt


def _render_lines(messages: Sequence[BaseMessage]) -> list[str]:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {_text(m.content)}")
        elif isinstance(m, AIMessage):
            if text := _text(m.content).strip():
                lines.append(f"You: {text}")
            for tc in m.tool_calls:
                lines.append(f"You called tool {tc['name']} with {json.dumps(tc['args'], ensure_ascii=False)}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result: {_text(m.content)}")
        else:
            lines.append(f"{m.type}: {_text(m.content)}")
    return lines


def _follow_up_prompt(new: Sequence[BaseMessage]) -> str:
    """New messages appended to an existing session's conversation."""
    if all(isinstance(m, HumanMessage) for m in new):
        return "\n".join(_text(m.content) for m in new)
    return "\n".join(_render_lines(new))


def _fingerprint(header: str, messages: Sequence[BaseMessage]) -> str:
    """Hash of a conversation as callers rebuild it: text only, tool traffic ignored.

    Callers (the eval harness, Retell) rebuild history from spoken text each turn, so tool calls and
    tool results never reappear; consecutive assistant lines are merged the same way.
    """
    turns: list[list[str]] = []
    for m in messages:
        if isinstance(m, (SystemMessage, ToolMessage)):
            continue
        role = "ai" if isinstance(m, AIMessage) else "user"
        text = " ".join(_text(m.content).split())
        if not text:
            continue
        if role == "ai" and turns and turns[-1][0] == "ai":
            turns[-1][1] = f"{turns[-1][1]} {text}"
        else:
            turns.append([role, text])
    return hashlib.sha256(json.dumps([header, turns]).encode()).hexdigest()


class _Pool:
    """One Copilot CLI process per event loop, shared by all sessions."""

    def __init__(self) -> None:
        self.client: Any = None
        self.home: str | None = None
        self.lock = asyncio.Lock()
        # conversation fingerprint -> live session whose history ends there
        self.sessions: OrderedDict[str, Any] = OrderedDict()


MAX_IDLE_SESSIONS = 64


_POOLS: dict[int, _Pool] = {}


def _pool() -> _Pool:
    return _POOLS.setdefault(id(asyncio.get_running_loop()), _Pool())


async def _disconnect(session: Any) -> None:
    try:
        await session.disconnect()
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass


async def _client() -> Any:
    pool = _pool()
    async with pool.lock:
        if pool.client is None:
            from copilot import CopilotClient

            # Keep eval sessions out of the user's real ~/.copilot.
            pool.home = tempfile.mkdtemp(prefix="muse-evals-copilot-")
            token = os.getenv("COPILOT_GITHUB_TOKEN") or None
            client = CopilotClient(github_token=token, base_directory=pool.home)
            await client.start()
            pool.client = client
    return pool.client


async def aclose() -> None:
    """Stop the Copilot CLI started for the current event loop (no-op if none)."""
    pool = _POOLS.pop(id(asyncio.get_running_loop()), None)
    if pool:
        for session in pool.sessions.values():
            await _disconnect(session)
        pool.sessions.clear()
    if pool and pool.client is not None:
        try:
            await pool.client.stop()
        finally:
            if pool.home:
                shutil.rmtree(pool.home, ignore_errors=True)


def make_chat_model(name: str, *, temperature: float | None = None, json_mode: bool = False) -> BaseChatModel:
    """Build a chat model from ``copilot:<model>[@effort]`` or any ``init_chat_model`` name.

    Copilot has no temperature knob, so ``temperature`` only applies to other providers.
    """
    if name.startswith(PREFIX):
        return ChatCopilot.from_name(name)
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode and name.startswith("openai:"):
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return init_chat_model(name, **kwargs)


def is_copilot(name: str | None) -> bool:
    return bool(name) and name.startswith(PREFIX)


class CopilotError(RuntimeError):
    """A Copilot session reported an error (usually a transient model/API failure)."""


def _ok(*_: Any) -> str:
    return "ok"


class ChatCopilot(BaseChatModel):
    model: str
    reasoning_effort: str | None = None
    timeout: float = Field(default=180.0, description="Seconds to wait for one completion")
    max_retries: int = Field(default=2, description="Fresh-session retries after a transient failure")
    retry_backoff: float = Field(default=10.0, description="Base seconds between retries")
    _usage_log: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> ChatCopilot:
        model, effort = parse_model_name(name)
        return cls(model=model, reasoning_effort=effort, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    @property
    def usage_log(self) -> list[dict[str, Any]]:
        return self._usage_log

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # type: ignore[override]
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("ChatCopilot is async-only; use ainvoke/astream")

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError("ChatCopilot is async-only; use ainvoke/astream")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(self._astream(messages, stop, run_manager, **kwargs))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # The SDK already retries HTTP calls, but transient Copilot outages still surface as
        # session errors. Retry on a fresh session as long as nothing has been streamed yet.
        for attempt in range(self.max_retries + 1):
            started = False
            try:
                async for chunk in self._attempt(messages, run_manager, kwargs, reuse=attempt == 0):
                    started = True
                    yield chunk
                return
            except (CopilotError, TimeoutError):
                if started or attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.retry_backoff * (attempt + 1))

    async def _attempt(
        self,
        messages: list[BaseMessage],
        run_manager: AsyncCallbackManagerForLLMRun | None,
        kwargs: dict[str, Any],
        reuse: bool,
    ) -> AsyncIterator[ChatGenerationChunk]:
        from copilot.generated.session_events import (
            AssistantMessageData,
            AssistantMessageDeltaData,
            AssistantUsageData,
            SessionErrorData,
            SessionIdleData,
        )
        from copilot.session import PermissionHandler
        from copilot.tools import Tool

        system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
        conv = [m for m in messages if not isinstance(m, SystemMessage)]
        tool_specs = [t["function"] for t in kwargs.get("tools") or []]
        header = json.dumps([self.model, self.reasoning_effort, system, sorted(f["name"] for f in tool_specs)])
        sdk_tools = [
            Tool(
                name=f["name"],
                description=f.get("description", ""),
                parameters=f.get("parameters") or {"type": "object", "properties": {}},
                handler=_ok,  # LangGraph executes the real tool; we only surface the call
                skip_permission=True,
                is_terminal=True,  # stop after the call so the graph can run it
            )
            for f in tool_specs
        ]
        client = await _client()
        pool = _pool()
        session = None
        # Resume the session that produced the latest assistant turn, sending only what's new.
        last_ai = max((i for i, m in enumerate(conv) if isinstance(m, AIMessage)), default=-1)
        if last_ai >= 0 and reuse:
            session = pool.sessions.pop(_fingerprint(header, conv[: last_ai + 1]), None)
        if session is not None:
            prompt = _follow_up_prompt(conv[last_ai + 1 :])
        else:
            _, prompt = render_messages(messages)
            session = await client.create_session(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                system_message={"mode": "replace", "content": system or "You are a helpful assistant."},
                tools=sdk_tools,
                available_tools=[t.name for t in sdk_tools],
                streaming=True,
                on_permission_request=PermissionHandler.approve_all,
            )
        queue: asyncio.Queue[Any] = asyncio.Queue()
        unsubscribe = session.on(lambda ev: queue.put_nowait(ev.data))
        reply: list[str] = []
        ok = False
        streamed: set[str] = set()
        last_msg: str | None = None
        usage = {"input_tokens": 0, "output_tokens": 0, "premium_requests": 0.0}
        try:
            await session.send(prompt)
            deadline = asyncio.get_running_loop().time() + self.timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Copilot {self.model} did not finish in {self.timeout:.0f}s")
                data = await asyncio.wait_for(queue.get(), remaining)
                if isinstance(data, AssistantMessageDeltaData) and data.delta_content:
                    text = data.delta_content
                    if last_msg is not None and data.message_id != last_msg:
                        text = " " + text.lstrip()  # separate consecutive assistant messages
                    last_msg = data.message_id
                    streamed.add(data.message_id)
                    reply.append(text)
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                    if run_manager:
                        await run_manager.on_llm_new_token(data.delta_content, chunk=chunk)
                    yield chunk
                elif isinstance(data, AssistantMessageData):
                    if data.content and data.message_id not in streamed:
                        text = data.content if last_msg is None else " " + data.content.lstrip()
                        last_msg = data.message_id
                        reply.append(text)
                        yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                    tool_chunks = [
                        {
                            "name": r.name,
                            "args": json.dumps(r.arguments or {}),
                            "id": r.tool_call_id,
                            "index": i,
                        }
                        for i, r in enumerate(data.tool_requests or [])
                    ]
                    if tool_chunks:
                        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=tool_chunks))
                elif isinstance(data, AssistantUsageData):
                    usage["input_tokens"] += data.input_tokens or 0
                    usage["output_tokens"] += data.output_tokens or 0
                    usage["premium_requests"] += data.cost or 0.0
                elif isinstance(data, SessionErrorData):
                    raise CopilotError(f"Copilot {self.model} error ({data.error_type}): {data.message}")
                elif isinstance(data, SessionIdleData):
                    ok = True
                    break
        finally:
            if callable(unsubscribe):
                unsubscribe()
            key = _fingerprint(header, [*conv, AIMessage(content="".join(reply))])
            if ok and "".join(reply).strip():
                if (old := pool.sessions.pop(key, None)) is not None:
                    await _disconnect(old)
                pool.sessions[key] = session
                while len(pool.sessions) > MAX_IDLE_SESSIONS:
                    await _disconnect(pool.sessions.popitem(last=False)[1])
            else:
                await _disconnect(session)
            self._usage_log.append(usage)  # failed attempts still consume premium requests
        total = usage["input_tokens"] + usage["output_tokens"]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": total,
                },
                response_metadata={"premium_requests": usage["premium_requests"], "model_name": self.model},
            )
        )
