"""LiveKit agent worker: dials the business over SIP and runs the LangGraph conversation.

Run:
    uv run muse-voice-agent dev       # connect to LiveKit and wait for calls dispatched by the MCP server
    uv run muse-voice-agent console   # talk to the agent in your terminal; you play the restaurant
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import langchain

from .config import get_settings
from .graph import CallControl, CallOutcome, build_call_graph
from .store import CallStore
from .tasks import RestaurantReservation, parse_task

logger = logging.getLogger("muse_voice_agent.agent")

settings = get_settings()
server = AgentServer()

DEMO_TASK = RestaurantReservation(
    business_name="Demo Bistro",
    phone_number="+14155550123",
    customer_name=settings.default_customer_name or "Alex",
    party_size=2,
    date="this Friday",
    time="7:30 PM",
    flexibility="anytime between 6:30 and 8:30 PM the same day",
)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    store = CallStore(settings.call_db_path)
    metadata = json.loads(ctx.job.metadata or "{}") if ctx.job.metadata else {}
    call_id: str | None = metadata.get("call_id")

    if call_id:
        record = store.get_call(call_id)
        if record is None:
            logger.error("unknown call_id %s", call_id)
            ctx.shutdown("unknown call")
            return
        task = parse_task(record["task"])
    else:
        task = DEMO_TASK
        logger.info("no call_id in job metadata; running demo task (console mode)")

    control = CallControl()
    result: dict[str, CallOutcome] = {}

    def on_outcome(outcome: CallOutcome) -> None:
        result["outcome"] = outcome
        logger.info("call outcome: %s", outcome.model_dump_json())
        if call_id:
            store.update_call(
                call_id,
                status="completed",
                outcome=outcome.outcome,
                summary=outcome.summary,
                details=outcome.model_dump(exclude_none=True),
                outcome_source="agent",
            )

    graph = build_call_graph(task, on_outcome, control=control)

    sip_identity = f"callee-{call_id}" if call_id else None
    if call_id:
        store.update_call(call_id, status="dialing")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=settings.sip_outbound_trunk_id,
                    sip_call_to=task.phone_number,
                    participant_identity=sip_identity,
                    participant_name=task.business_name,
                    wait_until_answered=True,
                )
            )
        except api.SipCallError as e:
            logger.warning("call failed: %s %s", e.sip_status_code, e.sip_status)
            status = "no_answer" if e.sip_status_code in (408, 480, 486, 487, 603) else "failed"
            store.update_call(call_id, status=status, error=f"SIP {e.sip_status_code} {e.sip_status}")
            ctx.shutdown("call not answered")
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("dial failed")
            store.update_call(call_id, status="failed", error=str(e))
            ctx.shutdown("dial failed")
            return
        await ctx.wait_for_participant(identity=sip_identity)
        store.update_call(call_id, status="in_progress", started_at=time.time())

    session = AgentSession(
        stt=inference.STT(model=settings.stt_model, language="en"),
        tts=inference.TTS(model=settings.tts_model, voice=settings.tts_voice),
        llm=langchain.LLMAdapter(graph=graph, stream_mode="custom"),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # noqa: ANN001
        item = ev.item
        text = getattr(item, "text_content", None)
        if call_id and text and getattr(item, "role", None) in ("user", "assistant"):
            store.append_transcript(call_id, item.role, text)

    hung_up = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_disconnect(p: rtc.RemoteParticipant) -> None:
        if sip_identity and p.identity == sip_identity:
            hung_up.set()

    last_activity = time.monotonic()

    @session.on("user_state_changed")
    @session.on("agent_state_changed")
    def _on_state(ev) -> None:  # noqa: ANN001
        nonlocal last_activity
        last_activity = time.monotonic()

    # The callee speaks first ("Hello, Luigi's"), so no greeting here.
    await session.start(agent=Agent(instructions=""), room=ctx.room)

    async def wait_for_playout() -> None:
        await asyncio.sleep(0.5)
        speech = session.current_speech
        if speech is not None:
            try:
                await asyncio.wait_for(speech.wait_for_playout(), timeout=15)
            except asyncio.TimeoutError:
                pass

    deadline = time.monotonic() + settings.max_call_seconds
    while not hung_up.is_set() and time.monotonic() < deadline:
        if control.pending_digits:
            digits, control.pending_digits = control.pending_digits, None
            await wait_for_playout()
            for d in digits:
                code = {"*": 10, "#": 11}.get(d, int(d) if d.isdigit() else -1)
                if code >= 0:
                    await ctx.room.local_participant.publish_dtmf(code=code, digit=d)
                    await asyncio.sleep(0.3)
        if control.end_requested:
            await wait_for_playout()
            await asyncio.sleep(1.0)
            break
        idle = session.current_speech is None and session.user_state != "speaking"
        if control.closing and idle and time.monotonic() - last_activity > 5:
            break  # they went quiet after our goodbye
        await asyncio.sleep(0.25)

    if hung_up.is_set():
        end_reason = "user_hangup"
    elif control.end_requested:
        end_reason = "agent_hangup"
    else:
        end_reason = "max_duration_reached"
    if call_id:
        store.update_call(call_id, ended_at=time.time(), end_reason=end_reason)
    if call_id and "outcome" not in result:
        reason = end_reason.replace("_", " ")
        store.update_call(
            call_id,
            status="completed",
            outcome="needs_followup",
            summary=f"Call ended without a recorded result ({reason}). See transcript.",
            outcome_source="call_system",
        )

    if call_id:
        try:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:  # noqa: BLE001
            logger.exception("failed to delete room")
    ctx.shutdown("call finished")


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
