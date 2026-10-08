from muse_voice_agent.report import align_answers, build_report, speaker_transcript


def _record(**overrides):
    record = {
        "status": "completed",
        "outcome": "info_received",
        "task": {
            "kind": "general",
            "business_name": "Hotel Zed",
            "phone_number": "+14155550100",
            "customer_name": "Angi",
            "goal": "Check rooms",
            "questions": ["Is a king room available?", "Nightly rate?", "Parking?"],
            "shareable_details": {},
            "callback_number": None,
        },
        "details": {},
        "transcript": [{"role": "user", "text": "Hotel Zed."}],
        "started_at": 1_700_000_000.0,
        "ended_at": 1_700_000_089.4,
        "end_reason": "agent_hangup",
        "outcome_source": "agent",
    }
    record.update(overrides)
    return record


def test_align_answers_matches_by_question_and_flags_unanswered():
    aligned, unanswered = align_answers(
        ["Is a king room available?", "Nightly rate?", "Parking?"],
        [
            {"question": "nightly rate", "answer": "$289 plus tax"},
            {"question": "Is a king room available?", "answer": "Yes"},
            {"question": "Parking?", "answer": "They didn't say"},
            {"question": "Check-in time", "answer": "3 PM"},
        ],
    )
    assert aligned == [
        {"question": "Is a king room available?", "answer": "Yes"},
        {"question": "Nightly rate?", "answer": "$289 plus tax"},
        {"question": "Parking?", "answer": "not answered"},
        {"question": "Check-in time", "answer": "3 PM"},
    ]
    assert unanswered == ["Parking?"]


def test_align_answers_falls_back_to_order_when_agent_rephrased():
    aligned, unanswered = align_answers(
        ["Open Sunday?", "Do you deliver?"],
        [
            {"question": "Are you open on Sundays", "answer": "Yes, 10-4"},
            {"question": "Delivery available", "answer": "not answered"},
        ],
    )
    assert aligned == [
        {"question": "Open Sunday?", "answer": "Yes, 10-4"},
        {"question": "Do you deliver?", "answer": "not answered"},
    ]
    assert unanswered == ["Do you deliver?"]


def test_report_fields_for_info_call():
    report = build_report(
        _record(details={"answers": [{"question": "Is a king room available?", "answer": "Yes"}]})
    )
    assert report["request"] == {
        "customer_name": "Angi",
        "goal": "Check rooms",
        "questions": ["Is a king room available?", "Nightly rate?", "Parking?"],
    }
    assert report["reached"] == "person"
    assert report["started_at"] == "2023-11-14T22:13:20+00:00"
    assert report["duration_seconds"] == 89
    assert (report["end_reason"], report["ended_by"]) == ("The assistant ended the call.", "assistant")
    assert report["committed_on_users_behalf"] is False
    assert report["unanswered_questions"] == ["Nightly rate?", "Parking?"]
    assert report["next_steps"] == [
        "Share the answers with the user.",
        "Still unanswered: Nightly rate?; Parking?",
    ]


def test_report_for_booking_offers_calendar():
    report = build_report(
        _record(
            outcome="booked",
            task={"kind": "restaurant_reservation", "party_size": 4, "date": "Fri", "time": "7pm"},
            details={"confirmed_date": "Friday Oct 10", "confirmed_time": "7:30 PM", "booked_under": "Angi"},
        )
    )
    assert report["committed_on_users_behalf"] is True
    assert report["unanswered_questions"] == []
    assert report["next_steps"] == [
        "Tell the user it's booked for Friday Oct 10 7:30 PM under Angi and offer to add it to their calendar."
    ]


def test_report_for_order_inferred_from_transcript():
    report = build_report(
        _record(
            outcome="ordered",
            task={"kind": "general", "goal": "order tea"},
            details={"pickup_time": "6:15 PM", "order_total": "$12.40"},
            outcome_source="transcript",
        )
    )
    assert report["next_steps"] == [
        "Tell the user the order was placed (pickup 6:15 PM, total $12.40) and offer a pickup reminder.",
        "This result was inferred from the transcript; double-check key details with the user.",
    ]


def test_report_for_unanswered_call():
    report = build_report(
        _record(
            status="no_answer",
            outcome=None,
            transcript=[],
            started_at=None,
            ended_at=1_700_000_030.0,
            end_reason="dial_no_answer",
            outcome_source=None,
        )
    )
    assert report["reached"] == "no_answer"
    assert report["duration_seconds"] is None
    assert report["ended_by"] == "no_answer"
    assert report["next_steps"][0] == "Nobody answered; offer to retry later or try another business."


def test_report_for_voicemail_and_unknown_reason():
    voicemail = build_report(_record(outcome="voicemail", end_reason="voicemail_reached", transcript=[]))
    assert voicemail["reached"] == "voicemail"
    assert voicemail["next_steps"][0].startswith("Reached voicemail")
    left = build_report(
        _record(
            outcome="voicemail",
            end_reason="voicemail_reached",
            details={"voicemail_message": "Hi, please call back.", "callback_number": "+14155550123"},
        )
    )
    assert left["voicemail_message"] == "Hi, please call back."
    assert left["callback_number"] == "+14155550123"
    assert left["next_steps"][0].startswith("Left voicemail")
    odd = build_report(_record(status="failed", outcome=None, end_reason="error_llm_websocket_open"))
    assert odd["reached"] == "not_connected"
    assert odd["end_reason"] == "Call ended (error llm websocket open)."


def test_unavailable_shares_alternative():
    report = build_report(_record(outcome="unavailable", details={"availability": "8:45 PM same day"}))
    assert report["next_steps"][0] == (
        "Tell the user their request wasn't available; the business offered: 8:45 PM same day. "
        "Ask whether to take it."
    )


def test_speaker_transcript_labels_roles():
    assert speaker_transcript(
        [{"role": "user", "text": "Hello?", "t": 1.0}, {"role": "assistant", "text": "Hi!"}, {"role": "user", "text": ""}]
    ) == [{"speaker": "business", "text": "Hello?"}, {"speaker": "assistant", "text": "Hi!"}]
