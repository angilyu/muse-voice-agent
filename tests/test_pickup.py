import pytest

from muse_voice_agent.pickup import classify_line, is_greeting, is_note


@pytest.mark.parametrize(
    "line, kind",
    [
        ("Hi, the person you're calling is using a screening service from Google. Go ahead and say your name and why you're calling.", "screener"),
        ("Hi, if you record your name and reason for calling, I'll see if this person is available.", "screener"),
        ("This call is being screened. What is the reason for your call?", "screener"),
        ("Hi, who's calling and what's this regarding?", "screener"),
        ("Thanks, please stay on the line.", "screener_wait"),
        ("Okay, connecting you now.", "screener_wait"),
        ("One sec, please hold.", "screener_wait"),
        ("Hi, you've reached Mike. Please leave your name and number after the beep.", "voicemail"),
        ("The person you are calling is unable to take your call.", "voicemail"),
        ("The person you are calling is not available. Please leave a message after the tone.", "voicemail"),
        ("Sorry, this mailbox is full and cannot accept new messages.", "voicemail_no_message"),
        ("This voicemail box has not been set up yet.", "voicemail_no_message"),
        ("Thank you for calling Luigi's. For reservations, press 1.", "menu"),
        ("This call may be recorded for quality. Para español, oprima dos.", "menu"),
        ("Hello?", "person"),
        ("Smoke Test Cafe, how can I help?", "person"),
        ("Sorry, 7 is not available, how about 5?", "person"),
        ("What is this regarding?", "person"),
        ("Sorry, I can't connect you.", "person"),
        ("Can you please hold?", "person"),
    ],
)
def test_classify_line(line, kind):
    assert classify_line(line) == kind


def test_is_greeting():
    assert is_greeting("Hello?")
    assert is_greeting("Hi, this is Dana.")
    assert is_greeting("Luigi's.", "Luigi's")
    assert is_greeting("Thanks for calling, how can I help?")
    assert is_greeting("Yes?")
    assert not is_greeting("Sorry, we only have 5 PM.")
    assert not is_greeting("Yeah, hold on, let me check.")
    assert not is_greeting("Hello, " + "word " * 20)


def test_is_note():
    assert is_note("[The call connected but nobody has spoken]")
    assert not is_note("Hello?")
