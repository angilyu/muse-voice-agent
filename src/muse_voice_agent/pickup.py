"""Tell call screeners, voicemail and phone menus apart from a person, from what they say.

Telephony reports all of them as "answered", so the only signal is the words on the line. This is a
cheap keyword classifier that runs on every business line before the model, so the graph can:

- answer a screener with who + why, then stay quiet while it "connects" us;
- skip the opener on a phone menu and press digits instead;
- introduce itself again when a person first says "Hello?" after we already spoke into silence, to a
  screener or to a menu, since that person never heard who we are.
"""

from __future__ import annotations

import re
from typing import Literal

LineKind = Literal["screener", "screener_wait", "voicemail", "voicemail_no_message", "menu", "person"]

_SCREENER = re.compile(
    r"screening service|call screen|(?:this call|calls?) (?:is|are) (?:being )?screened"
    r"|(?:state|say|record|tell (?:me|us))\s+(?:your|the)\s+name"
    r"|reason for (?:your |the )?call|why (?:you'?re|you are) calling"
    r"|who(?:'s| is) calling,? and what(?:'s| is) (?:this|it|the call) (?:regarding|about|for)\b"
    r"|(?:person|number|party) you(?:'re| are) (?:calling|trying to reach) (?:is|uses|has)"
    r"|google (?:assistant|voice)|before i (?:try to )?connect you|see if (?:this person|they|he|she)"
    r"(?:'re| are| is)? available",
    re.I,
)
_SCREENER_WAIT = re.compile(
    r"stay on the line|(?:will|'ll) be (?:right )?with you|being notified|connecting you\b"
    r"|(?:i'?ll|i will|let me|i'?m going to|now) (?:try to )?connect you\b|please (?:hold|wait)\b"
    r"|letting (?:them|him|her) know",
    re.I,
)
_VOICEMAIL = re.compile(
    r"leave (?:a|your) (?:brief |short )?message|after the (?:tone|beep)|at the (?:tone|beep)"
    r"|voice ?mail|record your message|(?:person|party|number|subscriber) you (?:are|were|'re) "
    r"(?:calling|trying to reach).*not available|(?:person|party|number|subscriber) .* is unavailable"
    r"|(?:unable to|can'?t|cannot) (?:take your call|come to the phone)",
    re.I,
)
_VOICEMAIL_NO_MESSAGE = re.compile(
    r"mailbox (?:is )?(?:full|not set up)|(?:mailbox|voicemail box) .* (?:full|not accepting|cannot accept|can't accept|not been set up|not set up)"
    r"|(?:not accepting|cannot accept|can't accept) (?:new )?messages"
    r"|(?:message|messages) (?:cannot|can't) be (?:left|recorded)"
    r"|no (?:tone|beep)|without (?:a )?(?:tone|beep)",
    re.I,
)
_MENU = re.compile(
    r"\bpress (?:\d|one|two|three|four|five|six|seven|eight|nine|zero|pound|star)\b"
    r"|para español|main menu|(?:this |your )?call (?:may|will) be (?:recorded|monitored)"
    r"|dial (?:the|your) (?:party'?s )?extension",
    re.I,
)
_GREETING = re.compile(
    r"^(?:hello|hi|hey|yo|good (?:morning|afternoon|evening))\b|\bhello\b|^(?:yes|yeah|yep)\W*$"
    r"|\bthis is \w+|\bspeaking\b|how (?:can|may) i (?:help|assist)|what can i do for you"
    r"|thanks? (?:you )?for calling|who(?:'s| is) (?:this|calling)",
    re.I,
)
# Retell/harness notes injected as HumanMessages ("[The call connected but nobody ...]").
NOTE_PREFIX = "["


def classify_line(text: str) -> LineKind:
    """Best guess at who produced one business line."""
    text = (text or "").strip()
    if _VOICEMAIL_NO_MESSAGE.search(text):
        return "voicemail_no_message"
    if _VOICEMAIL.search(text):
        return "voicemail"
    if _SCREENER.search(text):
        return "screener"
    if _MENU.search(text):
        return "menu"
    if _SCREENER_WAIT.search(text) and "?" not in text:
        return "screener_wait"
    return "person"


def is_greeting(text: str, business_name: str | None = None) -> bool:
    """A short pickup line ("Hello?", "Hi, this is Dana", "Luigi's, how can I help?")."""
    text = (text or "").strip()
    if not text or len(re.findall(r"[\w']+", text)) > 12:
        return False
    if _GREETING.search(text):
        return True
    return bool(business_name) and business_name.lower() in text.lower()


def is_note(text: str) -> bool:
    return (text or "").lstrip().startswith(NOTE_PREFIX)
