"""Sends sign-up and sign-in codes by email over SMTP (works with Gmail, Resend, SendGrid...)."""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage

from .config import Settings

logger = logging.getLogger("muse_voice_agent.mailer")


class EmailError(Exception):
    pass


class Mailer:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def send(self, to: str, subject: str, body: str) -> None:
        if self.settings.smtp_host == "log":
            logger.warning("email (SMTP_HOST=log) to %s: %s\n%s", to, subject, body)
            return
        try:
            await asyncio.to_thread(self._send, to, subject, body)
        except (OSError, smtplib.SMTPException) as e:
            logger.error("couldn't send email to %s: %s", to, e)
            raise EmailError(str(e)) from e

    def _send(self, to: str, subject: str, body: str) -> None:
        s = self.settings
        msg = EmailMessage()
        msg["From"] = s.email_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        context = ssl.create_default_context()
        if s.smtp_port == 465:
            server: smtplib.SMTP = smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=20, context=context)
        else:
            server = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20)
        with server:
            if s.smtp_port != 465:
                server.starttls(context=context)
            if s.smtp_username:
                server.login(s.smtp_username, s.smtp_password)
            server.send_message(msg)


class Outbox(Mailer):
    """Keeps emails in memory instead of sending them; for tests."""

    def __init__(self, settings: Settings | None = None):
        self.sent: list[tuple[str, str, str]] = []

    async def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))
