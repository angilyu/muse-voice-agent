"""Self-ping so free hosts (e.g. Render's Free plan) don't spin the server down when idle.

Render sleeps a free web service after 15 minutes without inbound traffic; a request to our own
public URL goes back in through Render's proxy, so it counts as traffic.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


async def keep_awake(
    url: str, interval: float, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    async with httpx.AsyncClient(timeout=30, transport=transport) as client:
        while True:
            await asyncio.sleep(interval)
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.warning("keepalive ping %s -> HTTP %s", url, resp.status_code)
            except httpx.HTTPError as exc:
                logger.warning("keepalive ping %s failed: %s", url, exc)


class KeepAliveMiddleware:
    """Starts keep_awake() once, when the ASGI server begins its lifespan."""

    def __init__(
        self,
        app: ASGIApp,
        url: str,
        interval: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.app = app
        self.url = url
        self.interval = interval
        self.transport = transport
        self.task: asyncio.Task | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" and self.task is None:
            logger.info("keepalive: pinging %s every %ss", self.url, self.interval)
            self.task = asyncio.create_task(keep_awake(self.url, self.interval, self.transport))
        await self.app(scope, receive, send)
