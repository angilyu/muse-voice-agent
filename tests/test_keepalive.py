import asyncio

import httpx

from muse_voice_agent.keepalive import KeepAliveMiddleware, keep_awake


async def test_keep_awake_pings_url_repeatedly_and_survives_errors():
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        if len(hits) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": True})

    task = asyncio.create_task(
        keep_awake("https://svc.example/healthz", 0.01, httpx.MockTransport(handler))
    )
    await asyncio.sleep(0.1)
    task.cancel()
    assert len(hits) >= 3
    assert set(hits) == {"https://svc.example/healthz"}


async def test_middleware_starts_pinger_once_on_lifespan():
    seen: list[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    hits: list[str] = []
    mw = KeepAliveMiddleware(
        inner,
        "https://svc.example/healthz",
        0.01,
        httpx.MockTransport(lambda r: hits.append(str(r.url)) or httpx.Response(200)),
    )
    await mw({"type": "http", "path": "/healthz"}, None, None)
    assert mw.task is None
    await mw({"type": "lifespan"}, None, None)
    first = mw.task
    await mw({"type": "lifespan"}, None, None)
    assert mw.task is first
    await asyncio.sleep(0.05)
    first.cancel()
    assert hits and seen == ["http", "lifespan", "lifespan"]


def test_build_app_wraps_keepalive_only_when_configured(tmp_path):
    from dataclasses import replace

    from muse_voice_agent.config import Settings
    from muse_voice_agent.mcp_server import build_app

    settings = replace(Settings(), mcp_auth_token="t", call_db_path=tmp_path / "calls.db")
    off = build_app(replace(settings, keepalive_seconds=0, public_base_url="https://x.example"))
    assert not isinstance(off, KeepAliveMiddleware)
    on = build_app(replace(settings, keepalive_seconds=600, public_base_url="https://x.example"))
    assert isinstance(on, KeepAliveMiddleware)
    assert on.url == "https://x.example/healthz"
