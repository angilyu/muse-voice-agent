"""Call the running MCP server over HTTP exactly like Muse would.

    uv run python scripts/smoke_test.py                                  # local server
    uv run python scripts/smoke_test.py https://<tunnel-host>/mcp        # through the tunnel
    uv run python scripts/smoke_test.py --call +14155550123              # also start a test call

With DRY_RUN=true the call is simulated. With DRY_RUN=false it places a REAL phone call to the
number you pass, so only use a number you own.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from muse_voice_agent.config import get_settings


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=f"http://{settings.mcp_host}:{settings.mcp_port}/mcp")
    parser.add_argument("--call", metavar="PHONE", help="start a test restaurant call to PHONE")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}
    async with httpx2.AsyncClient(headers=headers, timeout=30) as http:
        async with Client(streamable_http_client(args.url, http_client=http)) as client:
            tools = (await client.list_tools()).tools
            print("tools:", [t.name for t in tools])
            if not args.call:
                return
            started = await client.call_tool(
                "book_restaurant_reservation",
                {
                    "restaurant_name": "Smoke Test Cafe",
                    "phone_number": args.call,
                    "party_size": 2,
                    "date": "tomorrow",
                    "time": "7 PM",
                    "customer_name": settings.default_customer_name or "Test User",
                },
            )
            data = started.structured_content
            print("started:", json.dumps(data, indent=2))
            if "call_id" not in data:
                return
            while True:
                await asyncio.sleep(3)
                res = await client.call_tool(
                    "get_call_status", {"call_id": data["call_id"], "include_transcript": True}
                )
                status = res.structured_content
                print("status:", status["status"])
                if status["done"]:
                    print(json.dumps(status, indent=2))
                    return


if __name__ == "__main__":
    asyncio.run(main())
