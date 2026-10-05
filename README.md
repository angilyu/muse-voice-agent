# MuseVoiceAgent

A dummy AI phone agent, built with **LiveKit Agents** (voice and telephony) and **LangGraph**
(conversation logic). It's exposed to **Meta Muse** as an **MCP server**, so Muse can say
*"call Luigi's and book a table for 4 on Friday at 7"* or *"get a quote from Bob's Handyman to fix
my faucet"*, and this agent places the actual phone call.

```
 Muse (Meta cloud)
   │  MCP over HTTPS + Bearer token
   ▼
 cloudflared tunnel ──► MCP server (muse-voice-mcp, :8765)
                          │ book_restaurant_reservation / request_handyman_quote
                          │ get_call_status / list_calls
                          │
                          ├─ DRY_RUN=true  → simulated call (no phone needed)
                          └─ DRY_RUN=false → LiveKit agent dispatch (room "call-<id>")
                                               │
                                               ▼
                              LiveKit agent worker (muse-voice-agent)
                                 ├─ dials business via SIP outbound trunk (Twilio/Telnyx…)
                                 ├─ STT / TTS / turn detection: LiveKit Inference
                                 └─ LLM: LangGraph graph (LLMAdapter)
                                       caller node ─► record_outcome tool ─► goodbye ─► hang up
                          ▲
                          └── results + transcript in SQLite (data/calls.db)
```

## Layout

| File | What it does |
| --- | --- |
| `src/muse_voice_agent/mcp_server.py` | MCP server (streamable HTTP, bearer auth) that Muse connects to |
| `src/muse_voice_agent/dispatcher.py` | Starts a call: LiveKit agent dispatch, or a simulated call in dry-run mode |
| `src/muse_voice_agent/agent.py` | LiveKit worker: dials over SIP, runs the voice session, hangs up, saves the result |
| `src/muse_voice_agent/graph.py` | LangGraph conversation graph and the `record_outcome` tool |
| `src/muse_voice_agent/tasks.py` | Restaurant/handyman task schemas and the per-call system prompts |
| `src/muse_voice_agent/store.py` | SQLite call log shared by the MCP server and the worker |
| `scripts/smoke_test.py` | Calls the MCP server over HTTP the same way Muse does |

## 1. Install

```bash
uv sync
```

> `uv.lock` was generated against a corporate PyPI mirror. If you're outside that network, run
> `uv lock --default-index https://pypi.org/simple` once to regenerate it.

## 2. Configure `.env`

`.env` is git-ignored and already has a generated `MCP_AUTH_TOKEN`. Fill in:

| Key | Where to get it | Needed for |
| --- | --- | --- |
| `OPENAI_API_KEY` | platform.openai.com (or change `LLM_MODEL` to another `provider:model`) | live calls, console |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | cloud.livekit.io → Settings → API keys | live calls, console |
| `SIP_OUTBOUND_TRUNK_ID` | step 4 below | live calls |
| `DEFAULT_CUSTOMER_NAME`, `DEFAULT_CALLBACK_NUMBER` | you | optional |

`DRY_RUN=true` (the default) simulates every call, so you can wire Muse up before you have any keys.

## 3. Try it locally (no phone line)

```bash
uv run pytest                                    # unit + MCP tests
uv run muse-voice-mcp                            # terminal 1: MCP server on 127.0.0.1:8765
uv run python scripts/smoke_test.py --call +14155550123   # terminal 2: simulated booking
```

To talk to the voice agent yourself (you play the restaurant; needs the OpenAI and LiveKit keys):

```bash
uv run muse-voice-agent download-files   # one-time model download
uv run muse-voice-agent console
```

## 4. Enable real phone calls

1. Buy a number and create an Elastic SIP trunk with a provider, e.g.
   [Twilio](https://docs.livekit.io/telephony/start/providers/twilio.md) or
   [Telnyx](https://docs.livekit.io/telephony/start/providers/telnyx.md).
2. Create the LiveKit outbound trunk:
   ```bash
   cat > outbound-trunk.json <<'EOF'
   { "trunk": { "name": "muse-outbound", "address": "<your-trunk>.pstn.twilio.com",
                "numbers": ["+1XXXXXXXXXX"], "auth_username": "<user>", "auth_password": "<pass>" } }
   EOF
   lk sip outbound create outbound-trunk.json   # prints ST_xxx → SIP_OUTBOUND_TRUNK_ID
   ```
3. Set `DRY_RUN=false` in `.env`, then run both processes:
   ```bash
   uv run muse-voice-agent dev    # terminal 1: worker (registers as AGENT_NAME)
   uv run muse-voice-mcp          # terminal 2: MCP server
   ```
4. Test on **your own phone** first: `uv run python scripts/smoke_test.py --call +1YOURCELL`.
   Answer, pretend to be the restaurant, and watch the result come back.

## 5. Connect Muse

Muse runs in Meta's cloud, so it can't reach `localhost`. Expose the MCP server publicly:

```bash
cloudflared tunnel --url http://127.0.0.1:8765     # prints https://<random>.trycloudflare.com
uv run python scripts/smoke_test.py https://<random>.trycloudflare.com/mcp   # sanity check
```

(Use a named Cloudflare tunnel or deploy the server somewhere if you want a stable URL. Quick
tunnels change on every restart.)

Then send Muse one message:

> Build a custom integration to my phone-calling agent. It's an MCP server over streamable HTTP at
> `https://<random>.trycloudflare.com/mcp` and needs a bearer token (ask me through the secure
> credential flow). It can call restaurants to book tables and call handymen for quotes. Connect,
> list the tools, test `list_calls`, and save it as a reusable skill. Always confirm the business,
> number, and details with me before starting a call, and require my approval for call tools.

When Muse asks for the credential, enter the `MCP_AUTH_TOKEN` value from `.env` in the secure
prompt. Don't paste it into the chat. Then try: *"Use my phone agent to book a table for 2 at
<restaurant> (<phone>) this Friday at 7:30, flexible 6:30–8:30."*

## MCP tools

| Tool | Purpose |
| --- | --- |
| `book_restaurant_reservation(restaurant_name, phone_number, party_size, date, time, …)` | Starts a booking call, returns `call_id` immediately |
| `request_handyman_quote(business_name, phone_number, job_description, location, …)` | Starts a quote call (never books) |
| `get_call_status(call_id, include_transcript=False)` | Status, outcome (`booked`, `quote_received`, `unavailable`, `declined`, `needs_followup`, `voicemail`), details, transcript |
| `list_calls(limit=10)` | Recent calls |

Calls take minutes, so the tools return right away and Muse polls `get_call_status` until `done` is true.

## Safety built in

- Requires a bearer token on every request (`/healthz` is the only open path).
- `ALLOWED_DIAL_PREFIXES` (default `+1`) and `MAX_CONCURRENT_CALLS` limit who and how much it dials.
- The agent says it's an AI assistant in its first sentence. It never shares payment details or
  addresses, and it won't agree to deposits or fees; it records `needs_followup` instead.
- The handyman flow only gathers quotes and availability. It never books.
- Calls are capped at `MAX_CALL_SECONDS`, and the agent hangs up after recording an outcome.
- Check local laws on AI-voice disclosure and call recording before calling real businesses.
