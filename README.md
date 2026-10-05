# MuseVoiceAgent

A dummy AI phone agent: **Retell AI** handles the phone line and voice (Twilio SIP, speech-to-text,
text-to-speech, turn-taking), and **LangGraph** decides what to say. It's exposed to **Meta Muse** as
an **MCP server**, so Muse can say *"call Luigi's and book a table for 4 on Friday at 7"* or *"get a
quote from Bob's Handyman to fix my faucet"*, and this agent places the actual phone call. A LiveKit
Agents backend is kept as an alternative (`VOICE_BACKEND=livekit`).

```
 Muse (Meta cloud)
   │  MCP over HTTPS + bearer token
   ▼
 cloudflared tunnel ──► muse-voice-mcp (:8765)
                          │ book_restaurant_reservation / request_handyman_quote
                          │ get_call_status / list_calls
                          │
                          ├─ DRY_RUN=true  → simulated call (no phone needed)
                          └─ DRY_RUN=false, VOICE_BACKEND=retell
                               ├─ POST /v2/create-phone-call ─► Retell ─► Twilio SIP trunk ─► business
                               ├─ Retell ⇄ wss://<tunnel>/retell/llm/<secret>/<call>   (custom LLM)
                               │     each turn: transcript → LangGraph graph → streamed reply
                               │     caller node ─► record_outcome tool ─► goodbye ─► end_call
                               └─ monitor: GET /v2/get-call until ended (no answer, voicemail…)
                          ▲
                          └── results + transcript in SQLite (data/calls.db)
```

**Who does what.** Retell: dialing, STT, TTS (voice `RETELL_VOICE_ID`), turn-taking/interruptions,
voicemail detection, max duration. Your code: the MCP tools and safety limits, the LangGraph
conversation and the `record_outcome` result, and the call log Muse reads. Retell doesn't bill for
the LLM in custom-LLM mode; OpenAI does.

## Layout

| File | What it does |
| --- | --- |
| `src/muse_voice_agent/mcp_server.py` | MCP server (streamable HTTP, bearer auth) that Muse connects to |
| `src/muse_voice_agent/dispatcher.py` | Starts a call: Retell (or LiveKit) outbound call, or a simulated call in dry-run mode |
| `src/muse_voice_agent/retell.py` | Retell REST client, custom-LLM websocket (runs the graph per turn), call monitor |
| `src/muse_voice_agent/agent.py` | Optional LiveKit worker (`VOICE_BACKEND=livekit`) |
| `src/muse_voice_agent/graph.py` | LangGraph conversation graph and the `record_outcome` tool |
| `src/muse_voice_agent/tasks.py` | Restaurant/handyman task schemas and the per-call system prompts |
| `src/muse_voice_agent/store.py` | SQLite call log shared by the MCP server and the worker |
| `scripts/setup_twilio_trunk.py` | Provisions the Twilio SIP trunk (and the LiveKit outbound trunk) |
| `scripts/setup_retell.py` | Creates/updates the Retell custom-LLM agent and imports the Twilio number |
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
| `RETELL_API_KEY` | dashboard.retellai.com → Settings → API Keys | live calls |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | console.twilio.com → Account Info | trunk setup (step 4) |
| `RETELL_AGENT_ID`, `RETELL_FROM_NUMBER`, `RETELL_WS_SECRET`, `PUBLIC_BASE_URL` | written by `scripts/setup_retell.py` | live calls |
| `LIVEKIT_*`, `SIP_OUTBOUND_TRUNK_ID` | cloud.livekit.io / `setup_twilio_trunk.py` | only for `VOICE_BACKEND=livekit` or `console` |
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

1. In Twilio, get a voice-capable number and put `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` in `.env`.
2. Create the Twilio Elastic SIP trunk, credential list, and number association (safe to re-run):
   ```bash
   uv run python scripts/setup_twilio_trunk.py +1XXXXXXXXXX
   ```
   (It also creates a LiveKit outbound trunk if LiveKit keys are set; that's only used by the
   LiveKit backend.)
3. Start the public tunnel (Retell must reach the websocket, Muse must reach `/mcp`):
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8765     # prints https://<random>.trycloudflare.com
   ```
4. Create the Retell agent and import the number (safe to re-run):
   ```bash
   uv run python scripts/setup_retell.py +1XXXXXXXXXX --public-url https://<random>.trycloudflare.com
   ```
5. Set `DRY_RUN=false` in `.env` and start the server. On startup it re-points the Retell agent at
   `PUBLIC_BASE_URL`, so after a tunnel restart just update that value and restart:
   ```bash
   uv run muse-voice-mcp
   ```
6. Test on **your own phone** first: `uv run python scripts/smoke_test.py --call +1YOURCELL`.
   Answer as the restaurant ("Hello, Test Bistro"). The agent waits for you to speak first.

LiveKit instead: set `VOICE_BACKEND=livekit` and also run `uv run muse-voice-agent dev` (needs a
paid LiveKit plan or your own STT/TTS keys; the free plan has no Inference quota).

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

- Requires a bearer token on every HTTP request (`/healthz` is the only open path). Retell's
  websocket can't send headers, so it needs a secret path (`RETELL_WS_SECRET`) and only attaches to
  calls this server started.
- `ALLOWED_DIAL_PREFIXES` (default `+1`) and `MAX_CONCURRENT_CALLS` limit who and how much it dials.
- The agent says it's an AI assistant in its first sentence. It never shares payment details or
  addresses, and it won't agree to deposits or fees; it records `needs_followup` instead.
- The handyman flow only gathers quotes and availability. It never books.
- Calls are capped at `MAX_CALL_SECONDS`, and the agent hangs up after recording an outcome.
- Check local laws on AI-voice disclosure and call recording before calling real businesses.
