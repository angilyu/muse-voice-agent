# Fixed phone-agent safety rules

This is an engineering summary, **not legal advice**. Laws and provider settings change; review with counsel before enabling real calls or changing recording behavior.

## Non-overridable rules in code

`src/muse_voice_agent/safety.py` defines the fixed baseline. Task input and environment variables can make behavior stricter, but cannot disable these rules:

1. **No payment or secret numbers.** MCP/tool input is scanned for Luhn-valid card numbers and SSN patterns before a call is created. The prompt forbids reading card numbers, CVV, expiry, bank/routing/account numbers, SSNs, passwords, and one-time codes. The output guard redacts card/SSN-like data if the model tries to say it. If payment by phone, a deposit, prepayment, or a secret is required, the agent offers pay-on-arrival/payment-link/customer-callback alternatives and records `needs_followup`.
2. **Allow-listed personal details only.** Every task has `shareable_details`. The server always adds customer name and callback number; any other personal detail must be explicitly listed for that call. Prompt rules forbid unlisted DOBs, addresses, emails, insurance IDs, account IDs, etc. The output guard redacts common unapproved emails, phone numbers, DOBs and street addresses when feasible.
3. **Authority and limits.** `authority="info_only"` cannot report a booking/order even if the model attempts one. Commitment calls require structured/free-form limits (`max_spend`, `max_deposit`, `max_cancellation_fee`, `allowed_date_time_window`, party-size range and/or `limits`). Obvious over-limit recorded outcomes are downgraded to `needs_followup`; reports include `committed_on_users_behalf`, `commitment_within_limits`, `safety_flags`, and `needs_user_action`.
4. **AI identity.** The fixed opener says “an AI assistant,” and the prompt requires truthful answers to “robot/AI/real person” questions. The agent must not claim to be the customer or a human.
5. **Recording disclosure and objection handling.** When call recording is enabled, the opener includes “This call may be recorded.” The default scope is `always`; `required_states` still discloses for known all-party-consent states or unknown area codes. If the business objects, the graph ends politely and records `needs_followup` because Retell mid-call recording-stop support is not wired here.
6. **No telemarketing.** The MCP instructions and prompt limit use to user-requested errands to businesses, not promotional or sales calls.

## Legal research summary

- **California CIPA / Penal Code §632.** California Penal Code §632 is part of the California Invasion of Privacy Act and restricts intentionally recording or eavesdropping on a “confidential communication” without consent of all parties. The implementation therefore discloses recording early for California calls and, by default, for every call. Source: California Legislative Information, Penal Code §632, https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=632.
- **California B.O.T. Act / SB 1001.** California Business & Professions Code §§17940–17943 require certain bots communicating online with people in California to disclose that they are bots when used to incentivize a purchase/sale or influence voting. Phone errands here are not telemarketing, but the product still makes direct AI disclosure because the caller is automated. Source: California Legislative Information, Bus. & Prof. Code §§17940–17943, https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?division=7.&chapter=6.&part=3.&lawCode=BPC.
- **FCC TCPA AI voice ruling.** The FCC’s February 2024 Declaratory Ruling held that AI-generated voices can be an “artificial or prerecorded voice” under the TCPA. This project must not be used for telemarketing/sales calls; user-requested informational/transactional errands still disclose AI identity. Source: FCC Declaratory Ruling FCC-24-17, https://docs.fcc.gov/public/attachments/FCC-24-17A1.txt.
- **All-party consent states.** Recording-consent rules vary and can depend on “confidential,” “private,” in-person vs telephone, and interstate-call facts. The state-aware helper treats these as all-party/heightened-consent states for disclosure: CA, WA, FL, IL, MD, MA, MT, NH, PA, plus nuanced CT/OR/NV/MI. Because area-code mapping is imperfect and interstate calls are common, the default is always to disclose recording.
- **Newer California AI disclosure laws.** California’s AI Transparency Act (SB 942, as amended; Bus. & Prof. Code §§22757.1–22757.5) adds disclosure/watermark-style obligations for covered generative-AI systems. Even where those producer-focused provisions do not directly govern this phone-call MCP server, the implementation uses explicit AI self-identification in the opener and on request. Source: California Legislative Information, Bus. & Prof. Code Division 8, Chapter 25, https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?division=8.&chapter=25.&part=&lawCode=BPC.

## Operational notes

- `RECORDING_DISCLOSURE_SCOPE=always` is recommended and is the default.
- If Retell recording is truly disabled outside this app, set `CALL_RECORDING_ENABLED=false`; otherwise leave it true.
- If you need to continue calls after recording objections, add provider-supported mid-call recording disablement first; do not silently keep recording.
