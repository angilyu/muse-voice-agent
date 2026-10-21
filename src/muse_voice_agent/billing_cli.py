"""Admin commands for accounts and API keys.

Run where the call database lives: `uv run muse-voice-billing ...` locally, or on Render from the
service's Shell as `python -m muse_voice_agent.billing_cli ...`.

    muse-voice-billing list
    muse-voice-billing create-comp --email reviewer@example.com --minutes 60
    muse-voice-billing issue-key --email someone@example.com
    muse-voice-billing revoke --email someone@example.com
"""

from __future__ import annotations

import argparse
import sys

from .billing import Billing, format_date, minutes
from .config import get_settings
from .store import CallStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="muse-voice-billing", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List accounts with usage and subscription state")
    comp = sub.add_parser("create-comp", help="Create a complimentary account and print its key")
    comp.add_argument("--email", required=True)
    comp.add_argument("--minutes", type=int, required=True, help="Free call minutes")
    issue = sub.add_parser("issue-key", help="Replace an account's API key and print the new one")
    issue.add_argument("--email", required=True)
    revoke = sub.add_parser("revoke", help="Revoke all of an account's API keys")
    revoke.add_argument("--email", required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    billing = Billing(settings, CallStore(settings.call_db_path))

    if args.command == "list":
        for a in billing.list_accounts():
            if a.is_admin:
                continue
            u = billing.usage(a)
            paid = (
                "unlimited" if u.paid_total is None
                else f"{minutes(u.paid_used)}/{minutes(u.paid_total)} min" if u.paid_total
                else "-"
            )
            plan = billing.plan_for(a)
            label = plan.name if plan and a.plan == "subscription" else a.plan
            print(
                f"{a.id}  {a.email or '-':<32} {label:<12} {a.status:<9} "
                f"free {minutes(u.free_used)}/{minutes(u.free_total)} min  paid {paid}  "
                f"renews {format_date(a.current_period_end) or '-'}  "
                f"key …{billing.key_hint(a.id) or 'none'}"
            )
        return 0

    if args.command == "create-comp":
        if billing.find_by_email(args.email):
            print(f"An account for {args.email} already exists.", file=sys.stderr)
            return 1
        account = billing.create_account(args.email, plan="comp", free_seconds=args.minutes * 60)
        print(f"Created {account.id} with {args.minutes} free minutes.")
        print(billing.issue_key(account.id))
        return 0

    account = billing.find_by_email(args.email)
    if not account:
        print(f"No account for {args.email}.", file=sys.stderr)
        return 1
    if args.command == "issue-key":
        print(billing.issue_key(account.id))
    else:
        print(f"Revoked {billing.revoke_keys(account.id)} key(s) for {account.email}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
