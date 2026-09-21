#!/usr/bin/env python3
"""Exercise the Agent Lens read tools against a live deployment.

Unit tests mock the HTTP boundary, so they prove request shaping but not that
the shapes still match a running Agent Lens. This script makes the real calls
and reports what came back, which is what catches drift after a field rename on
the Agent Lens side.

Reads only. Nothing here mutates Agent Lens state.

Usage:
    AGENT_LENS_BASE_URL=https://<host> WANDB_API_KEY=<key> \\
        uv run python scripts/agent_lens_smoke.py --entity acme --project support-bot

Exits non-zero if any tool returns an error envelope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from typing import Any, Callable

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.agent_lens import (
    get_category_breakdowns,
    get_clustering_status,
    get_conversation_tags,
    get_insights_coverage,
    get_tag_distribution,
    list_category_example_turns,
    list_conversation_tag_names,
    list_matching_turns,
    list_tagged_conversations,
)
from wandb_mcp_server.utils import get_server_args


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_api_key() -> str | None:
    """Put a key in the request context, as the server does at startup.

    WandBApiManager.get_api_key() reads a contextvar with no environment
    fallback, so calling a tool outside a served request finds nothing. Resolve
    the key through the same precedence the console entrypoint uses (env, then
    .netrc, then .env) and seed the context for this process.
    """
    api_key = os.getenv("WANDB_API_KEY") or get_server_args().wandb_api_key
    if api_key:
        WandBApiManager.set_context_api_key(api_key)
    return api_key


def _run(label: str, call: Callable[[], str], verbose: bool) -> tuple[bool, Any]:
    """Invoke one tool and summarize its envelope."""
    try:
        payload = json.loads(call())
    except Exception as error:  # a tool should return an envelope, never raise
        print(f"  FAIL  {label}: raised {type(error).__name__}: {error}")
        return False, None

    if isinstance(payload, dict) and "error" in payload:
        print(f"  FAIL  {label}: {payload['error']} -- {payload.get('message', '')}")
        return False, None

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        shape = f"{len(data)} item(s)"
    elif isinstance(data, dict):
        shape = f"object with keys {sorted(data)}"
    else:
        shape = type(data).__name__
    truncated = " [truncated]" if isinstance(payload, dict) and "_truncation" in payload else ""
    print(f"  ok    {label}: {shape}{truncated}")
    if verbose:
        print(json.dumps(payload, indent=2)[:2000])
    return True, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", required=True, help="W&B entity (team or username)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--days", type=int, default=7, help="Insights window size in days (max 30, default 7)")
    parser.add_argument("--verbose", action="store_true", help="Print each full response")
    args = parser.parse_args()

    if not os.getenv("AGENT_LENS_BASE_URL"):
        print("AGENT_LENS_BASE_URL is required (absolute HTTPS origin, no path).", file=sys.stderr)
        return 2
    if not 1 <= args.days <= 30:
        print("--days must be between 1 and 30.", file=sys.stderr)
        return 2
    if not _seed_api_key():
        print("No W&B API key found in WANDB_API_KEY, .netrc, or .env.", file=sys.stderr)
        return 2

    entity, project = args.entity, args.project

    print(f"Agent Lens smoke: {entity}/{project}")

    results = []

    print("\nInsights")
    ok, coverage = _run("insights coverage", lambda: get_insights_coverage(entity, project), args.verbose)
    results.append(ok)

    # Anchor the window on the last classified week rather than on today. The
    # classification job lags, so a range ending now routinely misses every
    # turn in a project that does have data -- which would look like a broken
    # client instead of an empty range.
    end = datetime.now(timezone.utc)
    latest_week = coverage.get("latest_week") if ok and isinstance(coverage, dict) else None
    if latest_week:
        try:
            end = datetime.strptime(latest_week, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=7)
        except ValueError:
            print(f"  warn  could not parse latest_week {latest_week!r}; anchoring the window on today")
    elif ok:
        # Every ranged read below will be empty; say so once rather than five times.
        print("  note  project has no classified Insights turns; ranged reads will be empty")

    start = end - timedelta(days=args.days)
    window = {"start_at": _rfc3339(start), "end_at": _rfc3339(end)}
    print(f"  window {window['start_at']}..{window['end_at']}")

    results.append(_run("clustering status", lambda: get_clustering_status(entity, project), args.verbose)[0])

    ok, breakdowns = _run(
        "category breakdowns",
        lambda: get_category_breakdowns(entity, project, **window),
        args.verbose,
    )
    results.append(ok)

    # Drill into real categories from both families. The top-level `category` is
    # an intent; `failure_breakdowns[].category` is a failure. Querying one as
    # the other returns zero rows rather than an error, so exercising both is
    # what actually proves the drilldowns work.
    intent = failure = None
    if ok and isinstance(breakdowns, list) and breakdowns:
        intent = breakdowns[0].get("category")
        for entry in breakdowns:
            failures = entry.get("failure_breakdowns") or []
            if failures:
                failure = failures[0].get("category")
                break

    for signature_type, category in (("intent", intent), ("failure", failure)):
        if not category:
            print(f"  skip  example turns / matching turns ({signature_type}): none in range")
            continue
        results.append(
            _run(
                f"example turns ({signature_type}={category})",
                lambda s=signature_type, c=category: list_category_example_turns(
                    entity, project, s, c, **window, limit=5
                ),
                args.verbose,
            )[0]
        )
        filter_name = "intent_category" if signature_type == "intent" else "failure_category"
        results.append(
            _run(
                f"matching turns ({signature_type}={category})",
                lambda n=filter_name, c=category: list_matching_turns(entity, project, **window, **{n: c}),
                args.verbose,
            )[0]
        )

    print("\nConversation tags")
    ok, tag_names = _run("tag names", lambda: list_conversation_tag_names(entity, project), args.verbose)
    results.append(ok)

    conversation_ids = []
    if ok and tag_names:
        found, conversation_ids = _run(
            f"tagged conversations ({tag_names[0]})",
            lambda: list_tagged_conversations(entity, project, [tag_names[0]]),
            args.verbose,
        )
        results.append(found)
        conversation_ids = conversation_ids or []
    else:
        print("  skip  tagged conversations: project has no tags")

    if conversation_ids:
        results.append(
            _run(
                "conversation tags",
                lambda: get_conversation_tags(entity, project, conversation_ids[:20]),
                args.verbose,
            )[0]
        )
    else:
        print("  skip  conversation tags: no tagged conversations")

    results.append(
        _run(
            "tag distribution",
            lambda: get_tag_distribution(
                entity,
                project,
                after_ms=int(start.timestamp() * 1000),
                before_ms=int(end.timestamp() * 1000),
                time_bucket_seconds=86400,
            ),
            args.verbose,
        )[0]
    )

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
