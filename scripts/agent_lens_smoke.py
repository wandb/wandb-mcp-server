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


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

    entity, project = args.entity, args.project
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    window = {"start_at": _rfc3339(start), "end_at": _rfc3339(end)}

    print(f"Agent Lens smoke: {entity}/{project} over {window['start_at']}..{window['end_at']}")

    results = []

    print("\nInsights")
    ok, coverage = _run("insights coverage", lambda: get_insights_coverage(entity, project), args.verbose)
    results.append(ok)
    if ok and isinstance(coverage, dict) and not coverage.get("latest_week"):
        # Every ranged read below will be empty; say so once rather than nine times.
        print("  note  project has no classified Insights turns; ranged reads will be empty")

    results.append(_run("clustering status", lambda: get_clustering_status(entity, project), args.verbose)[0])

    ok, breakdowns = _run(
        "category breakdowns",
        lambda: get_category_breakdowns(entity, project, **window),
        args.verbose,
    )
    results.append(ok)

    # Drill into a real category when the breakdown named one; otherwise these
    # two tools would only ever be exercised against a guessed identifier.
    category = None
    if ok and isinstance(breakdowns, list) and breakdowns:
        counts = breakdowns[0].get("counts") or []
        category = counts[0].get("category") if counts else breakdowns[0].get("category")

    if category:
        results.append(
            _run(
                f"example turns ({category})",
                lambda: list_category_example_turns(entity, project, "intent", category, **window, limit=5),
                args.verbose,
            )[0]
        )
        results.append(
            _run(
                f"matching turns ({category})",
                lambda: list_matching_turns(entity, project, **window, intent_category=category),
                args.verbose,
            )[0]
        )
    else:
        print("  skip  example turns / matching turns: no category in range")

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
