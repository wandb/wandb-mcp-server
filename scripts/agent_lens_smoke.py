#!/usr/bin/env python3
"""Exercise the Agent Lens read tools against a live deployment.

Unit tests mock the HTTP boundary, so they prove request shaping but not that
the shapes still match a running Agent Lens. This script makes the real calls
and reports what came back, which is what catches drift after a field rename on
the Agent Lens side.

Reads only. Nothing here mutates Agent Lens state.

Usage:
    AGENT_LENS_BASE_URL=https://<host> WANDB_API_KEY=<key> \\
        uv run python scripts/agent_lens_smoke.py \\
        --entity acme --project support-bot \\
        --start-at 2026-09-01T00:00:00Z --end-at 2026-09-08T00:00:00Z \\
        --signature-type intent --category-id action_request \\
        --topic-id topic-123 --trace-id trace-123 \\
        --tag-id 30201f95-1221-433a-9ea5-1e513081962f \\
        --tag-name reviewed --conversation-id conv-123

The fixture arguments must identify populated, approved test data. The script
exits non-zero unless every one of the nine read endpoints returns matching
fixture data; empty results are not counted as qualification success.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Awaitable, Callable
from uuid import UUID

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.agent_lens import (
    get_category_breakdowns,
    get_clustering_status,
    get_conversation_tags,
    get_failure_attributions,
    get_insights_coverage,
    get_tag_distribution,
    list_category_example_turns,
    list_tags,
    list_tagged_conversations,
)
from wandb_mcp_server.utils import get_server_args


def _parse_rfc3339(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


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


async def _run(
    label: str,
    call: Callable[[], Awaitable[str]],
    verbose: bool,
    *,
    qualifies: Callable[[Any], bool],
) -> tuple[bool, Any]:
    """Invoke one tool and summarize its envelope."""
    try:
        payload = json.loads(await call())
    except Exception as error:  # a tool should return an envelope, never raise
        print(f"  FAIL  {label}: raised {type(error).__name__}")
        return False, None

    if isinstance(payload, dict) and "error" in payload:
        print(f"  FAIL  {label}: {payload['error']} -- {payload.get('message', '')}")
        return False, None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not qualifies(data):
        print(f"  FAIL  {label}: the qualified fixture returned no matching data")
        return False, data
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


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _category_present(value: Any, signature_type: str, category_id: str) -> bool:
    if not isinstance(value, list):
        return False
    if signature_type == "intent":
        return any(isinstance(row, dict) and row.get("category") == category_id for row in value)
    return any(
        isinstance(row, dict)
        and any(
            isinstance(item, dict) and item.get("category") == category_id
            for field in ("counts", "failure_breakdowns")
            for item in (row.get(field) or [])
        )
        for row in value
    )


def _conversation_tag_present(value: Any, conversation_id: str, tag_id: str, tag_name: str) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict)
        and item.get("conversation_id") == conversation_id
        and item.get("tag_id") == tag_id
        and item.get("tag") == tag_name
        for item in value
    )


def _distribution_has_tag(value: Any, tag_id: str) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("buckets"), list):
        return False
    return any(
        isinstance(bucket, dict) and isinstance(bucket.get("tag_counts"), dict) and tag_id in bucket["tag_counts"]
        for bucket in value["buckets"]
    )


def _tag_present(value: Any, tag_id: str, tag_name: str) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict) and item.get("id") == tag_id and item.get("name") == tag_name for item in value
    )


def _trace_present(value: Any, trace_id: str) -> bool:
    return isinstance(value, list) and any(
        isinstance(item, dict) and item.get("trace_id") == trace_id for item in value
    )


def _conversation_present(value: Any, conversation_id: str) -> bool:
    return isinstance(value, list) and conversation_id in value


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", required=True, help="W&B entity (team or username)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--start-at", required=True, help="Inclusive RFC 3339 fixture-window start")
    parser.add_argument("--end-at", required=True, help="Exclusive RFC 3339 fixture-window end")
    parser.add_argument(
        "--signature-type",
        required=True,
        choices=("intent", "failure"),
        help="Category family for the qualified category fixture",
    )
    parser.add_argument("--category-id", required=True, help="Qualified category in the selected family")
    parser.add_argument("--topic-id", required=True, help="Qualified topic ID refining the category fixture")
    parser.add_argument("--trace-id", required=True, help="Qualified trace with failure-attribution data")
    parser.add_argument("--tag-id", required=True, help="UUID of the qualified conversation-tag fixture")
    parser.add_argument("--tag-name", required=True, help="Qualified conversation-tag fixture")
    parser.add_argument("--conversation-id", required=True, help="Qualified conversation carrying that tag")
    parser.add_argument("--verbose", action="store_true", help="Print each full response")
    args = parser.parse_args()

    if not os.getenv("AGENT_LENS_BASE_URL"):
        print("AGENT_LENS_BASE_URL is required (absolute HTTPS origin, no path).", file=sys.stderr)
        return 2
    try:
        start = _parse_rfc3339(args.start_at, "--start-at")
        end = _parse_rfc3339(args.end_at, "--end-at")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if end <= start or (end - start).total_seconds() > 30 * 24 * 60 * 60:
        print("The fixture window must be nonempty and no longer than 30 days.", file=sys.stderr)
        return 2
    fixture_values = (
        args.category_id,
        args.topic_id,
        args.trace_id,
        args.tag_id,
        args.tag_name,
        args.conversation_id,
    )
    if any(not value.strip() for value in fixture_values):
        print("Category, topic, trace, tag, and conversation fixtures must be nonempty.", file=sys.stderr)
        return 2
    try:
        UUID(args.tag_id)
    except ValueError:
        print("--tag-id must be a UUID.", file=sys.stderr)
        return 2
    if not _seed_api_key():
        print("No W&B API key found in WANDB_API_KEY, .netrc, or .env.", file=sys.stderr)
        return 2

    entity, project = args.entity, args.project
    window = {"start_at": args.start_at, "end_at": args.end_at}
    print("Agent Lens smoke: exercising nine read endpoints against approved fixtures")

    results: list[bool] = []

    print("\nInsights")
    results.append(
        (
            await _run(
                "insights coverage",
                lambda: get_insights_coverage(entity, project),
                args.verbose,
                qualifies=_nonempty_mapping,
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "clustering status",
                lambda: get_clustering_status(entity, project),
                args.verbose,
                qualifies=_nonempty_list,
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "category breakdowns",
                lambda: get_category_breakdowns(entity, project, **window),
                args.verbose,
                qualifies=lambda data: _category_present(data, args.signature_type, args.category_id),
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "category example turns",
                lambda: list_category_example_turns(
                    entity,
                    project,
                    args.signature_type,
                    args.category_id,
                    **window,
                    topic_ids=[args.topic_id],
                    limit=5,
                ),
                args.verbose,
                qualifies=lambda data: _trace_present(data, args.trace_id),
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "failure attributions",
                lambda: get_failure_attributions(entity, project, [args.trace_id]),
                args.verbose,
                qualifies=lambda data: _trace_present(data, args.trace_id),
            )
        )[0]
    )

    print("\nConversation tags")
    results.append(
        (
            await _run(
                "tag catalog",
                lambda: list_tags(entity, project),
                args.verbose,
                qualifies=lambda data: _tag_present(data, args.tag_id, args.tag_name),
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "conversation tags",
                lambda: get_conversation_tags(entity, project, [args.conversation_id]),
                args.verbose,
                qualifies=lambda data: _conversation_tag_present(
                    data,
                    args.conversation_id,
                    args.tag_id,
                    args.tag_name,
                ),
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "tagged conversations",
                lambda: list_tagged_conversations(entity, project, [args.tag_id]),
                args.verbose,
                qualifies=lambda data: _conversation_present(data, args.conversation_id),
            )
        )[0]
    )
    results.append(
        (
            await _run(
                "tag distribution",
                lambda: get_tag_distribution(
                    entity,
                    project,
                    after_ms=int(start.timestamp() * 1000),
                    before_ms=int(end.timestamp() * 1000),
                    time_bucket_seconds=86400,
                ),
                args.verbose,
                qualifies=lambda data: _distribution_has_tag(data, args.tag_id),
            )
        )[0]
    )

    if len(results) != 9:
        print("\nFAIL: qualification did not exercise exactly nine endpoints", file=sys.stderr)
        return 1
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
