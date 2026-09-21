"""MCP tools for the **Agent Lens** Insights and conversation-tag read APIs.

Agent Lens is a separate W&B service with its own Huma-generated HTTP API. It
classifies agent turns into intent/failure categories and clusters, and carries
human- and judge-applied conversation tags. None of that is visible to the Weave
calls or Agents data planes, so these tools complement -- they do not duplicate
-- ``query_weave_traces_tool`` and the ``*_weave_agent_*`` family.

Every tool here is a read. Agent Lens authenticates a W&B API key as a bearer
token and derives project authorization from it server-side, so an MCP caller
can never read a project the key itself cannot reach. Requests carry the project
in ``X-Wandb-Entity`` / ``X-Wandb-Project`` headers, matching ``ProjectHeaders``
in ``internal/api/project.go`` of github.com/wandb/agent-lens.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

import requests

from wandb_mcp_server.api_client import WandBApiManager, raise_for_wandb_server_busy
from wandb_mcp_server.config import (
    AGENT_LENS_API_PREFIX,
    MAX_RESPONSE_TOKENS,
    resolve_agent_lens_base_url,
    structured_error,
)
from wandb_mcp_server.mcp_tools.tools_utils import get_no_retry_session, track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.processors import TraceProcessor

logger = get_rich_logger(__name__)

# Read paths on the Agent Lens API, relative to AGENT_LENS_API_PREFIX. Confirmed
# against internal/api/insights.go and internal/api/conversation_tags.go.
LATEST_WEEK_PATH = "/insights/latest-week"
CLUSTERING_STATUS_PATH = "/insights/clustering-status"
CATEGORY_BREAKDOWNS_PATH = "/insights/intent-category-breakdowns"
MATCHING_TURNS_PATH = "/insights/matching-turns"
CONVERSATION_TAG_NAMES_PATH = "/conversation-tags"
CONVERSATION_TAGS_QUERY_PATH = "/conversation-tags/query"
TAGGED_CONVERSATIONS_PATH = "/conversation-tags/conversations/query"
TAG_DISTRIBUTION_PATH = "/conversation-tags/distribution"


def _category_examples_path(signature_type: str, category_id: str) -> str:
    """Build the drilldown path, percent-encoding the caller-supplied category."""
    return f"/insights/{quote(signature_type, safe='')}/categories/{quote(category_id, safe='')}/example-turns"


_REQUEST_TIMEOUT_SECONDS = 30

# Agent Lens bounds every ranged Insights read to 30 days (insights.Window.Validate).
# Rejecting locally turns a 422 round-trip into an actionable message.
MAX_INSIGHTS_WINDOW_DAYS = 30

# Server-enforced request bounds, mirrored here so an oversized argument fails
# with a usable message instead of a generic 422.
MAX_CONVERSATION_IDS = 5000
MAX_TAG_FILTERS = 100
MAX_CLUSTER_IDS = 20
MAX_EXAMPLE_LIMIT = 50
MAX_TIME_BUCKET_SECONDS = 86400


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so Agent Lens applies its own field defaults."""
    return {k: v for k, v in values.items() if v is not None}


def _truncate_response(payload: Any) -> Any:
    """Trim the response's primary list so the serialized payload fits the budget.

    Agent Lens wraps every read in ``{"data": ...}``. That is a list for most
    reads and an object holding ``buckets`` for the tag distribution, so we trim
    whichever one is actually present and annotate ``_truncation`` to tell the
    caller to narrow the range rather than silently returning a partial answer.
    """
    if not isinstance(payload, dict):
        return payload
    if TraceProcessor.estimate_tokens(json.dumps(payload, default=str)) <= MAX_RESPONSE_TOKENS:
        return payload

    data = payload.get("data")
    if isinstance(data, list) and data:
        container, key = payload, "data"
    elif isinstance(data, dict) and isinstance(data.get("buckets"), list) and data["buckets"]:
        container, key = data, "buckets"
    else:
        return payload  # nothing structural to trim

    items = container[key]
    original = len(items)
    kept = list(items)

    def rebuilt(candidate: List[Any]) -> Dict[str, Any]:
        if container is payload:
            return {**payload, key: candidate}
        return {**payload, "data": {**data, key: candidate}}

    # Items vary in size, so re-measure and drop ~10% of the remainder (at least
    # one) each pass. This converges quickly while overshooting as little as
    # possible -- the same geometric shrink the Agents tools use.
    while kept and TraceProcessor.estimate_tokens(json.dumps(rebuilt(kept), default=str)) > MAX_RESPONSE_TOKENS:
        kept = kept[: -max(1, len(kept) // 10)]

    result = rebuilt(kept)
    result["_truncation"] = {
        "applied": True,
        "field": key,
        "returned": len(kept),
        "original": original,
        "note": (
            f"Response truncated to fit the {MAX_RESPONSE_TOKENS}-token budget; dropped "
            f"{original - len(kept)} '{key}' item(s). Narrow with a tighter time range, a "
            f"smaller limit, or more specific filters."
        ),
    }
    return result


def _parse_timestamp(value: str, field: str) -> datetime:
    """Parse an RFC 3339 bound the way Agent Lens does, normalized to UTC."""
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be an RFC 3339 timestamp")
    # datetime.fromisoformat accepts "Z" only from 3.11; the server uses RFC3339Nano.
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"{field} must be an RFC 3339 timestamp, for example 2026-09-01T00:00:00Z") from None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validated_window(start_at: str, end_at: str) -> Dict[str, str]:
    """Check the range locally against the server's own bounds before sending."""
    start = _parse_timestamp(start_at, "start_at")
    end = _parse_timestamp(end_at, "end_at")
    if end <= start:
        raise ValueError("start_at and end_at must define a nonempty time range")
    if end - start > timedelta(days=MAX_INSIGHTS_WINDOW_DAYS):
        raise ValueError(f"time range must not exceed {MAX_INSIGHTS_WINDOW_DAYS} days")
    return {"start_at": start_at, "end_at": end_at}


def _bounded_list(values: Sequence[str], maximum: int, field: str) -> List[str]:
    """Reject an oversized list locally rather than trading a 422 round-trip."""
    items = [str(value) for value in values]
    if not items:
        raise ValueError(f"{field} must contain at least one value")
    if len(items) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} values (got {len(items)})")
    return items


def _agent_lens_request(
    tool_name: str,
    method: str,
    path: str,
    entity_name: str,
    project_name: str,
    track_params: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
    """Call one Agent Lens read endpoint and return a JSON string.

    Follows the Agents tools: a no-retry request wrapped in
    ``track_tool_execution`` and the standard ``structured_error`` envelope, with
    the response trimmed to the token budget. Retrying a read belongs to the MCP
    caller; doing it here would amplify an already loaded service.

    Unlike the trace server, Agent Lens parses the credential as a bearer token
    (``internal/auth/middleware.go``). An explicit Authorization header there
    never falls back to a cookie identity and never grants admin privileges, so
    the key's own project authorization is the only access this can obtain.
    """
    try:
        base_url = resolve_agent_lens_base_url()
    except ValueError as error:
        return json.dumps(structured_error("agent_lens_not_configured", str(error)))

    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment.")
        return json.dumps(structured_error("auth_required", "A W&B API key is required to query the Agent Lens API."))

    url = f"{base_url}{AGENT_LENS_API_PREFIX}{path}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Wandb-Entity": entity_name,
        "X-Wandb-Project": project_name,
    }
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(_drop_none(body))

    with track_tool_execution(tool_name, None, track_params) as ctx:
        # Only the HTTP round-trip can raise here, so the try wraps just that;
        # the status-code branching below is plain control flow and stays outside.
        try:
            response = get_no_retry_session().request(
                method,
                url,
                headers=headers,
                params=params or None,
                data=data,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.error("Agent Lens request failed (%s)", type(e).__name__)
            ctx.mark_error(type(e).__name__)
            return json.dumps(structured_error("agent_lens_query_failed", "The Agent Lens request failed."))

        if response.status_code in {401, 403}:
            ctx.mark_error(f"http_{response.status_code}")
            return json.dumps(
                structured_error(
                    "agent_lens_forbidden",
                    "Agent Lens rejected the W&B API key for this project. Confirm the key can "
                    f"read {entity_name}/{project_name} and that Agent Lens is enabled for it.",
                    status_code=response.status_code,
                )
            )
        if response.status_code == 404:
            ctx.mark_error("agent_lens_unavailable")
            return json.dumps(
                structured_error(
                    "agent_lens_unavailable",
                    "The configured Agent Lens origin does not expose this endpoint (404); it may "
                    "predate this API or have the feature disabled.",
                    status_code=404,
                )
            )
        if response.status_code == 422:
            # Huma validation messages describe the caller's own arguments and
            # contain no project data, so passing one through is worth more than
            # a generic failure.
            ctx.mark_error("http_422")
            return json.dumps(
                structured_error(
                    "agent_lens_invalid_request",
                    f"Agent Lens rejected the request arguments: {_detail(response)}",
                    status_code=422,
                )
            )
        if response.status_code != 200:
            ctx.mark_error(f"http_{response.status_code}")
            if response.status_code in {429, 503}:
                overload = requests.HTTPError(
                    f"Agent Lens returned HTTP {response.status_code}",
                    response=response,
                )
                raise_for_wandb_server_busy(overload)
            return json.dumps(
                structured_error(
                    "agent_lens_query_failed",
                    f"The Agent Lens API returned HTTP {response.status_code}.",
                    status_code=response.status_code,
                )
            )

        # Parsing the body is the only other thing that can raise.
        try:
            result = response.json()
        except Exception as e:
            logger.error("Agent Lens response was not valid JSON")
            ctx.mark_error(type(e).__name__)
            return json.dumps(
                structured_error("agent_lens_query_failed", "The Agent Lens API returned an invalid response.")
            )

    return json.dumps(_truncate_response(result), default=str)


def _detail(response: Any) -> str:
    """Pull Huma's validation detail, falling back to a bounded generic message."""
    try:
        payload = response.json()
    except Exception:
        return "the request did not satisfy the API schema"
    if not isinstance(payload, dict):
        return "the request did not satisfy the API schema"
    detail = payload.get("detail") or payload.get("title")
    return str(detail)[:512] if detail else "the request did not satisfy the API schema"


def _invalid_argument(message: str) -> str:
    """Return the standard envelope for an argument this module rejected locally."""
    return json.dumps(structured_error("agent_lens_invalid_request", message))


def get_insights_coverage(entity_name: str, project_name: str) -> str:
    """Report the first and latest weeks that have classified Insights turns."""
    return _agent_lens_request(
        "get_insights_coverage",
        "GET",
        LATEST_WEEK_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_clustering_status(entity_name: str, project_name: str) -> str:
    """Report the latest successful clustering run per signature type."""
    return _agent_lens_request(
        "get_clustering_status",
        "GET",
        CLUSTERING_STATUS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_category_breakdowns(entity_name: str, project_name: str, start_at: str, end_at: str) -> str:
    """Return intent/failure category and cluster counts for a bounded range."""
    try:
        window = _validated_window(start_at, end_at)
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "get_category_breakdowns",
        "GET",
        CATEGORY_BREAKDOWNS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        params=window,
    )


def list_category_example_turns(
    entity_name: str,
    project_name: str,
    signature_type: str,
    category_id: str,
    start_at: str,
    end_at: str,
    cluster_ids: Optional[List[str]] = None,
    limit: int = 10,
    cursor: Optional[str] = None,
) -> str:
    """Page example turns for one intent or failure category."""
    if signature_type not in {"intent", "failure"}:
        return _invalid_argument('signature_type must be "intent" or "failure"')
    if not str(category_id).strip():
        return _invalid_argument("category_id must be a non-empty category identifier")
    if not 1 <= limit <= MAX_EXAMPLE_LIMIT:
        return _invalid_argument(f"limit must be between 1 and {MAX_EXAMPLE_LIMIT}")
    try:
        params: Dict[str, Any] = dict(_validated_window(start_at, end_at))
        if cluster_ids:
            params["cluster_ids[]"] = _bounded_list(cluster_ids, MAX_CLUSTER_IDS, "cluster_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    params["limit"] = limit
    if cursor:
        params["cursor"] = cursor
    return _agent_lens_request(
        "list_category_example_turns",
        "GET",
        _category_examples_path(signature_type, category_id),
        entity_name,
        project_name,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "signature_type": signature_type,
        },
        params=params,
    )


def list_matching_turns(
    entity_name: str,
    project_name: str,
    start_at: str,
    end_at: str,
    intent_category: Optional[str] = None,
    failure_category: Optional[str] = None,
    cluster_id: Optional[str] = None,
    cluster_kind: Optional[str] = None,
) -> str:
    """List turns matching a category or cluster, with failure attribution."""
    if cluster_kind is not None and cluster_kind not in {"intent", "failure"}:
        return _invalid_argument('cluster_kind must be "intent" or "failure"')
    if cluster_id and not cluster_kind:
        return _invalid_argument("cluster_kind is required when filtering by cluster_id")
    if not any((intent_category, failure_category, cluster_id)):
        return _invalid_argument("Provide at least one of intent_category, failure_category, or cluster_id")
    try:
        params: Dict[str, Any] = dict(_validated_window(start_at, end_at))
    except ValueError as error:
        return _invalid_argument(str(error))
    params.update(
        _drop_none(
            {
                "intent_category": intent_category,
                "failure_category": failure_category,
                "cluster_id": cluster_id,
                "cluster_kind": cluster_kind,
            }
        )
    )
    return _agent_lens_request(
        "list_matching_turns",
        "GET",
        MATCHING_TURNS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        params=params,
    )


def list_conversation_tag_names(entity_name: str, project_name: str) -> str:
    """List every conversation tag name in use in the project."""
    return _agent_lens_request(
        "list_conversation_tag_names",
        "GET",
        CONVERSATION_TAG_NAMES_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
    )


def get_conversation_tags(entity_name: str, project_name: str, conversation_ids: List[str]) -> str:
    """Return every tag applied to the given conversations, with provenance."""
    try:
        ids = _bounded_list(conversation_ids, MAX_CONVERSATION_IDS, "conversation_ids")
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "get_conversation_tags",
        "POST",
        CONVERSATION_TAGS_QUERY_PATH,
        entity_name,
        project_name,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "conversation_count": len(ids),
        },
        body={"conversation_ids": ids},
    )


def list_tagged_conversations(entity_name: str, project_name: str, tags: List[str]) -> str:
    """List conversation IDs carrying any of the given tags."""
    try:
        names = _bounded_list(tags, MAX_TAG_FILTERS, "tags")
    except ValueError as error:
        return _invalid_argument(str(error))
    return _agent_lens_request(
        "list_tagged_conversations",
        "POST",
        TAGGED_CONVERSATIONS_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name, "tag_count": len(names)},
        body={"tags": names},
    )


def get_tag_distribution(
    entity_name: str,
    project_name: str,
    after_ms: int,
    before_ms: int,
    time_bucket_seconds: int,
) -> str:
    """Return per-tag counts bucketed over time."""
    if after_ms < 0 or before_ms < 1:
        return _invalid_argument("after_ms must be >= 0 and before_ms must be >= 1")
    if before_ms <= after_ms:
        return _invalid_argument("before_ms must be greater than after_ms")
    if not 1 <= time_bucket_seconds <= MAX_TIME_BUCKET_SECONDS:
        return _invalid_argument(f"time_bucket_seconds must be between 1 and {MAX_TIME_BUCKET_SECONDS}")
    return _agent_lens_request(
        "get_tag_distribution",
        "POST",
        TAG_DISTRIBUTION_PATH,
        entity_name,
        project_name,
        {"entity_name": entity_name, "project_name": project_name},
        body={
            "after_ms": after_ms,
            "before_ms": before_ms,
            "time_bucket_seconds": time_bucket_seconds,
        },
    )


_WINDOW_PARAMS = """start_at : str
    Inclusive RFC 3339 start timestamp, e.g. "2026-09-01T00:00:00Z".
end_at : str
    Exclusive RFC 3339 end timestamp. The range must be nonempty and at most
    30 days; longer ranges are rejected before the request is sent."""


GET_INSIGHTS_COVERAGE_TOOL_DESCRIPTION = """Report which weeks have Agent Lens Insights data for a project.

<when_to_use>
Call this FIRST when answering any Insights question. Insights are produced by a
periodic classification job, so a project can have plenty of traces but no
classified turns, and the usable range rarely reaches today. Use the returned
first/latest weeks to choose a start_at/end_at that will actually contain data,
instead of guessing a range and getting an empty result.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": {"first_week", "latest_week"}}. Either may be null when the
project has no classified turns at all.
"""


GET_CLUSTERING_STATUS_TOOL_DESCRIPTION = """Report the latest successful Agent Lens clustering run per signature type.

<when_to_use>
Use to tell whether intent and failure clusters are fresh before relying on
cluster labels, and to explain a stale or missing breakdown. Each entry reports
the window that run covered, so a cluster absent from a recent range may simply
predate the latest run rather than having disappeared.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": [{"signature_type", "completed_at", "window_start",
"window_end"}]} -- at most one run per signature type.
"""


GET_CATEGORY_BREAKDOWNS_TOOL_DESCRIPTION = f"""Summarize Agent Lens intent and failure categories over a time range.

<when_to_use>
This is the main aggregate view: "what are users asking for, what is failing,
and how bad is it?". Returns per-category turn and conversation counts with
frustration counts, average latency and average cost per turn, plus the cluster
breakdowns beneath each category and failure-severity counts.

Start from get_agent_lens_insights_coverage_tool to pick a populated range. To
see the individual turns behind any number here, follow up with
list_agent_lens_matching_turns_tool or list_agent_lens_category_example_turns_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
{_WINDOW_PARAMS}

Returns
-------
JSON with {{"data": [CategoryBreakdown]}}, each holding "counts",
"cluster_breakdowns", "failure_breakdowns" and "failure_severity_counts".
"""


LIST_CATEGORY_EXAMPLE_TURNS_TOOL_DESCRIPTION = f"""Page example turns for one Agent Lens intent or failure category.

<when_to_use>
Use to ground a category or cluster in concrete traces -- "show me turns from
this failure category" -- after get_agent_lens_category_breakdowns_tool names it.
Returns identifiers only ("conversation_id", "trace_id"), so fetch the trace
content itself with get_weave_agent_trace_tool or query_weave_traces_tool.

Pass the returned "next_cursor" back as `cursor` to page. Treat the cursor as
opaque and keep every other argument identical across pages.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
signature_type : str
    "intent" or "failure" -- which category family `category_id` belongs to.
category_id : str
    The category identifier from get_agent_lens_category_breakdowns_tool.
{_WINDOW_PARAMS}
cluster_ids : list[str], optional
    Restrict to specific clusters within the category (at most {MAX_CLUSTER_IDS}).
limit : int, optional
    Turns per page, 1-{MAX_EXAMPLE_LIMIT} (default 10).
cursor : str, optional
    Opaque "next_cursor" from the previous page.

Returns
-------
JSON with {{"data": [{{"conversation_id", "trace_id"}}], "next_cursor"}}.
A null "next_cursor" means the last page.
"""


LIST_MATCHING_TURNS_TOOL_DESCRIPTION = f"""List Agent Lens turns matching a category or cluster, with failure attribution.

<when_to_use>
Use when you need the failure detail per turn, not just identifiers: each turn
carries "failure_signature", "failure_reason", "failure_severity" and
"failure_evidence_span_ids" (ordered most-important-first), which is what makes
this the right tool for "why did these turns fail?".

Provide at least one filter. `cluster_id` additionally requires `cluster_kind`
to say which family the cluster belongs to. Prefer
list_agent_lens_category_example_turns_tool when you only need paged identifiers.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
{_WINDOW_PARAMS}
intent_category : str, optional
    Restrict to one intent category.
failure_category : str, optional
    Restrict to one failure category.
cluster_id : str, optional
    Restrict to one cluster; requires `cluster_kind`.
cluster_kind : str, optional
    "intent" or "failure" -- which family `cluster_id` belongs to.

Returns
-------
JSON with {{"data": [MatchingTurn]}}, each holding "conversation_id",
"trace_id", "intent_signature", "started_at" and the failure attribution fields.
"""


LIST_CONVERSATION_TAG_NAMES_TOOL_DESCRIPTION = """List every Agent Lens conversation tag name used in a project.

<when_to_use>
Call this before filtering by tag. Tag names are project-defined free text, so
guessing one usually returns nothing; this is the only way to learn the exact
spellings that list_agent_lens_tagged_conversations_tool will match.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.

Returns
-------
JSON with {"data": [str]} -- the distinct tag names.
"""


GET_CONVERSATION_TAGS_TOOL_DESCRIPTION = f"""Fetch Agent Lens conversation tags, with provenance, for specific conversations.

<when_to_use>
Use when you already have conversation IDs -- from
list_agent_lens_tagged_conversations_tool, an Insights drilldown, or a Weave
trace query -- and need to know how each was labelled and by whom. Each tag
reports its "source" (human or judge), the applying "wb_user_id" or
"judge_version", and a "rationale", which is what distinguishes a reviewed
judgement from an automated one.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
conversation_ids : list[str]
    1-{MAX_CONVERSATION_IDS} conversation identifiers.

Returns
-------
JSON with {{"data": [ConversationTag]}} holding "conversation_id", "trace_id",
"tag", "source", "rationale", agent identity and timestamps. Untagged
conversations are simply absent.
"""


LIST_TAGGED_CONVERSATIONS_TOOL_DESCRIPTION = f"""List Agent Lens conversation IDs carrying any of the given tags.

<when_to_use>
Use to go from tag to conversations -- "which conversations were tagged
escalated?". Matching is OR across the supplied tags. Get exact tag spellings
from list_agent_lens_conversation_tag_names_tool first, then pass the returned
IDs to get_agent_lens_conversation_tags_tool for provenance or to the Weave
trace tools for content.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
tags : list[str]
    1-{MAX_TAG_FILTERS} tag names; a conversation matches if it carries any.

Returns
-------
JSON with {{"data": [str]}} -- the matching conversation IDs.
"""


GET_TAG_DISTRIBUTION_TOOL_DESCRIPTION = f"""Return Agent Lens conversation tag counts bucketed over time.

<when_to_use>
Use for trend questions -- "is this failure tag increasing?" -- rather than
fetching every tagged conversation and counting client-side. Bounds are epoch
milliseconds, unlike the RFC 3339 Insights tools.

Choose `time_bucket_seconds` to fit the range: 3600 for a day, 86400 for a
month. Very small buckets over a long range produce many buckets and may be
truncated to the response budget.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
after_ms : int
    Inclusive lower bound, epoch milliseconds.
before_ms : int
    Exclusive upper bound, epoch milliseconds; must exceed `after_ms`.
time_bucket_seconds : int
    Bucket width in seconds, 1-{MAX_TIME_BUCKET_SECONDS}.

Returns
-------
JSON with {{"data": {{"time_bucket_seconds", "after_ms", "before_ms",
"buckets": [{{"time_bucket_start_ms", "tag_counts"}}]}}}}.
"""
