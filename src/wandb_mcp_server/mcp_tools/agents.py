"""MCP tools for the Weave **Agents** observability and **Feedback** APIs (OTel/GenAI spans).

Agent spans are ingested via OTel into a data plane separate from classic Weave
calls, so these tools complement -- they do not duplicate -- the calls-based
``query_weave_traces_tool`` family. Each tool builds a request body and POSTs it
to one ``/agents/*`` or ``/feedback/*`` trace-server endpoint via
``_agents_request`` using basic auth, no automatic retries, and
``track_tool_execution``, then trims oversized responses to the token budget.

Request/response shapes mirror ``weave/trace_server/agents`` and feedback routes
in github.com/wandb/weave.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, List, Literal, Optional, Sequence

import requests

from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.api_client import WandBApiManager, raise_for_wandb_server_busy
from wandb_mcp_server.config import (
    MAX_RESPONSE_TOKENS,
    MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
    WF_TRACE_SERVER_URL,
    structured_error,
)
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.mcp_tools.tools_utils import get_no_retry_session, track_tool_execution
from wandb_mcp_server.tool_results import tool_error_json
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.processors import TraceProcessor

logger = get_rich_logger(__name__)

# /agents/* and /feedback/* endpoint paths on the Weave trace server.
AGENTS_QUERY_PATH = "/agents/query"
AGENT_VERSIONS_QUERY_PATH = "/agents/agent-versions/query"
AGENT_SPANS_QUERY_PATH = "/agents/spans/query"
AGENT_SPANS_STATS_PATH = "/agents/spans/stats"
AGENT_CUSTOM_ATTRS_SCHEMA_PATH = "/agents/spans/custom-attrs/schema"
AGENTS_SEARCH_PATH = "/agents/search"
AGENT_TRACE_CHAT_PATH = "/agents/traces/chat"
AGENT_CONVERSATION_CHAT_PATH = "/agents/conversations/chat"
AGENT_CONVERSATION_SPANS_PATH = "/agents/conversations/spans"
FEEDBACK_QUERY_PATH = "/feedback/query"
FEEDBACK_AGGREGATE_PATH = "/feedback/aggregate"
FEEDBACK_CREATE_PATH = "/feedback/create"
FEEDBACK_PURGE_PATH = "/feedback/purge"

_REQUEST_TIMEOUT_SECONDS = MCP_WANDB_REQUEST_TIMEOUT_SECONDS


def _effective_agents_timeout() -> float:
    base = float(_REQUEST_TIMEOUT_SECONDS)
    deadline = current_tool_deadline.get()
    if deadline is None:
        return base
    return max(0.1, min(base, deadline - time.monotonic()))


_MAX_CONVERSATION_SPANS_BATCH = 100
_MAX_FEEDBACK_QUERY_LIMIT = 200
_MAX_FEEDBACK_NOTE_CHARS = 10_000
_MAX_MESSAGE_FIELD_CHARS = 4_000
_COMPACTED_MESSAGE_FIELD_CHARS = 3_200

# Response list fields that may be trimmed to fit the token budget, ordered by
# how large each tends to be (most-verbose first).
_TRUNCATABLE_LIST_KEYS = (
    "spans",
    "conversations",
    "results",
    "result",
    "turns",
    "messages",
    "rows",
    "buckets",
    "groups",
    "agents",
    "versions",
    "attributes",
)

# Derived (computed) span metrics for the stats tool and their value types.
# Mirrors AGENT_SPAN_STATS_DERIVED_VALUE_TYPES in weave/trace_server/agents/types.py.
_DERIVED_METRIC_VALUE_TYPES = {
    "duration_ms": "number",
    "total_tokens": "number",
    "is_error": "boolean",
    "is_invocation": "boolean",
    "total_cost_usd": "number",
    "input_cost_usd": "number",
    "output_cost_usd": "number",
}


def _drop_none(body: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so the server applies its own field defaults."""
    return {k: v for k, v in body.items() if v is not None}


def _compact_message_blobs(messages: List[Any]) -> tuple[List[Any], int]:
    """Trim oversized tool_result / content strings before dropping whole turns."""
    compacted_count = 0
    out: List[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        updated: Optional[Dict[str, Any]] = None
        for field in ("tool_result", "arguments", "content"):
            val = msg.get(field)
            if isinstance(val, str) and len(val) > _MAX_MESSAGE_FIELD_CHARS:
                if updated is None:
                    updated = dict(msg)
                omitted = len(val) - _COMPACTED_MESSAGE_FIELD_CHARS
                updated[field] = f"{val[:_COMPACTED_MESSAGE_FIELD_CHARS]}\n...[truncated {omitted} chars]"
                compacted_count += 1
        out.append(updated if updated is not None else msg)
    return out, compacted_count


def _compact_chat_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact oversized per-message strings in trace/conversation chat views."""
    compacted_total = 0
    updated_payload = payload
    if isinstance(payload.get("messages"), list):
        msgs, count = _compact_message_blobs(payload["messages"])
        if count:
            updated_payload = {**updated_payload, "messages": msgs}
            compacted_total += count
    if isinstance(payload.get("turns"), list):
        new_turns: List[Any] = []
        turns_changed = False
        for turn in payload["turns"]:
            if isinstance(turn, dict) and isinstance(turn.get("messages"), list):
                msgs, count = _compact_message_blobs(turn["messages"])
                if count:
                    new_turns.append({**turn, "messages": msgs})
                    compacted_total += count
                    turns_changed = True
                    continue
            new_turns.append(turn)
        if turns_changed:
            updated_payload = {**updated_payload, "turns": new_turns}
    if compacted_total:
        updated_payload = {
            **updated_payload,
            "_field_compaction": {
                "applied": True,
                "compacted_fields": compacted_total,
                "max_chars": _COMPACTED_MESSAGE_FIELD_CHARS,
            },
        }
    return updated_payload


def _truncate_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trim oversized chat fields first, then trailing list items to fit the token budget."""
    if not isinstance(payload, dict):
        return payload
    if TraceProcessor.estimate_tokens(json.dumps(payload, default=str)) <= MAX_RESPONSE_TOKENS:
        return payload

    payload = _compact_chat_payload(payload)
    if TraceProcessor.estimate_tokens(json.dumps(payload, default=str)) <= MAX_RESPONSE_TOKENS:
        return payload

    list_key = next(
        (k for k in _TRUNCATABLE_LIST_KEYS if isinstance(payload.get(k), list) and payload[k]),
        None,
    )
    if list_key is None:
        return payload

    items = payload[list_key]
    original = len(items)
    kept = list(items)
    while (
        kept
        and TraceProcessor.estimate_tokens(json.dumps({**payload, list_key: kept}, default=str)) > MAX_RESPONSE_TOKENS
    ):
        kept = kept[: -max(1, len(kept) // 10)]

    result = {**payload, list_key: kept}
    result["_truncation"] = {
        "applied": True,
        "field": list_key,
        "returned": len(kept),
        "original": original,
        "note": (
            f"Response truncated to fit the {MAX_RESPONSE_TOKENS}-token budget; dropped "
            f"{original - len(kept)} '{list_key}' item(s). Narrow with filters, a smaller "
            f"limit, or a tighter time range."
        ),
    }
    return result


def _invalid_agents_argument(message: str) -> str:
    return tool_error_json(structured_error("agents_invalid_request", message))


def _agents_request(tool_name: str, path: str, body: Dict[str, Any], track_params: Dict[str, Any]) -> str:
    """POST to an ``/agents/*`` or ``/feedback/*`` endpoint and return a JSON string."""
    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment.")
        return tool_error_json(structured_error("auth_required", "A W&B API key is required to query the Agents API."))

    url = f"{WF_TRACE_SERVER_URL}{path}"
    auth_token = base64.b64encode(f":{api_key}".encode()).decode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {auth_token}",
    }
    data = json.dumps(_drop_none(body))

    with track_tool_execution(tool_name, None, track_params) as ctx:
        try:
            response = get_no_retry_session().post(
                url,
                headers=headers,
                data=data,
                timeout=_effective_agents_timeout(),
            )
        except Exception as e:
            logger.error("Agents API request failed (%s)", type(e).__name__)
            ctx.mark_error(type(e).__name__)
            return tool_error_json(
                structured_error(
                    "agents_query_failed",
                    "The Agents API request failed.",
                )
            )

        if response.status_code == 404:
            ctx.mark_error("agents_api_unavailable")
            return tool_error_json(
                structured_error(
                    "agents_api_unavailable",
                    "The configured trace server does not expose this Agents API endpoint (404); "
                    "it may predate the Weave Agents API.",
                    status_code=404,
                )
            )
        if response.status_code in {400, 422}:
            ctx.mark_error(f"http_{response.status_code}")
            detail = "the request did not satisfy the trace server schema"
            try:
                err_json = response.json()
                raw_detail = err_json.get("detail") if isinstance(err_json, dict) else None
                if isinstance(raw_detail, list) and raw_detail:
                    first = raw_detail[0]
                    if isinstance(first, dict) and first.get("msg"):
                        detail = sanitize_sensitive_text(str(first["msg"]), max_chars=512)
                elif isinstance(raw_detail, str) and raw_detail.strip():
                    detail = sanitize_sensitive_text(raw_detail.strip(), max_chars=512)
            except Exception:
                pass
            return tool_error_json(
                structured_error(
                    "agents_invalid_request",
                    f"Agents API rejected the request arguments: {detail}",
                    status_code=response.status_code,
                )
            )
        if response.status_code != 200:
            ctx.mark_error(f"http_{response.status_code}")
            if response.status_code in {429, 503}:
                overload = requests.HTTPError(
                    f"Agents API returned HTTP {response.status_code}",
                    response=response,
                )
                raise_for_wandb_server_busy(overload)
            return tool_error_json(
                structured_error(
                    "agents_query_failed",
                    f"The Agents API returned HTTP {response.status_code}.",
                    status_code=response.status_code,
                )
            )

        try:
            result = response.json()
        except Exception as e:
            logger.error("Agents API response was not valid JSON")
            ctx.mark_error(type(e).__name__)
            return tool_error_json(
                structured_error(
                    "agents_query_failed",
                    "The Agents API returned an invalid response.",
                )
            )

    return json.dumps(_truncate_response(result), default=str)


def _normalize_stats_metrics(
    metrics: Optional[List[Any]],
    aggregation: str,
    percentiles: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Turn user-friendly metric inputs into AgentSpanStatsMetricSpec dicts."""
    normalized: List[Dict[str, Any]] = []
    pct_list = [float(p) for p in percentiles] if percentiles else None
    for entry in metrics or []:
        if isinstance(entry, str):
            if entry in _DERIVED_METRIC_VALUE_TYPES:
                value_type = _DERIVED_METRIC_VALUE_TYPES[entry]
                source = "derived"
            else:
                value_type = "number"
                source = "field"
            aggs = ["count_true"] if value_type == "boolean" else [aggregation]
            spec: Dict[str, Any] = {
                "alias": entry,
                "value_type": value_type,
                "aggregations": aggs,
                "value": {"source": source, "key": entry},
            }
            if pct_list and value_type == "number":
                spec["percentiles"] = pct_list
            normalized.append(spec)
        elif isinstance(entry, dict):
            spec = dict(entry)
            spec.setdefault("value_type", "number")
            if "value" not in spec:
                key = spec.pop("key", None) or spec.pop("field", None) or spec.get("alias")
                if key:
                    spec["value"] = {"source": spec.pop("source", "field"), "key": key}
            spec.setdefault("alias", (spec.get("value") or {}).get("key", "metric"))
            spec.setdefault("aggregations", ["count_true"] if spec["value_type"] == "boolean" else [aggregation])
            if pct_list and spec["value_type"] == "number" and "percentiles" not in spec:
                spec["percentiles"] = pct_list
            normalized.append(spec)
        else:
            raise ValueError(f"Invalid metric entry: {entry!r}. Use a string name or a spec dict.")
    return normalized


def _group_by_refs(group_by: Optional[Sequence[Any]]) -> Optional[List[Dict[str, Any]]]:
    """Build AgentGroupByRef dicts from field names or custom-attribute ref dicts."""
    if not group_by:
        return None
    refs: List[Dict[str, Any]] = []
    for item in group_by:
        if isinstance(item, str):
            refs.append({"source": "field", "key": item})
        elif isinstance(item, dict) and item.get("key"):
            ref: Dict[str, Any] = {"source": item.get("source", "field"), "key": str(item["key"])}
            if item.get("alias"):
                ref["alias"] = str(item["alias"])
            refs.append(ref)
        else:
            raise ValueError(f"Invalid group_by entry: {item!r}. Use a field name string or {{'source', 'key'}} dict.")
    return refs


def _value_refs(columns: Optional[Sequence[Any]]) -> Optional[List[Dict[str, str]]]:
    """Normalize custom_attr_columns entries into AgentSpanValueRef dicts."""
    if not columns:
        return None
    refs: List[Dict[str, str]] = []
    for item in columns:
        if isinstance(item, str):
            refs.append({"source": "custom_attrs_string", "key": item})
        elif isinstance(item, dict) and item.get("key"):
            refs.append({"source": str(item.get("source", "custom_attrs_string")), "key": str(item["key"])})
        else:
            raise ValueError(f"Invalid custom_attr_columns entry: {item!r}.")
    return refs


def _combine_equality_query(
    base_query: Optional[Dict[str, Any]],
    equality_pairs: Sequence[tuple[str, Optional[str]]],
) -> Optional[Dict[str, Any]]:
    """Combine equality shortcuts with an optional Mongo-style Weave $expr query."""
    clauses: List[Dict[str, Any]] = []
    if base_query:
        clauses.append(base_query.get("$expr", base_query))
    for field_name, value in equality_pairs:
        if value is not None and str(value).strip():
            clauses.append({"$eq": [{"$getField": field_name}, {"$literal": str(value).strip()}]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return {"$expr": clauses[0]}
    return {"$expr": {"$and": clauses}}


def _sort_by(sort_field: Optional[str], sort_direction: str) -> Optional[List[Dict[str, str]]]:
    """Build an AgentSortBy / SortBy list from a field name + direction."""
    return [{"field": sort_field, "direction": sort_direction}] if sort_field else None


# =============================================================================
# Tool descriptions
# =============================================================================

LIST_AGENTS_TOOL_DESCRIPTION = """List GenAI agents in a Weave project with aggregated stats.

<when_to_use>
Entry point for the Weave Agents (OTel) data plane: returns per-agent invocation
counts, span counts, token totals, duration, error counts, and optional USD
costs (`include_costs=True`). Distinct from classic Weave calls (`query_weave_traces_tool`).
</when_to_use>

Returns `{"agents": [...], "total_count": int}`.
"""

LIST_AGENT_VERSIONS_TOOL_DESCRIPTION = """List per-version aggregated stats for a single GenAI agent (`agent_name`).

<when_to_use>
Use after `list_weave_agents_tool` to compare versions of one agent across
invocations, latency, tokens, cost, and error rate.
</when_to_use>

Returns `{"versions": [...], "total_count": int}`.
"""

QUERY_AGENT_SPANS_TOOL_DESCRIPTION = """Query or group GenAI agent spans with Insight, tag, signal, and custom-attribute filters.

<when_to_use>
Inspect raw OTel agent/LLM/tool spans or group spans (`group_by` field names or
`{"source": "custom_attrs_string", "key": ...}`, returning `span_count` plus
optional `measures` `{"alias", "aggregation", "value"?}`). Supports exact filters (`agent_name`,
`conversation_id`, `trace_id`, `operation_name`, `tool_name`, `status_code`),
Weave `$expr` `query`, Agent Lens `insight_filters` and `tag_filter`
(`{"tag_ids": [...], "exclude"?: bool}`), and feedback `signal_filters`.
</when_to_use>

Returns `{"spans": [...], "total_count": int}` or `{"groups": [...], "total_count": int}`.
"""

GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION = """Compute time- or value-bucketed metric and percentile series over GenAI agent spans.

<when_to_use>
Use for latency percentiles (`percentiles=[50, 95, 99]`), token/cost trends, or
error rates over `start`..`end` (`metrics`: `duration_ms`, `total_tokens`,
`total_cost_usd`, `is_error`, `is_invocation`, or numeric fields), optionally
filtered by `query`/`insight_filters`/`signal_filters` and grouped via `group_by`.
</when_to_use>

Returns `{"start", "end", "granularity", "columns", "rows"}`.
"""

LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION = """Discover user-defined custom attribute keys and value types on GenAI agent spans.

<when_to_use>
Call before grouping or filtering by custom attributes (`custom_attrs_string`,
`custom_attrs_int`, `custom_attrs_float`, `custom_attrs_bool`) in
`query_weave_agent_spans_tool` or `get_weave_agent_span_stats_tool`.
</when_to_use>

Returns `{"attributes": [{"source", "key", "value_type", "span_count"}], "has_more": bool}`.
"""

SEARCH_AGENTS_TOOL_DESCRIPTION = """Search agent messages by substring (`query`) and/or metadata filters, grouped by conversation.

<when_to_use>
Find conversations mentioning keywords or matching `roles`, `agent_name`,
`provider_name`, `request_model`, or time bounds. Follow up with
`get_weave_agent_conversation_tool` or `get_weave_agent_trace_tool`.
</when_to_use>

Returns `{"results": [{"conversation_id", "agent_name", "matched_messages", "last_activity"}], "total_conversations": int}`.
"""

GET_AGENT_TRACE_TOOL_DESCRIPTION = """Get the structured timeline/chat view for a single agent turn (`trace_id`).

<when_to_use>
Read user messages, assistant replies, tool calls/results, handoffs, and
feedback (`include_feedback=True`) for one turn. Set `include_model_tool_calls=True`
to keep raw model tool-call items alongside paired tool spans.
</when_to_use>

Returns `{trace_id, agent_name, status_code, total_duration_ms, messages, feedback?}`.
"""

GET_AGENT_CONVERSATION_TOOL_DESCRIPTION = """Get the chronological multi-turn chat view for an agent conversation (`conversation_id`).

<when_to_use>
Read a full conversation from Turn 1 onward (`sort_direction="asc"` default, or
`"desc"` for newest turns first; max 50 turns per call). Oversized tool outputs
are compacted before turns are dropped.
</when_to_use>

Returns `{conversation_id, turns: [...], total_turns, has_more, total_cost_usd, feedback?}`.
"""

GET_AGENT_CONVERSATION_SPANS_TOOL_DESCRIPTION = f"""Batch-load full span trees and span feedback for 1-{_MAX_CONVERSATION_SPANS_BATCH} `conversation_ids`.

<when_to_use>
Use after sampling conversations from an Agent Lens cluster or tag to load full
OTel span trees and feedback markers across many conversations in a single call.
</when_to_use>

Returns `{{"conversations": [{{"conversation_id", "spans", "feedback"}}]}}`.
"""

QUERY_AGENT_FEEDBACK_TOOL_DESCRIPTION = f"""Search or aggregate project-wide human notes (`wandb.note.1`), reactions (`wandb.agent_user_feedback`), and scorer feedback.

<when_to_use>
Filter project feedback by `conversation_id`, `trace_id`, `feedback_type`,
`wb_user_id`, `weave_ref`, or Weave `$expr` `query` (e.g. `$contains` on
`payload.note`, limit 1-{_MAX_FEEDBACK_QUERY_LIMIT}), or set `aggregate=True`
with `after_ms`/`before_ms` for bucketed feedback counts.
</when_to_use>

Returns `{{"result": [...], "total_count": int}}` or aggregated feedback buckets.
"""

MANAGE_AGENT_FEEDBACK_TOOL_DESCRIPTION = """Add or delete human notes/corrections (`wandb.note.1`) and reactions (`wandb.agent_user_feedback`) on agent turns or conversations.

<when_to_use>
Use in read-write mode (`action`: `add_note` with `note`, `add_reaction` with
`emoji`, or `delete_feedback` with `feedback_id`). Auto-derives canonical
`weave:///{entity}/{project}/agent_turn/{trace_id_hex}` (or `agent_conversation`
/ `agent_span`) refs from `trace_id`/`conversation_id`/`span_id` when `weave_ref` is omitted.
</when_to_use>

Returns created feedback record (`id`, `created_at`, `wb_user_id`) or purge result.
"""


# =============================================================================
# Tool functions
# =============================================================================


def list_agents(
    entity_name: str,
    project_name: str,
    agent_name: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_direction: str = "desc",
    limit: Optional[int] = None,
    offset: int = 0,
    include_costs: bool = False,
) -> str:
    """List GenAI agents with aggregated stats."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "filters": {"agent_name": agent_name} if agent_name else None,
        "sort_by": _sort_by(sort_by, sort_direction),
        "limit": limit,
        "offset": offset,
        "include_costs": include_costs,
    }
    return _agents_request(
        "list_agents",
        AGENTS_QUERY_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "agent_name": agent_name},
    )


def list_agent_versions(
    entity_name: str,
    project_name: str,
    agent_name: str,
    sort_by: Optional[str] = None,
    sort_direction: str = "desc",
    limit: Optional[int] = None,
    offset: int = 0,
    include_costs: bool = False,
) -> str:
    """List per-version stats for one agent."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "agent_name": agent_name,
        "sort_by": _sort_by(sort_by, sort_direction),
        "limit": limit,
        "offset": offset,
        "include_costs": include_costs,
    }
    return _agents_request(
        "list_agent_versions",
        AGENT_VERSIONS_QUERY_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "agent_name": agent_name},
    )


def query_agent_spans(
    entity_name: str,
    project_name: str,
    agent_name: Optional[str] = None,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    operation_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    status_code: Optional[str] = None,
    query: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[Any]] = None,
    measures: Optional[List[Dict[str, Any]]] = None,
    group_filters: Optional[List[Dict[str, Any]]] = None,
    insight_filters: Optional[List[Dict[str, Any]]] = None,
    tag_filter: Optional[Dict[str, Any]] = None,
    signal_filters: Optional[Dict[str, Any]] = None,
    custom_attr_columns: Optional[List[Any]] = None,
    started_after: Optional[str] = None,
    started_before: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_direction: str = "desc",
    limit: Optional[int] = None,
    offset: int = 0,
    include_details: bool = False,
    include_costs: bool = False,
) -> str:
    """Query or group GenAI agent spans."""
    try:
        grouped_refs = _group_by_refs(group_by)
        attr_cols = _value_refs(custom_attr_columns)
    except ValueError as error:
        return _invalid_agents_argument(str(error))

    effective_query = _combine_equality_query(
        query,
        (
            ("agent_name", agent_name),
            ("conversation_id", conversation_id),
            ("trace_id", trace_id),
            ("operation_name", operation_name),
            ("tool_name", tool_name),
            ("status_code", status_code),
        ),
    )

    body = {
        "project_id": f"{entity_name}/{project_name}",
        "query": effective_query,
        "group_by": grouped_refs,
        "measures": measures,
        "group_filters": group_filters,
        "insight_filters": insight_filters,
        "tag_filter": tag_filter,
        "signal_filters": signal_filters,
        "custom_attr_columns": attr_cols,
        "started_after": started_after,
        "started_before": started_before,
        "sort_by": _sort_by(sort_by, sort_direction),
        "limit": limit,
        "offset": offset,
        "include_details": include_details,
        "include_costs": include_costs,
    }
    return _agents_request(
        "query_agent_spans",
        AGENT_SPANS_QUERY_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "agent_name": agent_name},
    )


def get_agent_span_stats(
    entity_name: str,
    project_name: str,
    start: str,
    end: Optional[str] = None,
    metrics: Optional[List[Any]] = None,
    aggregation: str = "sum",
    percentiles: Optional[List[float]] = None,
    group_by: Optional[List[Any]] = None,
    granularity_seconds: Optional[int] = None,
    timezone: str = "UTC",
    query: Optional[Dict[str, Any]] = None,
    bucket_by: Optional[Dict[str, Any]] = None,
    group_limit: Optional[int] = None,
    group_filters: Optional[List[Dict[str, Any]]] = None,
    insight_filters: Optional[List[Dict[str, Any]]] = None,
    signal_filters: Optional[Dict[str, Any]] = None,
) -> str:
    """Time- or value-bucketed metric series over agent spans."""
    if not metrics:
        metrics = ["input_tokens", "output_tokens"]
    try:
        normalized_metrics = _normalize_stats_metrics(metrics, aggregation, percentiles=percentiles)
        grouped_refs = _group_by_refs(group_by)
    except ValueError as error:
        return _invalid_agents_argument(str(error))

    body = {
        "project_id": f"{entity_name}/{project_name}",
        "query": query,
        "start": start,
        "end": end,
        "metrics": normalized_metrics,
        "group_by": grouped_refs,
        "granularity": granularity_seconds,
        "timezone": timezone,
        "bucket_by": bucket_by,
        "group_limit": group_limit,
        "group_filters": group_filters,
        "insight_filters": insight_filters,
        "signal_filters": signal_filters,
    }
    return _agents_request(
        "get_agent_span_stats",
        AGENT_SPANS_STATS_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "start": start, "end": end},
    )


def list_agent_custom_attributes(
    entity_name: str,
    project_name: str,
    started_after: Optional[str] = None,
    started_before: Optional[str] = None,
    query: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> str:
    """Discover custom attribute keys on agent spans."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "query": query,
        "started_after": started_after,
        "started_before": started_before,
        "limit": limit,
        "offset": offset,
    }
    return _agents_request(
        "list_agent_custom_attributes",
        AGENT_CUSTOM_ATTRS_SCHEMA_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name},
    )


def search_agents(
    entity_name: str,
    project_name: str,
    query: str = "",
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    roles: Optional[List[str]] = None,
    provider_name: Optional[str] = None,
    request_model: Optional[str] = None,
    truncate_content: bool = True,
    started_after: Optional[str] = None,
    started_before: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> str:
    """Search agent messages; results grouped by conversation."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "query": query,
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "agent_name": agent_name,
        "roles": roles,
        "provider_name": provider_name,
        "request_model": request_model,
        "truncate_content": truncate_content,
        "started_after": started_after,
        "started_before": started_before,
        "limit": limit,
        "offset": offset,
    }
    return _agents_request(
        "search_agents",
        AGENTS_SEARCH_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "query": query},
    )


def get_agent_trace(
    entity_name: str,
    project_name: str,
    trace_id: str,
    include_feedback: bool = False,
    include_model_tool_calls: bool = False,
) -> str:
    """Get the structured chat/trajectory view for one trace."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "trace_id": trace_id,
        "include_feedback": include_feedback,
        "include_model_tool_calls": include_model_tool_calls if include_model_tool_calls else None,
    }
    return _agents_request(
        "get_agent_trace",
        AGENT_TRACE_CHAT_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "trace_id": trace_id},
    )


def get_agent_conversation(
    entity_name: str,
    project_name: str,
    conversation_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_direction: Literal["asc", "desc"] = "asc",
    include_feedback: bool = False,
    include_model_tool_calls: bool = False,
) -> str:
    """Get the multi-turn chat view for a conversation."""
    if sort_direction not in {"asc", "desc"}:
        return _invalid_agents_argument('sort_direction must be "asc" or "desc"')
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "conversation_id": conversation_id,
        "sort_by": [{"field": "started_at", "direction": sort_direction}],
        "limit": limit,
        "offset": offset,
        "include_feedback": include_feedback,
        "include_model_tool_calls": include_model_tool_calls if include_model_tool_calls else None,
    }
    return _agents_request(
        "get_agent_conversation",
        AGENT_CONVERSATION_CHAT_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "conversation_id": conversation_id},
    )


def get_agent_conversation_spans(
    entity_name: str,
    project_name: str,
    conversation_ids: List[str],
    started_after: Optional[str] = None,
    started_before: Optional[str] = None,
) -> str:
    """Batch-load full span trees and feedback markers for up to 100 conversations."""
    if not isinstance(conversation_ids, (list, tuple)) or not conversation_ids:
        return _invalid_agents_argument("conversation_ids must contain at least one conversation ID")
    cleaned = [str(cid).strip() for cid in conversation_ids]
    if any(not cid for cid in cleaned):
        return _invalid_agents_argument("conversation_ids must not contain blank values")
    if len(cleaned) > _MAX_CONVERSATION_SPANS_BATCH:
        return _invalid_agents_argument(
            f"conversation_ids must contain at most {_MAX_CONVERSATION_SPANS_BATCH} IDs per call"
        )
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "conversation_ids": cleaned,
        "started_after": started_after,
        "started_before": started_before,
    }
    return _agents_request(
        "get_agent_conversation_spans",
        AGENT_CONVERSATION_SPANS_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "conversation_count": len(cleaned)},
    )


def query_agent_feedback(
    entity_name: str,
    project_name: str,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    feedback_type: Optional[str] = None,
    wb_user_id: Optional[str] = None,
    weave_ref: Optional[str] = None,
    query: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort_by: Optional[str] = "created_at",
    sort_direction: str = "desc",
    limit: int = 50,
    offset: int = 0,
    aggregate: bool = False,
    after_ms: Optional[int] = None,
    before_ms: Optional[int] = None,
    time_bucket_seconds: Optional[int] = None,
    feedback_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    span_agent_names: Optional[List[str]] = None,
    span_types: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
) -> str:
    """Query or aggregate human notes, reactions, and scorer feedback across a project."""
    project_id = f"{entity_name}/{project_name}"
    if aggregate:
        if after_ms is None or before_ms is None:
            return _invalid_agents_argument("after_ms and before_ms are required when aggregate=True")
        if after_ms < 0 or before_ms <= after_ms:
            return _invalid_agents_argument("after_ms must be >= 0 and before_ms must be > after_ms")
        eff_types = feedback_types or ([feedback_type] if feedback_type else None)
        agg_body = {
            "project_id": project_id,
            "after_ms": after_ms,
            "before_ms": before_ms,
            "time_bucket_seconds": time_bucket_seconds,
            "feedback_types": eff_types,
            "tags": tags,
            "span_agent_names": span_agent_names,
            "span_types": span_types,
            "group_by": group_by,
        }
        return _agents_request(
            "query_agent_feedback",
            FEEDBACK_AGGREGATE_PATH,
            agg_body,
            {"entity_name": entity_name, "project_name": project_name, "aggregate": True},
        )

    if isinstance(limit, bool) or not 1 <= limit <= _MAX_FEEDBACK_QUERY_LIMIT:
        return _invalid_agents_argument(f"limit must be between 1 and {_MAX_FEEDBACK_QUERY_LIMIT}")
    if isinstance(offset, bool) or offset < 0:
        return _invalid_agents_argument("offset must be >= 0")
    if sort_direction not in {"asc", "desc"}:
        return _invalid_agents_argument('sort_direction must be "asc" or "desc"')

    effective_query = _combine_equality_query(
        query,
        (
            ("span_conversation_id", conversation_id),
            ("span_trace_id", trace_id),
            ("feedback_type", feedback_type),
            ("wb_user_id", wb_user_id),
            ("weave_ref", weave_ref),
        ),
    )
    query_body = {
        "project_id": project_id,
        "fields": fields,
        "query": effective_query,
        "sort_by": _sort_by(sort_by, sort_direction),
        "limit": limit,
        "offset": offset,
    }
    return _agents_request(
        "query_agent_feedback",
        FEEDBACK_QUERY_PATH,
        query_body,
        {"entity_name": entity_name, "project_name": project_name, "feedback_type": feedback_type},
    )


def _derive_agent_weave_ref(
    entity_name: str,
    project_name: str,
    *,
    weave_ref: Optional[str],
    conversation_id: Optional[str],
    trace_id: Optional[str],
    span_id: Optional[str],
) -> str:
    if weave_ref and weave_ref.strip():
        ref = weave_ref.strip()
        if not ref.startswith("weave:///"):
            raise ValueError("weave_ref must start with 'weave:///'")
        return ref
    prefix = f"weave:///{entity_name}/{project_name}"
    trace_clean = (trace_id or "").strip().replace("-", "")
    span_clean = (span_id or "").strip()
    conv_clean = (conversation_id or "").strip()
    if trace_clean and span_clean:
        return f"{prefix}/agent_span/{trace_clean}:{span_clean}"
    if trace_clean:
        return f"{prefix}/agent_turn/{trace_clean}"
    if conv_clean:
        return f"{prefix}/agent_conversation/{conv_clean}"
    raise ValueError("Provide trace_id, conversation_id, or weave_ref to target feedback")


def manage_agent_feedback(
    entity_name: str,
    project_name: str,
    action: Literal["add_note", "add_reaction", "delete_feedback"],
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    weave_ref: Optional[str] = None,
    note: Optional[str] = None,
    emoji: Optional[str] = None,
    feedback_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
) -> str:
    """Create or delete human notes (`wandb.note.1`) and reactions (`wandb.agent_user_feedback`)."""
    if action not in {"add_note", "add_reaction", "delete_feedback"}:
        return _invalid_agents_argument('action must be "add_note", "add_reaction", or "delete_feedback"')

    project_id = f"{entity_name}/{project_name}"
    track = {"entity_name": entity_name, "project_name": project_name}

    if action == "delete_feedback":
        fid = (feedback_id or "").strip()
        if not fid:
            return _invalid_agents_argument("feedback_id is required for action='delete_feedback'")
        purge_body = {
            "project_id": project_id,
            "query": {"$expr": {"$eq": [{"$getField": "id"}, {"$literal": fid}]}},
        }
        return _agents_request("manage_agent_feedback", FEEDBACK_PURGE_PATH, purge_body, track)

    try:
        target_ref = _derive_agent_weave_ref(
            entity_name,
            project_name,
            weave_ref=weave_ref,
            conversation_id=conversation_id,
            trace_id=trace_id,
            span_id=span_id,
        )
    except ValueError as error:
        return _invalid_agents_argument(str(error))

    create_body: Dict[str, Any] = {
        "project_id": project_id,
        "weave_ref": target_ref,
        "span_conversation_id": (conversation_id or "").strip() or None,
        "span_trace_id": (trace_id or "").strip().replace("-", "") or None,
        "span_agent_name": (agent_name or "").strip() or None,
        "span_agent_version": (agent_version or "").strip() or None,
    }

    if action == "add_note":
        trimmed_note = (note or "").strip()
        if not trimmed_note or len(trimmed_note) > _MAX_FEEDBACK_NOTE_CHARS:
            return _invalid_agents_argument(f"note must be 1 to {_MAX_FEEDBACK_NOTE_CHARS} non-whitespace characters")
        create_body["feedback_type"] = "wandb.note.1"
        create_body["payload"] = {"note": trimmed_note}
    else:
        trimmed_emoji = (emoji or "").strip()
        if not trimmed_emoji or len(trimmed_emoji) > 32:
            return _invalid_agents_argument("emoji must be a non-empty string up to 32 characters")
        create_body["feedback_type"] = "wandb.agent_user_feedback"
        create_body["payload"] = {"emoji": trimmed_emoji, "scorer_tags": [trimmed_emoji]}
        create_body["scorer_tags"] = [trimmed_emoji]
        create_body["scorer_tag_reasons"] = {trimmed_emoji: "Set by user"}
        create_body["scorer_tag_confidences"] = {trimmed_emoji: 1}

    return _agents_request("manage_agent_feedback", FEEDBACK_CREATE_PATH, create_body, track)
