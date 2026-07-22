"""MCP tools for the Weave **Agents** observability API (OTel/GenAI agent spans).

Agent spans are ingested via OTel into a data plane separate from classic Weave
calls, so these tools complement -- they do not duplicate -- the calls-based
``query_weave_traces_tool`` family. Each tool builds a request body and POSTs it
to one ``/agents/*`` trace-server endpoint via ``_agents_request`` -- the same
basic-auth + retry-session + ``track_tool_execution`` pattern as ``count_traces``
-- then trims the response to the token budget.

Request/response shapes mirror ``weave/trace_server/agents`` in github.com/wandb/weave.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import MAX_RESPONSE_TOKENS, WF_TRACE_SERVER_URL, structured_error
from wandb_mcp_server.mcp_tools.tools_utils import get_retry_session, track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.processors import TraceProcessor

logger = get_rich_logger(__name__)

# /agents/* endpoint paths on the Weave trace server. Confirmed against the
# wandb/weave generated client and the Node SDK request cassettes.
AGENTS_QUERY_PATH = "/agents/query"
AGENT_VERSIONS_QUERY_PATH = "/agents/agent-versions/query"
AGENT_SPANS_QUERY_PATH = "/agents/spans/query"
AGENT_SPANS_STATS_PATH = "/agents/spans/stats"
AGENT_CUSTOM_ATTRS_SCHEMA_PATH = "/agents/spans/custom-attrs/schema"
AGENTS_SEARCH_PATH = "/agents/search"
AGENT_TRACE_CHAT_PATH = "/agents/traces/chat"
AGENT_CONVERSATION_CHAT_PATH = "/agents/conversations/chat"

_REQUEST_TIMEOUT_SECONDS = 30

# Response list fields that may be trimmed to fit the token budget, ordered by
# how large each tends to be (most-verbose first).
_TRUNCATABLE_LIST_KEYS = (
    "spans",
    "results",
    "turns",
    "messages",
    "rows",
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


def _best_effort_viewer() -> Any:
    """Fetch the W&B viewer for analytics attribution; never fail the tool over it.

    The SDK viewer lookup can fail (auth quirks, network) independently of the
    data request, so a failure must not break the
    tool -- analytics identity is best-effort.
    """
    try:
        return WandBApiManager.get_api().viewer
    except Exception:
        return None


def _truncate_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the largest list field so the serialized response fits the budget.

    Agent span / chat payloads can be large. Rather than dropping fields (as the
    trace processor does for calls), we drop trailing items from the primary
    list and annotate ``_truncation`` so the caller knows to narrow the query.
    """
    if not isinstance(payload, dict):
        return payload
    if TraceProcessor.estimate_tokens(json.dumps(payload, default=str)) <= MAX_RESPONSE_TOKENS:
        return payload

    list_key = next(
        (k for k in _TRUNCATABLE_LIST_KEYS if isinstance(payload.get(k), list) and payload[k]),
        None,
    )
    if list_key is None:
        return payload  # nothing structural to trim

    items = payload[list_key]
    original = len(items)
    kept = list(items)
    # Shrink the list until the *whole* serialized payload fits the budget.
    # Items vary in size, so instead of computing a fixed cut we re-measure after
    # each pass and drop ~10% of whatever remains (at least one item) from the
    # end. This geometric shrink converges in a handful of iterations even for a
    # very large list, while overshooting as little as possible.
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


def _agents_request(tool_name: str, path: str, body: Dict[str, Any], track_params: Dict[str, Any]) -> str:
    """POST to an ``/agents/*`` endpoint and return a JSON string.

    Mirrors ``count_traces``: a basic-auth POST to the trace server via
    ``get_retry_session``, wrapped in ``track_tool_execution`` and the standard
    ``structured_error`` envelope, with the response trimmed to the token budget.
    """
    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment.")
        return json.dumps(structured_error("auth_required", "A W&B API key is required to query the Agents API."))

    url = f"{WF_TRACE_SERVER_URL}{path}"
    auth_token = base64.b64encode(f":{api_key}".encode()).decode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {auth_token}",
    }
    data = json.dumps(_drop_none(body))

    with track_tool_execution(tool_name, _best_effort_viewer(), track_params) as ctx:
        # Only the HTTP round-trip can raise here, so the try wraps just that;
        # the status-code branching below is plain control flow and stays outside.
        try:
            response = get_retry_session().post(url, headers=headers, data=data, timeout=_REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error(f"Agents API request to {path} failed: {e}", exc_info=True)
            ctx.mark_error(str(e))
            return json.dumps(structured_error("agents_query_failed", str(e)[:500]))

        if response.status_code == 404:
            ctx.mark_error("agents_api_unavailable")
            return json.dumps(
                structured_error(
                    "agents_api_unavailable",
                    f"The trace server at {WF_TRACE_SERVER_URL} has no {path} endpoint (404); "
                    "it may predate the Weave Agents API.",
                    status_code=404,
                )
            )
        if response.status_code != 200:
            ctx.mark_error(f"http_{response.status_code}")
            return json.dumps(
                structured_error(
                    "agents_query_failed",
                    f"Agents API {path} returned {response.status_code}: {response.text[:500]}",
                    status_code=response.status_code,
                )
            )

        # Parsing the body is the only other thing that can raise.
        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Agents API {path} returned a non-JSON body: {e}", exc_info=True)
            ctx.mark_error(str(e))
            return json.dumps(structured_error("agents_query_failed", str(e)[:500]))

    return json.dumps(_truncate_response(result), default=str)


def _normalize_stats_metrics(metrics: Optional[List[Any]], aggregation: str) -> List[Dict[str, Any]]:
    """Turn user-friendly metric inputs into AgentSpanStatsMetricSpec dicts.

    Each entry may be:
      - a string: a derived metric name (e.g. ``"duration_ms"``, ``"total_cost_usd"``,
        ``"is_error"``) or a raw numeric span field (e.g. ``"input_tokens"``);
      - a dict: a full metric spec, passed through (with light default-filling).
    """
    normalized: List[Dict[str, Any]] = []
    for entry in metrics or []:
        if isinstance(entry, str):
            if entry in _DERIVED_METRIC_VALUE_TYPES:
                value_type = _DERIVED_METRIC_VALUE_TYPES[entry]
                source = "derived"
            else:
                value_type = "number"
                source = "field"
            aggs = ["count_true"] if value_type == "boolean" else [aggregation]
            normalized.append(
                {
                    "alias": entry,
                    "value_type": value_type,
                    "aggregations": aggs,
                    "value": {"source": source, "key": entry},
                }
            )
        elif isinstance(entry, dict):
            spec = dict(entry)
            spec.setdefault("value_type", "number")
            if "value" not in spec:
                key = spec.pop("key", None) or spec.pop("field", None) or spec.get("alias")
                if key:
                    spec["value"] = {"source": spec.pop("source", "field"), "key": key}
            spec.setdefault("alias", (spec.get("value") or {}).get("key", "metric"))
            spec.setdefault("aggregations", ["count_true"] if spec["value_type"] == "boolean" else [aggregation])
            normalized.append(spec)
        else:
            raise ValueError(f"Invalid metric entry: {entry!r}. Use a string name or a spec dict.")
    return normalized


def _group_by_refs(group_by: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
    """Build AgentGroupByRef dicts from plain field names."""
    if not group_by:
        return None
    return [{"source": "field", "key": field} for field in group_by]


def _sort_by(sort_field: Optional[str], sort_direction: str) -> Optional[List[Dict[str, str]]]:
    """Build an AgentSortBy dict from a field name + direction."""
    return [{"field": sort_field, "direction": sort_direction}] if sort_field else None


# =============================================================================
# Tool descriptions
# =============================================================================

LIST_AGENTS_TOOL_DESCRIPTION = """List GenAI agents in a Weave project with aggregated stats.

<when_to_use>
Use as the entry point for the Weave Agents (OTel) data plane: "what agents are
running, and how are they doing?". Returns per-agent invocation counts, span
counts, token totals, duration, error counts, and first/last-seen timestamps.
These are OTel/GenAI agent spans, which the calls-based query_weave_traces_tool
does NOT see -- use this family for agent observability, and the trace tools for
classic Weave calls.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
agent_name : str, optional
    Restrict to a single agent by exact name.
sort_by : str, optional
    Field to sort by (e.g. "last_seen", "invocation_count", "total_input_tokens",
    "total_duration_ms", "error_count"). Defaults to server ordering.
sort_direction : str, optional
    "asc" or "desc" (default "desc").
limit : int, optional
    Max agents to return.
offset : int, optional
    Number of agents to skip (pagination).
include_costs : bool, optional
    Populate total_cost_usd per agent (default False; adds a cost lookup).

Returns
-------
str
    JSON: {"agents": [{agent_name, invocation_count, span_count,
    total_input_tokens, total_output_tokens, total_duration_ms, error_count,
    first_seen, last_seen, total_cost_usd?}], "total_count": int}.
"""

LIST_AGENT_VERSIONS_TOOL_DESCRIPTION = """List per-version aggregated stats for a single GenAI agent.

<when_to_use>
Use after list_weave_agents_tool to compare versions of one agent (e.g. did
v2 regress on latency or error rate?). Same stat shape as list_weave_agents_tool,
broken down by agent_version.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
agent_name : str
    The agent whose versions to list (required).
sort_by : str, optional
    Field to sort by (e.g. "last_seen", "invocation_count").
sort_direction : str, optional
    "asc" or "desc" (default "desc").
limit : int, optional
    Max versions to return.
offset : int, optional
    Number of versions to skip.
include_costs : bool, optional
    Populate total_cost_usd per version (default False).

Returns
-------
str
    JSON: {"versions": [{agent_version, ...same stats as agents...}], "total_count": int}.
"""

QUERY_AGENT_SPANS_TOOL_DESCRIPTION = """Query individual GenAI agent spans (raw rows) for a project.

<when_to_use>
Use to drill into the actual agent/LLM/tool spans behind the aggregates: filter
by agent, model, status, or time and inspect operation_name, provider/model,
tokens, costs, tool calls, and messages. This reads the OTel agent-spans table,
which is separate from classic Weave calls (query_weave_traces_tool). For the
readable conversation view of a turn, prefer get_weave_agent_trace_tool /
get_weave_agent_conversation_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
agent_name : str, optional
    Restrict to one agent (added as an equality filter on agent_name).
query : Dict[str, Any], optional
    Mongo-style Weave filter expression for advanced filtering on span fields
    (e.g. operation_name, provider_name, request_model, status_code). Combined
    (AND) with agent_name when both are given.
started_after : str, optional
    ISO-8601 timestamp; only spans with started_at >= this.
started_before : str, optional
    ISO-8601 timestamp; only spans with started_at < this.
sort_by : str, optional
    Span field to sort by (default "started_at").
sort_direction : str, optional
    "asc" or "desc" (default "desc").
limit : int, optional
    Max spans to return.
offset : int, optional
    Number of spans to skip.
include_details : bool, optional
    Include raw OTel dumps (large). Default False.
include_costs : bool, optional
    Compute per-span USD costs. Default False.

Returns
-------
str
    JSON: {"spans": [span objects], "total_count": int}. Large responses are
    truncated with a "_truncation" note.
"""

GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION = """Time-bucketed metric series over GenAI agent spans (chart-ready).

<when_to_use>
Use for trends and aggregates over time: token usage, cost, latency, error rate,
invocation volume -- optionally grouped (e.g. by agent_name or model). Returns
rows bucketed by time, suitable for plotting. For individual spans use
query_weave_agent_spans_tool instead.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
start : str
    ISO-8601 start of the time window (required).
end : str, optional
    ISO-8601 end of the window (defaults to now server-side).
metrics : List[Any], optional
    Metrics to compute. Each entry is either a name string or a full spec dict.
    Name strings: derived metrics "duration_ms", "total_tokens", "total_cost_usd",
    "input_cost_usd", "output_cost_usd", "is_error", "is_invocation"; or any raw
    numeric span field (e.g. "input_tokens", "output_tokens"). Defaults to
    ["input_tokens", "output_tokens"].
aggregation : str, optional
    Aggregation applied to numeric metric names (default "sum"; e.g. "avg", "max",
    "min", "count"). Boolean derived metrics (is_error/is_invocation) use count_true.
group_by : List[str], optional
    Span field names to group each bucket by (e.g. ["agent_name"]).
granularity_seconds : int, optional
    Bucket width in seconds (e.g. 3600 = hourly). Server picks a default if omitted.
timezone : str, optional
    Timezone for bucketing (default "UTC").

Returns
-------
str
    JSON: {"start", "end", "granularity", "timezone", "bucket_type",
    "columns": [...], "rows": [{timestamp, <metric columns>}]}.
"""

LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION = """Discover custom (user-defined) attribute keys on GenAI agent spans.

<when_to_use>
Use to learn which custom attributes exist before filtering/grouping by them in
query_weave_agent_spans_tool or get_weave_agent_span_stats_tool. Returns typed
keys (string/int/float/bool) and how many spans carry each.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
started_after : str, optional
    ISO-8601 timestamp lower bound on span start time.
started_before : str, optional
    ISO-8601 timestamp upper bound on span start time.
limit : int, optional
    Max attribute keys to return.
offset : int, optional
    Number of keys to skip.

Returns
-------
str
    JSON: {"attributes": [{source, key, value_type, span_count}], "has_more": bool}.
"""

SEARCH_AGENTS_TOOL_DESCRIPTION = """Search agent messages by content and/or filters; results grouped by conversation.

<when_to_use>
Use to find conversations: full-text search over message content ("find chats
mentioning Liverpool"), or structured retrieval by leaving query empty and
filtering (e.g. all messages in a trace_id or conversation_id, by role/model).
Returns conversations with their matched messages. To then read a full
conversation or turn, use get_weave_agent_conversation_tool /
get_weave_agent_trace_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
query : str, optional
    Substring to match in message content. Empty (default) = no content filter
    (structured retrieval over the filters below).
trace_id : str, optional
    Restrict to one trace.
conversation_id : str, optional
    Restrict to one conversation.
agent_name : str, optional
    Restrict to one agent.
roles : List[str], optional
    Message roles to include (e.g. ["user", "assistant", "tool"]).
provider_name : str, optional
    Restrict to one provider (e.g. "openai").
request_model : str, optional
    Restrict to one request model.
truncate_content : bool, optional
    Return previews instead of full content (default True).
started_after : str, optional
    ISO-8601 lower bound on message time.
started_before : str, optional
    ISO-8601 upper bound on message time.
limit : int, optional
    Max conversations to return.
offset : int, optional
    Number of conversations to skip.

Returns
-------
str
    JSON: {"results": [{conversation_id, conversation_name, agent_name,
    matched_messages: [{span_id, trace_id, role, content_preview, started_at}],
    last_activity}], "total_conversations": int}.
"""

GET_AGENT_TRACE_TOOL_DESCRIPTION = """Get the structured chat/trajectory view for one agent trace (a turn).

<when_to_use>
Use to read what an agent actually did in a single turn as an ordered timeline:
user messages, assistant replies (with model/tokens/cost), tool calls and
results, agent starts/handoffs, and context compaction. Identify the trace_id
from search_weave_agents_tool or query_weave_agent_spans_tool. For a whole
multi-turn conversation, use get_weave_agent_conversation_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
trace_id : str
    The trace whose chat view to render (required).
include_feedback : bool, optional
    Include feedback/annotations on the messages (default False).

Returns
-------
str
    JSON AgentTraceChatRes: {trace_id, agent_name, agent_version, status_code,
    total_duration_ms, total_cost_usd, messages: [typed timeline events], feedback?}.
"""

GET_AGENT_CONVERSATION_TOOL_DESCRIPTION = """Get the multi-turn chat view for an agent conversation.

<when_to_use>
Use to read a full conversation across turns as a sequence of chat views (each
turn is one trace). Identify the conversation_id from search_weave_agents_tool
or list_weave_agents_tool span data. For a single turn, use
get_weave_agent_trace_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
conversation_id : str
    The conversation to render (required).
limit : int, optional
    Max turns to return.
offset : int, optional
    Number of most-recent turns to skip (results are chronological within the page).
include_feedback : bool, optional
    Include feedback/annotations (default False).

Returns
-------
str
    JSON AgentConversationChatRes: {conversation_id, turns: [AgentTraceChatRes],
    total_turns, has_more, total_cost_usd, feedback?}.
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
    query: Optional[Dict[str, Any]] = None,
    started_after: Optional[str] = None,
    started_before: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_direction: str = "desc",
    limit: Optional[int] = None,
    offset: int = 0,
    include_details: bool = False,
    include_costs: bool = False,
) -> str:
    """Query individual GenAI agent spans."""
    # An agent_name shortcut becomes an equality filter, AND-combined with any
    # caller-supplied Mongo-style query (mirrors the wandb/weave Node client).
    effective_query = query
    if agent_name:
        name_clause = {"$expr": {"$eq": [{"$getField": "agent_name"}, {"$literal": agent_name}]}}
        if query:
            effective_query = {"$expr": {"$and": [query.get("$expr", query), name_clause["$expr"]]}}
        else:
            effective_query = name_clause

    body = {
        "project_id": f"{entity_name}/{project_name}",
        "query": effective_query,
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
    group_by: Optional[List[str]] = None,
    granularity_seconds: Optional[int] = None,
    timezone: str = "UTC",
) -> str:
    """Time-bucketed metric series over agent spans."""
    if not metrics:
        metrics = ["input_tokens", "output_tokens"]
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "start": start,
        "end": end,
        "metrics": _normalize_stats_metrics(metrics, aggregation),
        "group_by": _group_by_refs(group_by),
        "granularity": granularity_seconds,
        "timezone": timezone,
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
    limit: Optional[int] = None,
    offset: int = 0,
) -> str:
    """Discover custom attribute keys on agent spans."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
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
) -> str:
    """Get the structured chat/trajectory view for one trace."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "trace_id": trace_id,
        "include_feedback": include_feedback,
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
    include_feedback: bool = False,
) -> str:
    """Get the multi-turn chat view for a conversation."""
    body = {
        "project_id": f"{entity_name}/{project_name}",
        "conversation_id": conversation_id,
        "limit": limit,
        "offset": offset,
        "include_feedback": include_feedback,
    }
    return _agents_request(
        "get_agent_conversation",
        AGENT_CONVERSATION_CHAT_PATH,
        body,
        {"entity_name": entity_name, "project_name": project_name, "conversation_id": conversation_id},
    )
