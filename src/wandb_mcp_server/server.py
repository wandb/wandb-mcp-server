#!/usr/bin/env python
"""
Weights & Biases MCP Server - A Model Context Protocol server for querying Weights & Biases data.

This server provides tools for:
- Querying Weave traces and evaluations
- Counting traces efficiently
- Querying W&B experiment data through the public SDK
- Creating shareable reports with visualizations
- Searching official W&B documentation
- Discovering available entities and projects
- Starting and polling asynchronous conversations with ARIA
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import ipaddress
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Literal, Optional, Union

import wandb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import PositiveInt

from wandb_mcp_server.config import (
    MCP_COUNT_TOOL_WORKERS,
    MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
    WANDB_API_BASE_URL,
    _env_bool,
    resolve_aria_base_url,
)
from wandb_mcp_server.error_sanitizer import MAX_EXTERNAL_ERROR_CHARS, sanitize_sensitive_text, sanitize_sensitive_value
from wandb_mcp_server.instrumented_server import (
    InstrumentedFastMCP,
    register_current_sync_future,
)

# Import Weave for tracing MCP tool calls
try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    WEAVE_AVAILABLE = False

from wandb_mcp_server.mcp_tools.automations import (
    LIST_AUTOMATIONS_TOOL_DESCRIPTION,
    LIST_INTEGRATIONS_TOOL_DESCRIPTION,
    list_automations,
    list_integrations,
)
from wandb_mcp_server.mcp_tools.aria import (
    ARIA_GET_TURN_TOOL_DESCRIPTION,
    ARIA_GET_TURNS_TOOL_DESCRIPTION,
    ARIA_SEND_MESSAGE_TOOL_DESCRIPTION,
    get_aria_turn,
    get_aria_turns,
    normalize_aria_json_value,
    send_aria_message,
)
from wandb_mcp_server.mcp_tools.count_traces import (
    COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    count_traces,
)
from wandb_mcp_server.mcp_tools.create_report import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    create_report,
)
from wandb_mcp_server.mcp_tools.list_entities import (
    LIST_ENTITIES_TOOL_DESCRIPTION,
    list_entities,
)
from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import (
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    list_entity_projects,
)
from wandb_mcp_server.mcp_tools.query_artifacts import (
    COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION,
    LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    compare_artifact_versions,
    get_artifact_details,
    list_artifact_versions,
)
from wandb_mcp_server.mcp_tools.query_registry import (
    LIST_REGISTRIES_TOOL_DESCRIPTION,
    LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION,
    list_registries,
    list_registry_collections,
)
from wandb_mcp_server.mcp_tools.query_wandb import (
    QUERY_WANDB_TOOL_DESCRIPTION,
    query_wandb,
)

from wandb_mcp_server.mcp_tools.query_weave import (
    QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
    query_paginated_weave_traces,
)
from wandb_mcp_server.mcp_tools.agents import (
    GET_AGENT_CONVERSATION_TOOL_DESCRIPTION,
    GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION,
    GET_AGENT_TRACE_TOOL_DESCRIPTION,
    LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION,
    LIST_AGENT_VERSIONS_TOOL_DESCRIPTION,
    LIST_AGENTS_TOOL_DESCRIPTION,
    QUERY_AGENT_SPANS_TOOL_DESCRIPTION,
    SEARCH_AGENTS_TOOL_DESCRIPTION,
    get_agent_conversation,
    get_agent_span_stats,
    get_agent_trace,
    list_agent_custom_attributes,
    list_agent_versions,
    list_agents,
    query_agent_spans,
    search_agents,
)
from wandb_mcp_server.utils import ServerMCPArgs, get_rich_logger, get_server_args

# Export key functions for HF Spaces app
__all__ = [
    "validate_and_get_api_key",
    "validate_api_key",
    "configure_wandb_logging",
    "initialize_weave_tracing",
    "create_mcp_server",
    "register_tools",
    "InstrumentedFastMCP",
    "ServerMCPArgs",
    "cli",
]
from wandb_mcp_server.weave_api.models import QueryResult

# Configure logging (no side effects beyond logger setup)
logging.basicConfig(level=logging.INFO)
logger = get_rich_logger("weave-mcp-server", default_level_str="WARNING", env_var_name="MCP_SERVER_LOG_LEVEL")


@dataclass(frozen=True, slots=True)
class _AgentTool:
    """One Weave Agents (OTel/GenAI) tool: the MCP tool name, its implementation,
    and its description. Each implementation in mcp_tools.agents is a complete
    tool -- it builds its request and returns a JSON string -- so it is registered
    directly (no wrapper) and its parameter schema comes from the function signature.
    """

    name: str
    impl: Callable[..., str]
    description: str


# Single source of truth for the agent tools: the names feed the optional tool
# group registry below and the registration loop in register_tools().
_AGENT_TOOLS = (
    _AgentTool("list_weave_agents_tool", list_agents, LIST_AGENTS_TOOL_DESCRIPTION),
    _AgentTool("list_weave_agent_versions_tool", list_agent_versions, LIST_AGENT_VERSIONS_TOOL_DESCRIPTION),
    _AgentTool("query_weave_agent_spans_tool", query_agent_spans, QUERY_AGENT_SPANS_TOOL_DESCRIPTION),
    _AgentTool("get_weave_agent_span_stats_tool", get_agent_span_stats, GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION),
    _AgentTool(
        "list_weave_agent_custom_attributes_tool",
        list_agent_custom_attributes,
        LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION,
    ),
    _AgentTool("search_weave_agents_tool", search_agents, SEARCH_AGENTS_TOOL_DESCRIPTION),
    _AgentTool("get_weave_agent_trace_tool", get_agent_trace, GET_AGENT_TRACE_TOOL_DESCRIPTION),
    _AgentTool("get_weave_agent_conversation_tool", get_agent_conversation, GET_AGENT_CONVERSATION_TOOL_DESCRIPTION),
)

_AGENT_TOOL_NAMES = frozenset(tool.name for tool in _AGENT_TOOLS)

_WEAVE_TOOL_NAMES = frozenset(
    {
        "query_weave_traces_tool",
        "count_weave_traces_tool",
        "resolve_trace_roots_tool",
        "infer_trace_schema_tool",
        "summarize_evaluation_tool",
    }
)


@dataclass(frozen=True, slots=True)
class _OptionalToolGroup:
    """A group of tools controlled by one environment-backed feature flag."""

    key: str
    env_var: str
    default_enabled: bool
    tool_names: frozenset[str]


_OPTIONAL_TOOL_GROUPS = (
    _OptionalToolGroup(
        key="weave",
        env_var="WANDB_MCP_ENABLE_WEAVE_TOOLS",
        default_enabled=True,
        tool_names=_WEAVE_TOOL_NAMES,
    ),
    _OptionalToolGroup(
        key="weave_agents",
        env_var="WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
        default_enabled=False,
        tool_names=_AGENT_TOOL_NAMES,
    ),
)


def _remove_registered_tools(mcp_instance: FastMCP, tool_names: Collection[str]) -> None:
    """Remove tools from FastMCP's registry after decorator registration."""
    tool_manager = getattr(mcp_instance, "_tool_manager", None)
    tools = getattr(tool_manager, "_tools", None)
    if not isinstance(tools, dict):
        logger.warning("Could not remove disabled MCP tools; FastMCP internals changed")
        return
    for tool_name in tool_names:
        tools.pop(tool_name, None)


def _aria_result_or_tool_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a complete ARIA failure into a safe native MCP tool error."""
    result = normalize_aria_json_value(sanitize_sensitive_value(result, _error_context=result.get("ok") is False))
    if result.get("ok") is not False:
        return result
    safe_result = result
    payload = json.dumps(safe_result, default=str, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_EXTERNAL_ERROR_CHARS:
        error = safe_result.get("error") if isinstance(safe_result, dict) else None
        error = error if isinstance(error, dict) else {}
        error_type = sanitize_sensitive_text(error.get("type") or "aria_error", max_chars=64)
        if not error_type.isascii() or not all(character.isalnum() or character in "_.-" for character in error_type):
            error_type = "aria_error"
        compact_error: Dict[str, Any] = {
            "type": error_type,
            "message": sanitize_sensitive_text(
                error.get("message") or "ARIA returned an oversized error response.",
                max_chars=512,
            ),
            "retryable": error.get("retryable") is True,
        }
        for key in ("status_code", "retry_after_ms"):
            if isinstance(error.get(key), int) and 0 <= error[key] <= 60_000:
                compact_error[key] = error[key]
        compact: Dict[str, Any] = {
            "ok": False,
            "error": compact_error,
            "_truncation": {
                "applied": True,
                "reason": "mcp_error_budget",
            },
        }
        turn_id = safe_result.get("turn_id") if isinstance(safe_result, dict) else None
        if isinstance(turn_id, str) and len(turn_id) <= 512:
            candidate = {**compact, "turn_id": turn_id}
            candidate_payload = json.dumps(candidate, default=str, separators=(",", ":"))
            if len(candidate_payload.encode("utf-8")) <= MAX_EXTERNAL_ERROR_CHARS:
                compact = candidate
        payload = json.dumps(compact, default=str, separators=(",", ":"))
        if len(payload.encode("utf-8")) > MAX_EXTERNAL_ERROR_CHARS:
            compact_error["message"] = "ARIA returned an oversized error response."
            payload = json.dumps(compact, default=str, separators=(",", ":"))
    raise ToolError(payload)


_COUNT_EXECUTOR = ThreadPoolExecutor(
    max_workers=MCP_COUNT_TOOL_WORKERS,
    thread_name_prefix="mcp-count",
)


def _count_traces_with_context(
    api_key: Optional[str],
    session_id: Optional[str],
    entity_name: str,
    project_name: str,
    filters: Dict[str, Any],
    request_timeout: int,
) -> int:
    """Run count_traces inside a worker while preserving the request API key."""
    from wandb_mcp_server.api_client import WandBApiManager
    from wandb_mcp_server.session_manager import current_session_id

    token = WandBApiManager.set_context_api_key(api_key) if api_key else None
    session_token = current_session_id.set(session_id) if session_id else None
    try:
        return count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=filters,
            request_timeout=request_timeout,
        )
    finally:
        if session_token is not None:
            current_session_id.reset(session_token)
        if token is not None:
            WandBApiManager.reset_context_api_key(token)


async def _count_traces_with_deadline(
    api_key: Optional[str],
    session_id: Optional[str],
    entity_name: str,
    project_name: str,
    filters: Dict[str, Any],
    deadline_seconds: int,
) -> int:
    """Run count_traces without letting executor shutdown extend wall time."""
    request_timeout = min(
        MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
        max(1, deadline_seconds - 2),
    )
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        _COUNT_EXECUTOR,
        partial(
            _count_traces_with_context,
            api_key,
            session_id,
            entity_name,
            project_name,
            filters,
            request_timeout,
        ),
    )
    register_current_sync_future(future)
    try:
        # Shield the asyncio wrapper so timeout does not mark it done while
        # the underlying thread is still consuming physical capacity.
        return await asyncio.wait_for(asyncio.shield(future), timeout=deadline_seconds)
    except asyncio.TimeoutError:
        raise


async def _count_traces_or_none(
    api_key: Optional[str],
    session_id: Optional[str],
    entity_name: str,
    project_name: str,
    filters: Dict[str, Any],
    deadline_seconds: int,
) -> int | None:
    """Best-effort count for preflight/enrichment paths."""
    try:
        return await _count_traces_with_deadline(
            api_key,
            session_id,
            entity_name,
            project_name,
            filters,
            deadline_seconds,
        )
    except Exception as exc:
        # A preflight count is part of the same functional request. If W&B is
        # already applying backpressure, do not immediately amplify it with
        # the larger query that the count was intended to guard.
        from wandb_mcp_server.api_client import raise_for_wandb_server_busy

        raise_for_wandb_server_busy(exc)
        logger.info("Trace preflight count skipped after timeout or error")
        return None


# ===============================================================================
# SECTION 1: W&B AUTHENTICATION & API KEY SETUP
# ===============================================================================


def validate_api_key(api_key: str) -> bool:
    """
    Validate a W&B API key by attempting to use it.

    Args:
        api_key: The W&B API key to validate

    Returns:
        True if the API key is valid, False otherwise
    """
    try:
        # Try to create an API instance and fetch the viewer
        # This validates the key without setting any global state
        api = wandb.Api(
            api_key=api_key,
            overrides={"base_url": WANDB_API_BASE_URL},
            timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
        )
        api.viewer  # This will fail if the key is invalid
        logger.info("W&B API key validated successfully")
        return True
    except Exception as e:
        logger.error("W&B API key validation failed (%s)", type(e).__name__)
        return False


def validate_and_get_api_key(args: ServerMCPArgs) -> Optional[str]:
    """
    Validate and retrieve the W&B API key from various sources.

    The console entrypoint requires one server-side API key for both STDIO and
    loopback-only HTTP development. Authenticated multi-user HTTP is provided
    by the hosted wrapper and Helm image, not this standalone entrypoint.

    Priority order:
    1. Command-line argument (--wandb-api-key)
    2. Environment variable (WANDB_API_KEY)
    3. .netrc file
    4. .env file

    Args:
        args: Parsed command-line arguments

    Returns:
        The W&B API key if found, None otherwise

    Raises:
        ValueError: If no API key is found for STDIO transport
    """
    api_key = args.wandb_api_key or get_server_args().wandb_api_key

    if not api_key:
        transport_label = "STDIO" if args.transport == "stdio" else "standalone http"
        raise ValueError(
            f"WANDB_API_KEY must be set for {transport_label} transport. Options:\n"
            "1. Command-line: --wandb-api-key YOUR_KEY\n"
            "2. Environment: export WANDB_API_KEY=YOUR_KEY\n"
            "3. .env file: WANDB_API_KEY=YOUR_KEY\n"
            "4. .netrc file: machine api.wandb.ai login user password YOUR_KEY\n"
            "\nGet your API key at: https://wandb.ai/authorize"
        )

    return api_key


# ===============================================================================
# SECTION 2: W&B LOGGING CONFIGURATION
# ===============================================================================


def configure_wandb_logging() -> None:
    """
    Configure W&B and Weave logging behavior to avoid interference with MCP protocol.
    (because Weave outputs created or fetched traces)

    Environment variables that control logging:
    - WANDB_SILENT: Set to "True" to suppress all W&B output (default: True)
    - WEAVE_SILENT: Set to "True" to suppress all Weave output (default: True)
    - MCP_SERVER_LOG_LEVEL: Set server log level (DEBUG, INFO, WARNING, ERROR)
    - WANDB_CONSOLE: Set to "off" to disable console output (default: off)
    """
    # Ensure W&B operates silently by default to not interfere with MCP protocol
    os.environ.setdefault("WANDB_SILENT", "True")
    os.environ.setdefault("WEAVE_SILENT", "True")

    # Configure W&B to suppress console output
    try:
        wandb.setup(
            settings=wandb.Settings(
                silent=True,
                console="off",
                base_url=WANDB_API_BASE_URL,
                x_graphql_timeout_seconds=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
            )
        )
        logger.debug("W&B configured for silent operation")
    except Exception as e:
        logger.warning("Could not apply W&B SDK settings (%s)", type(e).__name__)

    # Silence specific loggers that might interfere with MCP
    weave_logger = get_rich_logger("weave")
    weave_logger.setLevel(logging.ERROR)

    gql_transport_logger = get_rich_logger("gql.transport.requests")
    gql_transport_logger.setLevel(logging.ERROR)

    # Allow users to enable more verbose W&B logging if needed for debugging
    if os.environ.get("WANDB_DEBUG", "").lower() == "true":
        logger.info("W&B debug logging enabled via WANDB_DEBUG=true")
        os.environ["WANDB_SILENT"] = "False"
        wandb_logger = get_rich_logger("wandb")
        wandb_logger.setLevel(logging.DEBUG)


def initialize_weave_tracing() -> bool:
    """
    Initialize Weave tracing for MCP operations using the official FastMCP integration.

    According to https://weave-docs.wandb.ai/guides/integrations/mcp, Weave automatically
    traces FastMCP operations (tools, resources, prompts) when weave.init() is called.

    Returns:
        True if Weave was successfully initialized, False otherwise
    """
    if not WEAVE_AVAILABLE:
        logger.debug("Weave not available - MCP operations will not be traced")
        return False

    # Check if Weave tracing is disabled
    if os.environ.get("WEAVE_DISABLED", "true").lower() == "true":
        logger.debug("Weave tracing disabled via WEAVE_DISABLED=true")
        return False

    # Get Weave project configuration
    entity = os.environ.get("MCP_LOGS_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
    project = os.environ.get("MCP_LOGS_WANDB_PROJECT", "wandb-mcp-logs")

    if not entity:
        logger.debug("No WANDB_ENTITY or MCP_LOGS_WANDB_ENTITY set - MCP operations will not be traced to Weave")
        return False

    try:
        weave_project = f"{entity}/{project}"
        logger.info("Initializing Weave tracing for MCP operations")

        # Set optional MCP configuration for list operations tracing
        if os.environ.get("MCP_TRACE_LIST_OPERATIONS", "").lower() == "true":
            os.environ["MCP_TRACE_LIST_OPERATIONS"] = "true"
            logger.info("MCP list operations tracing enabled")

        # Initialize Weave - this automatically enables tracing for FastMCP operations
        weave.init(weave_project)

        logger.info("Weave tracing initialized - FastMCP operations will be automatically traced")
        return True
    except Exception as e:
        logger.error("Failed to initialize Weave tracing (%s)", type(e).__name__)
        return False


# ===============================================================================
# SECTION 3: MCP TOOL REGISTRATION
# ===============================================================================


def register_tools(mcp_instance: FastMCP) -> None:
    """
    Register all W&B MCP tools on the given FastMCP instance.

    Available tools:
    - query_weave_traces_tool: Query LLM traces with filtering and pagination
    - count_weave_traces_tool: Efficiently count traces without returning data
    - resolve_trace_roots_tool: Batch-resolve root spans for child trace_ids
    - query_wandb_tool: Query W&B experiment data through the public SDK
    - create_wandb_report_tool: Create shareable reports with visualizations
    - log_analysis_to_wandb: Log analysis results as W&B runs
    - list_entities_tool: List W&B entities (user + teams)
    - query_wandb_entity_projects: List projects for an entity
    - infer_trace_schema_tool: Discover trace field schema by sampling
    - search_wandb_docs_tool: Search official W&B documentation
    - get_run_history_tool: Fetch run metric history with sampling
    - list_registries_tool: List W&B model registries
    - list_registry_collections_tool: List collections in a registry
    - list_artifact_versions_tool: List artifact versions
    - get_artifact_details_tool: Get artifact metadata and files
    - compare_artifact_versions_tool: Diff two artifact versions
    - compare_runs_tool: Structured diff between two runs
    - summarize_evaluation_tool: Aggregate Weave evaluation results
    - diagnose_run_tool: Automatic training health check
    - probe_project_tool: Run-side schema and structure discovery
    - list_wandb_automations_tool: List W&B Automations (rules that trigger
      notifications/webhooks on artifact, run-state, or run-metric events)
    - list_wandb_integrations_tool: List Slack and webhook integrations
      available as targets for Automation actions

    Weave Agents (OTel/GenAI) tools -- read the agent-spans data plane, which
    is separate from classic Weave calls:
    - list_weave_agents_tool: List agents with aggregated stats
    - list_weave_agent_versions_tool: Per-version stats for one agent
    - query_weave_agent_spans_tool: Query individual agent/LLM/tool spans
    - get_weave_agent_span_stats_tool: Time-bucketed metric series
    - list_weave_agent_custom_attributes_tool: Discover custom attribute keys
    - search_weave_agents_tool: Search messages, grouped by conversation
    - get_weave_agent_trace_tool: Chat/trajectory view for one trace (a turn)
    - get_weave_agent_conversation_tool: Multi-turn chat view for a conversation
    - aria_send_message: Start or continue an asynchronous ARIA conversation
    - aria_get_turn: Poll an ARIA turn and retrieve its current result
    - aria_get_turns: Poll several ARIA turns concurrently

    Args:
        mcp_instance: The FastMCP instance to register tools on
    """
    # The CLI intentionally loads .env after this module is imported. Resolve
    # registration gates here so late-loaded deployment configuration cannot
    # leave write tools enabled or raw GraphQL disabled unexpectedly.
    read_only = _env_bool("WANDB_MCP_READ_ONLY", False)
    raw_graphql_enabled = _env_bool("WANDB_MCP_ENABLE_RAW_GRAPHQL", False)

    @mcp_instance.tool(description=QUERY_WEAVE_TRACES_TOOL_DESCRIPTION)
    async def query_weave_traces_tool(
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        limit: Optional[int] = None,
        include_costs: bool = True,
        include_feedback: bool = True,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        truncate_length: int = 1000,
        return_full_data: bool = False,
        metadata_only: bool = False,
        detail_level: str = "summary",
    ) -> str:
        """Query traces with optional detail_level control.

        detail_level: "schema" (structural fields only), "summary" (truncated, default),
        "full" (everything untruncated, same as return_full_data=True).
        """
        detail_level = detail_level or "summary"
        _VALID_DETAIL_LEVELS = {"schema", "summary", "full"}
        if detail_level not in _VALID_DETAIL_LEVELS:
            raise ValueError(f"detail_level must be one of {_VALID_DETAIL_LEVELS}, got '{detail_level}'")
        if detail_level == "full":
            return_full_data = True

        from wandb_mcp_server.config import (
            MCP_HOSTED_MODE,
            MCP_MAX_FULL_TRACE_LIMIT,
            MCP_MAX_QUERY_LIMIT,
            COST_SORT_FIELDS,
            structured_error,
        )
        from wandb_mcp_server.api_client import WandBApiManager

        hosted_limit = MCP_MAX_FULL_TRACE_LIMIT if return_full_data else MCP_MAX_QUERY_LIMIT
        effective_limit = hosted_limit if MCP_HOSTED_MODE and limit is None else (1000 if limit is None else limit)
        if MCP_HOSTED_MODE and effective_limit > hosted_limit:
            return json.dumps(
                structured_error(
                    "quota_exceeded",
                    f"Hosted MCP trace queries are limited to {hosted_limit} traces for detail_level='{detail_level}'.",
                    limit=effective_limit,
                    max_limit=hosted_limit,
                    suggestions=[
                        "Use detail_level='schema' for broad discovery.",
                        f"Set limit={hosted_limit} or lower.",
                        "Use count_weave_traces_tool for aggregate counts without trace payloads.",
                        "Add filters to narrow the result set.",
                    ],
                )
            )
        if MCP_HOSTED_MODE and sort_by in COST_SORT_FIELDS:
            return json.dumps(
                structured_error(
                    "quota_exceeded",
                    f"Hosted MCP does not support sort_by='{sort_by}' because it requires a large first-pass scan.",
                    sort_by=sort_by,
                    suggestions=[
                        "Add time_range or op_name_contains filters.",
                        "Sort by started_at, then inspect costs in the bounded result set.",
                        "Use count_weave_traces_tool to size the query first.",
                    ],
                )
            )

        _SCHEMA_COLUMNS = [
            "id",
            "trace_id",
            "op_name",
            "started_at",
            "ended_at",
            "display_name",
            "parent_id",
            "summary",
        ]
        effective_columns = columns or []
        if detail_level == "schema" and not effective_columns:
            effective_columns = _SCHEMA_COLUMNS

        try:
            api_key = WandBApiManager.get_api_key()
            from wandb_mcp_server.session_manager import current_session_id

            session_id = current_session_id.get()
            from wandb_mcp_server.config import MCP_TOOL_TIMEOUT_SECONDS

            count_deadline = min(10, MCP_TOOL_TIMEOUT_SECONDS)
            if detail_level != "schema" and effective_limit > 100 and not metadata_only:
                pre_count = await _count_traces_or_none(
                    api_key,
                    session_id,
                    entity_name,
                    project_name,
                    filters or {},
                    count_deadline,
                )
                if pre_count and pre_count > 500:
                    return json.dumps(
                        {
                            "error": "query_too_large",
                            "message": f"Found {pre_count} matching traces. Queries over 500 traces "
                            f"risk server memory limits. Narrow your query.",
                            "trace_count": pre_count,
                            "suggestions": [
                                "detail_level='schema' (structural fields only, fast)",
                                f"limit={min(100, pre_count)} (reduce result count)",
                                "count_weave_traces_tool (counts and stats without trace data)",
                                "Add filters to narrow results",
                            ],
                        }
                    )

            result_model: QueryResult = await query_paginated_weave_traces(
                entity_name=entity_name,
                project_name=project_name,
                chunk_size=50,
                filters=filters or {},
                sort_by=sort_by,
                sort_direction=sort_direction,
                target_limit=effective_limit,
                include_costs=include_costs if detail_level != "schema" else False,
                include_feedback=include_feedback if detail_level != "schema" else False,
                columns=effective_columns,
                expand_columns=expand_columns or [],
                truncate_length=truncate_length,
                return_full_data=return_full_data,
                metadata_only=metadata_only,
            )

            try:
                matching_count = await _count_traces_or_none(
                    api_key,
                    session_id,
                    entity_name=entity_name,
                    project_name=project_name,
                    filters=filters or {},
                    deadline_seconds=count_deadline,
                )
                if matching_count is not None:
                    result_model.metadata.total_matching_count = matching_count
            except Exception:
                logger.debug("Trace total-count enrichment failed")

            # Normalize traces to plain dicts -- the processor may return
            # WeaveTrace Pydantic objects which aren't JSON-serializable by
            # json.dumps and don't support .items() for schema filtering.
            if result_model.traces:
                result_model.traces = [
                    t.model_dump() if hasattr(t, "model_dump") else (t if isinstance(t, dict) else {})
                    for t in result_model.traces
                ]

            if detail_level == "schema" and result_model.traces:
                schema_fields = {
                    "id",
                    "trace_id",
                    "op_name",
                    "started_at",
                    "ended_at",
                    "status",
                    "parent_id",
                    "display_name",
                }
                result_model.traces = [{k: v for k, v in t.items() if k in schema_fields} for t in result_model.traces]

            from wandb_mcp_server.config import MAX_RESPONSE_TOKENS
            from wandb_mcp_server.weave_api.processors import TraceProcessor

            response_json = result_model.model_dump_json()
            if result_model.traces:
                original_count = len(result_model.traces)
                metadata_json = result_model.metadata.model_dump_json()
                metadata_tokens = TraceProcessor.estimate_tokens(metadata_json)
                trace_budget = max(1, MAX_RESPONSE_TOKENS - metadata_tokens)
                truncated_traces, warning, level = TraceProcessor.enforce_token_budget(
                    response_json, result_model.traces, trace_budget
                )
                if level > 0:
                    result_model.traces = truncated_traces
                    result_model.metadata.truncation_applied = True
                    result_model.metadata.truncation_dropped_count = original_count - len(truncated_traces)
                    result_model.metadata.truncation_note = warning
                    response_json = result_model.model_dump_json()

            return response_json
        except MemoryError:
            logger.error("Memory limit reached in Weave trace query")
            return json.dumps(
                {
                    "error": "out_of_memory",
                    "message": "This query exceeded server memory limits. "
                    "Try: detail_level='schema', smaller limit, or metadata_only=True.",
                }
            )
        except Exception as e:
            from wandb_mcp_server.config import HostedLimitExceeded

            if isinstance(e, HostedLimitExceeded):
                return json.dumps(structured_error(e.error, str(e), **e.details))
            from wandb_mcp_server.api_client import raise_for_wandb_server_busy

            raise_for_wandb_server_busy(e)
            logger.error("Weave trace query failed (%s)", type(e).__name__)
            return json.dumps(
                {
                    "error": "query_failed",
                    "message": "The Weave trace query failed.",
                }
            )

    @mcp_instance.tool(description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION)
    async def count_weave_traces_tool(
        entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        from wandb_mcp_server.api_client import WandBApiManager
        from wandb_mcp_server.config import MCP_TOOL_TIMEOUT_SECONDS, structured_error

        try:
            root_filters = filters.copy() if filters else {}
            root_filters["trace_roots_only"] = True

            api_key = WandBApiManager.get_api_key()
            from wandb_mcp_server.session_manager import current_session_id

            session_id = current_session_id.get()
            total_count, root_traces_count = await asyncio.wait_for(
                asyncio.gather(
                    _count_traces_with_deadline(
                        api_key,
                        session_id,
                        entity_name,
                        project_name,
                        filters or {},
                        MCP_TOOL_TIMEOUT_SECONDS,
                    ),
                    _count_traces_with_deadline(
                        api_key,
                        session_id,
                        entity_name,
                        project_name,
                        root_filters,
                        MCP_TOOL_TIMEOUT_SECONDS,
                    ),
                ),
                timeout=MCP_TOOL_TIMEOUT_SECONDS,
            )

            return json.dumps({"total_count": total_count, "root_traces_count": root_traces_count})
        except asyncio.TimeoutError:
            logger.error("Timed out in count_weave_traces_tool")
            return json.dumps(
                structured_error(
                    "timeout",
                    f"Counting traces exceeded the {MCP_TOOL_TIMEOUT_SECONDS}s hosted timeout.",
                    timeout_seconds=MCP_TOOL_TIMEOUT_SECONDS,
                )
            )
        except Exception as e:
            from wandb_mcp_server.api_client import raise_for_wandb_server_busy

            raise_for_wandb_server_busy(e)
            logger.error("Weave trace count failed (%s)", type(e).__name__)
            return json.dumps(
                {
                    "error": "count_failed",
                    "message": "The Weave trace count failed.",
                }
            )

    from wandb_mcp_server.mcp_tools.resolve_trace_roots import (
        RESOLVE_TRACE_ROOTS_TOOL_DESCRIPTION,
        resolve_trace_roots,
    )

    @mcp_instance.tool(description=RESOLVE_TRACE_ROOTS_TOOL_DESCRIPTION)
    def resolve_trace_roots_tool(
        entity_name: str,
        project_name: str,
        trace_ids: List[str],
    ) -> str:
        """Batch-resolve root spans for child trace_ids."""
        return resolve_trace_roots(
            entity_name=entity_name,
            project_name=project_name,
            trace_ids=trace_ids,
        )

    @mcp_instance.tool(description=QUERY_WANDB_TOOL_DESCRIPTION)
    def query_wandb_tool(
        entity_name: str,
        project_name: str,
        resource: Literal["project", "run", "runs", "sweep", "sweeps", "reports"],
        run_id: Optional[str] = None,
        sweep_id: Optional[str] = None,
        report_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        order: str = "-created_at",
        limit: int = 50,
        include: Optional[List[str]] = None,
        summary_keys: Optional[List[str]] = None,
        config_keys: Optional[List[str]] = None,
        response_mode: Literal["items", "count"] = "items",
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        return query_wandb(
            entity_name=entity_name,
            project_name=project_name,
            resource=resource,
            run_id=run_id,
            sweep_id=sweep_id,
            report_name=report_name,
            filters=filters,
            order=order,
            limit=limit,
            include=include,
            summary_keys=summary_keys,
            config_keys=config_keys,
            response_mode=response_mode,
            cursor=cursor,
        )

    if raw_graphql_enabled:
        from wandb_mcp_server.mcp_tools.query_wandb_gql import (
            QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION,
            query_paginated_wandb_gql,
        )

        @mcp_instance.tool(description=QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION)
        def query_wandb_graphql_tool(
            query: str,
            variables: Optional[Dict[str, Any]] = None,
            max_items: int = 100,
            items_per_page: int = 20,
        ) -> Dict[str, Any]:
            return query_paginated_wandb_gql(query, variables, max_items, items_per_page)

    if not read_only:

        @mcp_instance.tool(description=CREATE_WANDB_REPORT_TOOL_DESCRIPTION)
        def create_wandb_report_tool(
            entity_name: str,
            project_name: str,
            title: str,
            description: Optional[str] = None,
            markdown_report_text: str = "",
            plots_html: Optional[Union[Dict[str, str], str]] = None,
            panels: Optional[List[Dict[str, Any]]] = None,
        ) -> str:
            try:
                result = create_report(
                    entity_name=entity_name,
                    project_name=project_name,
                    title=title,
                    description=description,
                    markdown_report_text=markdown_report_text,
                    plots_html=plots_html,
                    panels=panels,
                )

                return f"The report was saved here: {result['url']}"
            except Exception:
                raise

        from wandb_mcp_server.mcp_tools.log_analysis import (
            LOG_ANALYSIS_TOOL_DESCRIPTION,
            log_analysis,
        )

        @mcp_instance.tool(description=LOG_ANALYSIS_TOOL_DESCRIPTION)
        def log_analysis_to_wandb(
            entity_name: str,
            project_name: str,
            analysis_name: str,
            data: List[Dict[str, Any]],
            charts: Optional[List[Dict[str, Any]]] = None,
            scalars: Optional[Dict[str, float]] = None,
        ) -> str:
            try:
                result = log_analysis(
                    entity_name=entity_name,
                    project_name=project_name,
                    analysis_name=analysis_name,
                    data=data,
                    charts=charts,
                    scalars=scalars,
                )
                return json.dumps(result)
            except Exception as e:
                from wandb_mcp_server.api_client import raise_for_wandb_server_busy

                raise_for_wandb_server_busy(e)
                logger.error(
                    "W&B analysis logging failed (error_type=%s)",
                    type(e).__name__[:64],
                )
                return json.dumps(
                    {
                        "error": "log_failed",
                        "message": "The W&B analysis run could not be logged.",
                    }
                )

    @mcp_instance.tool(description=LIST_ENTITIES_TOOL_DESCRIPTION)
    def list_entities_tool() -> str:
        """List W&B entities (user + teams) accessible with the current API key."""
        return list_entities()

    @mcp_instance.tool(description=LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION)
    def query_wandb_entity_projects(
        entity: Optional[str] = None,
        max_projects: int = 50,
    ) -> str:
        """List projects for a W&B entity."""
        return list_entity_projects(entity=entity, max_projects=max_projects)

    @mcp_instance.tool(description=LIST_AUTOMATIONS_TOOL_DESCRIPTION)
    def list_wandb_automations_tool(
        entity: Optional[str] = None,
        name: Optional[str] = None,
        max_items: int = 50,
    ) -> str:
        """List W&B Automations accessible with the current API key."""
        return list_automations(entity=entity, name=name, max_items=max_items)

    @mcp_instance.tool(description=LIST_INTEGRATIONS_TOOL_DESCRIPTION)
    def list_wandb_integrations_tool(
        entity: str | None = None,
        kind: str | None = None,
        max_items: PositiveInt = 50,
    ) -> str:
        """List Slack and webhook integrations for a W&B entity."""
        return list_integrations(entity=entity, kind=kind, max_items=max_items)

    from wandb_mcp_server.mcp_tools.infer_schema import (
        INFER_TRACE_SCHEMA_TOOL_DESCRIPTION,
        infer_trace_schema,
    )

    @mcp_instance.tool(description=INFER_TRACE_SCHEMA_TOOL_DESCRIPTION)
    def infer_trace_schema_tool(
        entity_name: str,
        project_name: str,
        sample_size: int = 20,
        top_n_values: int = 5,
    ) -> str:
        """Discover the schema of Weave traces in a project."""
        return infer_trace_schema(
            entity_name=entity_name,
            project_name=project_name,
            sample_size=sample_size,
            top_n_values=top_n_values,
        )

    from wandb_mcp_server.mcp_tools.docs_search import (
        SEARCH_WANDB_DOCS_TOOL_DESCRIPTION,
        is_docs_proxy_enabled,
        search_wandb_docs,
    )

    if is_docs_proxy_enabled():

        @mcp_instance.tool(description=SEARCH_WANDB_DOCS_TOOL_DESCRIPTION)
        async def search_wandb_docs_tool(query: str) -> str:
            """Search the official W&B documentation."""
            return await search_wandb_docs(query)

    from wandb_mcp_server.mcp_tools.run_history import (
        GET_RUN_HISTORY_TOOL_DESCRIPTION,
        get_run_history,
    )

    @mcp_instance.tool(description=GET_RUN_HISTORY_TOOL_DESCRIPTION)
    def get_run_history_tool(
        entity_name: str,
        project_name: str,
        run_id: str,
        keys: Optional[List[str]] = None,
        samples: int = 500,
        min_step: Optional[int] = None,
        max_step: Optional[int] = None,
        x_axis: str = "_step",
        target_x: Optional[float] = None,
        tolerance: Optional[float] = None,
        stream: Literal["default", "system"] = "default",
    ) -> str:
        """Retrieve sampled time-series metric data from a W&B run."""
        try:
            return get_run_history(
                entity_name=entity_name,
                project_name=project_name,
                run_id=run_id,
                keys=keys,
                samples=samples,
                min_step=min_step,
                max_step=max_step,
                x_axis=x_axis,
                target_x=target_x,
                tolerance=tolerance,
                stream=stream,
            )
        except Exception as e:
            from wandb_mcp_server.api_client import raise_for_wandb_server_busy

            raise_for_wandb_server_busy(e)
            logger.error("Run-history query failed (%s)", type(e).__name__)
            return json.dumps(
                {
                    "error": "history_query_failed",
                    "message": "The W&B run-history query failed.",
                }
            )

    # --- Registry & Artifact tools ---

    @mcp_instance.tool(description=LIST_REGISTRIES_TOOL_DESCRIPTION)
    def list_registries_tool(
        organization: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        max_items: int = 50,
    ) -> str:
        """List W&B registries for an organization."""
        return list_registries(
            organization=organization,
            filter=filter,
            max_items=max_items,
        )

    @mcp_instance.tool(description=LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION)
    def list_registry_collections_tool(
        registry_name: str,
        organization: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        max_items: int = 50,
    ) -> str:
        """List collections within a W&B registry."""
        return list_registry_collections(
            registry_name=registry_name,
            organization=organization,
            filter=filter,
            max_items=max_items,
        )

    @mcp_instance.tool(description=LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION)
    def list_artifact_versions_tool(
        collection_name: str,
        entity_name: Optional[str] = None,
        project_name: Optional[str] = None,
        registry_name: Optional[str] = None,
        organization: Optional[str] = None,
        type_name: Optional[str] = None,
        source: str = "project",
        max_items: int = 50,
        order: str = "-created_at",
        tags: Optional[List[str]] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> str:
        """List versions of an artifact collection."""
        return list_artifact_versions(
            collection_name=collection_name,
            entity_name=entity_name,
            project_name=project_name,
            registry_name=registry_name,
            organization=organization,
            type_name=type_name,
            source=source,
            max_items=max_items,
            order=order,
            tags=tags,
            created_after=created_after,
            created_before=created_before,
        )

    @mcp_instance.tool(description=GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION)
    def get_artifact_details_tool(
        artifact_name: str,
        type_name: Optional[str] = None,
        include_files: bool = False,
        max_files: int = 50,
    ) -> str:
        """Get full details for a specific artifact version."""
        return get_artifact_details(
            artifact_name=artifact_name,
            type_name=type_name,
            include_files=include_files,
            max_files=max_files,
        )

    @mcp_instance.tool(description=COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION)
    def compare_artifact_versions_tool(
        artifact_name_a: str,
        artifact_name_b: str,
        type_name: Optional[str] = None,
        include_file_diff: bool = True,
        max_file_diff_entries: int = 50,
    ) -> str:
        """Compare two artifact versions side-by-side."""
        return compare_artifact_versions(
            artifact_name_a=artifact_name_a,
            artifact_name_b=artifact_name_b,
            type_name=type_name,
            include_file_diff=include_file_diff,
            max_file_diff_entries=max_file_diff_entries,
        )

    # ----- wb_agent-inspired analysis tools (v0.3.2) -----

    from wandb_mcp_server.mcp_tools.compare_runs import (
        COMPARE_RUNS_TOOL_DESCRIPTION,
        compare_runs,
    )

    @mcp_instance.tool(description=COMPARE_RUNS_TOOL_DESCRIPTION)
    def compare_runs_tool(
        entity_name: str,
        project_name: str,
        run_id_a: str,
        run_id_b: str,
        include_history_overlap: bool = False,
        history_keys: Optional[List[str]] = None,
        history_samples: int = 50,
        config_keys: Optional[List[str]] = None,
        summary_keys: Optional[List[str]] = None,
        x_axis: str = "_step",
    ) -> str:
        """Compare two W&B runs side-by-side."""
        return compare_runs(
            entity_name=entity_name,
            project_name=project_name,
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            include_history_overlap=include_history_overlap,
            history_keys=history_keys,
            history_samples=history_samples,
            config_keys=config_keys,
            summary_keys=summary_keys,
            x_axis=x_axis,
        )

    from wandb_mcp_server.mcp_tools.summarize_evaluation import (
        SUMMARIZE_EVALUATION_TOOL_DESCRIPTION,
        summarize_evaluation,
    )

    @mcp_instance.tool(description=SUMMARIZE_EVALUATION_TOOL_DESCRIPTION)
    def summarize_evaluation_tool(
        entity_name: str,
        project_name: str,
        eval_name: Optional[str] = None,
        max_evals: int = 5,
        include_per_task: bool = False,
    ) -> str:
        """Summarize Weave evaluation results."""
        return summarize_evaluation(
            entity_name=entity_name,
            project_name=project_name,
            eval_name=eval_name,
            max_evals=max_evals,
            include_per_task=include_per_task,
        )

    from wandb_mcp_server.mcp_tools.diagnose_run import (
        DIAGNOSE_RUN_TOOL_DESCRIPTION,
        diagnose_run,
    )

    @mcp_instance.tool(description=DIAGNOSE_RUN_TOOL_DESCRIPTION)
    def diagnose_run_tool(
        entity_name: str,
        project_name: str,
        run_id: str,
        loss_key: Optional[str] = None,
        val_loss_key: Optional[str] = None,
        config_keys: Optional[List[str]] = None,
        summary_keys: Optional[List[str]] = None,
        x_axis: str = "_step",
        samples: int = 500,
    ) -> str:
        """Diagnose a W&B run's training health."""
        return diagnose_run(
            entity_name=entity_name,
            project_name=project_name,
            run_id=run_id,
            loss_key=loss_key,
            val_loss_key=val_loss_key,
            config_keys=config_keys,
            summary_keys=summary_keys,
            x_axis=x_axis,
            samples=samples,
        )

    from wandb_mcp_server.mcp_tools.probe_project import (
        PROBE_PROJECT_TOOL_DESCRIPTION,
        probe_project,
    )

    @mcp_instance.tool(description=PROBE_PROJECT_TOOL_DESCRIPTION)
    def probe_project_tool(
        entity_name: str,
        project_name: str,
        sample_runs: int = 6,
        field_pattern: Optional[str] = None,
        include_artifacts: bool = False,
    ) -> str:
        """Probe a W&B project to discover its structure."""
        return probe_project(
            entity_name=entity_name,
            project_name=project_name,
            sample_runs=sample_runs,
            field_pattern=field_pattern,
            include_artifacts=include_artifacts,
        )

    # ----- Weave Agents (OTel/GenAI) tools -----
    # These read the OTel agent-spans data plane (separate from classic Weave
    # calls), so they are registered directly and independently gated below.
    # Each implementation is a complete tool; its parameter schema is derived
    # from the function signature.
    for tool in _AGENT_TOOLS:
        mcp_instance.tool(name=tool.name, description=tool.description)(tool.impl)

    # ARIA forwards the caller's credential to a distinct hosted service and
    # includes a write operation. Register it only after explicit opt-in;
    # never rely on removing it through private FastMCP internals.
    aria_enabled = _env_bool("WANDB_MCP_ENABLE_ARIA_TOOLS", False)
    if aria_enabled:
        # The CLI loads .env after module imports. Resolve and validate again
        # here so a late configuration can never silently fall back to a
        # different credential-bearing origin.
        resolve_aria_base_url()
        if not read_only:

            @mcp_instance.tool(description=ARIA_SEND_MESSAGE_TOOL_DESCRIPTION)
            async def aria_send_message(
                message: str,
                entity: Optional[str] = None,
                project: Optional[str] = None,
                parent_turn_id: Optional[str] = None,
                wait_seconds: int = 0,
                include_turn: bool = False,
            ) -> Dict[str, Any]:
                return _aria_result_or_tool_error(
                    await send_aria_message(
                        message=message,
                        entity=entity,
                        project=project,
                        parent_turn_id=parent_turn_id,
                        wait_seconds=wait_seconds,
                        include_turn=include_turn,
                    )
                )

        @mcp_instance.tool(description=ARIA_GET_TURN_TOOL_DESCRIPTION)
        async def aria_get_turn(
            turn_id: str,
            wait_seconds: int = 0,
            include_turn: bool = False,
        ) -> Dict[str, Any]:
            return _aria_result_or_tool_error(
                await get_aria_turn(
                    turn_id=turn_id,
                    wait_seconds=wait_seconds,
                    include_turn=include_turn,
                )
            )

        @mcp_instance.tool(description=ARIA_GET_TURNS_TOOL_DESCRIPTION)
        async def aria_get_turns(
            turn_ids: List[str],
            wait_seconds: int = 0,
            include_turn: bool = False,
        ) -> Dict[str, Any]:
            return _aria_result_or_tool_error(
                await get_aria_turns(
                    turn_ids=turn_ids,
                    wait_seconds=wait_seconds,
                    include_turn=include_turn,
                )
            )

    # Optional groups are registered first and removed last so each feature
    # gate controls the complete public surface without bypassing the shared
    # InstrumentedFastMCP dispatch boundary.
    for group in _OPTIONAL_TOOL_GROUPS:
        if not _env_bool(group.env_var, group.default_enabled):
            logger.info(
                "Optional MCP tool group '%s' disabled via %s",
                group.key,
                group.env_var,
            )
            _remove_registered_tools(mcp_instance, group.tool_names)


# ===============================================================================
# SECTION 4: MCP SERVER SETUP (STDIO & HTTP)
# ===============================================================================


def _validate_standalone_transport(transport: str, host: str) -> str:
    """Validate standalone transport safety and return the concrete bind host."""
    if transport == "stdio":
        return host
    if transport != "http":
        raise ValueError(f"Invalid transport type: {transport}. Must be 'stdio' or 'http'")

    normalized_host = host.strip().strip("[]").rstrip(".").lower()
    if normalized_host == "localhost":
        # Avoid relying on mutable hostname resolution for the security
        # boundary. A literal loopback address is passed to the socket binder.
        bind_host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(normalized_host)
        except ValueError:
            address = None
        if address is None or not address.is_loopback:
            raise ValueError(
                "Standalone --transport http is development-only and must bind "
                "to a literal loopback host. Use the authenticated hosted wrapper "
                "or Helm image for production HTTP."
            )
        bind_host = normalized_host

    if os.environ.get("MCP_AUTH_DISABLED", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise ValueError(
            "Standalone --transport http has no bearer-authentication "
            "middleware. Set MCP_AUTH_DISABLED=true for loopback development, "
            "or use the authenticated hosted wrapper or Helm image."
        )
    return bind_host


def create_mcp_server(transport: str, host: str = "localhost", port: Optional[int] = None) -> FastMCP:
    """
    Create and configure a FastMCP server for the specified transport.

    Args:
        transport: Transport type ("stdio" or "http")
        host: Host for HTTP transport (default: "localhost")
        port: Port for HTTP transport (default: 8080)

    Returns:
        Configured FastMCP instance with all tools registered

    Raises:
        ValueError: If transport type is invalid

    Authentication:
        - STDIO transport: Uses environment variables (WANDB_API_KEY required)
        - Standalone HTTP transport: unauthenticated, loopback-only development
          server. Set MCP_AUTH_DISABLED=true to acknowledge this explicitly.
          Production HTTP uses the hosted wrapper or Helm deployment.
    """
    host = _validate_standalone_transport(transport, host)

    from wandb_mcp_server.analytics import configure_analytics_runtime

    configure_analytics_runtime(transport)

    if transport == "http":
        port = port if port is not None else 8080
        logger.info(f"Configuring HTTP server on {host}:{port}")
        mcp = InstrumentedFastMCP("weave-mcp-server", host=host, port=port, stateless_http=True)
        logger.warning("Standalone HTTP development server is unauthenticated and loopback-only")

    else:
        logger.info("Configuring stdio server")
        mcp = InstrumentedFastMCP("weave-mcp-server")
        logger.info("STDIO transport uses environment variable authentication")

    # Register all tools
    register_tools(mcp)

    return mcp


# ===============================================================================
# SECTION 5: MAIN CLI ENTRY POINT
# ===============================================================================


def cli():
    """
    Main command-line interface for starting the Weights & Biases MCP Server.

    Usage:
        wandb_mcp_server [OPTIONS]

    Options:
        --transport {stdio,http}     Transport type (default: stdio)
        --host HOST                  Host for HTTP transport (default: localhost)
        --port PORT                  Port for HTTP transport (default: 8080)
        --wandb-api-key KEY         W&B API key (can also use env var)

    Environment Variables:
        WANDB_API_KEY               W&B API key (required for STDIO and development HTTP)
        MCP_SERVER_LOG_LEVEL        Server log level (DEBUG, INFO, WARNING, ERROR)
        WANDB_SILENT                Set to "False" to enable W&B output (default: True)
        WEAVE_SILENT                Set to "False" to enable Weave output (default: True)
        WANDB_DEBUG                 Set to "true" to enable W&B debug logging
        MCP_AUTH_DISABLED           Must be "true" for loopback HTTP development
        WB_AGENT_BASE_URL           ARIA service URL (default: https://wb-agent.wandb.ai)
    """
    print("Starting W&B MCP Server...", file=sys.stderr)

    # Load .env only when running as CLI, not on import
    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

    # Normalize process-wide logging BEFORE any logger.info call fires. When
    # MCP_LOG_FORMAT=json this installs our JSON handler on root + uvicorn.* + mcp,
    # so access logs and MCP SDK logs are structured too (not just wandb_mcp_server.*
    # loggers that go through get_rich_logger). No-op when MCP_LOG_FORMAT is unset
    # or set to "rich" -- preserves today's Cloud Run behavior verbatim.
    from wandb_mcp_server.utils import configure_process_logging

    configure_process_logging()

    # Parse command line arguments
    import simple_parsing

    args = simple_parsing.parse(ServerMCPArgs)

    # Reject unsafe standalone HTTP before credential validation or any
    # optional Weave initialization can perform network work.
    bind_host = _validate_standalone_transport(args.transport, args.host)

    # Configure W&B logging behavior
    configure_wandb_logging()

    # Validate and get API key
    api_key = validate_and_get_api_key(args)

    # Validate API key if we have one (but don't set global state)
    if api_key:
        from wandb_mcp_server.api_client import WandBApiManager

        # Upstream SDK errors can echo request details. Make the candidate key
        # available to the shared sanitizer while validation is in flight, then
        # discard that temporary context regardless of the result.
        validation_token = WandBApiManager.set_context_api_key(api_key)
        try:
            api_key_is_valid = validate_api_key(api_key)
        finally:
            WandBApiManager.reset_context_api_key(validation_token)

        if not api_key_is_valid:
            raise ValueError("WANDB_API_KEY validation failed")

        # The console entrypoint is single-actor. Authenticated hosted HTTP
        # supplies a per-request context in its wrapper instead.
        WandBApiManager.set_context_api_key(api_key)
        logger.info("API key set in context for standalone session")

    # Initialize Weave tracing for MCP tool calls
    initialize_weave_tracing()

    logger.info("Starting Weights & Biases MCP Server")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"API Key configured: {'Yes' if api_key else 'No'}")

    # Create and run the MCP server
    server = create_mcp_server(args.transport, bind_host, args.port)

    if args.transport == "http":
        logger.info(f"Starting HTTP server on {bind_host}:{args.port or 8080}")
        server.run(transport="streamable-http")
    else:
        logger.info("Starting stdio server")
        server.run(transport="stdio")


if __name__ == "__main__":
    cli()
