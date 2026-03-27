#!/usr/bin/env python
"""
Weights & Biases MCP Server - A Model Context Protocol server for querying Weights & Biases data.

This server provides tools for:
- Querying Weave traces and evaluations
- Counting traces efficiently
- Executing GraphQL queries against W&B experiment data
- Creating shareable reports with visualizations
- Getting help via wandbot support agent
- Discovering available entities and projects
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from wandb_mcp_server.config import WANDB_BASE_URL

# Import Weave for tracing MCP tool calls
try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    WEAVE_AVAILABLE = False

from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import (
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    list_entity_projects,
)
from wandb_mcp_server.mcp_tools.create_report import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    create_report,
)
from wandb_mcp_server.mcp_tools.count_traces import (
    COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    count_traces,
)
from wandb_mcp_server.mcp_tools.query_wandb_gql import (
    QUERY_WANDB_GQL_TOOL_DESCRIPTION,
    query_paginated_wandb_gql,
)

# wandbot removed -- zero usage across 400+ benchmark runs, superseded by search_wandb_docs_tool
from wandb_mcp_server.mcp_tools.query_weave import (
    QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
    query_paginated_weave_traces,
)
from wandb_mcp_server.utils import get_rich_logger, get_server_args, ServerMCPArgs

# Export key functions for HF Spaces app
__all__ = [
    "validate_and_get_api_key",
    "validate_api_key",
    "configure_wandb_logging",
    "initialize_weave_tracing",
    "create_mcp_server",
    "register_tools",
    "ServerMCPArgs",
    "cli",
]
from wandb_mcp_server.weave_api.models import QueryResult

# Configure logging (no side effects beyond logger setup)
logging.basicConfig(level=logging.INFO)
logger = get_rich_logger("weave-mcp-server", default_level_str="WARNING", env_var_name="MCP_SERVER_LOG_LEVEL")


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
        api = wandb.Api(api_key=api_key, overrides={"base_url": WANDB_BASE_URL})
        viewer = api.viewer  # This will fail if the key is invalid
        viewer_id = getattr(viewer, "username", None) or getattr(viewer, "entity", None) or "<unknown>"
        logger.info(f"W&B API key validated successfully. Viewer: {viewer_id}")
        return True
    except Exception as e:
        logger.error(f"Invalid W&B API key: {e}")
        return False


def validate_and_get_api_key(args: ServerMCPArgs) -> Optional[str]:
    """
    Validate and retrieve the W&B API key from various sources.

    For HTTP transport: API key is optional (clients provide their own)
    For STDIO transport: API key is required from environment

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

    # For HTTP transport, API key is optional (clients provide their own)
    if args.transport == "http":
        if api_key:
            logger.info("Server W&B API key configured (for server operations)")
        else:
            logger.info("No server W&B API key configured (clients will provide their own)")
        return api_key

    # For STDIO transport, API key is required
    if not api_key:
        raise ValueError(
            "WANDB_API_KEY must be set for STDIO transport. Options:\n"
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
        wandb.setup(settings=wandb.Settings(silent=True, console="off", base_url=WANDB_BASE_URL))
        logger.debug("W&B configured for silent operation")
    except Exception as e:
        logger.warning(f"Could not apply wandb.setup settings: {e}")

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
        logger.info(f"Initializing Weave tracing for MCP operations: {weave_project}")

        # Set optional MCP configuration for list operations tracing
        if os.environ.get("MCP_TRACE_LIST_OPERATIONS", "").lower() == "true":
            os.environ["MCP_TRACE_LIST_OPERATIONS"] = "true"
            logger.info("MCP list operations tracing enabled")

        # Initialize Weave - this automatically enables tracing for FastMCP operations
        weave.init(weave_project)

        logger.info("Weave tracing initialized - FastMCP operations will be automatically traced")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Weave tracing: {e}")
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
    - query_wandb_tool: Execute GraphQL queries against W&B experiment data
    - create_wandb_report_tool: Create shareable reports with visualizations
    - query_wandb_entity_projects: List available entities and projects
    - search_wandb_docs_tool: Search official W&B documentation

    Args:
        mcp_instance: The FastMCP instance to register tools on
    """

    @mcp_instance.tool(description=QUERY_WEAVE_TRACES_TOOL_DESCRIPTION)
    async def query_weave_traces_tool(
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        limit: int = 1000,
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
        _VALID_DETAIL_LEVELS = {"schema", "summary", "full"}
        if detail_level not in _VALID_DETAIL_LEVELS:
            raise ValueError(f"detail_level must be one of {_VALID_DETAIL_LEVELS}, got '{detail_level}'")
        if detail_level == "full":
            return_full_data = True

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
            result_model: QueryResult = await query_paginated_weave_traces(
                entity_name=entity_name,
                project_name=project_name,
                chunk_size=50,
                filters=filters or {},
                sort_by=sort_by,
                sort_direction=sort_direction,
                target_limit=limit,
                include_costs=include_costs if detail_level != "schema" else False,
                include_feedback=include_feedback if detail_level != "schema" else False,
                columns=effective_columns,
                expand_columns=expand_columns or [],
                truncate_length=truncate_length,
                return_full_data=return_full_data,
                metadata_only=metadata_only,
            )

            try:
                matching_count = count_traces(
                    entity_name=entity_name,
                    project_name=project_name,
                    filters=filters or {},
                )
                result_model.metadata.total_matching_count = matching_count
            except Exception:
                logger.debug("count_traces for total_matching_count failed", exc_info=True)

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
        except Exception as e:
            logger.error(f"Error in query_weave_traces_tool: {e}", exc_info=True)
            raise e

    @mcp_instance.tool(description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION)
    async def count_weave_traces_tool(
        entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        from concurrent.futures import ThreadPoolExecutor
        from wandb_mcp_server.api_client import WandBApiManager

        try:
            root_filters = filters.copy() if filters else {}
            root_filters["trace_roots_only"] = True

            api_key = WandBApiManager.get_api_key()

            def _count_with_context(**kwargs):
                WandBApiManager.set_context_api_key(api_key)
                return count_traces(**kwargs)

            with ThreadPoolExecutor(max_workers=2) as executor:
                total_future = executor.submit(
                    _count_with_context, entity_name=entity_name, project_name=project_name, filters=filters or {}
                )
                root_future = executor.submit(
                    _count_with_context, entity_name=entity_name, project_name=project_name, filters=root_filters
                )
                total_count = total_future.result()
                root_traces_count = root_future.result()

            return json.dumps({"total_count": total_count, "root_traces_count": root_traces_count})
        except Exception as e:
            logger.error(f"Error in count_weave_traces_tool: {e}")
            return json.dumps({"error": f"Error counting traces: {str(e)}"})

    @mcp_instance.tool(description=QUERY_WANDB_GQL_TOOL_DESCRIPTION)
    async def query_wandb_tool(
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        max_items: int = 100,
        items_per_page: int = 20,
    ) -> Dict[str, Any]:
        return query_paginated_wandb_gql(query, variables, max_items, items_per_page)

    @mcp_instance.tool(description=CREATE_WANDB_REPORT_TOOL_DESCRIPTION)
    async def create_wandb_report_tool(
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
        except Exception as e:
            raise e

    @mcp_instance.tool(description=LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION)
    def query_wandb_entity_projects(entity: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        return list_entity_projects(entity)

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
            )
        except Exception as e:
            logger.error(f"Error in get_run_history_tool: {e}")
            return json.dumps({"error": str(e)})


# ===============================================================================
# SECTION 4: MCP SERVER SETUP (STDIO & HTTP)
# ===============================================================================


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
        - HTTP transport: Clients provide W&B API key as Bearer token
          Set MCP_AUTH_DISABLED=true to disable auth (development only)
    """
    if transport == "http":
        port = port if port is not None else 8080
        logger.info(f"Configuring HTTP server on {host}:{port}")
        mcp = FastMCP("weave-mcp-server", host=host, port=port, stateless_http=True)

        # Log authentication status for HTTP
        if os.environ.get("MCP_AUTH_DISABLED", "false").lower() == "true":
            logger.warning("⚠️  MCP authentication is DISABLED - server is publicly accessible")
        else:
            logger.info("🔒 MCP authentication enabled - clients must provide W&B API key as Bearer token")

    elif transport == "stdio":
        logger.info("Configuring stdio server")
        mcp = FastMCP("weave-mcp-server")
        logger.info("STDIO transport uses environment variable authentication")
    else:
        raise ValueError(f"Invalid transport type: {transport}. Must be 'stdio' or 'http'")

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
        WANDB_API_KEY               W&B API key (required for STDIO, optional for HTTP)
        MCP_SERVER_LOG_LEVEL        Server log level (DEBUG, INFO, WARNING, ERROR)
        WANDB_SILENT                Set to "False" to enable W&B output (default: True)
        WEAVE_SILENT                Set to "False" to enable Weave output (default: True)
        WANDB_DEBUG                 Set to "true" to enable W&B debug logging
        MCP_AUTH_DISABLED           Set to "true" to disable HTTP auth (dev only)
    """
    print("Starting W&B MCP Server...", file=sys.stderr)

    # Load .env only when running as CLI, not on import
    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

    # Parse command line arguments
    import simple_parsing

    args = simple_parsing.parse(ServerMCPArgs)

    # Configure W&B logging behavior
    configure_wandb_logging()

    # Validate and get API key
    api_key = validate_and_get_api_key(args)

    # Validate API key if we have one (but don't set global state)
    if api_key:
        validate_api_key(api_key)

        # For STDIO transport, set the API key in context for all operations
        # This is essential since STDIO doesn't have per-request auth
        if args.transport == "stdio":
            from wandb_mcp_server.api_client import WandBApiManager

            # Set the API key in context for the entire session
            # No need to reset since STDIO runs for the whole session
            WandBApiManager.set_context_api_key(api_key)
            logger.info("API key set in context for STDIO session")

            try:
                api = WandBApiManager.get_api()
                viewer = api.viewer
                viewer_id = getattr(viewer, "username", None) or getattr(viewer, "entity", None) or "<unknown>"
                logger.info(f"Authenticated W&B viewer: {viewer_id}")
            except Exception as viewer_err:
                logger.warning(f"Could not fetch W&B viewer: {viewer_err}")

    # Initialize Weave tracing for MCP tool calls
    initialize_weave_tracing()

    logger.info("Starting Weights & Biases MCP Server")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"API Key configured: {'Yes' if api_key else 'No (clients provide their own)'}")

    # Validate transport type
    if args.transport not in ["stdio", "http"]:
        raise ValueError(f"Invalid transport type: {args.transport}. Must be 'stdio' or 'http'")

    # Create and run the MCP server
    server = create_mcp_server(args.transport, args.host, args.port)

    if args.transport == "http":
        logger.info(f"Starting HTTP server on {args.host}:{args.port or 8080}")
        server.run(transport="streamable-http")
    else:
        logger.info("Starting stdio server")
        server.run(transport="stdio")


if __name__ == "__main__":
    cli()
