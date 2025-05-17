#!/usr/bin/env python
"""
Weights & Biases MCP Server - A Model Context Protocol server for querying Weights & Biases data.
"""

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.query_models import list_entity_projects
from wandb_mcp_server.report import create_report
from wandb_mcp_server.tool_prompts import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
)
from wandb_mcp_server.mcp_tools.count_traces import (
    COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    count_traces,
)
from wandb_mcp_server.mcp_tools.query_wandb_gql import (
    QUERY_WANDB_GQL_TOOL_DESCRIPTION,
    query_paginated_wandb_gql,
)
from wandb_mcp_server.mcp_tools.query_wandbot import (
    WANDBOT_TOOL_DESCRIPTION,
    query_wandbot_api,
)
from wandb_mcp_server.mcp_tools.query_weave import (
    QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
    query_paginated_weave_traces,
)
from wandb_mcp_server.trace_utils import DateTimeEncoder
from wandb_mcp_server.utils import get_rich_logger, get_server_args

# Silence logging to avoid interfering with MCP server
os.environ["WANDB_SILENT"] = "True"
os.environ["WEAVE_SILENT"] = "True"
weave_logger = get_rich_logger("weave")
weave_logger.setLevel(logging.ERROR)
gql_transport_logger = get_rich_logger("gql.transport.requests")
gql_transport_logger.setLevel(logging.ERROR)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_rich_logger("weave-mcp-server")

# Create an MCP server using FastMCP
mcp = FastMCP("weave-mcp-server")


@mcp.tool(description=QUERY_WEAVE_TRACES_TOOL_DESCRIPTION)
async def query_weave_traces_tool(
    entity_name: str,
    project_name: str,
    filters: Dict = {},
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = None,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: list | None = None,
    expand_columns: list | None = None,
    truncate_length: int | None = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
) -> str:
    try:
        # Use paginated query with chunks of 20
        result = await query_paginated_weave_traces(
            entity_name=entity_name,
            project_name=project_name,
            chunk_size=50,
            filters=filters,
            sort_by=sort_by,
            sort_direction=sort_direction,
            target_limit=limit,
            include_costs=include_costs,
            include_feedback=include_feedback,
            columns=columns,
            expand_columns=expand_columns,
            truncate_length=truncate_length,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )

        return json.dumps(result, cls=DateTimeEncoder)

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error querying traces: {str(e)}"


@mcp.tool(description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION)
async def count_weave_traces_tool(
    entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
) -> str:
    try:
        # Call the synchronous count_traces function
        total_count = count_traces(
            entity_name=entity_name, project_name=project_name, filters=filters
        )

        # Create a copy of filters and ensure trace_roots_only is True
        root_filters = filters.copy() if filters else {}
        root_filters["trace_roots_only"] = True
        root_traces_count = count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=root_filters,
        )

        return json.dumps(
            {"total_count": total_count, "root_traces_count": root_traces_count}
        )

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error counting traces: {str(e)}"


@mcp.tool(description=QUERY_WANDB_GQL_TOOL_DESCRIPTION)
def query_wandb_gql_tool(
    query: str,
    variables: Dict[str, Any] = None,
    max_items: int = 100,
    items_per_page: int = 20,
) -> Dict[str, Any]:
    return query_paginated_wandb_gql(query, variables, max_items, items_per_page)


@mcp.tool(description=CREATE_WANDB_REPORT_TOOL_DESCRIPTION)
async def create_wandb_report_tool(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: str = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
) -> str:
    # Handle plot_htmls if it's a JSON string
    if isinstance(plots_html, str):
        try:
            plots_html = json.loads(plots_html)
        except json.JSONDecodeError:
            # If it's not valid JSON, keep it as is (though this will likely cause other errors)
            pass

    report_link = create_report(
        entity_name=entity_name,
        project_name=project_name,
        title=title,
        description=description,
        markdown_report_text=markdown_report_text,
        plots_html=plots_html,
    )
    return f"The report was saved here: {report_link}"


@mcp.tool(description=LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION)
def query_wandb_entity_projects(entity: Optional[str] = None) -> List[Dict[str, Any]]:
    return list_entity_projects(entity)


@mcp.tool(description=WANDBOT_TOOL_DESCRIPTION)
def query_wandb_support_bot(question: str) -> str:
    return query_wandbot_api(question)


def cli():
    """Command-line interface for starting the Weave MCP Server."""
    # Ensure WANDB_SILENT is set, and attempt to configure wandb for silent operation globally
    os.environ["WANDB_SILENT"] = "True"
    try:
        wandb.setup(settings=wandb.Settings(silent=True, console="off"))
    except Exception as e:
        logger.warning(f"Could not apply wandb.setup settings: {e}")

    # Attempt to explicitly login to W&B and suppress its stdout messages
    # This is to ensure login happens before mcp.run() and to capture login confirmations.
    api_key = get_server_args().wandb_api_key
    if api_key:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()
        try:
            logger.info("Attempting explicit W&B login in cli()...")
            wandb.login(key=api_key)
            login_msg_stdout = captured_stdout.getvalue().strip()
            login_msg_stderr = captured_stderr.getvalue().strip()
            if login_msg_stdout:
                logger.info(f"Suppressed stdout during W&B login: {login_msg_stdout}")
            if login_msg_stderr:
                logger.info(f"Suppressed stderr during W&B login: {login_msg_stderr}")
            logger.info("Explicit W&B login attempt finished.")
        except Exception as e:
            logger.error(f"Error during explicit W&B login: {e}")
            # Potentially re-raise or handle as a fatal error if login is critical
        finally:
            sys.stdout = original_stdout # Always restore stdout
            sys.stderr = original_stderr # Always restore stderr
    else:
        logger.warning("WANDB_API_KEY not found via get_server_args(). Skipping explicit login.")

    # Validate that we have the required API key (may be redundant if explicit login was attempted)
    if not get_server_args().wandb_api_key: # Re-check, as get_server_args might have complex logic or state
        raise ValueError(
            "WANDB_API_KEY must be set either as an environment variable, in .env file, or as a command-line argument"
        )

    logger.info(f"Starting Weights & Biases MCP Server.")
    logger.info(
        f"API Key configured: {'Yes' if get_server_args().wandb_api_key else 'No'}"
    )

    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
