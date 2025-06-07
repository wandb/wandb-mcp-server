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
import asyncio
import signal
import atexit

import wandb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import (
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    list_entity_projects
)
from wandb_mcp_server.mcp_tools.create_report import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    create_report
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
from wandb_mcp_server.mcp_tools.code_sandbox import (
    EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION,
    execute_sandbox_code,
    check_sandbox_availability,
)
from wandb_mcp_server.mcp_tools.code_sandbox.sandbox_models import (
    SandboxExecutionRequest,
    SandboxExecutionResult
)
from wandb_mcp_server.mcp_tools.tools_utils import save_result_to_sandbox_if_requested
from wandb_mcp_server.utils import get_rich_logger, get_server_args
from wandb_mcp_server.weave_api.models import QueryResult

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
logging.basicConfig(level=logging.INFO) # Sets root logger level and default handler
logger = get_rich_logger(
    "weave-mcp-server", 
    default_level_str="WARNING", 
    env_var_name="MCP_SERVER_LOG_LEVEL"
)

# Create an MCP server using FastMCP
mcp = FastMCP("weave-mcp-server")

# --------------- MCP TOOLS ---------------

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
    columns: list = [],
    expand_columns: list = [],
    truncate_length: int = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
    save_filename: str = "",
) -> str:
    try:
        # Use paginated query with chunks of 20
        result_model: QueryResult = await query_paginated_weave_traces(
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
        json_output_string = result_model.model_dump_json()

        # Save to sandbox if requested
        await save_result_to_sandbox_if_requested(
            result=json_output_string,
            save_filename=save_filename,
            logger=logger
        )
        
        return json_output_string

    except Exception as e:
        logger.error(f"Error in query_weave_traces_tool: {e}", exc_info=True)
        raise e


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
async def query_wandb_tool(
    query: str,
    variables: Dict[str, Any] = None,
    max_items: int = 100,
    items_per_page: int = 20,
    save_filename: str = "",
) -> Dict[str, Any]:
    gql_result = query_paginated_wandb_gql(query, variables, max_items, items_per_page)

    # Save to sandbox if requested
    await save_result_to_sandbox_if_requested(
        result=gql_result,
        save_filename=save_filename,
        logger=logger
    )

    return gql_result


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


# Check sandbox availability before registering the tool
_sandbox_available, _sandbox_types, _sandbox_reason = check_sandbox_availability()

if _sandbox_available:
    @mcp.tool(description=EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION)
    async def execute_sandbox_code_tool(
        code: str,
        timeout: int = 30,
        install_packages: Optional[List[str]] = None,
    ) -> str:
        """Execute Python code in a secure sandbox environment."""
        try:
            # Validate input using Pydantic model
            request = SandboxExecutionRequest(
                code=code,
                timeout=timeout,
                sandbox_type=None,
                install_packages=install_packages,
            )
            
            result_dict = await execute_sandbox_code(
                code=request.code,
                timeout=request.timeout,
                sandbox_type=request.sandbox_type.value if request.sandbox_type else None,
                install_packages=request.install_packages,
            )
            
            # Validate output using Pydantic model
            result = SandboxExecutionResult(**result_dict)
            return result.model_dump_json()
            
        except Exception as e:
            logger.error(f"Error in execute_sandbox_code_tool: {e}", exc_info=True)
            error_result = SandboxExecutionResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                output="",
                logs=[],
                sandbox_used="none"
            )
            return error_result.model_dump_json()


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

    logger.info("Starting Weights & Biases MCP Server.")
    logger.info(
        f"API Key configured: {'Yes' if get_server_args().wandb_api_key else 'No'}"
    )
    
    # Log sandbox availability
    if _sandbox_available:
        logger.info(f"Code sandbox available: {', '.join(_sandbox_types)}")
        
        # Initialize Pyodide sandbox on startup if available
        if "pyodide" in _sandbox_types:
            logger.info("Pre-initializing Pyodide sandbox and downloading packages...")
            async def init_pyodide():
                try:
                    from wandb_mcp_server.mcp_tools.code_sandbox import PyodideSandbox
                    # This will initialize Pyodide and download packages
                    await PyodideSandbox.initialize_early()
                except Exception as e:
                    logger.warning(f"Failed to pre-initialize Pyodide sandbox: {e}")
            
            loop = asyncio.new_event_loop()
            loop.run_until_complete(init_pyodide())
            loop.close()
    else:
        logger.info(f"Code sandbox not available: {_sandbox_reason}")
        
        # Provide helpful instructions if sandbox is desired but not available
        if not os.getenv("DISABLE_CODE_SANDBOX"):
            logger.info("\n" + "="*60)
            logger.info("To enable code execution, you can:")
            logger.info("1. For local execution: Install Deno")
            logger.info("   - macOS/Linux: curl -fsSL https://deno.land/install.sh | sh")
            logger.info("   - Windows: irm https://deno.land/install.ps1 | iex")
            logger.info("   - Add to PATH: export PATH=\"$HOME/.deno/bin:$PATH\"")
            logger.info("2. For cloud execution: Set E2B_API_KEY")
            logger.info("   - Sign up at https://e2b.dev")
            logger.info("   - Get your API key from the dashboard")
            logger.info("3. To disable this message: Set DISABLE_CODE_SANDBOX=1")
            logger.info("="*60 + "\n")
    
    # Set up cleanup handlers
    async def cleanup_sandboxes():
        """Clean up sandbox resources on shutdown."""
        logger.info("Cleaning up sandbox resources...")
        try:
            from wandb_mcp_server.mcp_tools.code_sandbox import E2BSandbox, PyodideSandbox
            # Clean up E2B sandbox
            await E2BSandbox.cleanup_shared_sandbox()
            # Clean up Pyodide sandbox
            await PyodideSandbox.cleanup_shared_process()
        except Exception as e:
            logger.error(f"Error during sandbox cleanup: {e}")
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cleanup_sandboxes())
        loop.close()
        sys.exit(0)
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Also register with atexit for normal exits
    def atexit_cleanup():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cleanup_sandboxes())
        loop.close()
    
    atexit.register(atexit_cleanup)

    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
