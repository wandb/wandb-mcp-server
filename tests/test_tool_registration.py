import importlib
import os

from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.server import register_tools


WEAVE_TOOLS = {
    "query_weave_traces_tool",
    "count_weave_traces_tool",
    "resolve_trace_roots_tool",
    "infer_trace_schema_tool",
    "summarize_evaluation_tool",
}

NON_WEAVE_TOOLS = {
    "query_wandb_tool",
    "create_wandb_report_tool",
    "get_run_history_tool",
    "list_artifact_versions_tool",
    "compare_runs_tool",
    "probe_project_tool",
}

# Agents (OTel) tools read the same trace backend, so they are gated with the
# Weave tools.
AGENT_TOOLS = {
    "list_weave_agents_tool",
    "list_weave_agent_versions_tool",
    "query_weave_agent_spans_tool",
    "get_weave_agent_span_stats_tool",
    "list_weave_agent_custom_attributes_tool",
    "search_weave_agents_tool",
    "get_weave_agent_trace_tool",
    "get_weave_agent_conversation_tool",
}


def _registered_tool_names() -> set[str]:
    mcp = FastMCP("test")
    register_tools(mcp)
    return set(mcp._tool_manager._tools)


def test_weave_tools_registered_by_default():
    os.environ.pop("WANDB_MCP_ENABLE_WEAVE_TOOLS", None)
    import wandb_mcp_server.config as cfg

    importlib.reload(cfg)
    names = _registered_tool_names()

    assert WEAVE_TOOLS.issubset(names)
    assert NON_WEAVE_TOOLS.issubset(names)
    assert AGENT_TOOLS.issubset(names)


def test_weave_tools_can_be_disabled():
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = "false"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        os.environ.pop("WANDB_MCP_ENABLE_WEAVE_TOOLS", None)
        importlib.reload(cfg)

    assert WEAVE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.isdisjoint(names)
    assert NON_WEAVE_TOOLS.issubset(names)
