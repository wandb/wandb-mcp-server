import importlib
import os

import pytest
from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.server import create_mcp_server, register_tools


WEAVE_TOOLS = {
    "query_weave_traces_tool",
    "count_weave_traces_tool",
    "resolve_trace_roots_tool",
    "infer_trace_schema_tool",
    "summarize_evaluation_tool",
}

MODELS_TOOLS = {
    "query_wandb_tool",
    "get_run_history_tool",
    "list_artifact_versions_tool",
    "get_artifact_details_tool",
    "compare_artifact_versions_tool",
    "compare_runs_tool",
    "diagnose_run_tool",
    "probe_project_tool",
    "list_entities_tool",
    "query_wandb_entity_projects",
    "list_registries_tool",
    "list_registry_collections_tool",
    "list_wandb_automations_tool",
    "list_wandb_integrations_tool",
    "search_wandb_docs_tool",
    "create_wandb_report_tool",
    "log_analysis_to_wandb",
}

NON_WEAVE_TOOLS = MODELS_TOOLS

BASE_WRITE_TOOLS = {
    "create_wandb_report_tool",
    "log_analysis_to_wandb",
}

ARIA_TOOLS = {
    "aria_send_message",
    "aria_get_turn",
    "aria_get_turns",
}

WRITE_TOOLS = BASE_WRITE_TOOLS | {"aria_send_message"}

# Agents (OTel) tools read a separate agent-spans data plane and are opt-in.
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


def _reset_tool_gate_env() -> None:
    os.environ.pop("WANDB_MCP_ENABLE_WEAVE_TOOLS", None)
    os.environ.pop("WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS", None)
    os.environ.pop("WANDB_MCP_ENABLE_ARIA_TOOLS", None)
    os.environ.pop("WANDB_MCP_READ_ONLY", None)
    os.environ.pop("WANDB_MCP_ENABLE_RAW_GRAPHQL", None)


def _registered_tool_names() -> set[str]:
    mcp = FastMCP("test")
    register_tools(mcp)
    return set(mcp._tool_manager._tools)


def test_weave_tools_registered_by_default():
    _reset_tool_gate_env()
    import wandb_mcp_server.config as cfg

    importlib.reload(cfg)
    names = _registered_tool_names()

    assert WEAVE_TOOLS.issubset(names)
    assert NON_WEAVE_TOOLS.issubset(names)
    assert BASE_WRITE_TOOLS.issubset(names)
    assert AGENT_TOOLS.isdisjoint(names)
    assert ARIA_TOOLS.isdisjoint(names)
    assert "query_wandb_graphql_tool" not in names
    assert len(names) == 22


def test_raw_graphql_tool_is_opt_in_and_independent():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_RAW_GRAPHQL"] = "true"
    os.environ["WANDB_MCP_READ_ONLY"] = "true"
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = "false"
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert "query_wandb_tool" in names
    assert "query_wandb_graphql_tool" in names
    assert WRITE_TOOLS.isdisjoint(names)
    assert ARIA_TOOLS.isdisjoint(names)
    assert WEAVE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.issubset(names)


def test_weave_tools_can_be_disabled():
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = "false"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert WEAVE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.isdisjoint(names)
    assert NON_WEAVE_TOOLS.issubset(names)
    assert BASE_WRITE_TOOLS.issubset(names)
    assert ARIA_TOOLS.isdisjoint(names)


def test_agent_tools_enabled_via_flag():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert WEAVE_TOOLS.issubset(names)
    assert AGENT_TOOLS.issubset(names)
    assert NON_WEAVE_TOOLS.issubset(names)
    assert BASE_WRITE_TOOLS.issubset(names)
    assert ARIA_TOOLS.isdisjoint(names)
    assert len(names) == 30


def test_agent_and_weave_flags_are_independent():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = "false"
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert WEAVE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.issubset(names)
    assert NON_WEAVE_TOOLS.issubset(names)
    assert BASE_WRITE_TOOLS.issubset(names)
    assert ARIA_TOOLS.isdisjoint(names)


def test_agent_and_aria_tools_have_exact_full_manifest():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    os.environ["WANDB_MCP_ENABLE_ARIA_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert AGENT_TOOLS.issubset(names)
    assert ARIA_TOOLS.issubset(names)
    assert len(names) == 33


@pytest.mark.parametrize(
    ("weave", "agents", "aria", "expected"),
    [
        (False, False, False, MODELS_TOOLS),
        (True, False, False, MODELS_TOOLS | WEAVE_TOOLS),
        (False, True, False, MODELS_TOOLS | AGENT_TOOLS),
        (True, False, True, MODELS_TOOLS | WEAVE_TOOLS | ARIA_TOOLS),
        (True, True, False, MODELS_TOOLS | WEAVE_TOOLS | AGENT_TOOLS),
        (True, True, True, MODELS_TOOLS | WEAVE_TOOLS | AGENT_TOOLS | ARIA_TOOLS),
    ],
)
def test_feature_profiles_have_exact_manifests(weave, agents, aria, expected):
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = str(weave).lower()
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = str(agents).lower()
    os.environ["WANDB_MCP_ENABLE_ARIA_TOOLS"] = str(aria).lower()
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert names == expected
    assert len(names) in {17, 22, 25, 30, 33}


@pytest.mark.parametrize(
    ("weave", "agents", "aria"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_every_feature_profile_has_exact_read_only_variant(weave, agents, aria):
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = str(weave).lower()
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = str(agents).lower()
    os.environ["WANDB_MCP_ENABLE_ARIA_TOOLS"] = str(aria).lower()
    os.environ["WANDB_MCP_READ_ONLY"] = "false"
    import wandb_mcp_server.config as cfg

    importlib.reload(cfg)
    writable = _registered_tool_names()
    os.environ["WANDB_MCP_READ_ONLY"] = "true"
    try:
        importlib.reload(cfg)
        read_only = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    expected_removed = BASE_WRITE_TOOLS | ({"aria_send_message"} if aria else set())
    assert writable - read_only == expected_removed
    assert read_only == writable - expected_removed


def test_read_only_mode_removes_exactly_the_write_tools():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    os.environ["WANDB_MCP_ENABLE_ARIA_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    importlib.reload(cfg)
    default_names = _registered_tool_names()
    os.environ["WANDB_MCP_READ_ONLY"] = "true"
    try:
        importlib.reload(cfg)
        read_only_names = _registered_tool_names()
        assert cfg.WANDB_MCP_READ_ONLY is True
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert default_names - read_only_names == WRITE_TOOLS
    assert read_only_names - default_names == set()
    assert WRITE_TOOLS.isdisjoint(read_only_names)
    assert (NON_WEAVE_TOOLS - BASE_WRITE_TOOLS).issubset(read_only_names)
    assert "query_wandb_tool" in read_only_names
    assert WEAVE_TOOLS.issubset(read_only_names)
    assert AGENT_TOOLS.issubset(read_only_names)
    assert ARIA_TOOLS - {"aria_send_message"} <= read_only_names
    assert len(default_names) == 33
    assert len(read_only_names) == 30


def test_read_only_mode_is_independent_of_weave_and_agent_gates():
    _reset_tool_gate_env()
    os.environ["WANDB_MCP_READ_ONLY"] = "true"
    os.environ["WANDB_MCP_ENABLE_WEAVE_TOOLS"] = "false"
    os.environ["WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS"] = "true"
    os.environ["WANDB_MCP_ENABLE_ARIA_TOOLS"] = "true"
    import wandb_mcp_server.config as cfg

    try:
        importlib.reload(cfg)
        names = _registered_tool_names()
    finally:
        _reset_tool_gate_env()
        importlib.reload(cfg)

    assert WRITE_TOOLS.isdisjoint(names)
    assert WEAVE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.issubset(names)
    assert (NON_WEAVE_TOOLS - BASE_WRITE_TOOLS).issubset(names)
    assert ARIA_TOOLS - {"aria_send_message"} <= names
    assert "query_wandb_tool" in names


def test_late_dotenv_registration_gates_fail_closed(monkeypatch):
    """CLI-loaded flags must apply even when config was imported already."""
    _reset_tool_gate_env()
    monkeypatch.setenv("WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS", "true")
    monkeypatch.setenv("WANDB_MCP_ENABLE_ARIA_TOOLS", "true")
    monkeypatch.setenv("WANDB_MCP_READ_ONLY", "true")

    names = _registered_tool_names()

    assert len(names) == 30
    assert WRITE_TOOLS.isdisjoint(names)
    assert AGENT_TOOLS.issubset(names)
    assert ARIA_TOOLS - {"aria_send_message"} <= names


def test_late_dotenv_raw_graphql_flag_is_applied(monkeypatch):
    """The opt-in compatibility tool also follows CLI-loaded configuration."""
    _reset_tool_gate_env()
    monkeypatch.setenv("WANDB_MCP_ENABLE_RAW_GRAPHQL", "true")
    monkeypatch.setenv("WANDB_MCP_READ_ONLY", "true")

    names = _registered_tool_names()

    assert "query_wandb_graphql_tool" in names
    assert WRITE_TOOLS.isdisjoint(names)


def test_constructed_servers_use_public_boundary_instrumentation(monkeypatch):
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    import wandb_mcp_server.analytics as analytics

    monkeypatch.setattr(analytics, "_configured_transport", None)

    stdio = create_mcp_server("stdio")
    assert isinstance(stdio, InstrumentedFastMCP)
    assert analytics._resolve_transport() == "stdio"

    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")
    http = create_mcp_server("http")
    assert isinstance(http, InstrumentedFastMCP)
    assert analytics._resolve_transport() == "http"
