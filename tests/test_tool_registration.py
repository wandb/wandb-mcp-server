from __future__ import annotations

import asyncio
import inspect

import pytest
from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.runtime_contract import load_runtime_contract, tools_for_profile
from wandb_mcp_server.server import create_mcp_server, register_tools


LEGACY_TOOL_VARIABLES = (
    "WANDB_MCP_ENABLE_WEAVE_TOOLS",
    "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
    "WANDB_MCP_ENABLE_ARIA_TOOLS",
    "WANDB_MCP_ENABLE_RAW_GRAPHQL",
    "WANDB_MCP_READ_ONLY",
)


def _registered_tool_names() -> set[str]:
    mcp = FastMCP("test")
    register_tools(mcp)
    return {tool.name for tool in asyncio.run(mcp.list_tools())}


@pytest.fixture(autouse=True)
def _clean_runtime_selectors(monkeypatch: pytest.MonkeyPatch):
    for name in LEGACY_TOOL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    monkeypatch.setenv("MCP_CAPACITY_CLASS", "small")
    monkeypatch.delenv("WANDB_MCP_TOOL_PROFILE", raising=False)
    monkeypatch.delenv("WANDB_MCP_ACCESS_MODE", raising=False)
    monkeypatch.delenv("WB_AGENT_BASE_URL", raising=False)


def test_default_public_profile_is_exact_models_weave():
    contract = load_runtime_contract()
    assert _registered_tool_names() == set(tools_for_profile(contract, "models-weave", "read-write"))


@pytest.mark.parametrize(
    ("tool_profile", "access_mode", "expected_count"),
    [
        ("models-only", "read-write", 17),
        ("models-only", "read-only", 15),
        ("models-weave", "read-write", 22),
        ("models-weave", "read-only", 20),
        ("models-weave-agents", "read-write", 30),
        ("models-weave-agents", "read-only", 28),
        ("models-weave-agents-aria", "read-write", 33),
        ("models-weave-agents-aria", "read-only", 30),
        ("models-weave-graphql-compat", "read-write", 23),
        ("models-weave-graphql-compat", "read-only", 21),
    ],
)
def test_every_supported_profile_has_one_exact_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tool_profile: str,
    access_mode: str,
    expected_count: int,
):
    contract = load_runtime_contract()
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", tool_profile)
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", access_mode)
    if "aria" in tool_profile:
        monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")

    names = _registered_tool_names()

    assert names == set(tools_for_profile(contract, tool_profile, access_mode))
    assert len(names) == expected_count


@pytest.mark.parametrize("name", LEGACY_TOOL_VARIABLES)
def test_legacy_tool_variables_fail_startup(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setenv(name, "false")
    with pytest.raises(ValueError, match="Legacy MCP tool flags are not supported"):
        register_tools(FastMCP("test"))


@pytest.mark.parametrize("value", ["", "unknown", "MODELS_WEAVE", "MODELS-WEAVE"])
def test_unknown_or_malformed_profile_fails_startup(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", value)
    with pytest.raises(ValueError, match="WANDB_MCP_TOOL_PROFILE"):
        register_tools(FastMCP("test"))


@pytest.mark.parametrize("value", ["", "readonly", "true", "READ-WRITE"])
def test_malformed_access_mode_fails_startup(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", value)
    with pytest.raises(ValueError, match="WANDB_MCP_ACCESS_MODE"):
        register_tools(FastMCP("test"))


def test_endpoint_presence_never_enables_aria(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")
    assert {"aria_send_message", "aria_get_turn", "aria_get_turns"}.isdisjoint(_registered_tool_names())


def test_aria_profile_requires_explicit_safe_https_origin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-agents-aria")
    with pytest.raises(ValueError, match="explicit WB_AGENT_BASE_URL"):
        register_tools(FastMCP("test"))
    monkeypatch.setenv("WB_AGENT_BASE_URL", "http://unsafe.example")
    with pytest.raises(ValueError, match="WB_AGENT_BASE_URL"):
        register_tools(FastMCP("test"))


@pytest.mark.parametrize(
    ("workload", "profile"),
    [
        ("shared", "models-weave-agents-aria"),
        ("shared", "models-weave-graphql-compat"),
        ("dedicated", "models-weave-agents"),
        ("dedicated", "models-weave-agents-aria"),
        ("dedicated", "models-weave-graphql-compat"),
    ],
)
def test_managed_profiles_reject_aria_raw_graphql_and_unsupported_groups(
    monkeypatch: pytest.MonkeyPatch,
    workload: str,
    profile: str,
):
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", workload)
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", profile)
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")
    with pytest.raises(ValueError, match="not allowed with managed"):
        register_tools(FastMCP("test"))


def test_registration_never_mutates_private_fastmcp_registry():
    source = inspect.getsource(register_tools) + inspect.getsource(create_mcp_server)
    assert "_tool_manager" not in source
    assert "._tools" not in source
    assert "_remove_registered_tools" not in source


def test_constructed_servers_use_public_boundary_instrumentation(monkeypatch: pytest.MonkeyPatch):
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
