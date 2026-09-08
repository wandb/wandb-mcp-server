"""Regressions for malformed input and fail-closed runtime configuration."""

from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from wandb_mcp_server.admission import WeightedAdmissionController, tool_cost
from wandb_mcp_server.harness import current_harness_context
from wandb_mcp_server import instrumented_server as boundary


@pytest.mark.parametrize(
    "arguments",
    [
        42,
        "bad",
        [],
        [1],
        {"resource": []},
        {"include": [{}]},
        {"include": [[]]},
        {"include": [1]},
        {"include": "summary"},
        {"include": False},
    ],
)
def test_malformed_cost_is_conservatively_heavy(arguments):
    assert tool_cost("query_wandb_tool", arguments) == ("heavy", 4)


def test_cyclic_cost_input_is_bounded():
    include = []
    include.append(include)
    assert tool_cost("query_wandb_tool", {"include": include}) == ("heavy", 4)


@pytest.mark.asyncio
async def test_invalid_call_restores_context_and_permits(monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", lambda: tracker)
    server = boundary.InstrumentedFastMCP("audit")
    server._admission_controller = WeightedAdmissionController(
        actor_capacity=4, process_capacity=4, wait_timeout_seconds=0
    )
    invoked = []

    @server.tool()
    async def query_wandb_tool(include: list[str]) -> str:
        invoked.append(include)
        return "unexpected"

    variables = [current_harness_context, boundary._current_sync_call_state, boundary._current_sync_executor]
    before = [variable.get() for variable in variables]
    with pytest.raises(ToolError):
        await server.call_tool("query_wandb_tool", {"include": [{}]})
    assert [variable.get() for variable in variables] == before
    assert server._admission_controller.process_in_use == 0
    assert not invoked
    tracker.track_tool_call.assert_called_once()
    assert tracker.track_tool_call.call_args.kwargs["success"] is False


@pytest.mark.asyncio
async def test_classifier_failure_still_cleans_up_and_records_error(monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", lambda: tracker)
    server = boundary.InstrumentedFastMCP("audit")
    monkeypatch.setattr(boundary, "_dispatch_tool_cost", MagicMock(side_effect=TypeError("private-canary")))
    before = current_harness_context.get()
    with pytest.raises(TypeError):
        await server.call_tool("unknown_tool", {})
    assert current_harness_context.get() is before
    tracker.track_tool_call.assert_called_once()
    assert tracker.track_tool_call.call_args.kwargs["success"] is False
    assert "private-canary" not in str(tracker.track_tool_call.call_args)


@pytest.mark.parametrize("value", ["", " ", "stict", "private-config-canary"])
def test_invalid_privacy_fails_without_echo(monkeypatch, value):
    from wandb_mcp_server.analytics import _resolve_privacy_level

    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", value)
    with pytest.raises(ValueError, match="MCP_LOG_PRIVACY_LEVEL") as caught:
        _resolve_privacy_level()
    assert "private-config-canary" not in str(caught.value)


def test_registration_rejects_invalid_privacy_before_registering(monkeypatch):
    from wandb_mcp_server.server import register_tools

    server = boundary.InstrumentedFastMCP("audit")
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "stict")
    monkeypatch.setattr(server, "add_tool", MagicMock())
    with pytest.raises(ValueError, match="MCP_LOG_PRIVACY_LEVEL"):
        register_tools(server)
    server.add_tool.assert_not_called()
