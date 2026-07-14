"""End-to-end analytics tests at the public FastMCP tool boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import patch

import pytest

from wandb_mcp_server.analytics import reset_analytics_tracker
from wandb_mcp_server.analytics_datadog import (
    get_datadog_forwarder,
    reset_datadog_forwarder,
)
from wandb_mcp_server.analytics_segment import (
    get_segment_forwarder,
    reset_segment_forwarder,
)
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP, structured_result_error
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution


@pytest.fixture(autouse=True)
def _reset_all():
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()
    yield
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()


@pytest.fixture()
def _enable_analytics(monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    monkeypatch.setenv("MCP_SEGMENT_DRY_RUN", "true")
    monkeypatch.setenv("MCP_DATADOG_FORWARD", "true")
    monkeypatch.setenv("DD_API_KEY", "test-key-e2e")
    monkeypatch.setenv("DD_ENV", "test")
    monkeypatch.setenv("DD_SERVICE", "test-mcp")
    monkeypatch.setenv("DD_VERSION", "0.3.7")
    api_key = "k" * 40
    token = WandBApiManager.set_context_api_key(api_key)
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()
    yield api_key
    WandBApiManager.reset_context_api_key(token)


def _server() -> InstrumentedFastMCP:
    server = InstrumentedFastMCP("analytics-test")

    @server.tool(name="query_public_tool")
    async def query_public_tool(
        entity_name: str,
        query: str,
        max_items: int = 50,
        include_files: bool = False,
    ) -> str:
        with track_tool_execution(
            "query_internal_helper",
            None,
            {"entity_name": entity_name, "query": query},
        ):
            return json.dumps({"ok": True, "max_items": max_items, "include_files": include_files})

    @server.tool(name="nested_helper_tool")
    async def nested_helper_tool() -> str:
        with track_tool_execution("outer_helper", None, {}):
            with track_tool_execution("inner_helper", None, {}):
                return json.dumps({"ok": True})

    @server.tool(name="structured_error_tool")
    async def structured_error_tool() -> str:
        return json.dumps({"error": "permission_denied", "message": "access denied"})

    @server.tool(name="exception_tool")
    async def exception_tool() -> str:
        raise ValueError("bad query")

    @server.tool(name="slow_tool")
    async def slow_tool() -> str:
        await asyncio.sleep(0.05)
        return "ok"

    return server


@pytest.mark.usefixtures("_enable_analytics")
@pytest.mark.asyncio
async def test_public_success_reaches_both_sinks_once() -> None:
    server = _server()
    dd_forwarder = get_datadog_forwarder()
    with patch.object(dd_forwarder, "_post"):
        await server.call_tool(
            "query_public_tool",
            {
                "entity_name": "private-team",
                "query": "private query text",
                "max_items": 50,
                "include_files": True,
            },
        )

    segment = get_segment_forwarder().get_forwarded_payloads()
    datadog = dd_forwarder.get_forwarded_payloads()
    assert len(segment) == len(datadog) == 1
    assert segment[0]["properties"]["tool_name"] == "query_public_tool"
    assert segment[0]["properties"]["mcp_tool_name"] == "query_public_tool"
    assert segment[0]["properties"]["call_type"] == "tools/call"
    assert segment[0]["properties"]["success"] is True
    assert "error" not in segment[0]["properties"]
    assert segment[0]["properties"]["usage_dimensions"] == {
        "include_files": True,
        "max_items_bucket": "26-50",
    }
    expected_actor = f"wandb_key:{hashlib.sha256(('k' * 40).encode()).hexdigest()[:24]}"
    assert segment[0]["userId"] == expected_actor
    assert "private-team" not in str(segment[0])
    assert "private query text" not in str(segment[0])
    assert datadog[0]["attributes"]["tool"]["name"] == "query_public_tool"
    assert datadog[0]["attributes"]["usage_dimensions"] == {
        "include_files": True,
        "max_items_bucket": "26-50",
    }
    assert "params" not in datadog[0]["attributes"]


@pytest.mark.usefixtures("_enable_analytics")
@pytest.mark.asyncio
async def test_nested_implementation_helpers_do_not_double_count() -> None:
    server = _server()
    dd_forwarder = get_datadog_forwarder()
    with patch.object(dd_forwarder, "_post"):
        await server.call_tool("nested_helper_tool", {})

    segment = get_segment_forwarder().get_forwarded_payloads()
    assert len(segment) == 1
    assert segment[0]["properties"]["tool_name"] == "nested_helper_tool"


@pytest.mark.usefixtures("_enable_analytics")
@pytest.mark.asyncio
async def test_exception_is_emitted_as_one_failed_public_call() -> None:
    server = _server()
    dd_forwarder = get_datadog_forwarder()
    with patch.object(dd_forwarder, "_post"):
        with pytest.raises(Exception, match="bad query"):
            await server.call_tool("exception_tool", {})

    segment = get_segment_forwarder().get_forwarded_payloads()
    datadog = dd_forwarder.get_forwarded_payloads()
    assert len(segment) == len(datadog) == 1
    assert segment[0]["properties"]["success"] is False
    assert "ToolError" in segment[0]["properties"]["error"]
    assert datadog[0]["status"] == "error"


@pytest.mark.usefixtures("_enable_analytics")
@pytest.mark.asyncio
async def test_structured_error_result_is_failed() -> None:
    server = _server()
    dd_forwarder = get_datadog_forwarder()
    with patch.object(dd_forwarder, "_post"):
        await server.call_tool("structured_error_tool", {})

    segment = get_segment_forwarder().get_forwarded_payloads()[0]
    datadog = dd_forwarder.get_forwarded_payloads()[0]
    assert segment["properties"]["success"] is False
    assert "access denied" in segment["properties"]["error"]
    assert datadog["status"] == "error"


def test_mcp_is_error_result_is_failed() -> None:
    class ErrorResult:
        isError = True

    assert structured_result_error(ErrorResult()) == "ToolError: MCP result marked as an error"


@pytest.mark.usefixtures("_enable_analytics")
@pytest.mark.asyncio
async def test_duration_is_measured_at_public_boundary() -> None:
    server = _server()
    dd_forwarder = get_datadog_forwarder()
    with patch.object(dd_forwarder, "_post"):
        await server.call_tool("slow_tool", {})

    segment = get_segment_forwarder().get_forwarded_payloads()[0]
    datadog = dd_forwarder.get_forwarded_payloads()[0]
    assert segment["properties"]["duration_ms"] >= 40
    assert datadog["attributes"]["duration"] >= 40_000_000


@pytest.mark.usefixtures("_enable_analytics")
def test_implementation_context_is_non_emitting() -> None:
    with track_tool_execution("internal_helper", None, {"query": "secret"}):
        pass
    assert get_segment_forwarder().get_forwarded_payloads() == []
    assert get_datadog_forwarder().get_forwarded_payloads() == []
