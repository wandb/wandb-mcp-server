"""Exercise failure provenance through the official MCP HTTP client."""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from wandb_mcp_server.config import structured_error
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.tool_results import ToolErrorJSON, tool_error_json


@pytest.mark.asyncio
async def test_owned_failures_and_customer_data_have_distinct_protocol_outcomes(monkeypatch):
    tracker = Mock()
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", lambda: tracker)
    server = InstrumentedFastMCP("result-contract", stateless_http=True, json_response=True)
    server._admission_controller = None

    @server.tool()
    def owned_dictionary(code: str) -> dict:
        return structured_error(code, "Safe server-authored explanation.")

    @server.tool()
    async def owned_json(code: str) -> str:
        return tool_error_json({"error": code, "message": "Safe server-authored explanation."})

    @server.tool()
    def customer_data() -> dict:
        return {"error": "permission_denied", "result": {"error": "sample metric", "value": 3}}

    @server.tool()
    async def customer_json() -> str:
        return json.dumps({"error": "sample metric", "result": {"error": "authentication_failed"}})

    app = server.streamable_http_app()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        for code in ("authentication_failed", "permission_denied", "invalid_input", "upstream_error"):
                            for tool in ("owned_dictionary", "owned_json"):
                                result = await client.call_tool(tool, {"code": code})
                                assert result.isError is True
                                assert result.content[0].text.startswith("{"), result
                                assert json.loads(result.content[0].text)["error"] == code
                                event = tracker.track_tool_call.call_args.kwargs
                                assert event["success"] is False
                                assert event["error"] == f"{code}: tool failed"
                        for tool in ("customer_data", "customer_json"):
                            result = await client.call_tool(tool, {})
                            assert result.isError is False
                            assert "error" in json.loads(result.content[0].text)
                            assert tracker.track_tool_call.call_args.kwargs["success"] is True
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_history_schema_exposes_effective_limits(monkeypatch):
    from wandb_mcp_server import server as registration

    monkeypatch.setattr(registration, "MCP_MAX_HISTORY_KEYS", 7)
    monkeypatch.setattr(registration, "MCP_MAX_HISTORY_RANGE_STEPS", 321)
    server = InstrumentedFastMCP("history-schema")
    try:
        registration.register_tools(server)
        tool = next(t for t in await server.list_tools() if t.name == "get_run_history_tool")
        fields = tool.inputSchema["properties"]
        assert fields["keys"]["maxItems"] == 7
        assert fields["min_step"]["minimum"] == 0
        assert fields["max_step"]["minimum"] == 0
        assert "321" in fields["max_step"]["description"]
        assert "7 metric keys" in tool.description
        assert "321 steps" in tool.description
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_owned_result_state_is_isolated_between_concurrent_calls(monkeypatch):
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", Mock)
    server = InstrumentedFastMCP("concurrent-results")
    server._admission_controller = None
    ready = asyncio.Event()

    @server.tool()
    async def owned_failure() -> dict:
        ready.set()
        return structured_error("permission_denied", "Permission denied.")

    @server.tool()
    async def customer_result() -> dict:
        await ready.wait()
        await asyncio.sleep(0)
        return {"error": "customer metric", "result": {"error": "customer value"}}

    try:
        failure, success = await asyncio.gather(
            server.call_tool("owned_failure", {}), server.call_tool("customer_result", {})
        )
        assert failure.isError is True
        assert not getattr(success, "isError", False)
        content = success[0] if isinstance(success, tuple) else success
        assert json.loads(content[0].text)["error"] == "customer metric"
    finally:
        server._sync_executor.shutdown(wait=True)


def test_registry_budget_error_keeps_provenance_after_serialization(monkeypatch):
    from wandb_mcp_server.mcp_tools import query_registry

    api = Mock()
    api.registries.return_value = []
    monkeypatch.setattr(query_registry.WandBApiManager, "get_api", lambda: api)
    monkeypatch.setattr(query_registry, "resolve_registry_organization", lambda *args: "synthetic-org")
    monkeypatch.setattr(
        query_registry,
        "fit_registry_response",
        lambda *args, **kwargs: structured_error("response_too_large", "Use a smaller page."),
    )
    result = query_registry.list_registries(organization="synthetic-org")
    assert isinstance(result, ToolErrorJSON)
    assert json.loads(result)["error"] == "response_too_large"
