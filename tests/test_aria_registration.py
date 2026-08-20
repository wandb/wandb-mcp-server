import asyncio
import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

import wandb_mcp_server.server as server
from wandb_mcp_server.instrumented_server import (
    InstrumentedFastMCP,
    _dispatch_tool_cost,
    _NON_IDEMPOTENT_WRITE_TOOLS,
)
from wandb_mcp_server.server import register_tools


def _enable_aria(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-agents-aria")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-write")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    monkeypatch.setenv("MCP_CAPACITY_CLASS", "small")
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")


def test_aria_tools_are_registered_with_self_guiding_descriptions(monkeypatch) -> None:
    _enable_aria(monkeypatch)
    mcp = FastMCP("test")
    register_tools(mcp)

    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

    assert {"aria_send_message", "aria_get_turn", "aria_get_turns"} <= tools.keys()
    assert "hand off W&B-native work" in tools["aria_send_message"].description
    assert "parent_turn_id" in tools["aria_send_message"].description
    assert "bounded interval" in tools["aria_get_turn"].description
    assert "concurrently" in tools["aria_get_turns"].description
    for name in ("aria_send_message", "aria_get_turn", "aria_get_turns"):
        assert "<when_to_use>" in tools[name].description
        assert "include_turn" in tools[name].inputSchema["properties"]
    assert tools["aria_send_message"].inputSchema["required"] == ["message"]
    assert tools["aria_get_turn"].inputSchema["required"] == ["turn_id"]
    assert tools["aria_get_turns"].inputSchema["required"] == ["turn_ids"]


def test_default_profile_never_registers_aria(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-write")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    mcp = FastMCP("test")

    register_tools(mcp)

    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert {"aria_send_message", "aria_get_turn", "aria_get_turns"}.isdisjoint(names)


def test_enabled_aria_revalidates_late_base_url_before_registration(monkeypatch) -> None:
    _enable_aria(monkeypatch)
    monkeypatch.setenv("WB_AGENT_BASE_URL", "http://unsafe.example")

    with pytest.raises(ValueError, match="WB_AGENT_BASE_URL"):
        register_tools(FastMCP("test"))


def test_aria_failure_sets_native_mcp_error_with_json_payload(monkeypatch) -> None:
    expected = {
        "ok": False,
        "turn_id": "turn-1",
        "error": {
            "type": "service_unavailable",
            "message": "ARIA is unavailable.",
            "retryable": True,
            "details": {"region": "us"},
        },
        "next_action": "Retry with the same turn_id.",
    }

    async def failed_get(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setattr(server, "get_aria_turn", failed_get)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turn",
                            "arguments": {"turn_id": "turn-1"},
                        },
                    },
                )

    response = asyncio.run(run())
    result = response.json()["result"]

    assert result["isError"] is True
    error_text = result["content"][0]["text"]
    assert error_text.startswith("Error executing tool aria_get_turn: ")
    assert json.loads(error_text.partition(": ")[2]) == expected
    assert result.get("structuredContent") is None


def test_aria_dispatch_costs_are_bounded() -> None:
    assert _dispatch_tool_cost("aria_send_message", {}) == ("heavy", 4)
    assert _dispatch_tool_cost("aria_get_turn", {}) == ("light", 1)
    assert _dispatch_tool_cost("aria_get_turns", {}) == ("heavy", 4)
    assert "aria_send_message" in _NON_IDEMPOTENT_WRITE_TOOLS


def test_aria_rate_limit_preserves_its_retry_after_at_mcp_boundary(monkeypatch) -> None:
    expected = {
        "ok": False,
        "error": {
            "type": "rate_limited",
            "message": "ARIA is rate limiting requests.",
            "retryable": True,
            "status_code": 429,
            "retry_after_ms": 7_000,
        },
        "turn_id": "turn-1",
        "next_action": "Retry aria_get_turn with the same turn_id.",
    }

    async def failed_get(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setattr(server, "get_aria_turn", failed_get)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turn",
                            "arguments": {"turn_id": "turn-1"},
                        },
                    },
                )

    response = asyncio.run(run())
    assert response.status_code == 200
    error_text = response.json()["result"]["content"][0]["text"]
    payload = json.loads(error_text.partition(": ")[2])
    assert payload == expected


def test_oversized_aria_error_remains_bounded_parseable_and_sanitized(monkeypatch) -> None:
    secret = "oversized-error-secret-token"
    internal_host = "wandb-api.default.svc.cluster.local"
    expected = {
        "ok": False,
        "error": {
            "type": "batch_lookup_failed",
            "message": "ARIA could not retrieve any requested turns.",
            "retryable": True,
            "details": {
                "results": [
                    {
                        "turn_id": f"turn-{index}",
                        "detail": f"{secret} https://{internal_host}:8081 " + ("x" * 1_000),
                    }
                    for index in range(20)
                ]
            },
        },
    }

    async def failed_batch(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", secret)
    monkeypatch.setattr(server, "get_aria_turns", failed_batch)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turns",
                            "arguments": {"turn_ids": ["turn-1"]},
                        },
                    },
                )

    response = asyncio.run(run())
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    error_text = result["content"][0]["text"]
    json_text = error_text.partition(": ")[2]
    assert len(json_text) <= 4_096
    payload = json.loads(json_text)
    assert payload["error"]["type"] == "batch_lookup_failed"
    assert payload["error"]["retryable"] is True
    assert payload["_truncation"] == {"applied": True, "reason": "mcp_error_budget"}
    assert secret not in json_text
    assert internal_host not in json_text


def test_unicode_oversized_aria_error_still_fits_utf8_budget(monkeypatch) -> None:
    expected = {
        "ok": False,
        "error": {
            "type": "batch_lookup_failed",
            "message": "😀" * 512,
            "retryable": True,
        },
        "turn_id": "😀" * 512,
    }

    async def failed_get(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setattr(server, "get_aria_turn", failed_get)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turn",
                            "arguments": {"turn_id": "turn-1"},
                        },
                    },
                )

    response = asyncio.run(run())
    json_text = response.json()["result"]["content"][0]["text"].partition(": ")[2]
    assert len(json_text.encode("utf-8")) <= 4_096
    payload = json.loads(json_text)
    assert payload["error"]["type"] == "batch_lookup_failed"
    assert payload["error"]["retryable"] is True
    assert payload["_truncation"]["reason"] == "mcp_error_budget"


def test_mcp_boundary_normalizes_nonfinite_success_content_and_structure(monkeypatch) -> None:
    expected = {
        "ok": True,
        "turn_id": "turn-1",
        "state": "completed",
        "turn": {
            "score": float("nan"),
            "values": [float("inf"), float("-inf")],
        },
    }

    async def successful_get(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setattr(server, "get_aria_turn", successful_get)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turn",
                            "arguments": {"turn_id": "turn-1", "include_turn": True},
                        },
                    },
                )

    response = asyncio.run(run())
    result = response.json(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))["result"]
    assert result["isError"] is False
    structured = result["structuredContent"]["result"]
    assert structured["turn"] == {
        "score": "<non-finite float: NaN>",
        "values": [
            "<non-finite float: Infinity>",
            "<non-finite float: -Infinity>",
        ],
    }
    assert "<non-finite float: NaN>" in result["content"][0]["text"]


def test_mcp_boundary_normalizes_nonfinite_error_details(monkeypatch) -> None:
    expected = {
        "ok": False,
        "error": {
            "type": "upstream_error",
            "message": "ARIA failed.",
            "retryable": False,
            "details": {"score": float("nan"), "limit": float("inf")},
        },
    }

    async def failed_get(**kwargs):
        return expected

    _enable_aria(monkeypatch)
    monkeypatch.setattr(server, "get_aria_turn", failed_get)
    mcp = InstrumentedFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
                return await client.post(
                    "/mcp",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "aria_get_turn",
                            "arguments": {"turn_id": "turn-1"},
                        },
                    },
                )

    response = asyncio.run(run())
    body = response.json(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    error_text = body["result"]["content"][0]["text"]
    payload = json.loads(
        error_text.partition(": ")[2],
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert payload["error"]["details"] == {
        "score": "<non-finite float: NaN>",
        "limit": "<non-finite float: Infinity>",
    }
