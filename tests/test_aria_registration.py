import asyncio
import json

import httpx
from mcp.server.fastmcp import FastMCP

import wandb_mcp_server.server as server
from wandb_mcp_server.server import WandBFastMCP, register_tools


def test_aria_tools_are_registered_with_self_guiding_descriptions() -> None:
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

    monkeypatch.setattr(server, "get_aria_turn", failed_get)
    mcp = WandBFastMCP("test", stateless_http=True, json_response=True)
    register_tools(mcp)

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="https://mcp.example") as client:
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
    assert json.loads(result["content"][0]["text"]) == expected
    assert result.get("structuredContent") is None
