import asyncio
import json
from typing import Any, Dict

import httpx

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.auth import MCPAuthASGIMiddleware
from wandb_mcp_server.server import AuthenticatedFastMCP, create_mcp_server


async def _context_echo_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
    await asyncio.sleep(0)
    api_key = WandBApiManager.get_api_key()
    body = json.dumps({"api_key": api_key}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


def test_http_bearer_tokens_are_isolated_per_concurrent_request() -> None:
    app = MCPAuthASGIMiddleware(_context_echo_app)

    async def run() -> list[str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://mcp.example") as client:
            responses = await asyncio.gather(
                client.get(
                    "/mcp",
                    headers={"Authorization": "Bearer user-one-token-123456"},
                ),
                client.get(
                    "/mcp",
                    headers={"Authorization": "Bearer user-two-token-123456"},
                ),
            )
        return [response.json()["api_key"] for response in responses]

    assert asyncio.run(run()) == [
        "user-one-token-123456",
        "user-two-token-123456",
    ]
    assert WandBApiManager.get_api_key() is None


def test_http_mcp_rejects_missing_bearer_token() -> None:
    app = MCPAuthASGIMiddleware(_context_echo_app)

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://mcp.example") as client:
            return await client.get("/mcp")

    response = asyncio.run(run())

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="W&B MCP"'
    assert "Authorization required" in response.json()["error"]


def test_http_server_attaches_auth_middleware() -> None:
    mcp = create_mcp_server("http")

    assert isinstance(mcp, AuthenticatedFastMCP)
    app = mcp.streamable_http_app()
    assert any(middleware.cls is MCPAuthASGIMiddleware for middleware in app.user_middleware)


def test_http_tool_call_sees_callers_bearer_token() -> None:
    mcp = AuthenticatedFastMCP("auth-integration", stateless_http=True, json_response=True)

    @mcp.tool()
    async def current_key_length() -> int:
        return len(WandBApiManager.get_api_key() or "")

    async def run() -> httpx.Response:
        app = mcp.streamable_http_app()
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="https://mcp.example") as client:
                return await client.post(
                    "/mcp",
                    headers={
                        "Authorization": "Bearer smoke-test-token-123456789",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "current_key_length",
                            "arguments": {},
                        },
                    },
                )

    response = asyncio.run(run())

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["content"][0]["text"] == "26"
    assert WandBApiManager.get_api_key() is None
