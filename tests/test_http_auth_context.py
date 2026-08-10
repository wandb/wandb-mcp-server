import asyncio
from types import SimpleNamespace

import httpx
from fastapi.responses import JSONResponse
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Route

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.auth import mcp_auth_middleware
from wandb_mcp_server.server import AuthenticatedFastMCP, create_mcp_server


class _FakeSessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, str] = {}

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_session(self, api_key: str, session_id: str) -> str:
        existing = self.sessions.get(session_id)
        if existing is not None and existing != api_key:
            raise ValueError("Session API key mismatch")
        self.sessions[session_id] = api_key
        return session_id


class _FakeTracker:
    def track_user_session(self, **kwargs) -> None:
        pass

    def track_request(self, **kwargs) -> None:
        pass


def _patch_auth_dependencies(monkeypatch) -> None:
    fake_api = SimpleNamespace(viewer=SimpleNamespace(username="tester", entity="test-team"))
    monkeypatch.setattr(WandBApiManager, "get_api", staticmethod(lambda api_key=None: fake_api))

    import wandb_mcp_server.analytics as analytics
    import wandb_mcp_server.session_manager as session_manager

    manager = _FakeSessionManager()
    tracker = _FakeTracker()
    monkeypatch.setattr(session_manager, "get_session_manager", lambda: manager)
    monkeypatch.setattr(analytics, "get_analytics_tracker", lambda: tracker)


async def _context_echo(request: Request) -> JSONResponse:
    await asyncio.sleep(0)
    return JSONResponse({"api_key": WandBApiManager.get_api_key()})


def _auth_app() -> Starlette:
    app = Starlette(routes=[Route("/mcp", _context_echo)])
    app.add_middleware(BaseHTTPMiddleware, dispatch=mcp_auth_middleware)
    return app


def test_http_bearer_tokens_are_isolated_per_concurrent_request(monkeypatch) -> None:
    _patch_auth_dependencies(monkeypatch)
    app = _auth_app()

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
    app = _auth_app()

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://mcp.example") as client:
            return await client.get("/mcp")

    response = asyncio.run(run())

    assert response.status_code == 401
    assert "www-authenticate" not in response.headers
    assert "Authorization required" in response.json()["error"]


def test_http_server_attaches_existing_auth_middleware() -> None:
    mcp = create_mcp_server("http")

    assert isinstance(mcp, AuthenticatedFastMCP)
    app = mcp.streamable_http_app()
    middleware = next(item for item in app.user_middleware if item.cls is BaseHTTPMiddleware)
    assert middleware.kwargs["dispatch"] is mcp_auth_middleware


def test_http_tool_call_sees_callers_bearer_token(monkeypatch) -> None:
    _patch_auth_dependencies(monkeypatch)
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
    assert response.headers["mcp-session-id"].startswith("sess_")
    result = response.json()["result"]
    assert result["content"][0]["text"] == "26"
    assert WandBApiManager.get_api_key() is None
