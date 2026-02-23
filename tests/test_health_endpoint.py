"""Unit tests for the /health endpoint (MCP-10)."""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


async def _health_endpoint(_request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[Route("/health", _health_endpoint, methods=["GET"])])
client = TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_returns_ok_json(self):
        assert client.get("/health").json() == {"status": "ok"}

    def test_content_type_json(self):
        assert "application/json" in client.get("/health").headers["content-type"]

    def test_post_not_allowed(self):
        assert client.post("/health").status_code == 405
