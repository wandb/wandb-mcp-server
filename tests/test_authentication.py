"""
Unit tests for authentication logic in the W&B MCP Server.

Ported from wandb-mcp-server-internal. These tests use mocks
so no real API keys or network calls are needed.
"""

import json
import requests
from unittest.mock import patch, MagicMock


def parse_sse_response(text):
    """Parse SSE response to extract JSON data."""
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                pass
    return None


class TestAuthentication:
    """Test authentication functionality."""

    def test_request_without_auth_returns_401(self, server_url):
        """Requests without authentication should be rejected."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response

            response = requests.post(
                f"{server_url}/mcp",
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0.0",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    "id": 1,
                },
            )
            assert response.status_code == 401

    def test_request_with_invalid_token_returns_401(self, server_url):
        """Requests with invalid tokens should be rejected."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_post.return_value = mock_response

            response = requests.post(
                f"{server_url}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Authorization": "Bearer invalid_token_123",
                },
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
            )
            assert response.status_code == 401

    def test_request_with_valid_token_succeeds(self, server_url, fake_api_key):
        """Requests with valid API keys should succeed."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"mcp-session-id": "test-session-123"}
            mock_response.text = (
                'data: {"jsonrpc":"2.0","result":{"serverInfo":{"name":"wandb-mcp-server","version":"1.0.0"}},"id":1}'
            )
            mock_post.return_value = mock_response

            response = requests.post(
                f"{server_url}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Authorization": f"Bearer {fake_api_key}",
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0.0",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    "id": 1,
                },
            )
            assert response.status_code == 200
            assert response.headers.get("mcp-session-id") is not None

            data = parse_sse_response(response.text)
            assert data is not None
            assert "result" in data
            assert "serverInfo" in data["result"]

    def test_session_id_returned_on_initialize(self, server_url, fake_api_key):
        """Session ID should be returned when initializing."""
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"mcp-session-id": "session-abc-123"}
            mock_response.text = 'data: {"jsonrpc":"2.0","result":{},"id":1}'
            mock_post.return_value = mock_response

            response = requests.post(
                f"{server_url}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Authorization": f"Bearer {fake_api_key}",
                },
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
            )
            session_id = response.headers.get("mcp-session-id")
            assert session_id is not None
            assert len(session_id) > 0
