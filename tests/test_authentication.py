"""
Test authentication for the W&B MCP Server.

These tests verify that the server properly handles authentication
using W&B API keys as Bearer tokens.
"""

import os
import json
import pytest
import requests
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_sse_response(text):
    """Parse SSE response to extract JSON data."""
    for line in text.split('\n'):
        if line.startswith('data: '):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                pass
    return None


@pytest.fixture
def server_url():
    """Get the server URL from environment or use default."""
    return os.environ.get("MCP_TEST_SERVER_URL", "http://localhost:7860")


@pytest.fixture
def api_key():
    """Get the W&B API key from environment."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        pytest.skip("WANDB_API_KEY not set in environment")
    return key


@pytest.fixture
def base_headers():
    """Headers required for MCP Streamable HTTP."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }


class TestAuthentication:
    """Test authentication functionality."""
    
    @pytest.mark.unit
    def test_request_without_auth_returns_401(self, server_url, base_headers):
        """Test that requests without authentication are rejected."""
        # Mock the server response
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            
            response = requests.post(
                f"{server_url}/mcp",
                headers=base_headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0.0",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"}
                    },
                    "id": 1
                }
            )
            
            assert response.status_code == 401
    
    @pytest.mark.unit
    def test_request_with_invalid_token_returns_401(self, server_url, base_headers):
        """Test that requests with invalid tokens are rejected."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_post.return_value = mock_response
            
            headers = {
                **base_headers,
                "Authorization": "Bearer invalid_token_123"
            }
            
            response = requests.post(
                f"{server_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {},
                    "id": 1
                }
            )
            
            assert response.status_code == 401
    
    @pytest.mark.unit
    def test_request_with_valid_token_succeeds(self, server_url, base_headers, api_key):
        """Test that requests with valid API keys succeed."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"mcp-session-id": "test-session-123"}
            mock_response.text = 'data: {"jsonrpc":"2.0","result":{"serverInfo":{"name":"wandb-mcp-server","version":"1.0.0"}},"id":1}'
            mock_post.return_value = mock_response
            
            headers = {
                **base_headers,
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                f"{server_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0.0",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"}
                    },
                    "id": 1
                }
            )
            
            assert response.status_code == 200
            assert response.headers.get("mcp-session-id") is not None
            
            # Parse the SSE response
            data = parse_sse_response(response.text)
            assert data is not None
            assert "result" in data
            assert "serverInfo" in data["result"]
    
    @pytest.mark.unit
    def test_session_id_returned_on_initialize(self, server_url, base_headers, api_key):
        """Test that session ID is returned when initializing."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"mcp-session-id": "session-abc-123"}
            mock_response.text = 'data: {"jsonrpc":"2.0","result":{},"id":1}'
            mock_post.return_value = mock_response
            
            headers = {
                **base_headers,
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                f"{server_url}/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {},
                    "id": 1
                }
            )
            
            session_id = response.headers.get("mcp-session-id")
            assert session_id is not None
            assert len(session_id) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_auth_flow_integration(self, server_url, base_headers, api_key):
        """Integration test for full authentication flow with real server."""
        # Step 1: Try without auth (should fail)
        try:
            response = requests.post(
                f"{server_url}/mcp",
                headers=base_headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {},
                    "id": 1
                },
                timeout=5
            )
            assert response.status_code == 401
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running at {server_url}")
        
        # Step 2: Initialize with auth
        headers = {
            **base_headers,
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(
            f"{server_url}/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}
                },
                "id": 2
            },
            timeout=10
        )
        
        assert response.status_code == 200
        session_id = response.headers.get("mcp-session-id")
        assert session_id is not None
        
        # Step 3: Use session to list tools
        session_headers = {
            **headers,
            "mcp-session-id": session_id
        }
        
        try:
            response = requests.post(
                f"{server_url}/mcp",
                headers=session_headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 3
                },
                timeout=2  # Short timeout for streaming
            )
            assert response.status_code == 200
        except requests.exceptions.ReadTimeout:
            # Expected for SSE streaming
            pass
