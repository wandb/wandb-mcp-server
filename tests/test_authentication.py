"""Unit tests for authentication logic in the W&B MCP Server.

Tests directly exercise is_valid_wandb_api_key(), validate_bearer_token(),
and _resolve_session_id() rather than mocking HTTP responses.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from wandb_mcp_server.auth import (
    MCPAuthConfig,
    _resolve_session_id,
    is_valid_wandb_api_key,
    mcp_auth_middleware,
    validate_bearer_token,
)


class TestIsValidWandbApiKey:
    def test_valid_40_char_key(self):
        assert is_valid_wandb_api_key("a" * 40) is True

    def test_valid_20_char_boundary(self):
        assert is_valid_wandb_api_key("a" * 20) is True

    def test_valid_100_char_boundary(self):
        assert is_valid_wandb_api_key("a" * 100) is True

    def test_too_short_19_chars(self):
        assert is_valid_wandb_api_key("a" * 19) is False

    def test_too_long_101_chars(self):
        assert is_valid_wandb_api_key("a" * 101) is False

    def test_empty_string(self):
        assert is_valid_wandb_api_key("") is False

    def test_none_returns_false(self):
        assert is_valid_wandb_api_key(None) is False

    def test_whitespace_only(self):
        assert is_valid_wandb_api_key("   ") is False

    def test_key_with_dashes_and_underscores(self):
        assert is_valid_wandb_api_key("abc_DEF-123.456" + "x" * 10) is True

    def test_key_with_special_chars_rejected(self):
        assert is_valid_wandb_api_key("a" * 20 + "!@#$") is False

    def test_key_with_unicode_rejected(self):
        assert is_valid_wandb_api_key("a" * 19 + "\u00e9") is False

    def test_key_with_spaces_rejected(self):
        assert is_valid_wandb_api_key("a" * 10 + " " + "b" * 10) is False

    def test_whitespace_stripped_before_check(self):
        assert is_valid_wandb_api_key("  " + "a" * 40 + "  ") is True


class TestValidateBearerToken:
    @pytest.mark.asyncio
    async def test_missing_credentials_raises_401(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await validate_bearer_token(None, MCPAuthConfig())
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_credentials_raises_401(self):
        from fastapi import HTTPException

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        with pytest.raises(HTTPException) as exc_info:
            await validate_bearer_token(creds, MCPAuthConfig())
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_short_key_raises_401(self):
        from fastapi import HTTPException

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="short")
        with pytest.raises(HTTPException) as exc_info:
            await validate_bearer_token(creds, MCPAuthConfig())
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_error_does_not_leak_key_length(self):
        from fastapi import HTTPException

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="short")
        with pytest.raises(HTTPException) as exc_info:
            await validate_bearer_token(creds, MCPAuthConfig())
        assert "characters" not in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_valid_key_returns_stripped(self):
        key = "a" * 40
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=f"  {key}  ")
        result = await validate_bearer_token(creds, MCPAuthConfig())
        assert result == key


class TestResolveSessionIdValidation:
    def _make_request(self, headers: dict) -> MagicMock:
        req = MagicMock()
        req.headers = headers
        return req

    def test_valid_session_id_accepted(self):
        req = self._make_request({"Mcp-Session-Id": "sess_abc123"})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid == "sess_abc123"
        assert is_new is False

    def test_oversized_session_id_rejected(self):
        req = self._make_request({"Mcp-Session-Id": "x" * 200})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid.startswith("sess_")
        assert is_new is True

    def test_session_id_with_newlines_rejected(self):
        req = self._make_request({"Mcp-Session-Id": "sess_abc\ninjected"})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid.startswith("sess_")
        assert is_new is True

    def test_session_id_with_spaces_rejected(self):
        req = self._make_request({"Mcp-Session-Id": "sess abc"})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid.startswith("sess_")
        assert is_new is True

    def test_session_id_at_128_chars_accepted(self):
        req = self._make_request({"Mcp-Session-Id": "a" * 128})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid == "a" * 128
        assert is_new is False

    def test_session_id_at_129_chars_rejected(self):
        req = self._make_request({"Mcp-Session-Id": "a" * 129})
        sid, is_new = _resolve_session_id(req, "fake_key_12345678901234567890")
        assert sid.startswith("sess_")
        assert is_new is True


class TestAuthMiddlewareBypass:
    @pytest.mark.asyncio
    async def test_non_mcp_path_bypasses_auth(self):
        req = MagicMock()
        req.url.path = "/health"
        resp = MagicMock()

        async def call_next(_):
            return resp

        result = await mcp_auth_middleware(req, call_next)
        assert result is resp

    @pytest.mark.asyncio
    async def test_auth_disabled_bypasses(self):
        req = MagicMock()
        req.url.path = "/mcp"
        resp = MagicMock()

        async def call_next(_):
            return resp

        with patch.dict("os.environ", {"MCP_AUTH_DISABLED": "true"}):
            result = await mcp_auth_middleware(req, call_next)
        assert result is resp
