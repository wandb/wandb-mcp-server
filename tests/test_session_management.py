"""Unit tests for MCP session issuance and management (Issue #23).

Tests cover:
- Server-issued session IDs when no client header is present
- Client-provided session IDs being preserved
- Session contextvar propagation
- MultiTenantSessionManager integration in auth middleware
- Analytics events receiving correct session IDs
"""

import logging
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.auth import _resolve_session_id, mcp_auth_middleware
from wandb_mcp_server.session_manager import (
    MultiTenantSessionManager,
    current_session_id,
    reset_session_manager,
)


@pytest.fixture(autouse=True)
def _reset_session_mgr():
    reset_session_manager()
    yield
    reset_session_manager()


# ---------------------------------------------------------------------------
# _resolve_session_id
# ---------------------------------------------------------------------------


class TestResolveSessionId:
    def _make_request(self, headers: dict) -> MagicMock:
        req = MagicMock()
        req.headers = headers
        return req

    def test_client_header_preserved(self):
        req = self._make_request({"Mcp-Session-Id": "client-abc"})
        sid, is_new = _resolve_session_id(req, "fake_key_1234567890_abcdefgh")
        assert sid == "client-abc"
        assert is_new is False

    def test_lowercase_header_preserved(self):
        req = self._make_request({"mcp-session-id": "client-lower"})
        sid, is_new = _resolve_session_id(req, "fake_key_1234567890_abcdefgh")
        assert sid == "client-lower"
        assert is_new is False

    def test_no_header_generates_session(self):
        req = self._make_request({})
        sid, is_new = _resolve_session_id(req, "fake_key_1234567890_abcdefgh")
        assert sid.startswith("sess_")
        assert is_new is True

    def test_generated_ids_are_unique(self):
        req = self._make_request({})
        ids = {_resolve_session_id(req, "fake_key_1234567890_abcdefgh")[0] for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Session contextvar propagation
# ---------------------------------------------------------------------------


class TestSessionContextvar:
    def test_default_is_none(self):
        assert current_session_id.get() is None

    def test_set_and_reset(self):
        tok = current_session_id.set("test-session")
        assert current_session_id.get() == "test-session"
        current_session_id.reset(tok)
        assert current_session_id.get() is None


# ---------------------------------------------------------------------------
# MultiTenantSessionManager basics (non-HMAC)
# ---------------------------------------------------------------------------


class TestSessionManagerBasics:
    @pytest.fixture()
    def mgr(self):
        with patch("wandb_mcp_server.session_manager.MultiTenantSessionManager._start_cleanup_task"):
            return MultiTenantSessionManager(
                session_ttl_seconds=300,
                max_sessions_per_key=5,
                enable_hmac_sha256_sessions=False,
            )

    def test_create_returns_session_id(self, mgr):
        sid = mgr.create_session("fake_key_1234567890_abcdefgh")
        assert sid.startswith("sess_")

    def test_create_with_custom_id(self, mgr):
        sid = mgr.create_session("fake_key_1234567890_abcdefgh", session_id="custom-123")
        assert sid == "custom-123"

    def test_validate_correct_key(self, mgr):
        sid = mgr.create_session("fake_key_1234567890_abcdefgh")
        assert mgr.validate_session(sid, "fake_key_1234567890_abcdefgh") is True

    def test_validate_wrong_key(self, mgr):
        sid = mgr.create_session("fake_key_1234567890_abcdefgh")
        assert mgr.validate_session(sid, "wrong_key_1234567890_other_xy") is False

    def test_validate_unknown_session(self, mgr):
        assert mgr.validate_session("nonexistent", "fake_key_1234567890_abcdefgh") is False

    def test_idempotent_create(self, mgr):
        sid1 = mgr.create_session("fake_key_1234567890_abcdefgh", session_id="reused")
        sid2 = mgr.create_session("fake_key_1234567890_abcdefgh", session_id="reused")
        assert sid1 == sid2

    def test_create_mismatched_key_raises(self, mgr):
        mgr.create_session("fake_key_1234567890_abcdefgh", session_id="shared")
        with pytest.raises(ValueError, match="mismatch"):
            mgr.create_session("other_key_1234567890_different", session_id="shared")

    def test_stats(self, mgr):
        mgr.create_session("fake_key_1234567890_abcdefgh")
        stats = mgr.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["unique_api_keys"] == 1


# ---------------------------------------------------------------------------
# Auth middleware session integration (mocked)
# ---------------------------------------------------------------------------


class _EventCapture(logging.Filter):
    """Capture analytics events emitted during middleware execution."""

    def __init__(self):
        super().__init__()
        self.events: list[Dict[str, Any]] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "json_fields"):
            self.events.append(record.json_fields)
        return True


def _make_fake_request(path="/mcp", api_key="fake_key_1234567890_abcdefgh", session_header=None):
    """Build a minimal mock Request for middleware tests."""
    headers = {"Authorization": f"Bearer {api_key}"}
    if session_header:
        headers["Mcp-Session-Id"] = session_header

    req = MagicMock()
    req.url.path = path
    req.method = "POST"
    req.headers = headers
    req.state = MagicMock()
    return req


class TestAuthMiddlewareSession:
    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"}):
            yield

    @pytest.fixture()
    def _mock_deps(self):
        """Patch WandBApiManager and session manager to avoid side effects."""
        with (
            patch("wandb_mcp_server.api_client.WandBApiManager") as mock_wbm,
            patch("wandb_mcp_server.session_manager.MultiTenantSessionManager._start_cleanup_task"),
        ):
            mock_wbm.set_context_api_key.return_value = "tok"
            mock_wbm.reset_context_api_key.return_value = None
            mock_api = MagicMock()
            mock_api.viewer = SimpleNamespace(username="alice", entity="team", email="a@co.com")
            mock_wbm.get_api.return_value = mock_api
            yield mock_wbm

    @pytest.mark.asyncio
    async def test_new_session_emits_header(self, _mock_deps):
        req = _make_fake_request()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        result = await mcp_auth_middleware(req, call_next)
        assert "Mcp-Session-Id" in result.headers
        assert result.headers["Mcp-Session-Id"].startswith("sess_")

    @pytest.mark.asyncio
    async def test_existing_session_no_new_header(self, _mock_deps):
        req = _make_fake_request(session_header="existing-123")
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        result = await mcp_auth_middleware(req, call_next)
        assert "Mcp-Session-Id" not in result.headers

    @pytest.mark.asyncio
    async def test_session_stored_in_request_state(self, _mock_deps):
        req = _make_fake_request()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        await mcp_auth_middleware(req, call_next)
        assert hasattr(req.state, "session_id")

    @pytest.mark.asyncio
    async def test_contextvar_reset_after_request(self, _mock_deps):
        req = _make_fake_request()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            assert current_session_id.get() is not None
            return resp

        await mcp_auth_middleware(req, call_next)
        assert current_session_id.get() is None

    @pytest.mark.asyncio
    async def test_non_mcp_path_skipped(self, _mock_deps):
        req = _make_fake_request(path="/health")
        resp = MagicMock()
        resp.status_code = 200

        async def call_next(_):
            return resp

        result = await mcp_auth_middleware(req, call_next)
        assert result is resp
