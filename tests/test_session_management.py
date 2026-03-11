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
    SessionCapacityError,
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


# ---------------------------------------------------------------------------
# Session fixation prevention (cross-tenant session ID reuse)
# ---------------------------------------------------------------------------


class TestSessionFixationPrevention:
    """Verify that a client cannot hijack another tenant's session ID.

    Scenario: User A authenticates and gets sess_aaa. User B authenticates
    with a different API key but sends Mcp-Session-Id: sess_aaa. The
    middleware must detect the mismatch and issue a fresh session for B,
    never allowing B's actions to correlate with A's session.
    """

    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"}):
            yield

    @pytest.fixture()
    def _mock_deps(self):
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
    async def test_stolen_session_id_replaced(self, _mock_deps):
        """User B sending User A's session ID gets a fresh session, not A's."""
        tenant_a_key = "key_AAAA_1234567890_abcdefgh"
        tenant_b_key = "key_BBBB_1234567890_xyzwvuts"
        stolen_session = "sess_aaa_owned_by_tenant_a"

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        req_a = _make_fake_request(api_key=tenant_a_key, session_header=stolen_session)
        await mcp_auth_middleware(req_a, call_next)
        assert req_a.state.session_id == stolen_session

        req_b = _make_fake_request(api_key=tenant_b_key, session_header=stolen_session)
        resp_b = MagicMock()
        resp_b.status_code = 200
        resp_b.headers = {}

        async def call_next_b(_):
            return resp_b

        result_b = await mcp_auth_middleware(req_b, call_next_b)

        assert req_b.state.session_id != stolen_session
        assert req_b.state.session_id.startswith("sess_")
        assert "Mcp-Session-Id" in result_b.headers
        assert result_b.headers["Mcp-Session-Id"] != stolen_session

    @pytest.mark.asyncio
    async def test_stolen_session_contextvar_not_leaked(self, _mock_deps):
        """The contextvar during B's request must hold B's fresh session, not A's."""
        tenant_a_key = "key_AAAA_1234567890_abcdefgh"
        tenant_b_key = "key_BBBB_1234567890_xyzwvuts"
        stolen_session = "sess_aaa_for_leak_test"

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        await mcp_auth_middleware(
            _make_fake_request(api_key=tenant_a_key, session_header=stolen_session),
            call_next,
        )

        captured_session_during_b = None

        async def call_next_capture(_):
            nonlocal captured_session_during_b
            captured_session_during_b = current_session_id.get()
            r = MagicMock()
            r.status_code = 200
            r.headers = {}
            return r

        await mcp_auth_middleware(
            _make_fake_request(api_key=tenant_b_key, session_header=stolen_session),
            call_next_capture,
        )

        assert captured_session_during_b is not None
        assert captured_session_during_b != stolen_session
        assert captured_session_during_b.startswith("sess_")

    @pytest.mark.asyncio
    async def test_legitimate_session_reuse_works(self, _mock_deps):
        """Same tenant re-sending their own session ID should keep it."""
        tenant_key = "key_SAME_1234567890_abcdefgh"
        session = "sess_my_own_session"

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        await mcp_auth_middleware(
            _make_fake_request(api_key=tenant_key, session_header=session),
            call_next,
        )

        req2 = _make_fake_request(api_key=tenant_key, session_header=session)
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.headers = {}

        async def call_next2(_):
            return resp2

        result2 = await mcp_auth_middleware(req2, call_next2)
        assert req2.state.session_id == session
        assert "Mcp-Session-Id" not in result2.headers


# ---------------------------------------------------------------------------
# SessionCapacityError and per-key cap enforcement
# ---------------------------------------------------------------------------


class TestSessionCapacityError:
    @pytest.fixture()
    def mgr(self):
        with patch("wandb_mcp_server.session_manager.MultiTenantSessionManager._start_cleanup_task"):
            return MultiTenantSessionManager(
                session_ttl_seconds=300,
                max_sessions_per_key=2,
                enable_hmac_sha256_sessions=False,
            )

    def test_capacity_raises_correct_type(self, mgr):
        """When cleanup cannot free enough sessions, SessionCapacityError is raised."""
        key = "fake_key_1234567890_abcdefgh"
        mgr.create_session(key, session_id="s1")
        mgr.create_session(key, session_id="s2")
        with patch.object(mgr, "_cleanup_api_key_sessions"):
            with pytest.raises(SessionCapacityError, match="Maximum concurrent sessions"):
                mgr.create_session(key, session_id="s3")

    def test_capacity_error_is_subclass_of_value_error(self):
        assert issubclass(SessionCapacityError, ValueError)

    def test_mismatch_still_raises_plain_value_error(self, mgr):
        """API key mismatch must remain a plain ValueError, not SessionCapacityError."""
        mgr.create_session("fake_key_1234567890_abcdefgh", session_id="shared")
        with pytest.raises(ValueError, match="mismatch") as exc_info:
            mgr.create_session("other_key_1234567890_different", session_id="shared")
        assert not isinstance(exc_info.value, SessionCapacityError)


class TestSessionCapacityMiddleware:
    """Auth middleware returns 429 when session capacity is exhausted."""

    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"}):
            yield

    @pytest.fixture()
    def _mock_deps(self):
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
    async def test_capacity_returns_429(self, _mock_deps):
        call_next_called = False

        async def call_next(_):
            nonlocal call_next_called
            call_next_called = True
            r = MagicMock()
            r.status_code = 200
            r.headers = {}
            return r

        with patch("wandb_mcp_server.session_manager.get_session_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_mgr.create_session.side_effect = SessionCapacityError("cap exceeded")
            mock_get_mgr.return_value = mock_mgr

            req = _make_fake_request()
            result = await mcp_auth_middleware(req, call_next)

        assert result.status_code == 429
        assert not call_next_called


# ---------------------------------------------------------------------------
# Downstream crash analytics
# ---------------------------------------------------------------------------


class TestDownstreamCrashAnalytics:
    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "false"}):
            yield

    @pytest.fixture()
    def _mock_deps(self):
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
    async def test_crash_emits_500_analytics(self, _mock_deps):
        async def crashing_call_next(_):
            raise RuntimeError("handler exploded")

        with patch("wandb_mcp_server.auth._track_request_event") as mock_track:
            req = _make_fake_request()
            with pytest.raises(RuntimeError, match="handler exploded"):
                await mcp_auth_middleware(req, crashing_call_next)

            mock_track.assert_called_once()
            call_args = mock_track.call_args
            assert call_args[0][4] == 500  # status_code positional arg


# ---------------------------------------------------------------------------
# Contextvar reset correctness after mismatch recovery
# ---------------------------------------------------------------------------


class TestContextvarResetAfterMismatch:
    @pytest.fixture(autouse=True)
    def _env(self):
        with patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"}):
            yield

    @pytest.fixture()
    def _mock_deps(self):
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
    async def test_contextvar_is_none_after_mismatch_request(self, _mock_deps):
        """After mismatch recovery, contextvar must be None, not the stolen session."""
        tenant_a_key = "key_AAAA_1234567890_abcdefgh"
        tenant_b_key = "key_BBBB_1234567890_xyzwvuts"
        stolen = "sess_stolen_id_from_a"

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}

        async def call_next(_):
            return resp

        req_a = _make_fake_request(api_key=tenant_a_key, session_header=stolen)
        await mcp_auth_middleware(req_a, call_next)

        req_b = _make_fake_request(api_key=tenant_b_key, session_header=stolen)
        resp_b = MagicMock()
        resp_b.status_code = 200
        resp_b.headers = {}

        async def call_next_b(_):
            return resp_b

        await mcp_auth_middleware(req_b, call_next_b)

        assert current_session_id.get() is None
