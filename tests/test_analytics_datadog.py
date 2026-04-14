"""Unit tests for analytics_datadog.py -- Datadog HTTP Logs Intake forwarder.

Tests the mapper (severity, attributes, PII exclusion), the gated forwarder
(off, enabled, missing key), the singleton lifecycle, and E2E integration
where AnalyticsTracker._emit() feeds the DatadogForwarder.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.analytics import AnalyticsTracker, reset_analytics_tracker
from wandb_mcp_server.analytics_datadog import (
    DatadogForwarder,
    get_datadog_forwarder,
    map_to_datadog_log,
    reset_datadog_forwarder,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_datadog_forwarder()
    reset_analytics_tracker()
    yield
    reset_datadog_forwarder()
    reset_analytics_tracker()


def _make_event(event_type: str, **overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "schema_version": "1.0",
        "event_type": event_type,
        "timestamp": "2026-04-13T12:00:00+00:00",
        "user_id": "alice",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# map_to_datadog_log -- severity mapping
# ---------------------------------------------------------------------------


class TestSeverityMapping:
    """Datadog status field must reflect operational severity."""

    def test_success_tool_call_is_info(self):
        event = _make_event("tool_call", tool_name="query_traces", success=True)
        entry = map_to_datadog_log(event, dd_env="staging", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "info"

    def test_failed_tool_call_is_error(self):
        event = _make_event("tool_call", tool_name="query_traces", success=False)
        entry = map_to_datadog_log(event, dd_env="staging", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "error"

    def test_tool_call_with_error_string_is_error(self):
        event = _make_event("tool_call", tool_name="create_report", success=False, error="CommError: 404 Not Found")
        entry = map_to_datadog_log(event, dd_env="staging", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "error"

    def test_request_200_is_info(self):
        event = _make_event("request", method="POST", path="/mcp", status_code=200)
        entry = map_to_datadog_log(event, dd_env="prod", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "info"

    def test_request_401_is_warn(self):
        event = _make_event("request", method="POST", path="/mcp", status_code=401)
        entry = map_to_datadog_log(event, dd_env="prod", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "warn"

    def test_request_500_is_error(self):
        event = _make_event("request", method="POST", path="/mcp", status_code=500)
        entry = map_to_datadog_log(event, dd_env="prod", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "error"

    def test_session_is_info(self):
        event = _make_event("user_session", session_id="s1")
        entry = map_to_datadog_log(event, dd_env="prod", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "info"

    def test_error_field_overrides_success_true(self):
        event = _make_event("tool_call", tool_name="x", success=True, error="ValueError: bad")
        entry = map_to_datadog_log(event, dd_env="prod", dd_version="0.3.0", dd_service="mcp")
        assert entry["status"] == "error"


# ---------------------------------------------------------------------------
# map_to_datadog_log -- DD reserved attributes
# ---------------------------------------------------------------------------


class TestDatadogAttributes:
    """Structured attributes for automatic DD faceting and dashboards."""

    def test_duration_in_nanoseconds(self):
        event = _make_event("tool_call", tool_name="query_traces", success=True, duration_ms=245.5)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["attributes"]["duration"] == 245_500_000

    def test_duration_absent_when_none(self):
        event = _make_event("tool_call", tool_name="x", success=True)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "duration" not in entry["attributes"]

    def test_http_attributes_on_request(self):
        event = _make_event("request", method="POST", path="/mcp", status_code=200, duration_ms=50)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        http = entry["attributes"]["http"]
        assert http["status_code"] == 200
        assert http["method"] == "POST"
        assert http["url_details"]["path"] == "/mcp"

    def test_http_attributes_absent_for_tool_call(self):
        event = _make_event("tool_call", tool_name="x", success=True)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "http" not in entry["attributes"]

    def test_error_kind_and_message_parsed(self):
        event = _make_event("tool_call", tool_name="create_report", success=False, error="CommError: 404 Not Found")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        err = entry["attributes"]["error"]
        assert err["kind"] == "CommError"
        assert err["message"] == "404 Not Found"

    def test_error_without_colon_uses_default_kind(self):
        event = _make_event("tool_call", tool_name="x", success=False, error="something went wrong")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["attributes"]["error"]["kind"] == "Error"
        assert entry["attributes"]["error"]["message"] == "something went wrong"

    def test_usr_id_present(self):
        event = _make_event("tool_call", tool_name="x", success=True, user_id="alice")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["attributes"]["usr"]["id"] == "alice"

    def test_usr_id_absent_when_none(self):
        event = _make_event("tool_call", tool_name="x", success=True, user_id=None)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "usr" not in entry["attributes"]

    def test_tool_attributes(self):
        event = _make_event("tool_call", tool_name="count_traces", success=True)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        tool = entry["attributes"]["tool"]
        assert tool["name"] == "count_traces"
        assert tool["success"] is True

    def test_session_id_forwarded(self):
        event = _make_event("tool_call", tool_name="x", success=True, session_id="sess_abc")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["attributes"]["session_id"] == "sess_abc"


# ---------------------------------------------------------------------------
# map_to_datadog_log -- PII exclusion
# ---------------------------------------------------------------------------


class TestPIIExclusion:
    """Datadog must NOT receive params, api_key_hash, metadata, or email_domain."""

    def test_params_excluded(self):
        event = _make_event(
            "tool_call",
            tool_name="query_traces",
            success=True,
            params={"entity": "team", "project": "proj"},
        )
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "team" not in entry["attributes"]
        assert "params" not in entry["attributes"]

    def test_api_key_hash_excluded(self):
        event = _make_event("user_session", session_id="s1", api_key_hash="abcdef1234567890")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "api_key_hash" not in entry["attributes"]

    def test_metadata_excluded(self):
        event = _make_event("user_session", session_id="s1", metadata={"client": "cursor"})
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "metadata" not in entry["attributes"]

    def test_email_domain_excluded(self):
        event = _make_event("tool_call", tool_name="x", success=True, email_domain="acme.com")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "email_domain" not in entry["attributes"]


# ---------------------------------------------------------------------------
# map_to_datadog_log -- tags and top-level fields
# ---------------------------------------------------------------------------


class TestTagsAndTopLevel:
    """Verify ddtags, ddsource, hostname, service, message."""

    def test_tags_contain_env_service_version(self):
        event = _make_event("tool_call", tool_name="count_traces", success=True)
        entry = map_to_datadog_log(event, dd_env="staging", dd_version="0.3.0", dd_service="wandb-mcp-server")
        tags = entry["ddtags"]
        assert "env:staging" in tags
        assert "service:wandb-mcp-server" in tags
        assert "version:0.3.0" in tags
        assert "event_type:tool_call" in tags
        assert "tool_name:count_traces" in tags
        assert "success:true" in tags

    def test_ddsource_is_wandb(self):
        event = _make_event("tool_call", tool_name="x", success=True)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["ddsource"] == "wandb-mcp-server"

    @patch.dict("os.environ", {"K_REVISION": "mcp-server-00042-abc"})
    def test_hostname_from_k_revision(self):
        event = _make_event("tool_call", tool_name="x", success=True)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert entry["hostname"] == "mcp-server-00042-abc"

    def test_message_is_human_readable(self):
        event = _make_event("tool_call", tool_name="query_traces", success=True, duration_ms=100)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        msg = entry["message"]
        assert "mcp.tool_call.query_traces" in msg
        assert "100ms" in msg

    def test_message_for_error_tool_call(self):
        event = _make_event("tool_call", tool_name="create_report", success=False, error="CommError: timeout")
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "ERROR" in entry["message"]

    def test_message_for_request(self):
        event = _make_event("request", method="POST", path="/mcp", status_code=200, duration_ms=50)
        entry = map_to_datadog_log(event, dd_env="s", dd_version="v", dd_service="svc")
        assert "POST /mcp -> 200" in entry["message"]


# ---------------------------------------------------------------------------
# DatadogForwarder -- gating and modes
# ---------------------------------------------------------------------------


class TestDatadogForwarder:
    """Forwarder gating via env vars."""

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "false"}, clear=False)
    def test_disabled_by_default(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        assert not fwd.enabled

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "abc123"}, clear=False)
    def test_enabled_with_key(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        assert fwd.enabled

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": ""}, clear=False)
    def test_disabled_without_key(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        assert not fwd.enabled

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "false"}, clear=False)
    def test_forward_returns_none_when_disabled(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        result = fwd.forward(_make_event("tool_call", tool_name="x", success=True))
        assert result is None
        assert fwd.get_forwarded_payloads() == []

    @patch.dict(
        "os.environ",
        {
            "MCP_DATADOG_FORWARD": "true",
            "DD_API_KEY": "testkey",
            "DD_ENV": "test",
            "DD_VERSION": "0.3.0",
            "DD_SERVICE": "test-mcp",
        },
        clear=False,
    )
    def test_forward_builds_entry_and_records(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        with patch.object(fwd, "_post"):
            event = _make_event("tool_call", tool_name="query_traces", success=True, duration_ms=100)
            entry = fwd.forward(event)
            assert entry is not None
            assert entry["status"] == "info"
            assert entry["service"] == "test-mcp"
            assert len(fwd.get_forwarded_payloads()) == 1

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey"},
        clear=False,
    )
    def test_post_called_in_background(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        mock_post = MagicMock()
        with patch.object(fwd, "_post", mock_post):
            event = _make_event("tool_call", tool_name="x", success=True)
            fwd.forward(event)
            fwd._executor.shutdown(wait=True)
            mock_post.assert_called_once()

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey"},
        clear=False,
    )
    def test_clear_forwarded_payloads(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        with patch.object(fwd, "_post"):
            fwd.forward(_make_event("tool_call", tool_name="x", success=True))
            assert len(fwd.get_forwarded_payloads()) == 1
            fwd.clear_forwarded_payloads()
            assert fwd.get_forwarded_payloads() == []


# ---------------------------------------------------------------------------
# DatadogForwarder._post -- HTTP behavior
# ---------------------------------------------------------------------------


class TestDatadogPost:
    """Verify HTTP POST behavior with mocked requests."""

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey123", "DD_SITE": "us5.datadoghq.com"},
        clear=False,
    )
    def test_post_sends_to_correct_url(self):
        reset_datadog_forwarder()
        fwd = DatadogForwarder()
        assert fwd._intake_url == "https://http-intake.logs.us5.datadoghq.com/api/v2/logs"

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey123"},
        clear=False,
    )
    @patch("wandb_mcp_server.analytics_datadog._build_retry_session")
    def test_post_uses_dd_api_key_header(self, mock_build_session):
        reset_datadog_forwarder()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_session.post.return_value = mock_resp
        mock_build_session.return_value = mock_session

        fwd = DatadogForwarder()
        entry = {"message": "test", "status": "info"}
        fwd._post(entry)

        call_args = mock_session.post.call_args
        assert call_args.kwargs["headers"]["DD-API-KEY"] == "testkey123"
        assert call_args.kwargs["json"] == [entry]

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey123"},
        clear=False,
    )
    @patch("wandb_mcp_server.analytics_datadog._build_retry_session")
    def test_post_warns_on_non_success_status(self, mock_build_session):
        reset_datadog_forwarder()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_session.post.return_value = mock_resp
        mock_build_session.return_value = mock_session

        fwd = DatadogForwarder()
        with patch("wandb_mcp_server.analytics_datadog.logger") as mock_logger:
            fwd._post({"message": "test"})
            mock_logger.warning.assert_called_once()

    @patch.dict(
        "os.environ",
        {"MCP_DATADOG_FORWARD": "true", "DD_API_KEY": "testkey123"},
        clear=False,
    )
    @patch("wandb_mcp_server.analytics_datadog._build_retry_session")
    def test_post_handles_network_error(self, mock_build_session):
        reset_datadog_forwarder()
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("network down")
        mock_build_session.return_value = mock_session

        fwd = DatadogForwarder()
        with patch("wandb_mcp_server.analytics_datadog.logger") as mock_logger:
            fwd._post({"message": "test"})
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    """get_datadog_forwarder / reset_datadog_forwarder."""

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "false"}, clear=False)
    def test_singleton_returns_same_instance(self):
        reset_datadog_forwarder()
        a = get_datadog_forwarder()
        b = get_datadog_forwarder()
        assert a is b

    @patch.dict("os.environ", {"MCP_DATADOG_FORWARD": "false"}, clear=False)
    def test_reset_creates_new_instance(self):
        reset_datadog_forwarder()
        a = get_datadog_forwarder()
        reset_datadog_forwarder()
        b = get_datadog_forwarder()
        assert a is not b


# ---------------------------------------------------------------------------
# E2E: AnalyticsTracker -> DatadogForwarder
# ---------------------------------------------------------------------------


class TestE2ETrackerToDatadog:
    """End-to-end: tracker._emit() feeds the DatadogForwarder."""

    @patch.dict(
        "os.environ",
        {
            "MCP_ANALYTICS_DISABLED": "false",
            "MCP_DATADOG_FORWARD": "true",
            "DD_API_KEY": "testkey",
            "DD_ENV": "test",
            "DD_SERVICE": "test-mcp",
            "DD_VERSION": "1.0.0",
        },
        clear=False,
    )
    def test_tool_call_reaches_datadog(self):
        reset_datadog_forwarder()
        reset_analytics_tracker()
        fwd = get_datadog_forwarder()
        with patch.object(fwd, "_post"):
            tracker = AnalyticsTracker(enabled=True)
            tracker.track_tool_call(
                tool_name="query_traces",
                session_id="sess_1",
                viewer_info="alice",
                params={"entity": "team"},
                success=True,
                duration_ms=150.5,
            )
            payloads = fwd.get_forwarded_payloads()
            assert len(payloads) == 1
            entry = payloads[0]
            assert entry["status"] == "info"
            assert entry["attributes"]["tool"]["name"] == "query_traces"
            assert entry["attributes"]["duration"] == 150_500_000
            assert "params" not in entry["attributes"]

    @patch.dict(
        "os.environ",
        {
            "MCP_ANALYTICS_DISABLED": "false",
            "MCP_DATADOG_FORWARD": "true",
            "DD_API_KEY": "testkey",
        },
        clear=False,
    )
    def test_failed_tool_call_reaches_datadog_as_error(self):
        reset_datadog_forwarder()
        reset_analytics_tracker()
        fwd = get_datadog_forwarder()
        with patch.object(fwd, "_post"):
            tracker = AnalyticsTracker(enabled=True)
            tracker.track_tool_call(
                tool_name="create_report",
                session_id="sess_2",
                viewer_info="bob",
                success=False,
                error="CommError: 404 Not Found",
                duration_ms=1823,
            )
            payloads = fwd.get_forwarded_payloads()
            assert len(payloads) == 1
            entry = payloads[0]
            assert entry["status"] == "error"
            assert entry["attributes"]["error"]["kind"] == "CommError"
            assert entry["attributes"]["tool"]["success"] is False

    @patch.dict(
        "os.environ",
        {
            "MCP_ANALYTICS_DISABLED": "false",
            "MCP_DATADOG_FORWARD": "true",
            "DD_API_KEY": "testkey",
        },
        clear=False,
    )
    def test_request_500_reaches_datadog_as_error(self):
        reset_datadog_forwarder()
        reset_analytics_tracker()
        fwd = get_datadog_forwarder()
        with patch.object(fwd, "_post"):
            tracker = AnalyticsTracker(enabled=True)
            tracker.track_request(
                request_id="req_1",
                session_id="sess_3",
                method="POST",
                path="/mcp",
                status_code=500,
                duration_ms=5000,
                user_id="carol",
            )
            payloads = fwd.get_forwarded_payloads()
            assert len(payloads) == 1
            entry = payloads[0]
            assert entry["status"] == "error"
            assert entry["attributes"]["http"]["status_code"] == 500
            assert entry["attributes"]["usr"]["id"] == "carol"
