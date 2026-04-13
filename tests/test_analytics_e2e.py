"""E2E analytics tests: real tool calls -> Segment + Datadog payloads.

Exercises the full pipeline from track_tool_execution through _emit() to
both SegmentForwarder (dry-run) and DatadogForwarder (dry-run), asserting
that success/failure/duration propagate correctly to each sink.

Uses mocked WandBApiManager so no real API calls are made, but the analytics
pipeline is exercised end-to-end through the real code paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.analytics import reset_analytics_tracker
from wandb_mcp_server.analytics_datadog import get_datadog_forwarder, reset_datadog_forwarder
from wandb_mcp_server.analytics_segment import get_segment_forwarder, reset_segment_forwarder


@pytest.fixture(autouse=True)
def _reset_all():
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()
    yield
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()


@pytest.fixture()
def _enable_analytics(monkeypatch):
    """Enable both forwarders in dry-run / capture mode."""
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    monkeypatch.setenv("MCP_SEGMENT_DRY_RUN", "true")
    monkeypatch.setenv("MCP_DATADOG_FORWARD", "true")
    monkeypatch.setenv("DD_API_KEY", "test-key-e2e")
    monkeypatch.setenv("DD_ENV", "test")
    monkeypatch.setenv("DD_SERVICE", "test-mcp")
    monkeypatch.setenv("DD_VERSION", "0.3.1")
    reset_analytics_tracker()
    reset_segment_forwarder()
    reset_datadog_forwarder()


def _mock_viewer():
    v = MagicMock()
    v.username = "testuser"
    v.entity = "testorg"
    v.email = "test@wandb.com"
    return v


# ---------------------------------------------------------------------------
# Successful tool call -> both sinks
# ---------------------------------------------------------------------------


class TestSuccessfulToolCall:
    """A tool call that completes successfully should record success, duration, and no error."""

    @pytest.mark.usefixtures("_enable_analytics")
    def test_success_reaches_segment(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("query_traces", viewer, {"entity": "org", "project": "proj"}):
                pass  # tool "succeeds" instantly

        seg_fwd = get_segment_forwarder()
        seg_payloads = seg_fwd.get_forwarded_payloads()
        assert len(seg_payloads) == 1
        seg = seg_payloads[0]
        assert seg["event"] == "mcp_server.tool_call"
        assert seg["properties"]["tool_name"] == "query_traces"
        assert seg["properties"]["success"] is True
        assert seg["properties"]["error"] is None
        assert seg["properties"]["duration_ms"] is not None
        assert seg["properties"]["duration_ms"] >= 0
        assert seg["userId"] == "testuser"

    @pytest.mark.usefixtures("_enable_analytics")
    def test_success_reaches_datadog(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("query_traces", viewer, {"entity": "org", "project": "proj"}):
                pass

        dd_payloads = dd_fwd.get_forwarded_payloads()
        assert len(dd_payloads) == 1
        dd = dd_payloads[0]
        assert dd["status"] == "info"
        assert dd["attributes"]["tool"]["name"] == "query_traces"
        assert dd["attributes"]["tool"]["success"] is True
        assert "error" not in dd["attributes"]
        assert dd["attributes"]["duration"] >= 0
        assert dd["attributes"]["usr"]["id"] == "testuser"
        assert "params" not in dd["attributes"]

    @pytest.mark.usefixtures("_enable_analytics")
    def test_success_segment_excludes_email_domain(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("count_traces", viewer, {"entity": "org"}):
                pass

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert "email_domain" not in seg["properties"]


# ---------------------------------------------------------------------------
# Failed tool call (exception) -> both sinks
# ---------------------------------------------------------------------------


class TestFailedToolCallException:
    """A tool that raises should record success=False, the error string, and duration."""

    @pytest.mark.usefixtures("_enable_analytics")
    def test_exception_reaches_segment(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with pytest.raises(ValueError, match="bad query"):
                with track_tool_execution("query_traces", viewer, {"entity": "org"}):
                    raise ValueError("bad query")

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert seg["properties"]["success"] is False
        assert "ValueError" in seg["properties"]["error"]
        assert "bad query" in seg["properties"]["error"]
        assert seg["properties"]["duration_ms"] is not None

    @pytest.mark.usefixtures("_enable_analytics")
    def test_exception_reaches_datadog_as_error(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with pytest.raises(RuntimeError):
                with track_tool_execution("create_report", viewer, {"title": "test"}):
                    raise RuntimeError("CommError: 404 Not Found")

        dd = dd_fwd.get_forwarded_payloads()[0]
        assert dd["status"] == "error"
        assert dd["attributes"]["error"]["kind"] == "RuntimeError"
        assert "CommError" in dd["attributes"]["error"]["message"]
        assert dd["attributes"]["tool"]["success"] is False
        assert dd["attributes"]["duration"] >= 0


# ---------------------------------------------------------------------------
# Failed tool call (mark_error, no raise) -> both sinks
# ---------------------------------------------------------------------------


class TestFailedToolCallMarkError:
    """A tool that catches and returns an error dict should use ctx.mark_error()."""

    @pytest.mark.usefixtures("_enable_analytics")
    def test_mark_error_reaches_segment(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("list_registries", viewer, {"organization": "org"}) as ctx:
                ctx.mark_error("PermissionError: access denied")

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert seg["properties"]["success"] is False
        assert "PermissionError" in seg["properties"]["error"]
        assert seg["properties"]["duration_ms"] is not None

    @pytest.mark.usefixtures("_enable_analytics")
    def test_mark_error_reaches_datadog(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("list_registries", viewer, {"organization": "org"}) as ctx:
                ctx.mark_error("PermissionError: access denied")

        dd = dd_fwd.get_forwarded_payloads()[0]
        assert dd["status"] == "error"
        assert dd["attributes"]["error"]["kind"] == "PermissionError"
        assert dd["attributes"]["error"]["message"] == "access denied"
        assert dd["attributes"]["tool"]["success"] is False


# ---------------------------------------------------------------------------
# Duration tracking
# ---------------------------------------------------------------------------


class TestDurationTracking:
    """Duration should reflect actual execution time, not zero."""

    @pytest.mark.usefixtures("_enable_analytics")
    def test_duration_is_nonzero_for_slow_tool(self):
        import time

        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("slow_tool", viewer, {}):
                time.sleep(0.05)

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert seg["properties"]["duration_ms"] >= 40

        dd = dd_fwd.get_forwarded_payloads()[0]
        assert dd["attributes"]["duration"] >= 40_000_000  # 40ms in nanoseconds


# ---------------------------------------------------------------------------
# PII and data separation between sinks
# ---------------------------------------------------------------------------


class TestDataSeparation:
    """Segment and Datadog must receive different data per their purposes."""

    @pytest.mark.usefixtures("_enable_analytics")
    def test_segment_gets_params_datadog_does_not(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()
        params = {"entity_name": "wandb-smle", "project_name": "email-agent", "limit": 10}

        with patch.object(dd_fwd, "_post"):
            with track_tool_execution("query_traces", viewer, params):
                pass

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert "params" in seg["properties"]
        assert seg["properties"]["params"]["entity_name"] == "wandb-smle"

        dd = dd_fwd.get_forwarded_payloads()[0]
        assert "params" not in dd["attributes"]
        assert "wandb-smle" not in str(dd["attributes"])

    @pytest.mark.usefixtures("_enable_analytics")
    def test_datadog_has_structured_severity_segment_does_not(self):
        from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

        viewer = _mock_viewer()
        dd_fwd = get_datadog_forwarder()

        with patch.object(dd_fwd, "_post"):
            with pytest.raises(Exception):
                with track_tool_execution("create_report", viewer, {}):
                    raise Exception("timeout")

        dd = dd_fwd.get_forwarded_payloads()[0]
        assert dd["status"] == "error"

        seg = get_segment_forwarder().get_forwarded_payloads()[0]
        assert "status" not in seg
        assert seg["properties"]["success"] is False
