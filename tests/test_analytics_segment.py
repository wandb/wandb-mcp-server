"""Unit tests for analytics_segment.py -- Gorilla /analytics/t compatibility.

Tests the pure mapper, the gated forwarder (dry-run, live, off),
the singleton lifecycle, and the end-to-end integration where
AnalyticsTracker._emit() automatically feeds the SegmentForwarder.
"""

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.analytics import AnalyticsTracker, reset_analytics_tracker
from wandb_mcp_server.analytics_segment import (
    SEGMENT_EVENT_PREFIX,
    SegmentForwarder,
    get_segment_forwarder,
    map_to_segment_track,
    reset_segment_forwarder,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_segment_forwarder()
    reset_analytics_tracker()
    yield
    reset_segment_forwarder()
    reset_analytics_tracker()


# ---------------------------------------------------------------------------
# map_to_segment_track (pure mapper)
# ---------------------------------------------------------------------------


class TestMapToSegmentTrack:
    """Pure mapper function tests -- no I/O."""

    def _make_event(self, event_type: str, **overrides) -> Dict[str, Any]:
        base = {
            "schema_version": "1.0",
            "event_type": event_type,
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "alice",
        }
        base.update(overrides)
        return base

    def test_tool_call_basic(self):
        event = self._make_event(
            "tool_call",
            session_id="s1",
            tool_name="query_wandb_gql",
            params={"entity": "team"},
            success=True,
        )
        result = map_to_segment_track(event)
        assert result is not None
        assert result["userId"] == "alice"
        assert result["event"] == f"{SEGMENT_EVENT_PREFIX}.tool_call"
        assert result["properties"]["tool_name"] == "query_wandb_gql"
        assert result["properties"]["source"] == "wandb-mcp-server"
        assert result["properties"]["schema_version"] == "1.0"

    def test_tool_call_preserves_timestamp(self):
        event = self._make_event("tool_call", tool_name="t")
        result = map_to_segment_track(event)
        assert "timestamp" in result
        assert "2026-02-27" in result["timestamp"]

    def test_user_session_basic(self):
        event = self._make_event(
            "user_session",
            session_id="sess",
            email_domain="wandb.com",
            api_key_hash="abcd1234",
        )
        result = map_to_segment_track(event)
        assert result["event"] == f"{SEGMENT_EVENT_PREFIX}.session_start"
        assert result["properties"]["email_domain"] == "wandb.com"

    def test_request_basic(self):
        event = self._make_event(
            "request",
            request_id="r1",
            method="POST",
            path="/mcp/sse",
            status_code=200,
        )
        result = map_to_segment_track(event)
        assert result["event"] == f"{SEGMENT_EVENT_PREFIX}.http_request"
        assert result["properties"]["method"] == "POST"

    def test_returns_none_for_unknown_event_type(self):
        event = self._make_event("unknown_type")
        assert map_to_segment_track(event) is None

    def test_returns_none_when_user_id_empty(self):
        event = self._make_event("tool_call", user_id="")
        assert map_to_segment_track(event) is None

    def test_returns_none_when_user_id_missing(self):
        event = {"event_type": "tool_call", "timestamp": "t"}
        assert map_to_segment_track(event) is None

    def test_invalid_timestamp_omitted(self):
        event = self._make_event("tool_call", timestamp="not-a-date", tool_name="t")
        result = map_to_segment_track(event)
        assert result is not None
        assert "timestamp" not in result

    def test_extra_fields_not_leaked(self):
        event = self._make_event(
            "tool_call",
            tool_name="t",
            email_domain="evil.com",
        )
        result = map_to_segment_track(event)
        assert "email_domain" not in result["properties"]


# ---------------------------------------------------------------------------
# SegmentForwarder unit tests
# ---------------------------------------------------------------------------


class TestSegmentForwarder:
    def test_disabled_by_default(self):
        f = SegmentForwarder()
        assert f.enabled is False
        assert f.forward({"event_type": "tool_call", "user_id": "x"}) is None

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_dry_run_returns_payload(self):
        f = SegmentForwarder()
        assert f.dry_run is True
        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "bob",
            "tool_name": "query_weave",
            "success": True,
        }
        result = f.forward(event)
        assert result is not None
        assert result["userId"] == "bob"

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_dry_run_skips_unmappable(self):
        f = SegmentForwarder()
        result = f.forward({"event_type": "unknown", "user_id": "x"})
        assert result is None

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_dry_run_records_forwarded_payloads(self):
        f = SegmentForwarder()
        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "carol",
            "tool_name": "t",
            "success": True,
        }
        f.forward(event)
        payloads = f.get_forwarded_payloads()
        assert len(payloads) == 1
        assert payloads[0]["userId"] == "carol"

    @patch.dict("os.environ", {"MCP_SEGMENT_FORWARD": "true"})
    def test_live_dispatches_post_in_background(self):
        f = SegmentForwarder(base_url="https://api.wandb.test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        f._session = mock_session

        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "carol",
            "tool_name": "t",
            "success": True,
        }
        result = f.forward(event)
        assert result is not None

        time.sleep(0.2)
        mock_session.post.assert_called_once()
        url_arg = mock_session.post.call_args.args[0]
        assert "analytics/t" in url_arg

    @patch.dict("os.environ", {"MCP_SEGMENT_FORWARD": "true"})
    def test_live_handles_post_failure_gracefully(self):
        f = SegmentForwarder(base_url="https://api.wandb.test")
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("network")
        f._session = mock_session

        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "dan",
            "tool_name": "t",
            "success": True,
        }
        result = f.forward(event)
        assert result is not None

        time.sleep(0.2)
        mock_session.post.assert_called_once()

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_clear_forwarded_payloads(self):
        f = SegmentForwarder()
        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "eve",
            "tool_name": "t",
            "success": True,
        }
        f.forward(event)
        assert len(f.get_forwarded_payloads()) == 1
        f.clear_forwarded_payloads()
        assert len(f.get_forwarded_payloads()) == 0


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_returns_same_instance(self):
        assert get_segment_forwarder() is get_segment_forwarder()

    def test_reset_creates_new_instance(self):
        f1 = get_segment_forwarder()
        reset_segment_forwarder()
        assert get_segment_forwarder() is not f1


# ---------------------------------------------------------------------------
# End-to-end: AnalyticsTracker._emit() -> SegmentForwarder
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """Verify that events emitted by AnalyticsTracker flow through to the
    SegmentForwarder when it is enabled."""

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_tracker_tool_call_reaches_forwarder(self):
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()
        assert forwarder.enabled is True

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_tool_call(
            tool_name="query_weave",
            session_id="s1",
            viewer_info="alice",
            params={"project": "test"},
            success=True,
        )

        payloads = forwarder.get_forwarded_payloads()
        assert len(payloads) == 1
        p = payloads[0]
        assert p["event"] == f"{SEGMENT_EVENT_PREFIX}.tool_call"
        assert p["properties"]["tool_name"] == "query_weave"

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_tracker_user_session_reaches_forwarder(self):
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()

        from types import SimpleNamespace

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_user_session(
            session_id="sess-1",
            viewer_info=SimpleNamespace(username="bob", email="b@co.com"),
            api_key_hash="a" * 64,
        )

        payloads = forwarder.get_forwarded_payloads()
        assert len(payloads) == 1
        assert payloads[0]["userId"] == "bob"
        assert payloads[0]["event"] == f"{SEGMENT_EVENT_PREFIX}.session_start"

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_tracker_request_reaches_forwarder(self):
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_request(
            request_id="r1",
            session_id="s1",
            method="POST",
            path="/mcp/sse",
            status_code=200,
            duration_ms=55.0,
            user_id="carol",
        )

        payloads = forwarder.get_forwarded_payloads()
        assert len(payloads) == 1
        assert payloads[0]["event"] == f"{SEGMENT_EVENT_PREFIX}.http_request"
        assert payloads[0]["properties"]["status_code"] == 200

    def test_forwarder_inactive_does_not_accumulate(self):
        """When forwarder is off (default), no payloads should accumulate."""
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()
        assert forwarder.enabled is False

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_tool_call(
            tool_name="t",
            session_id="s",
            viewer_info="v",
        )
        assert len(forwarder.get_forwarded_payloads()) == 0

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_unmappable_events_not_forwarded(self):
        """Events without user_id should not reach the forwarder."""
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_tool_call(
            tool_name="t",
            session_id="s",
            viewer_info=None,
        )
        assert len(forwarder.get_forwarded_payloads()) == 0

    @patch.dict("os.environ", {"MCP_SEGMENT_DRY_RUN": "true"})
    def test_sanitised_params_in_forwarded_payload(self):
        """Params should be sanitised by the tracker before reaching Segment."""
        reset_segment_forwarder()
        forwarder = get_segment_forwarder()

        tracker = AnalyticsTracker(enabled=True)
        tracker.track_tool_call(
            tool_name="gql",
            session_id="s",
            viewer_info="alice",
            params={"api_key": "super_secret", "entity": "my-team"},
        )

        payloads = forwarder.get_forwarded_payloads()
        assert len(payloads) == 1
        seg_params = payloads[0]["properties"]["params"]
        assert seg_params["api_key"] == "<redacted>"
        assert seg_params["entity"] == "my-team"
