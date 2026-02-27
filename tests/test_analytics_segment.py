"""Unit tests for analytics_segment.py -- Gorilla /analytics/t compatibility."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch


from wandb_mcp_server.analytics_segment import (
    SEGMENT_EVENT_PREFIX,
    SegmentForwarder,
    map_to_segment_track,
)


# ---------------------------------------------------------------------------
# map_to_segment_track
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

    # -- tool_call --

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

    # -- user_session --

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

    # -- request --

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

    # -- edge cases --

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
# SegmentForwarder
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

    @patch.dict("os.environ", {"MCP_SEGMENT_FORWARD": "true"})
    def test_live_posts_to_gorilla(self):
        f = SegmentForwarder(base_url="https://api.wandb.test")
        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "carol",
            "tool_name": "t",
            "success": True,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("wandb_mcp_server.analytics_segment.requests.post", return_value=mock_resp) as mock_post:
            result = f.forward(event)
            assert result is not None
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "analytics/t" in call_kwargs.args[0]

    @patch.dict("os.environ", {"MCP_SEGMENT_FORWARD": "true"})
    def test_live_handles_post_failure_gracefully(self):
        f = SegmentForwarder(base_url="https://api.wandb.test")
        event = {
            "schema_version": "1.0",
            "event_type": "tool_call",
            "timestamp": "2026-02-27T18:30:00+00:00",
            "user_id": "dan",
            "tool_name": "t",
            "success": True,
        }
        with patch("wandb_mcp_server.analytics_segment.requests.post", side_effect=Exception("network")):
            result = f.forward(event)
            assert result is not None  # still returns payload despite failure
