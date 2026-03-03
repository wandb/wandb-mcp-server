"""Unit tests for analytics.py -- AnalyticsTracker (MCP-12).

Rewritten from Nico's PR #2 with improved datetime handling and
cleaner param sanitisation.
"""

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from wandb_mcp_server.analytics import (
    SCHEMA_VERSION,
    AnalyticsTracker,
    get_analytics_tracker,
    reset_analytics_tracker,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_analytics_tracker()
    yield
    reset_analytics_tracker()


class _EventCapture(logging.Filter):
    """Logging filter that captures the last analytics event payload."""

    def __init__(self):
        super().__init__()
        self.event: Optional[Dict[str, Any]] = None
        self.labels: Optional[Dict[str, str]] = None

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "json_fields"):
            self.event = record.json_fields
        if hasattr(record, "labels"):
            self.labels = record.labels
        return True


@pytest.fixture()
def capture():
    """Yield a capture filter attached to the analytics logger."""
    log = logging.getLogger("wandb_mcp_server.analytics")
    cap = _EventCapture()
    log.addFilter(cap)
    yield cap
    log.removeFilter(cap)


# -- Enable / disable --------------------------------------------------------


class TestEnableDisable:
    def test_enabled_by_default(self):
        assert AnalyticsTracker(enabled=True).enabled is True

    def test_disabled_via_constructor(self):
        assert AnalyticsTracker(enabled=False).enabled is False

    @patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"})
    def test_disabled_via_env_var(self):
        assert AnalyticsTracker(enabled=True).enabled is False

    @patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "false"})
    def test_env_false_keeps_enabled(self):
        assert AnalyticsTracker(enabled=True).enabled is True


# -- Email domain extraction --------------------------------------------------


class TestExtractEmailDomain:
    def setup_method(self):
        self.t = AnalyticsTracker(enabled=True)

    def test_from_string(self):
        assert self.t._extract_email_domain("user@anthropic.com") == "anthropic.com"

    def test_from_attr(self):
        assert self.t._extract_email_domain(SimpleNamespace(email="e@wandb.com")) == "wandb.com"

    def test_from_dict(self):
        assert self.t._extract_email_domain({"email": "t@nvidia.com"}) == "nvidia.com"

    def test_no_at_returns_none(self):
        assert self.t._extract_email_domain("no-at-sign") is None

    def test_none_returns_none(self):
        assert self.t._extract_email_domain(None) is None

    def test_lowercased(self):
        assert self.t._extract_email_domain("U@UPPER.COM") == "upper.com"

    def test_empty_string(self):
        assert self.t._extract_email_domain("") is None

    def test_at_only(self):
        assert self.t._extract_email_domain("@") == ""


# -- User ID extraction -------------------------------------------------------


class TestExtractUserId:
    def setup_method(self):
        self.t = AnalyticsTracker(enabled=True)

    def test_username_attr(self):
        assert self.t._extract_user_id(SimpleNamespace(username="jdoe")) == "jdoe"

    def test_entity_attr(self):
        assert self.t._extract_user_id(SimpleNamespace(entity="team")) == "team"

    def test_email_attr_returns_domain_not_full_address(self):
        """Email-only viewers must return domain, never the full address (PII)."""
        assert self.t._extract_user_id(SimpleNamespace(email="e@x.com")) == "x.com"

    def test_string(self):
        assert self.t._extract_user_id("raw") == "raw"

    def test_string_email_returns_domain(self):
        """String inputs that look like emails must return domain only."""
        assert self.t._extract_user_id("alice@wandb.com") == "wandb.com"

    def test_username_priority(self):
        v = SimpleNamespace(username="u", entity="e", email="x@y.com")
        assert self.t._extract_user_id(v) == "u"

    def test_none_returns_none(self):
        assert self.t._extract_user_id(None) is None

    def test_unrecognised_type_returns_none(self):
        """Arbitrary objects should not be str()-ified (data leakage guard)."""
        assert self.t._extract_user_id(42) is None

    def test_empty_username_falls_through(self):
        v = SimpleNamespace(username="", entity="team")
        assert self.t._extract_user_id(v) == "team"

    def test_email_only_viewer_never_leaks_full_email(self):
        """Regression: _extract_user_id must never return a full email address."""
        viewer = SimpleNamespace(email="secret@corp.com")
        result = self.t._extract_user_id(viewer)
        assert "@" not in (result or ""), f"Full email leaked: {result}"


# -- Param sanitisation -------------------------------------------------------


class TestSanitiseParams:
    def test_redacts_api_key(self):
        assert AnalyticsTracker._sanitise_params({"api_key": "secret"})["api_key"] == "<redacted>"

    def test_redacts_token(self):
        assert AnalyticsTracker._sanitise_params({"bearer_token": "x"})["bearer_token"] == "<redacted>"

    def test_truncates_long_string(self):
        result = AnalyticsTracker._sanitise_params({"query": "x" * 500})
        assert "<truncated:500 chars>" in result["query"]

    def test_passes_safe_params(self):
        assert AnalyticsTracker._sanitise_params({"project": "ok"})["project"] == "ok"

    def test_empty_params(self):
        assert AnalyticsTracker._sanitise_params(None) == {}

    def test_nested_dict_redaction(self):
        result = AnalyticsTracker._sanitise_params({"config": {"api_key": "s3cr3t", "model": "gpt-4"}})
        assert result["config"]["api_key"] == "<redacted>"
        assert result["config"]["model"] == "gpt-4"

    def test_deeply_nested_stops_at_depth_limit(self):
        deep = {"l1": {"l2": {"l3": {"l4": {"api_key": "leak"}}}}}
        result = AnalyticsTracker._sanitise_params(deep)
        assert result["l1"]["l2"]["l3"]["l4"] == {"api_key": "leak"}

    def test_integer_values_pass_through(self):
        assert AnalyticsTracker._sanitise_params({"count": 42})["count"] == 42

    def test_boolean_values_pass_through(self):
        assert AnalyticsTracker._sanitise_params({"flag": True})["flag"] is True

    def test_list_of_dicts_sanitised(self):
        result = AnalyticsTracker._sanitise_params({"items": [{"api_key": "leak", "name": "ok"}]})
        assert result["items"][0]["api_key"] == "<redacted>"
        assert result["items"][0]["name"] == "ok"

    def test_nested_list_of_dicts_sanitised(self):
        result = AnalyticsTracker._sanitise_params({"filters": [{"config": {"token": "secret", "model": "gpt-4"}}]})
        assert result["filters"][0]["config"]["token"] == "<redacted>"
        assert result["filters"][0]["config"]["model"] == "gpt-4"

    def test_list_of_primitives_unchanged(self):
        result = AnalyticsTracker._sanitise_params({"ids": [1, 2, 3]})
        assert result["ids"] == [1, 2, 3]

    def test_deeply_nested_list_stops_at_depth_limit(self):
        deep = {"l1": [{"l2": [{"l3": [{"api_key": "leak"}]}]}]}
        result = AnalyticsTracker._sanitise_params(deep)
        assert result["l1"][0]["l2"][0]["l3"] == [{"api_key": "leak"}]


# -- Schema version & base fields -------------------------------------------


class TestSchemaVersion:
    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "1.0"

    def test_user_session_has_schema_version(self, capture):
        AnalyticsTracker(enabled=True).track_user_session(session_id="s", viewer_info="u@t.com")
        assert capture.event is not None
        assert capture.event["schema_version"] == SCHEMA_VERSION

    def test_tool_call_has_schema_version(self, capture):
        AnalyticsTracker(enabled=True).track_tool_call(tool_name="t", session_id="s", viewer_info="v")
        assert capture.event is not None
        assert capture.event["schema_version"] == SCHEMA_VERSION

    def test_request_has_schema_version(self, capture):
        AnalyticsTracker(enabled=True).track_request(
            request_id="r",
            session_id="s",
            method="GET",
            path="/",
            status_code=200,
        )
        assert capture.event is not None
        assert capture.event["schema_version"] == SCHEMA_VERSION

    def test_emit_rejects_missing_required_fields(self, capture):
        """Events missing schema_version/timestamp should be dropped, not emitted."""
        t = AnalyticsTracker(enabled=True)
        t._emit({"event_type": "test"}, {})
        assert capture.event is None


# -- track_user_session -------------------------------------------------------


class TestTrackUserSession:
    def test_no_raise_enabled(self):
        AnalyticsTracker(enabled=True).track_user_session(
            session_id="s1",
            viewer_info=SimpleNamespace(username="u", email="u@x.com"),
            api_key_hash="a" * 64,
        )

    def test_skipped_disabled(self, caplog):
        t = AnalyticsTracker(enabled=False)
        with caplog.at_level(logging.INFO, logger="wandb_mcp_server.analytics"):
            t.track_user_session(session_id="s", viewer_info="v")
        assert not any("ANALYTICS_EVENT" in r.message for r in caplog.records)

    def test_api_key_hash_truncated(self, capture):
        AnalyticsTracker(enabled=True).track_user_session(
            session_id="s",
            viewer_info="u@t.com",
            api_key_hash="abcdef1234567890extra",
        )
        assert capture.event is not None
        assert len(capture.event["api_key_hash"]) == 16

    def test_uses_utc_timestamps(self, capture):
        AnalyticsTracker(enabled=True).track_user_session(session_id="s", viewer_info="v")
        assert capture.event is not None
        assert "+00:00" in capture.event["timestamp"] or "Z" in capture.event["timestamp"]

    def test_none_api_key_hash(self, capture):
        AnalyticsTracker(enabled=True).track_user_session(session_id="s", viewer_info="v", api_key_hash=None)
        assert capture.event is not None
        assert capture.event["api_key_hash"] is None

    def test_event_fields_complete(self, capture):
        AnalyticsTracker(enabled=True).track_user_session(
            session_id="sess-1",
            viewer_info=SimpleNamespace(username="alice", email="a@co.com"),
            api_key_hash="a" * 64,
            metadata={"client": "cursor"},
        )
        e = capture.event
        assert e["event_type"] == "user_session"
        assert e["session_id"] == "sess-1"
        assert e["user_id"] == "alice"
        assert e["email_domain"] == "co.com"
        assert e["metadata"] == {"client": "cursor"}


# -- track_tool_call ----------------------------------------------------------


class TestTrackToolCall:
    def test_no_raise(self):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="query_weave",
            session_id="s1",
            viewer_info="v",
            params={"project": "test", "api_key": "secret"},
        )

    def test_skipped_disabled(self):
        AnalyticsTracker(enabled=False).track_tool_call(tool_name="t", session_id="s", viewer_info="v")

    def test_sanitises_api_key(self, capture):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="t",
            session_id="s",
            viewer_info="v",
            params={"api_key": "super_secret", "project": "ok"},
        )
        assert capture.event["params"]["api_key"] == "<redacted>"
        assert capture.event["params"]["project"] == "ok"

    def test_error_field_recorded(self, capture):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="t",
            session_id="s",
            viewer_info="v",
            success=False,
            error="timeout",
        )
        assert capture.event["success"] is False
        assert capture.event["error"] == "timeout"

    def test_duration_ms_recorded(self, capture):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="t",
            session_id="s",
            viewer_info="v",
            duration_ms=123.4,
        )
        assert capture.event["duration_ms"] == 123.4

    def test_labels_include_tool_name(self, capture):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="query_gql",
            session_id="s",
            viewer_info="v",
        )
        assert capture.labels["tool_name"] == "query_gql"


# -- track_request -------------------------------------------------------------


class TestTrackRequest:
    def test_no_raise(self):
        AnalyticsTracker(enabled=True).track_request(
            request_id="r1",
            session_id="s1",
            method="POST",
            path="/mcp",
            status_code=200,
            duration_ms=42.5,
        )

    def test_skipped_disabled(self):
        AnalyticsTracker(enabled=False).track_request(
            request_id="r",
            session_id="s",
            method="GET",
            path="/",
            status_code=200,
        )

    def test_event_fields_complete(self, capture):
        AnalyticsTracker(enabled=True).track_request(
            request_id="r1",
            session_id="s1",
            method="POST",
            path="/mcp/sse",
            status_code=200,
            duration_ms=55.0,
            user_id="alice",
            email_domain="co.com",
        )
        e = capture.event
        assert e["event_type"] == "request"
        assert e["method"] == "POST"
        assert e["path"] == "/mcp/sse"
        assert e["status_code"] == 200
        assert e["duration_ms"] == 55.0


# -- Global tracker singleton --------------------------------------------------


class TestGlobalTracker:
    def test_singleton(self):
        assert get_analytics_tracker() is get_analytics_tracker()

    def test_reset(self):
        t1 = get_analytics_tracker()
        reset_analytics_tracker()
        assert get_analytics_tracker() is not t1

    @patch.dict("os.environ", {"MCP_ANALYTICS_DISABLED": "true"})
    def test_singleton_respects_env(self):
        t = get_analytics_tracker()
        assert t.enabled is False
