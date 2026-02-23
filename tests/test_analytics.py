"""Unit tests for analytics.py -- AnalyticsTracker (PR #2)."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from wandb_mcp_server.analytics import (
    AnalyticsTracker,
    get_analytics_tracker,
    reset_analytics_tracker,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_analytics_tracker()
    yield
    reset_analytics_tracker()


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


# -- User ID extraction -------------------------------------------------------

class TestExtractUserId:

    def setup_method(self):
        self.t = AnalyticsTracker(enabled=True)

    def test_username_attr(self):
        assert self.t._extract_user_id(SimpleNamespace(username="jdoe")) == "jdoe"

    def test_entity_attr(self):
        assert self.t._extract_user_id(SimpleNamespace(entity="team")) == "team"

    def test_email_attr(self):
        assert self.t._extract_user_id(SimpleNamespace(email="e@x.com")) == "e@x.com"

    def test_string(self):
        assert self.t._extract_user_id("raw") == "raw"

    def test_username_priority(self):
        v = SimpleNamespace(username="u", entity="e", email="x@y.com")
        assert self.t._extract_user_id(v) == "u"


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

    def test_api_key_hash_truncated(self):
        class Cap(logging.Filter):
            event = None
            def filter(self, record):
                if hasattr(record, "json_fields"):
                    Cap.event = record.json_fields
                return True
        log = logging.getLogger("wandb_mcp_server.analytics")
        c = Cap()
        log.addFilter(c)
        try:
            AnalyticsTracker(enabled=True).track_user_session(
                session_id="s", viewer_info="u@t.com",
                api_key_hash="abcdef1234567890extra",
            )
            assert Cap.event is not None
            assert len(Cap.event["api_key_hash"]) == 16
        finally:
            log.removeFilter(c)


# -- track_tool_call ----------------------------------------------------------

class TestTrackToolCall:

    def test_no_raise(self):
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="query_weave", session_id="s1", viewer_info="v",
            params={"project": "test", "api_key": "secret"},
        )

    def test_skipped_disabled(self):
        AnalyticsTracker(enabled=False).track_tool_call(
            tool_name="t", session_id="s", viewer_info="v",
        )

    def test_sanitizes_api_key(self):
        class Cap(logging.Filter):
            event = None
            def filter(self, record):
                if hasattr(record, "json_fields"):
                    Cap.event = record.json_fields
                return True
        log = logging.getLogger("wandb_mcp_server.analytics")
        c = Cap()
        log.addFilter(c)
        try:
            AnalyticsTracker(enabled=True).track_tool_call(
                tool_name="t", session_id="s", viewer_info="v",
                params={"api_key": "super_secret", "project": "ok"},
            )
            assert Cap.event["params"]["api_key"] == "<redacted>"
            assert Cap.event["params"]["project"] == "ok"
        finally:
            log.removeFilter(c)

    def test_truncates_large_strings(self):
        class Cap(logging.Filter):
            event = None
            def filter(self, record):
                if hasattr(record, "json_fields"):
                    Cap.event = record.json_fields
                return True
        log = logging.getLogger("wandb_mcp_server.analytics")
        c = Cap()
        log.addFilter(c)
        try:
            AnalyticsTracker(enabled=True).track_tool_call(
                tool_name="t", session_id="s", viewer_info="v",
                params={"query": "x" * 500},
            )
            assert "<truncated:500 chars>" in Cap.event["params"]["query"]
        finally:
            log.removeFilter(c)


# -- track_request -------------------------------------------------------------

class TestTrackRequest:

    def test_no_raise(self):
        AnalyticsTracker(enabled=True).track_request(
            request_id="r1", session_id="s1", method="POST",
            path="/mcp", status_code=200, duration_ms=42.5,
        )

    def test_skipped_disabled(self):
        AnalyticsTracker(enabled=False).track_request(
            request_id="r", session_id="s", method="GET",
            path="/", status_code=200,
        )


# -- Global tracker singleton --------------------------------------------------

class TestGlobalTracker:

    def test_singleton(self):
        assert get_analytics_tracker() is get_analytics_tracker()

    def test_reset(self):
        t1 = get_analytics_tracker()
        reset_analytics_tracker()
        assert get_analytics_tracker() is not t1
