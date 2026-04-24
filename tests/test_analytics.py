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


# -- Privacy levels ------------------------------------------------------------


class TestPrivacyLevel:
    """MCP_LOG_PRIVACY_LEVEL gates how much customer content is redacted.

    Three levels: off (today's behavior, Cloud Run default), standard (free-text
    redaction, customer K8s default), strict (also hash identifiers).
    """

    # ---- _resolve_privacy_level --------------------------------------------

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("MCP_LOG_PRIVACY_LEVEL", raising=False)
        from wandb_mcp_server.analytics import _resolve_privacy_level

        assert _resolve_privacy_level() == "off"

    def test_standard_recognised(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
        from wandb_mcp_server.analytics import _resolve_privacy_level

        assert _resolve_privacy_level() == "standard"

    def test_strict_recognised(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
        from wandb_mcp_server.analytics import _resolve_privacy_level

        assert _resolve_privacy_level() == "strict"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "STANDARD")
        from wandb_mcp_server.analytics import _resolve_privacy_level

        assert _resolve_privacy_level() == "standard"

    def test_unknown_falls_back_to_off(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "paranoid")
        from wandb_mcp_server.analytics import _resolve_privacy_level

        assert _resolve_privacy_level() == "off"

    # ---- _sanitise_params: off mode is byte-identical to pre-amend ---------

    def test_off_preserves_free_text_values(self):
        """BigQuery contract: off-mode must pass free-text values through unchanged."""
        params = {"query": "SELECT * FROM runs", "description": "my analysis"}
        result = AnalyticsTracker._sanitise_params(params, level="off")
        assert result["query"] == "SELECT * FROM runs"
        assert result["description"] == "my analysis"

    def test_off_preserves_identifiers(self):
        params = {"entity_name": "acme-corp", "project_name": "llm-eval"}
        result = AnalyticsTracker._sanitise_params(params, level="off")
        assert result["entity_name"] == "acme-corp"
        assert result["project_name"] == "llm-eval"

    def test_off_still_redacts_sensitive_keys(self):
        """The existing api_key/token redaction must still fire at off level."""
        result = AnalyticsTracker._sanitise_params({"api_key": "leak"}, level="off")
        assert result["api_key"] == "<redacted>"

    # ---- _sanitise_params: standard mode redacts free-text values ----------

    def test_standard_redacts_free_text_values(self):
        params = {
            "query": "SELECT * FROM runs",
            "question": "How does W&B work?",
            "description": "my private analysis",
            "title": "Internal eval",
        }
        result = AnalyticsTracker._sanitise_params(params, level="standard")
        assert result["query"] == f"<redacted: text len={len(params['query'])}>"
        assert result["question"] == f"<redacted: text len={len(params['question'])}>"
        assert result["description"] == f"<redacted: text len={len(params['description'])}>"
        assert result["title"] == f"<redacted: text len={len(params['title'])}>"

    def test_standard_preserves_identifiers(self):
        """standard should leave entity/project plaintext -- only strict hashes."""
        params = {"entity_name": "acme-corp", "project_name": "llm-eval"}
        result = AnalyticsTracker._sanitise_params(params, level="standard")
        assert result["entity_name"] == "acme-corp"
        assert result["project_name"] == "llm-eval"

    def test_standard_still_redacts_sensitive_keys(self):
        result = AnalyticsTracker._sanitise_params({"api_key": "leak"}, level="standard")
        assert result["api_key"] == "<redacted>"

    def test_standard_preserves_non_string_free_text_values(self):
        """Redaction only applies to string values; dict/list/int pass through."""
        params = {"title": {"nested": "data"}, "body": 42}
        result = AnalyticsTracker._sanitise_params(params, level="standard")
        # Non-string values still go through the dict/list/primitive branches
        assert result["title"] == {"nested": "data"}
        assert result["body"] == 42

    # ---- _sanitise_params: strict mode hashes identifiers ------------------

    def test_strict_hashes_identifier_keys(self):
        params = {
            "entity_name": "acme-corp",
            "project_name": "llm-eval",
            "run_id": "abc123",
        }
        result = AnalyticsTracker._sanitise_params(params, level="strict")
        for key in ("entity_name", "project_name", "run_id"):
            assert result[key].startswith("<h:")
            assert result[key].endswith(">")
            assert len(result[key]) == len("<h:>") + 12

    def test_strict_hash_is_deterministic_per_value(self):
        """Same input must hash to the same digest so cohort analytics still work."""
        r1 = AnalyticsTracker._sanitise_params({"entity_name": "acme"}, level="strict")
        r2 = AnalyticsTracker._sanitise_params({"entity_name": "acme"}, level="strict")
        assert r1["entity_name"] == r2["entity_name"]

    def test_strict_different_values_produce_different_hashes(self):
        r1 = AnalyticsTracker._sanitise_params({"entity_name": "acme"}, level="strict")
        r2 = AnalyticsTracker._sanitise_params({"entity_name": "beta"}, level="strict")
        assert r1["entity_name"] != r2["entity_name"]

    def test_strict_also_redacts_free_text(self):
        """Strict is additive on top of standard."""
        result = AnalyticsTracker._sanitise_params({"query": "SELECT *", "entity_name": "acme"}, level="strict")
        assert result["query"] == "<redacted: text len=8>"
        assert result["entity_name"].startswith("<h:")

    # ---- Nested structures propagate level --------------------------------

    def test_level_propagates_through_nested_dicts(self):
        result = AnalyticsTracker._sanitise_params({"filters": {"query": "private text"}}, level="standard")
        assert result["filters"]["query"] == "<redacted: text len=12>"

    def test_level_propagates_through_nested_lists(self):
        result = AnalyticsTracker._sanitise_params({"filters": [{"query": "private text"}]}, level="standard")
        assert result["filters"][0]["query"] == "<redacted: text len=12>"

    # ---- Integration through track_tool_call ------------------------------

    def test_track_tool_call_redacts_at_standard(self, capture, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="query_wandb_gql",
            session_id="s",
            viewer_info="alice",
            params={"query": "{ viewer { id } }", "entity_name": "acme"},
        )
        assert capture.event["params"]["query"].startswith("<redacted: text len=")
        # entity_name is an identifier, not free-text -> plaintext at standard
        assert capture.event["params"]["entity_name"] == "acme"

    def test_track_tool_call_hashes_identifiers_at_strict(self, capture, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="count_traces",
            session_id="s",
            viewer_info=SimpleNamespace(username="alice", email="a@co.com"),
            params={"entity_name": "acme", "project_name": "eval"},
        )
        assert capture.event["params"]["entity_name"].startswith("<h:")
        assert capture.event["params"]["project_name"].startswith("<h:")
        # Strict also hashes user_id and email_domain
        assert capture.event["user_id"].startswith("<h:")
        assert capture.event["email_domain"].startswith("<h:")

    def test_track_tool_call_off_mode_preserves_identity(self, capture, monkeypatch):
        """BigQuery contract: off-mode must emit user_id/email_domain plaintext."""
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "off")
        AnalyticsTracker(enabled=True).track_tool_call(
            tool_name="count_traces",
            session_id="s",
            viewer_info=SimpleNamespace(username="alice", email="a@co.com"),
            params={"entity_name": "acme"},
        )
        assert capture.event["user_id"] == "alice"
        assert capture.event["email_domain"] == "co.com"
        assert capture.event["params"]["entity_name"] == "acme"

    # ---- is_verbose_log_site_gated ----------------------------------------

    def test_verbose_gated_is_false_at_off(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "off")
        from wandb_mcp_server.analytics import is_verbose_log_site_gated

        assert is_verbose_log_site_gated() is False

    def test_verbose_gated_is_true_at_standard(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
        from wandb_mcp_server.analytics import is_verbose_log_site_gated

        assert is_verbose_log_site_gated() is True

    def test_verbose_gated_is_true_at_strict(self, monkeypatch):
        monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
        from wandb_mcp_server.analytics import is_verbose_log_site_gated

        assert is_verbose_log_site_gated() is True
