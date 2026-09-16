"""Request rejection reasons are fixed vocabulary, never exception details."""

import json
from unittest.mock import patch

import pytest

from wandb_mcp_server.analytics import AnalyticsTracker, _prepare_event
from wandb_mcp_server.analytics_datadog import map_to_datadog_log


@pytest.mark.parametrize("level", ["off", "standard", "strict"])
@pytest.mark.parametrize("reason", ["auth_missing", "session_actor_mismatch", "session_capacity", "internal_error"])
def test_request_reason_reaches_canonical_and_datadog(level, reason, monkeypatch):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    tracker = AnalyticsTracker(enabled=True)
    with patch.object(tracker, "_emit") as emit:
        tracker.track_request("synthetic", None, "POST", "/mcp", 403, request_reason=reason)
    event, labels = emit.call_args.args
    assert event["request_reason"] == reason
    assert "request_reason" not in labels
    mapped = map_to_datadog_log(event, dd_env="test", dd_version="test", dd_service="test")
    assert mapped["attributes"]["request_reason"] == reason


@pytest.mark.parametrize("reason", ["private-query-canary", {"value": "secret-canary"}, ["session_capacity"], None])
def test_untrusted_request_reason_is_omitted(reason):
    event = _prepare_event({"event_type": "request", "request_reason": reason})
    assert "request_reason" not in event
    assert "canary" not in json.dumps(event)


def test_request_reason_survives_event_compaction():
    event = _prepare_event({"event_type": "request", "request_reason": "internal_error", "unknown": "x" * 10000})
    assert event["request_reason"] == "internal_error"
