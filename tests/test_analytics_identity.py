"""Authenticated usernames and strict pseudonyms agree across analytics sinks."""

import json
from types import SimpleNamespace

import pytest

from wandb_mcp_server import analytics
from wandb_mcp_server.analytics_datadog import map_to_datadog_log
from wandb_mcp_server.analytics_segment import map_to_segment_track
from wandb_mcp_server.api_client import WandBApiManager


FINGERPRINT = "wandb_key:" + "a" * 24


@pytest.fixture(autouse=True)
def identity_context(monkeypatch):
    monkeypatch.setattr(analytics, "current_actor_id", lambda: analytics.actor_id_from_api_key_hash("a" * 64))
    monkeypatch.setattr(WandBApiManager, "get_cached_viewer_info", lambda: None, raising=False)
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    monkeypatch.setattr(analytics, "MCP_REQUEST_SUCCESS_SAMPLE_RATE", 1.0)

    def no_api_lookup(*args, **kwargs):
        pytest.fail("analytics must not initialize a client or fetch a viewer")

    monkeypatch.setattr(WandBApiManager, "get_api", no_api_lookup)


def emit(monkeypatch, event_type, viewer=None):
    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    if event_type == "tool_call":
        tracker.track_tool_call("query_wandb_tool", "synthetic-session", viewer, duration_ms=1)
    elif event_type == "user_session":
        tracker.track_user_session("synthetic-session", viewer, api_key_hash="a" * 64)
    else:
        tracker.track_request("synthetic-request", "synthetic-session", "POST", "/mcp", 200)
    assert len(events) == 1
    return events[0]


def mapped(event):
    dd = map_to_datadog_log(event, dd_env="test", dd_version="test", dd_service="test")
    segment = map_to_segment_track(event)
    return dd, segment


@pytest.mark.parametrize("level", ["off", "standard"])
@pytest.mark.parametrize("event_type", ["tool_call", "user_session", "request"])
def test_cached_username_is_preferred_only_in_approved_sinks(monkeypatch, level, event_type):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    monkeypatch.setattr(WandBApiManager, "get_cached_viewer_info", lambda: {"username": "cached-user"})
    event = emit(monkeypatch, event_type)
    dd, segment = mapped(event)
    assert event["user_id"] == event["actor_id"] == "cached-user"
    assert dd["attributes"]["usr"]["id"] == dd["attributes"]["actor_id"] == "cached-user"
    if segment is not None:
        assert segment["userId"] == FINGERPRINT
        assert "cached-user" not in json.dumps(segment)
    assert "_segment" not in json.dumps(event), "private routing provenance must not serialize"


@pytest.mark.parametrize("level", ["off", "standard", "strict"])
def test_missing_cached_username_keeps_fingerprint(monkeypatch, level):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    event = emit(monkeypatch, "tool_call")
    dd, segment = mapped(event)
    assert event["actor_id"] == FINGERPRINT
    assert dd["attributes"]["usr"]["id"] == FINGERPRINT
    assert segment["userId"] == FINGERPRINT


@pytest.mark.parametrize("event_type", ["tool_call", "user_session", "request"])
def test_strict_cached_username_is_hashed_consistently(monkeypatch, event_type):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
    monkeypatch.setattr(
        WandBApiManager,
        "get_cached_viewer_info",
        lambda: {"username": "private-user", "email": "private@example.invalid"},
    )
    event = emit(monkeypatch, event_type)
    dd, segment = mapped(event)
    pseudonym = event["user_id"]
    assert pseudonym == FINGERPRINT
    assert event["actor_id"] == dd["attributes"]["usr"]["id"] == pseudonym
    if segment is not None:
        assert segment["userId"] == pseudonym
    retained = json.dumps([event, dd, segment])
    assert "private-user" not in retained
    assert "private@example.invalid" not in retained
    assert "example.invalid" not in retained


def test_strict_explicit_username_without_fingerprint_is_hashed(monkeypatch):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
    monkeypatch.setattr(analytics, "current_actor_id", lambda: None)
    event = emit(monkeypatch, "tool_call", {"username": "private-user"})
    assert event["user_id"].startswith("<h:")
    assert event["actor_id"] == event["user_id"]
    assert "private-user" not in json.dumps([event, *mapped(event)])


@pytest.mark.parametrize("sink", ["canonical", "datadog", "segment"])
def test_strict_direct_sink_cannot_bypass_identity_privacy(monkeypatch, sink):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
    event = {
        "event_type": "user_session",
        "actor_id": "raw-actor",
        "user_id": "raw-user",
        "email_domain": "private.example",
        "metadata": {"username": "nested-user", "email": "nested@private.example"},
    }
    if sink == "canonical":
        result = analytics._prepare_event(event)
    elif sink == "datadog":
        result = mapped(event)[0]
    else:
        result = mapped(event)[1]
    retained = json.dumps(result)
    for raw in ("raw-actor", "raw-user", "private.example", "nested-user"):
        assert raw not in retained
    assert event["actor_id"] == "raw-actor", "privacy projection must not mutate caller data"


@pytest.mark.parametrize("raw_identity", ["<h:123456abcdef>", FINGERPRINT])
@pytest.mark.parametrize("sink", ["canonical", "datadog", "segment"])
def test_strict_hash_shaped_plain_identity_is_not_trusted(monkeypatch, raw_identity, sink):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
    event = {
        "event_type": "user_session",
        "actor_id": raw_identity,
        "user_id": raw_identity,
        "metadata": {"username": raw_identity},
    }
    if sink == "canonical":
        result = analytics._prepare_event(event)
    else:
        result = mapped(event)[0 if sink == "datadog" else 1]
    assert raw_identity not in json.dumps(result)
    assert analytics._hash_identifier(raw_identity) in json.dumps(result)


@pytest.mark.parametrize("has_key", [False, True])
def test_strict_internal_pseudonyms_remain_stable_across_sinks(monkeypatch, has_key):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "strict")
    if not has_key:
        monkeypatch.setattr(analytics, "current_actor_id", lambda: None)
    event = emit(monkeypatch, "tool_call", {"username": "private-user"})
    again = analytics._prepare_event(event)
    dd, segment = mapped(again)
    expected = FINGERPRINT if has_key else analytics._hash_identifier("private-user")
    assert event["actor_id"] == again["actor_id"] == expected
    assert event["user_id"] == again["user_id"] == expected
    assert dd["attributes"]["usr"]["id"] == segment["userId"] == expected


def test_username_extraction_does_not_invoke_properties(monkeypatch):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")

    class NetworkBackedViewer:
        @property
        def username(self):
            pytest.fail("analytics read a potentially network-backed property")

    event = emit(monkeypatch, "tool_call", NetworkBackedViewer())
    assert event["actor_id"] == FINGERPRINT


def test_supplied_authenticated_username_wins_over_cached_value(monkeypatch):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
    monkeypatch.setattr(WandBApiManager, "get_cached_viewer_info", lambda: {"username": "other-user"})
    event = emit(monkeypatch, "tool_call", SimpleNamespace(username="request-user"))
    assert event["user_id"] == event["actor_id"] == "request-user"
    assert mapped(event)[1]["userId"] == FINGERPRINT


@pytest.mark.parametrize("level", ["off", "standard", "strict"])
def test_segment_without_key_hashes_username_even_in_direct_mapper(monkeypatch, level):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    monkeypatch.setattr(analytics, "current_actor_id", lambda: None)
    event = emit(monkeypatch, "tool_call", {"username": "private-user"})
    assert mapped(event)[1]["userId"] == analytics._hash_identifier("private-user")
    direct = {
        "event_type": "user_session",
        "actor_id": "private-user",
        "user_id": "private-user",
        "metadata": {"username": "private-user", "email": "private@example.invalid"},
        "_segment_identity": "forged-raw-identity",
    }
    result = mapped(direct)[1]
    assert result["userId"] == analytics._hash_identifier("private-user")
    for raw in ("private-user", "private@example.invalid", "forged-raw-identity"):
        assert raw not in json.dumps(result)


@pytest.mark.parametrize("viewer", [{"entity": "team"}, {"email": "member@example.invalid"}, "member@example.invalid"])
def test_username_never_falls_back_to_team_or_email(monkeypatch, viewer):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
    event = emit(monkeypatch, "tool_call", viewer)
    assert event["actor_id"] == event["user_id"] == FINGERPRINT
    assert mapped(event)[0]["attributes"]["usr"]["id"] == FINGERPRINT


@pytest.mark.parametrize("level", ["off", "standard", "strict"])
def test_diagnostic_error_event_obeys_identity_destination_policy(monkeypatch, level):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    monkeypatch.setattr(WandBApiManager, "get_cached_viewer_info", lambda: {"username": "diagnostic-user"})
    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    tracker.track_tool_call(
        "query_wandb_tool",
        "synthetic-session",
        None,
        success=False,
        error_diagnostics={"category": "input_validation", "validation_fields": ["resource"]},
    )
    event = events[0]
    dd, segment = mapped(event)
    display = FINGERPRINT if level == "strict" else "diagnostic-user"
    assert event["actor_id"] == dd["attributes"]["usr"]["id"] == display
    assert segment["userId"] == FINGERPRINT
    assert segment["properties"]["error_diagnostics"]["category"] == "input_validation"
    assert "diagnostic-user" not in json.dumps(segment)


@pytest.mark.parametrize("level", ["off", "standard", "strict"])
def test_safe_diagnostics_reach_each_sink_without_raw_details(monkeypatch, level):
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    tracker.track_tool_call(
        "query_wandb_tool",
        "synthetic-session",
        None,
        success=False,
        error_diagnostics={
            "category": "input_validation",
            "exception_type": "ValidationError",
            "validation_fields": ["resource", "customer-private-field"],
            "validation_codes": ["literal_error", "raw-secret-code"],
            "message": "private-upstream-payload",
            "input": "private-argument-value",
        },
    )
    event = events[0]
    expected = {
        "category": "input_validation",
        "exception_type": "ValidationError",
        "validation_fields": ["resource"],
        "validation_codes": ["literal_error"],
    }
    dd, segment = mapped(event)
    assert event["error_diagnostics"] == expected
    assert dd["attributes"]["error_diagnostics"] == expected
    assert segment["properties"]["error_diagnostics"] == expected
    retained = json.dumps([event, dd, segment])
    for private in ("customer-private-field", "raw-secret-code", "private-upstream-payload", "private-argument-value"):
        assert private not in retained
