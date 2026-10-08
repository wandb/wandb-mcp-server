"""Authenticated usernames are resolved once per actor and reused by telemetry."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from wandb_mcp_server import analytics
from wandb_mcp_server.analytics_segment import map_to_segment_track
from wandb_mcp_server.api_client import WandBApiManager

KEY = "k" * 40


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    WandBApiManager._clear_api_cache()
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", "standard")
    yield
    WandBApiManager._clear_api_cache()


class FakeApi:
    def __init__(self, username="synthetic-user", delay=0.0, error=None):
        self.username, self.delay, self.error, self.lookups = username, delay, error, 0

    @property
    def viewer(self):
        self.lookups += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(_attrs={"username": self.username})


def use_api(monkeypatch, api):
    monkeypatch.setattr(WandBApiManager, "get_api", lambda key=None: api)
    return api


def test_lookup_is_cached_per_actor(monkeypatch):
    api = use_api(monkeypatch, FakeApi())
    assert WandBApiManager.resolve_viewer_username(KEY) == "synthetic-user"
    assert WandBApiManager.resolve_viewer_username(KEY) == "synthetic-user"
    assert WandBApiManager.peek_viewer_username(KEY) == (True, "synthetic-user")
    assert WandBApiManager.peek_viewer_username("other" * 8) == (False, None)
    assert api.lookups == 1


def test_concurrent_lookups_share_one_request(monkeypatch):
    api = use_api(monkeypatch, FakeApi(delay=0.05))
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(WandBApiManager.resolve_viewer_username(KEY))) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["synthetic-user"] * 8
    assert api.lookups == 1


@pytest.mark.parametrize("api", [FakeApi(error=RuntimeError("HTTP 503")), FakeApi(username="bad name!")])
def test_failures_and_invalid_usernames_are_negatively_cached(monkeypatch, api):
    use_api(monkeypatch, api)
    assert WandBApiManager.resolve_viewer_username(KEY) is None
    assert WandBApiManager.resolve_viewer_username(KEY) is None
    assert api.lookups == 1
    monkeypatch.setattr(WandBApiManager, "_viewer_identity_negative_ttl_seconds", 0.0)
    WandBApiManager._clear_api_cache()
    WandBApiManager.resolve_viewer_username(KEY)
    WandBApiManager.resolve_viewer_username(KEY)
    assert api.lookups == 3


@pytest.mark.parametrize("env", [{"MCP_LOG_PRIVACY_LEVEL": "strict"}, {"MCP_ANALYTICS_DISABLED": "true"}])
def test_strict_or_disabled_analytics_never_looks_up(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def fail(*args, **kwargs):
        pytest.fail("lookup must not run")

    monkeypatch.setattr(WandBApiManager, "get_api", fail)
    assert asyncio.run(analytics.resolve_authenticated_viewer(KEY)) is None


def test_slow_lookup_does_not_block_request_and_fills_cache(monkeypatch):
    use_api(monkeypatch, FakeApi(delay=0.2))
    monkeypatch.setattr(analytics, "_VIEWER_LOOKUP_TIMEOUT_SECONDS", 0.01)
    assert asyncio.run(analytics.resolve_authenticated_viewer(KEY)) is None
    deadline = time.monotonic() + 2
    while WandBApiManager.peek_viewer_username(KEY) != (True, "synthetic-user"):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert asyncio.run(analytics.resolve_authenticated_viewer(KEY)) == {"username": "synthetic-user"}


def test_tool_telemetry_uses_resolved_username(monkeypatch):
    use_api(monkeypatch, FakeApi())
    monkeypatch.setattr(analytics, "MCP_REQUEST_SUCCESS_SAMPLE_RATE", 1.0)
    WandBApiManager.resolve_viewer_username(KEY)
    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    token = WandBApiManager.set_context_api_key(KEY)
    try:
        tracker.track_tool_call("query_wandb_tool", "synthetic-session", None, duration_ms=1)
    finally:
        WandBApiManager.reset_context_api_key(token)
    assert events[0]["user_id"] == "synthetic-user"
    assert map_to_segment_track(analytics._prepare_event(events[0]))["userId"] == "synthetic-user"
