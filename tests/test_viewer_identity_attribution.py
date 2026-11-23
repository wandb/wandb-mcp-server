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
    def __init__(self, username="synthetic-user", delay=0.0, error=None, attrs=None):
        self.username, self.delay, self.error, self.attrs, self.lookups = username, delay, error, attrs, 0

    @property
    def viewer(self):
        self.lookups += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        if self.attrs is not None:
            return SimpleNamespace(_attrs=dict(self.attrs))
        return SimpleNamespace(_attrs={"username": self.username})


def use_api(monkeypatch, api):
    monkeypatch.setattr(WandBApiManager, "_viewer_lookup_client", lambda key: api)
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

    monkeypatch.setattr(WandBApiManager, "_viewer_lookup_client", fail)
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


def test_bogus_key_misses_cannot_evict_resolved_users(monkeypatch):
    use_api(monkeypatch, FakeApi())
    WandBApiManager.resolve_viewer_username(KEY)
    use_api(monkeypatch, FakeApi(error=RuntimeError("HTTP 401")))
    monkeypatch.setattr(WandBApiManager, "_viewer_identity_max_misses", 3)
    for index in range(10):
        assert WandBApiManager.resolve_viewer_username(f"bogus-{index:034d}") is None
    assert len(WandBApiManager._viewer_identity_misses) == 3
    assert WandBApiManager.peek_viewer_username(KEY) == (True, "synthetic-user")


def test_lookup_is_skipped_without_caching_when_slots_are_busy(monkeypatch):
    api = use_api(monkeypatch, FakeApi())
    monkeypatch.setattr(WandBApiManager, "_viewer_identity_slots", threading.BoundedSemaphore(1))
    assert WandBApiManager._viewer_identity_slots.acquire(blocking=False)
    try:
        assert WandBApiManager.resolve_viewer_username(KEY) is None
    finally:
        WandBApiManager._viewer_identity_slots.release()
    assert api.lookups == 0
    assert WandBApiManager.peek_viewer_username(KEY) == (False, None)
    assert WandBApiManager.resolve_viewer_username(KEY) == "synthetic-user"


def test_lookup_client_is_not_added_to_the_functional_client_cache(monkeypatch):
    from wandb_mcp_server import api_client

    built = []
    monkeypatch.setattr(api_client.wandb, "Api", lambda **kwargs: built.append(kwargs) or FakeApi())
    assert WandBApiManager.resolve_viewer_username(KEY) == "synthetic-user"
    assert len(built) == 1
    assert not WandBApiManager._api_cache


def test_startup_validation_seeds_attribution_without_second_lookup(monkeypatch):
    from wandb_mcp_server import server

    api = FakeApi()
    monkeypatch.setattr(server.wandb, "Api", lambda **kwargs: api)
    assert server.validate_api_key(KEY)
    assert api.lookups == 1
    assert WandBApiManager.peek_viewer_username(KEY) == (True, "synthetic-user")


def test_team_service_account_is_cached_positively_and_tagged(monkeypatch):
    api = use_api(
        monkeypatch,
        FakeApi(attrs={"id": "User:svc-123", "entity": "mistral", "username": None, "email": "bot@mistral.ai"}),
    )
    identity = WandBApiManager.resolve_viewer_identity(KEY)
    assert identity is not None
    assert identity.actor_type == "service_account"
    assert identity.username is None
    assert identity.email_domain == "mistral.ai"
    assert WandBApiManager.resolve_viewer_username(KEY) is None
    assert WandBApiManager.peek_viewer_username(KEY) == (True, None)
    assert api.lookups == 1
    assert len(WandBApiManager._viewer_identity_cache) == 1
    assert len(WandBApiManager._viewer_identity_misses) == 0
    assert asyncio.run(analytics.resolve_authenticated_viewer(KEY)) == {
        "actor_type": "service_account",
        "email_domain": "mistral.ai",
    }


def test_unauthenticated_lookup_marks_actor_unauthenticated_and_suppresses_segment_tool_call(monkeypatch):
    api = use_api(monkeypatch, FakeApi(error=RuntimeError("HTTP 401 Unauthorized")))
    monkeypatch.setattr(analytics, "MCP_REQUEST_SUCCESS_SAMPLE_RATE", 1.0)
    viewer_info = asyncio.run(analytics.resolve_authenticated_viewer(KEY))
    assert viewer_info == {"actor_type": "unauthenticated"}
    assert api.lookups == 1
    assert len(WandBApiManager._viewer_identity_cache) == 0
    assert len(WandBApiManager._viewer_identity_misses) == 1

    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    token = WandBApiManager.set_context_api_key(KEY)
    try:
        tracker.track_tool_call("search_wandb_docs_tool", None, viewer_info, duration_ms=1, success=True)
        tracker.track_request("req-1", None, "POST", "/mcp", 401, duration_ms=1)
    finally:
        WandBApiManager.reset_context_api_key(token)

    tool_event = analytics._prepare_event(events[0])
    req_event = analytics._prepare_event(events[1])
    assert tool_event["actor_type"] == "unauthenticated"
    assert map_to_segment_track(tool_event) is None
    assert req_event["actor_type"] == "unauthenticated"


def test_resolved_human_email_domain_is_cached_without_retaining_raw_email(monkeypatch):
    import json

    use_api(
        monkeypatch,
        FakeApi(attrs={"username": "researcher", "email": "researcher@example.org"}),
    )
    monkeypatch.setattr(analytics, "MCP_REQUEST_SUCCESS_SAMPLE_RATE", 1.0)
    viewer_info = asyncio.run(analytics.resolve_authenticated_viewer(KEY))
    assert viewer_info == {"username": "researcher", "email_domain": "example.org"}

    events = []
    tracker = analytics.AnalyticsTracker()
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    token = WandBApiManager.set_context_api_key(KEY)
    try:
        tracker.track_tool_call("query_wandb_tool", "sess-1", viewer_info, duration_ms=1)
    finally:
        WandBApiManager.reset_context_api_key(token)

    prepared = analytics._prepare_event(events[0])
    segment = map_to_segment_track(prepared)
    assert prepared["user_id"] == "researcher"
    assert prepared["actor_type"] == "user"
    assert prepared["email_domain"] == "example.org"
    assert segment is not None
    assert segment["userId"] == "researcher"
    assert segment["properties"]["actor_type"] == "user"
    assert "email_domain" not in segment["properties"]
    assert "researcher@example.org" not in json.dumps([viewer_info, prepared, segment])


def test_cold_lookup_slots_handle_sixteen_concurrent_distinct_actors(monkeypatch):
    barrier = threading.Barrier(16)

    class ConcurrentApi:
        @property
        def viewer(self):
            barrier.wait(timeout=2.0)
            return SimpleNamespace(_attrs={"username": "burst-user"})

    monkeypatch.setattr(WandBApiManager, "_viewer_lookup_client", lambda key: ConcurrentApi())
    results = []
    threads = [
        threading.Thread(
            target=lambda idx=idx: results.append(WandBApiManager.resolve_viewer_username(f"k-{idx:038d}"))
        )
        for idx in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert results == ["burst-user"] * 16
