"""Telemetry observes an actor's already-populated SDK viewer without a lookup."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wandb_mcp_server import api_client
from wandb_mcp_server.api_client import WandBApiManager


@pytest.fixture(autouse=True)
def clear_clients():
    WandBApiManager._clear_api_cache()
    yield
    WandBApiManager._clear_api_cache()


def cached_identity(key):
    token = WandBApiManager.set_context_api_key(key)
    try:
        return WandBApiManager.get_cached_viewer_info()
    finally:
        WandBApiManager.reset_context_api_key(token)


def test_cache_miss_does_not_create_api(monkeypatch):
    constructor = Mock(side_effect=AssertionError("must not authenticate for telemetry"))
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    assert cached_identity("synthetic-key") is None
    constructor.assert_not_called()


def test_viewer_is_discovered_only_after_real_operation(monkeypatch):
    class ExistingApi:
        _viewer = None

        @property
        def viewer(self):
            pytest.fail("telemetry invoked the SDK viewer property")

    api = ExistingApi()
    constructor = Mock(return_value=api)
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    WandBApiManager.get_api("synthetic-key")
    assert cached_identity("synthetic-key") is None
    api._viewer = SimpleNamespace(_attrs={"username": "known-user", "email": "not-for-telemetry@example.invalid"})
    assert cached_identity("synthetic-key") == {"username": "known-user"}
    constructor.assert_called_once()


def test_cached_viewer_is_isolated_by_actor_and_endpoint(monkeypatch):
    constructor = Mock(
        side_effect=[
            SimpleNamespace(_viewer=SimpleNamespace(_attrs={"username": "first-user"})),
            SimpleNamespace(_viewer=SimpleNamespace(_attrs={"username": "second-user"})),
        ]
    )
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    WandBApiManager.get_api("first-key")
    WandBApiManager.get_api("second-key")
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(cached_identity, ("first-key", "second-key"))) == [
            {"username": "first-user"},
            {"username": "second-user"},
        ]
    assert cached_identity("unknown-key") is None
    monkeypatch.setattr(api_client, "WANDB_API_BASE_URL", "https://other.invalid")
    assert cached_identity("first-key") is None


def test_expired_viewer_is_not_reused_or_refreshed(monkeypatch):
    constructor = Mock(return_value=SimpleNamespace(_viewer=SimpleNamespace(_attrs={"username": "old-user"})))
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    monkeypatch.setattr(WandBApiManager, "_api_cache_ttl_seconds", 0)
    WandBApiManager.get_api("synthetic-key")
    assert cached_identity("synthetic-key") is None
    constructor.assert_called_once()


@pytest.mark.parametrize(
    "attrs",
    [
        None,
        {},
        {"entity": "not-a-username"},
        {"email": "not-a-username@example.invalid"},
        {"username": "not-an-email@example.invalid"},
        {"username": "x" * 1000},
    ],
)
def test_absent_or_invalid_username_does_not_use_other_identity(monkeypatch, attrs):
    monkeypatch.setattr(
        api_client.wandb, "Api", lambda **kwargs: SimpleNamespace(_viewer=SimpleNamespace(_attrs=attrs))
    )
    WandBApiManager.get_api("synthetic-key")
    assert cached_identity("synthetic-key") is None
