from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import MagicMock

import pytest

from wandb_mcp_server import api_client
from wandb_mcp_server.api_client import WandBApiManager


@pytest.fixture(autouse=True)
def clear_api_cache(monkeypatch):
    WandBApiManager._clear_api_cache()
    monkeypatch.setattr(WandBApiManager, "_api_cache_ttl_seconds", 300.0)
    monkeypatch.setattr(WandBApiManager, "_api_cache_max_entries", 128)
    yield
    WandBApiManager._clear_api_cache()


def test_get_api_reuses_actor_client_without_storing_raw_key(monkeypatch) -> None:
    api = MagicMock()
    constructor = MagicMock(return_value=api)
    monkeypatch.setattr(api_client.wandb, "Api", constructor)

    assert WandBApiManager.get_api("actor-secret-key") is api
    assert WandBApiManager.get_api("actor-secret-key") is api

    constructor.assert_called_once()
    assert "actor-secret-key" not in repr(WandBApiManager._api_cache)


def test_get_api_keeps_actors_isolated(monkeypatch) -> None:
    constructor = MagicMock(side_effect=[MagicMock(), MagicMock()])
    monkeypatch.setattr(api_client.wandb, "Api", constructor)

    first = WandBApiManager.get_api("first-actor-key")
    second = WandBApiManager.get_api("second-actor-key")

    assert first is not second
    assert constructor.call_count == 2


def test_get_api_refreshes_expired_clients(monkeypatch) -> None:
    constructor = MagicMock(side_effect=[MagicMock(), MagicMock()])
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    monkeypatch.setattr(WandBApiManager, "_api_cache_ttl_seconds", 0.0)

    first = WandBApiManager.get_api("actor-key")
    second = WandBApiManager.get_api("actor-key")

    assert first is not second
    assert constructor.call_count == 2


def test_get_api_cache_is_bounded_and_lru(monkeypatch) -> None:
    constructor = MagicMock(side_effect=lambda **_: MagicMock())
    monkeypatch.setattr(api_client.wandb, "Api", constructor)
    monkeypatch.setattr(WandBApiManager, "_api_cache_max_entries", 2)

    first = WandBApiManager.get_api("first-key")
    WandBApiManager.get_api("second-key")
    assert WandBApiManager.get_api("first-key") is first
    WandBApiManager.get_api("third-key")
    WandBApiManager.get_api("second-key")

    assert constructor.call_count == 4
    assert len(WandBApiManager._api_cache) == 2


def test_get_api_single_flights_same_actor_initialization(monkeypatch) -> None:
    construction_started = threading.Event()
    release_construction = threading.Event()
    api = MagicMock()
    calls = 0

    def construct(**_):
        nonlocal calls
        calls += 1
        construction_started.set()
        assert release_construction.wait(timeout=2)
        return api

    monkeypatch.setattr(api_client.wandb, "Api", construct)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(WandBApiManager.get_api, "shared-actor-key")
        assert construction_started.wait(timeout=2)
        second = pool.submit(WandBApiManager.get_api, "shared-actor-key")
        release_construction.set()

    assert first.result() is api
    assert second.result() is api
    assert calls == 1


def test_get_api_initializes_different_actors_concurrently(monkeypatch) -> None:
    both_started = threading.Barrier(2)

    def construct(**_):
        both_started.wait(timeout=2)
        return MagicMock()

    monkeypatch.setattr(api_client.wandb, "Api", construct)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(WandBApiManager.get_api, "first-actor-key")
        second = pool.submit(WandBApiManager.get_api, "second-actor-key")

    assert first.result() is not second.result()
