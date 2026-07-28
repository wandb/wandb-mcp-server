from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests
import wandb

from wandb_mcp_server import api_client
from wandb_mcp_server.api_client import WandBApiManager, wandb_server_busy_from_exception


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
    assert constructor.call_args.kwargs["overrides"]["x_extra_http_headers"] == {"X-WandB-Workload": "mcp"}


def test_shared_overrides_leave_user_agent_to_sdk(monkeypatch) -> None:
    monkeypatch.setattr(api_client, "WANDB_API_BASE_URL", "http://wandb-api:8081")

    assert api_client.wandb_api_overrides() == {
        "base_url": "http://wandb-api:8081",
        "x_extra_http_headers": {"X-WandB-Workload": "mcp"},
    }
    assert "User-Agent" not in api_client.wandb_workload_headers()


def test_real_wandb_transport_preserves_sdk_user_agent_and_workload_header() -> None:
    try:
        with patch("wandb.sdk.wandb_login._verify_login"):
            api = WandBApiManager._new_api("k" * 40)

        headers = api._service_api._settings.x_extra_http_headers
        assert headers["X-WandB-Workload"] == "mcp"
        assert headers["User-Agent"] == f"W&B Public Client {wandb.__version__}"
    finally:
        wandb.teardown()


@pytest.mark.parametrize("status_code", [429, 503])
def test_overload_errors_honor_retry_after(status_code: int) -> None:
    response = requests.Response()
    response.status_code = status_code
    response.headers["Retry-After"] = "2.5"
    http_error = requests.HTTPError(f"HTTP {status_code}", response=response)
    wrapped = RuntimeError("W&B request failed")
    wrapped.__cause__ = http_error

    busy = wandb_server_busy_from_exception(wrapped)

    assert busy is not None
    assert busy.status_code == status_code
    assert busy.as_dict() == {
        "error": "server_busy",
        "message": "The W&B service is busy; retry this tool call.",
        "retryable": True,
        "retry_after_ms": 2500,
    }


def test_non_overload_errors_are_not_reclassified() -> None:
    assert wandb_server_busy_from_exception(RuntimeError("HTTP 500")) is None


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
