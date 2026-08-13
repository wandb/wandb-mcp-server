from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from unittest.mock import MagicMock

import pytest
import requests
from wandb.proto import wandb_api_pb2
from wandb.sdk.lib.service.service_connection import WandbApiFailedError

from wandb_mcp_server import api_client
from wandb_mcp_server.api_client import (
    WandBApiManager,
    wandb_server_busy_from_exception,
)


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


def _http_error(status_code: int, *, body: str = "", retry_after: str | None = None):
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode()
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError(f"{status_code} {body}", response=response)


def test_rate_limit_maps_to_bounded_retryable_server_busy() -> None:
    busy = wandb_server_busy_from_exception(_http_error(429, body="Too many requests", retry_after="120"))

    assert busy is not None
    assert busy.status_code == 429
    assert busy.as_dict() == {
        "error": "server_busy",
        "message": "The W&B service is busy; retry this tool call.",
        "retryable": True,
        "retry_after_ms": 60_000,
    }


def test_overload_service_unavailable_maps_to_server_busy() -> None:
    busy = wandb_server_busy_from_exception(_http_error(503, body="Service unavailable: capacity exhausted"))

    assert busy is not None
    assert busy.status_code == 503
    assert busy.retry_after_ms == 1_000


@pytest.mark.parametrize("status", [429, 503])
def test_real_wandb_service_error_shape_maps_to_server_busy(status: int) -> None:
    response = wandb_api_pb2.ApiErrorResponse(
        http_status=status,
        message="Service unavailable: capacity exhausted",
    )

    busy = wandb_server_busy_from_exception(WandbApiFailedError("W&B API request failed", response))

    assert busy is not None
    assert busy.status_code == status
    assert busy.retry_after_ms == 1_000


def test_unrelated_503_is_not_misclassified_as_capacity() -> None:
    assert wandb_server_busy_from_exception(_http_error(503, body="upstream certificate validation failed")) is None


def test_runtime_release_does_not_add_deferred_core_workload_header() -> None:
    package_root = Path(__file__).parents[1] / "src" / "wandb_mcp_server"
    application_source = "\n".join(
        path.read_text() for path in package_root.rglob("*.py") if "__pycache__" not in path.parts
    )

    assert "X-WandB-Workload" not in application_source
    assert "GORILLA_MCP_" not in application_source
