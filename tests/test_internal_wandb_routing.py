"""Dedicated routing tests for internal W&B API traffic and public links."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import wandb_mcp_server.config as config
from wandb_mcp_server import analytics
from wandb_mcp_server.analytics_segment import SegmentForwarder
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools import query_wandb
from wandb_mcp_server import utils
from wandb_mcp_server import wandb_urls


def test_internal_base_url_precedes_public_for_backend_calls(monkeypatch) -> None:
    with monkeypatch.context() as context:
        context.setenv("WANDB_BASE_URL", "https://customer.wandb.io/")
        context.setenv("WANDB_INTERNAL_BASE_URL", "http://wandb-api:8081/")
        reloaded = importlib.reload(config)
        assert reloaded.WANDB_BASE_URL == "https://customer.wandb.io"
        assert reloaded.WANDB_INTERNAL_BASE_URL == "http://wandb-api:8081"
        assert reloaded.WANDB_API_BASE_URL == "http://wandb-api:8081"
    importlib.reload(config)


def test_public_base_is_backend_fallback_only_when_internal_is_unset(monkeypatch) -> None:
    with monkeypatch.context() as context:
        context.setenv("WANDB_BASE_URL", "https://customer.wandb.io/")
        context.delenv("WANDB_INTERNAL_BASE_URL", raising=False)
        reloaded = importlib.reload(config)
        assert reloaded.WANDB_INTERNAL_BASE_URL is None
        assert reloaded.WANDB_API_BASE_URL == "https://customer.wandb.io"
    importlib.reload(config)


def test_api_manager_uses_resolved_internal_url_without_retrying_public(monkeypatch) -> None:
    monkeypatch.setattr("wandb_mcp_server.api_client.WANDB_API_BASE_URL", "http://wandb-api:8081")
    api_error = RuntimeError("internal service unavailable")
    with patch("wandb_mcp_server.api_client.wandb.Api", side_effect=api_error) as api_constructor:
        with pytest.raises(RuntimeError, match="internal service unavailable"):
            WandBApiManager.get_api("k" * 40)

    api_constructor.assert_called_once()
    assert api_constructor.call_args.kwargs["overrides"] == {
        "base_url": "http://wandb-api:8081",
        "x_extra_http_headers": {"X-WandB-Workload": "mcp"},
    }


def test_segment_forwarder_prefers_internal_url(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://customer.wandb.io")
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", "http://wandb-api:8081")

    assert SegmentForwarder().base_url == "http://wandb-api:8081"


def test_analytics_dimension_keeps_public_host(monkeypatch) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://customer.wandb.io")
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", "http://wandb-api:8081")

    assert analytics._safe_wandb_base_host() == "customer.wandb.io"


def test_netrc_lookup_uses_public_host(monkeypatch) -> None:
    observed_hosts: list[str] = []

    class FakeNetrc:
        def authenticators(self, host: str):
            observed_hosts.append(host)
            return ("user", None, "k" * 40)

    monkeypatch.setenv("WANDB_BASE_URL", "https://customer.wandb.io")
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", "http://wandb-api:8081")
    monkeypatch.setattr("wandb_mcp_server.utils.os.path.exists", lambda path: True)
    monkeypatch.setattr("wandb_mcp_server.utils.netrc.netrc", lambda path: FakeNetrc())

    assert utils._wandb_api_key_via_netrc_file("/fake/netrc") == "k" * 40
    assert observed_hosts == ["customer.wandb.io"]


def test_public_link_helpers_rewrite_internal_sdk_urls(monkeypatch) -> None:
    monkeypatch.setattr(wandb_urls, "WANDB_BASE_URL", "https://customer.wandb.io")
    monkeypatch.setattr(wandb_urls, "WANDB_API_BASE_URL", "http://wandb-api:8081")

    assert wandb_urls.public_wandb_url("team", "project", "runs", "abc") == (
        "https://customer.wandb.io/team/project/runs/abc"
    )
    assert (
        wandb_urls.publicize_wandb_url("http://wandb-api:8081/team/project/reports/opaque?view=1")
        == "https://customer.wandb.io/team/project/reports/opaque?view=1"
    )
    assert ".svc" not in wandb_urls.publicize_wandb_url(
        "http://wandb-api.default.svc.cluster.local:8081/team/project/runs/abc"
    )


def test_structured_query_never_returns_internal_sdk_links(monkeypatch) -> None:
    monkeypatch.setattr(wandb_urls, "WANDB_BASE_URL", "https://customer.wandb.io")
    monkeypatch.setattr(wandb_urls, "WANDB_API_BASE_URL", "http://wandb-api:8081")
    run = SimpleNamespace(
        id="abc",
        name="display",
        state="finished",
        entity="team",
        project="project",
        url="http://wandb-api:8081/team/project/runs/abc",
        created_at=None,
        heartbeat_at=None,
        duration=None,
        group=None,
        job_type=None,
        tags=[],
        user=None,
        summary={"loss": 0.1},
    )

    serialized = query_wandb._serialize_run(
        run,
        frozenset(),
        summary_keys=None,
        include_summary=True,
    )

    assert serialized["url"] == "https://customer.wandb.io/team/project/runs/abc"
    assert "wandb-api" not in str(serialized)


def test_backend_wandb_construction_is_centralized() -> None:
    package_root = Path(query_wandb.__file__).parents[1]
    constructor_locations: list[Path] = []

    for module in package_root.rglob("*.py"):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "wandb"
                and node.func.attr == "Api"
            ):
                constructor_locations.append(module.relative_to(package_root))

    assert constructor_locations == [Path("api_client.py")]


def test_backend_call_sites_use_shared_api_manager() -> None:
    package_root = Path(query_wandb.__file__).parents[1]
    expected_boundary = {
        "server.py": "WandBApiManager",
        "mcp_tools/create_report.py": "WandBApiManager",
        "mcp_tools/log_analysis.py": "WandBApiManager",
        "mcp_tools/run_history.py": "WandBApiManager",
        "mcp_tools/query_wandb.py": "WandBApiManager",
        "mcp_tools/query_wandb_gql.py": "get_wandb_api",
    }
    for relative_path, boundary in expected_boundary.items():
        source = (package_root / relative_path).read_text()
        assert boundary in source, relative_path
