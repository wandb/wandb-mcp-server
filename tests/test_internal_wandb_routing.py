"""Dedicated routing tests for internal W&B API traffic and public links."""

from __future__ import annotations

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
    assert api_constructor.call_args.kwargs["overrides"] == {"base_url": "http://wandb-api:8081"}


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


def test_backend_wandb_constructors_use_only_resolved_api_url() -> None:
    package_root = Path(query_wandb.__file__).parents[1]
    constructor_modules = (
        package_root / "api_client.py",
        package_root / "server.py",
        package_root / "mcp_tools" / "log_analysis.py",
        package_root / "mcp_tools" / "run_history.py",
    )

    for module in constructor_modules:
        source = module.read_text()
        assert "WANDB_API_BASE_URL" in source, module
        assert "WANDB_BASE_URL" not in source, module

    create_report_source = (package_root / "mcp_tools" / "create_report.py").read_text()
    report_writer_source = (package_root / "wandb_report_writer.py").read_text()
    assert "WandBApiManager.get_api" in create_report_source
    assert "wandb.Api(" not in create_report_source
    assert "WANDB_BASE_URL" not in create_report_source
    assert "WANDB_API_BASE_URL" not in create_report_source
    assert "WANDB_BASE_URL" not in report_writer_source
    assert "WANDB_API_BASE_URL" not in report_writer_source
