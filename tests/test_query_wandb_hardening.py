"""Regression tests for bounded v0.4 W&B query behavior."""

from __future__ import annotations

from contextlib import contextmanager
import math
from types import SimpleNamespace

import pytest
from wandb.apis.public import Runs

from wandb_mcp_server.mcp_tools import query_wandb as query_module
from wandb_mcp_server.wandb_selective_reads import (
    PROJECTED_REPORTS_QUERY,
    PROJECTED_RUNS_QUERY,
    PROJECTED_SWEEPS_QUERY,
)


@contextmanager
def _tracking(*args, **kwargs):
    yield SimpleNamespace(mark_error=lambda error: None)


class FakeServiceApi:
    app_url = "https://wandb.ai/"

    def __init__(self, handler):
        self.handler = handler
        self.executions: list[tuple[str, dict]] = []

    def execute_graphql(self, query, variables=None):
        variables = dict(variables or {})
        self.executions.append((query, variables))
        return self.handler(query, variables)


class SelectiveApi:
    def __init__(self, handler):
        self._service_api = FakeServiceApi(handler)


def _run_node(run_id: str, *, summary=None, config=None, sweep_name=None):
    node = {
        "id": f"storage-{run_id}",
        "name": run_id,
        "displayName": f"Display {run_id}",
        "state": "finished",
        "createdAt": "2026-01-01T00:00:00Z",
        "heartbeatAt": "2026-01-01T01:00:00Z",
        "computeSeconds": 60,
        "historyLineCount": 10,
        "group": "g",
        "jobType": "train",
        "tags": ["release"],
        "user": {"username": "alice", "name": "Alice"},
    }
    if summary is not None:
        node["summaryMetrics"] = summary
    if config is not None:
        node["config"] = config
    if sweep_name is not None:
        node["sweepName"] = sweep_name
    return node


def _install_api(monkeypatch, api):
    monkeypatch.setattr(query_module.WandBApiManager, "get_api", lambda: api)
    monkeypatch.setattr(query_module, "track_tool_execution", _tracking)


def test_real_wandb_028_runs_paginator_is_bounded_to_one_page(monkeypatch):
    def handler(query, variables):
        assert variables["perPage"] == 50
        return {
            "project": {
                "runCount": 50,
                "runs": {
                    "edges": [{"cursor": f"c-{index}", "node": _run_node(f"run-{index}")} for index in range(50)],
                    "pageInfo": {"hasNextPage": False, "endCursor": "c-49"},
                },
            }
        }

    service = FakeServiceApi(handler)

    class Api:
        def __init__(self):
            self.flush_calls = 0

        def flush(self):
            self.flush_calls += 1

        def runs(self, path, **kwargs):
            entity, project = path.split("/", 1)
            return Runs(
                service,
                entity,
                project,
                filters=kwargs["filters"],
                order=kwargs["order"],
                per_page=kwargs["per_page"],
                include_sweeps=kwargs["include_sweeps"],
                lazy=kwargs["lazy"],
            )

    api = Api()
    _install_api(monkeypatch, api)

    result = query_module.query_wandb("entity", "project", "runs", limit=50)

    assert result["returned_count"] == 50
    assert result["has_more"] is False
    assert len(service.executions) == 1
    assert api.flush_calls == 1


def test_config_only_single_projection_preserves_default_summary_and_nulls(monkeypatch):
    def handler(query, variables):
        assert variables["includeSummary"] is True
        assert variables["summaryKeys"] is None
        return {
            "project": {
                "run": _run_node(
                    "run-1",
                    summary={"accuracy": None, "loss": 0.1},
                    config={"learning_rate": {"value": None}},
                )
            }
        }

    api = SelectiveApi(handler)
    _install_api(monkeypatch, api)

    result = query_module.query_wandb(
        "entity",
        "project",
        "run",
        run_id="run-1",
        config_keys=["learning_rate", "missing"],
    )

    assert result["item"]["summary"] == {"accuracy": None, "loss": 0.1}
    assert result["item"]["config"] == {"learning_rate": None}
    assert result["item"]["missing_config_keys"] == ["missing"]
    assert "graphql_id" not in result["item"]


def test_projected_run_cursor_and_lightweight_sweep_have_no_n_plus_one(monkeypatch):
    def handler(query, variables):
        assert query == PROJECTED_RUNS_QUERY
        assert variables["includeSweep"] is True
        after = variables["after"]
        if after is None:
            nodes = [("cursor-1", "run-1"), ("cursor-2", "run-2")]
            has_more = True
            end_cursor = "cursor-2"
        else:
            assert after == "cursor-2"
            nodes = [("cursor-3", "run-3")]
            has_more = False
            end_cursor = "cursor-3"
        return {
            "project": {
                "runCount": 3,
                "runs": {
                    "edges": [
                        {"cursor": item_cursor, "node": _run_node(run_id, sweep_name="sweep-1")}
                        for item_cursor, run_id in nodes
                    ],
                    "pageInfo": {"hasNextPage": has_more, "endCursor": end_cursor},
                },
            }
        }

    api = SelectiveApi(handler)
    _install_api(monkeypatch, api)

    first = query_module.query_wandb("entity", "project", "runs", limit=2, include=["sweep"])
    second = query_module.query_wandb(
        "entity",
        "project",
        "runs",
        limit=2,
        include=["sweep"],
        cursor=first["next_cursor"],
    )

    assert first["next_cursor"] == "cursor-2"
    assert first["truncation"]["applied"] is True
    assert second["has_more"] is False
    assert second["next_cursor"] is None
    assert second["project_exhaustive"] is False
    assert first["items"][0]["sweep"]["id"] == "sweep-1"
    assert first["items"][0]["sweep"]["url"].endswith("/sweeps/sweep-1")
    assert len(api._service_api.executions) == 2


def test_fifty_sweeps_and_reports_each_require_one_bounded_request(monkeypatch):
    def handler(query, variables):
        if query == PROJECTED_SWEEPS_QUERY:
            assert variables["first"] == 50
            assert variables["includeConfig"] is False
            return {
                "project": {
                    "totalSweeps": 50,
                    "sweeps": {
                        "edges": [
                            {
                                "cursor": f"s-{index}",
                                "node": {
                                    "id": f"storage-s-{index}",
                                    "name": f"sweep-{index}",
                                    "displayName": f"Sweep {index}",
                                    "state": "FINISHED",
                                    "runCountExpected": 2,
                                    "runCount": 2,
                                },
                            }
                            for index in range(50)
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": "s-49"},
                    },
                }
            }
        assert query == PROJECTED_REPORTS_QUERY
        assert variables["first"] == 50
        assert variables["includeSpec"] is False
        return {
            "project": {
                "allViews": {
                    "edges": [
                        {
                            "cursor": f"r-{index}",
                            "node": {
                                "id": f"report-{index}",
                                "name": f"report-{index}",
                                "displayName": f"Report {index}",
                            },
                        }
                        for index in range(50)
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": "r-49"},
                }
            }
        }

    api = SelectiveApi(handler)
    _install_api(monkeypatch, api)

    sweeps = query_module.query_wandb("entity", "project", "sweeps", limit=50)
    reports = query_module.query_wandb("entity", "project", "reports", limit=50)

    assert len(sweeps["items"]) == 50
    assert len(reports["items"]) == 50
    assert all("config" not in item for item in sweeps["items"])
    assert all("spec" not in item for item in reports["items"])
    assert len(api._service_api.executions) == 2


def test_report_total_count_is_unknown_when_more_pages_exist(monkeypatch):
    def handler(query, variables):
        return {
            "project": {
                "allViews": {
                    "edges": [{"cursor": "r-1", "node": {"id": "report-1", "name": "report-1"}}],
                    "pageInfo": {"hasNextPage": True, "endCursor": "r-1"},
                }
            }
        }

    api = SelectiveApi(handler)
    _install_api(monkeypatch, api)

    result = query_module.query_wandb("entity", "project", "reports", limit=1)

    assert result["total_count"] is None
    assert result["next_cursor"] == "r-1"


def test_clamped_limit_does_not_claim_more_when_backend_is_exhausted(monkeypatch):
    monkeypatch.setattr(query_module, "MCP_MAX_WANDB_QUERY_ITEMS", 1)

    class Api:
        def flush(self):
            pass

        def runs(self, *args, **kwargs):
            return [_sdk_run("run-1")]

    _install_api(monkeypatch, Api())
    result = query_module.query_wandb("entity", "project", "runs", limit=50)

    assert result["requested_limit"] == 50
    assert result["limit"] == 1
    assert result["limit_clamped"] is True
    assert result["has_more"] is False
    assert result["truncated"] is False


def _sdk_run(run_id):
    return SimpleNamespace(
        id=run_id,
        name=run_id,
        state="finished",
        entity="entity",
        project="project",
        url=f"https://wandb.ai/entity/project/runs/{run_id}",
        created_at=None,
        heartbeat_at=None,
        duration=None,
        group=None,
        job_type=None,
        tags=[],
        user=None,
    )


def test_json_normalization_handles_cycles_and_non_finite_numbers():
    cyclic: dict[str, object] = {"nan": math.nan, "infinity": math.inf}
    cyclic["self"] = cyclic

    result = query_module._json_safe(cyclic)

    assert result["nan"] is None
    assert result["infinity"] is None
    assert result["self"] == {"_truncated": "cycle"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "authentication_failed"), (403, "permission_denied"), (404, "resource_not_found"), (429, "server_busy")],
)
def test_sdk_errors_map_to_stable_categories(monkeypatch, status, expected):
    class Error(RuntimeError):
        status_code = status

    monkeypatch.setattr(query_module.WandBApiManager, "get_api", lambda: (_ for _ in ()).throw(Error("secret")))
    monkeypatch.setattr(query_module, "track_tool_execution", _tracking)

    result = query_module.query_wandb("entity", "project", "project")

    assert result["error"] == expected
    assert "secret" not in result["message"]
