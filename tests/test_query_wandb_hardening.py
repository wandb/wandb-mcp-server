"""Regression tests for bounded v0.4 W&B query behavior."""

from __future__ import annotations

from contextlib import contextmanager
import math
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from wandb.apis.public import Run, Runs
from wandb.apis.public.reports import BetaReport

from wandb_mcp_server.mcp_tools import query_wandb as query_module
from wandb_mcp_server.wandb_selective_reads import (
    PROJECTED_REPORTS_QUERY,
    PROJECTED_RUNS_QUERY,
    PROJECTED_SWEEPS_QUERY,
    SelectiveReadUnavailable,
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


def _projected_cursor_for(**overrides):
    query = {
        "entity_name": "entity",
        "project_name": "project",
        "resource": "runs",
        "filters": {"state": "finished"},
        "order": "-created_at",
        "report_name": None,
        "include": frozenset({"summary", "config"}),
        "summary_keys": ["accuracy"],
        "config_keys": ["learning_rate"],
    }
    query.update(overrides)
    fingerprint = query_module._collection_cursor_fingerprint(**query)
    return query_module._encode_query_cursor(
        kind="projected",
        position="private-backend-position",
        fingerprint=fingerprint,
    )


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


def test_projected_run_cursor_allows_changed_limit_and_lightweight_sweep_has_no_n_plus_one(monkeypatch):
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
        limit=1,
        include=["sweep"],
        cursor=first["next_cursor"],
    )

    assert first["next_cursor"].startswith("mcp-query-v1:")
    assert "cursor-2" not in first["next_cursor"]
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
    assert result["next_cursor"].startswith("mcp-query-v1:")
    assert "r-1" not in result["next_cursor"]


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


def test_json_normalization_does_not_treat_shared_values_as_cycles():
    shared = {"metric": 1}

    result = query_module._json_safe({"left": shared, "right": shared})

    assert result == {"left": {"metric": 1}, "right": {"metric": 1}}


def test_real_wandb_028_run_summary_serializes_without_attribute_probe():
    service = FakeServiceApi(lambda query, variables: {})
    run = Run(
        service,
        "entity",
        "project",
        "run-1",
        attrs={
            "name": "run-1",
            "displayName": "Real SDK run",
            "state": "finished",
            "summaryMetrics": '{"accuracy": 0.9, "nullable": null}',
            "config": "{}",
        },
        lazy=False,
    )

    result = query_module._serialize_run(
        run,
        frozenset(),
        include_summary=True,
    )

    assert result["summary"] == {"accuracy": 0.9, "nullable": None}


def test_dense_unicode_uses_real_token_budget(monkeypatch):
    monkeypatch.setattr(query_module, "MAX_RESPONSE_TOKENS", 100)
    payload = {
        "source": "wandb_sdk",
        "resource": "project",
        "entity": "entity",
        "project": "project",
        "item": {
            "id": "project",
            "display_name": "界" * 200,
        },
        "truncated": False,
        "truncation": {"applied": False},
    }

    result = query_module._fit_single_to_budget(payload)

    assert query_module._estimate_tokens(result) <= 100
    assert "界" * 200 not in str(result)


def test_projected_report_url_matches_wandb_028_beta_report():
    service = FakeServiceApi(lambda query, variables: {})
    attrs = {
        "id": "VmlldzoxMjM=",
        "name": "quarterly-results",
        "displayName": "Quarterly Results / Café",
        "spec": "{}",
    }
    sdk_report = BetaReport(
        service,
        attrs,
        entity="entity",
        project="project",
    )
    projected = query_module._decorate_projected_report(
        {
            "id": attrs["id"],
            "name": attrs["name"],
            "display_name": attrs["displayName"],
        },
        "entity",
        "project",
    )

    assert urlsplit(projected["url"]).path == urlsplit(sdk_report.url).path
    assert projected["url"].endswith("/reports/Quarterly-Results-Caf%C3%A9--VmlldzoxMjM")


def test_partial_sdk_projection_fallback_cursor_continues_without_skipping(monkeypatch):
    class Api:
        def flush(self):
            pass

        def runs(self, *args, **kwargs):
            return [_sdk_run("run-1"), _sdk_run("run-2")]

    _install_api(monkeypatch, Api())
    monkeypatch.setattr(
        query_module,
        "fetch_projected_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )

    first = query_module.query_wandb(
        "entity",
        "project",
        "runs",
        limit=1,
        include=["summary"],
        summary_keys=["accuracy"],
    )
    second = query_module.query_wandb(
        "entity",
        "project",
        "runs",
        limit=1,
        include=["summary"],
        summary_keys=["accuracy"],
        cursor=first["next_cursor"],
    )

    assert first["items"][0]["id"] == "run-1"
    assert first["next_cursor"].startswith("mcp-query-v1:")
    assert second["items"][0]["id"] == "run-2"
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_count_mode_flushes_cached_runs_before_construction(monkeypatch):
    class Api:
        def __init__(self):
            self.flushed = False

        def flush(self):
            self.flushed = True

        def runs(self, *args, **kwargs):
            assert self.flushed
            return [_sdk_run("run-1")]

    api = Api()
    _install_api(monkeypatch, api)

    result = query_module.query_wandb(
        "entity",
        "project",
        "runs",
        response_mode="count",
    )

    assert result["total_count"] == 1


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource": "runs", "cursor": "x" * 5000},
        {"resource": "runs", "filters": {"name": "x" * (64 * 1024)}},
        {"resource": "runs", "filters": {"metric": float("nan")}},
        {"resource": "run", "run_id": "x" * 1025},
        {"resource": "run", "run_id": "run-1", "summary_keys": ["x" * 1025]},
        {"resource": "reports", "report_name": "x" * 1025},
        {"resource": "runs", "include": ["x" * 65]},
    ],
)
def test_oversized_or_nonfinite_typed_inputs_fail_before_api(monkeypatch, kwargs):
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for invalid typed input"),
    )

    result = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        **kwargs,
    )

    expected_error = "invalid_cursor" if "cursor" in kwargs else "invalid_request"
    assert result["error"] == expected_error


@pytest.mark.parametrize(
    "resource",
    [["runs"], "x" * (64 * 1024)],
    ids=["unhashable", "oversized"],
)
def test_invalid_resource_errors_are_bounded_and_do_not_echo_input(monkeypatch, resource):
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an invalid resource"),
    )

    result = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource=resource,
    )

    assert result["error"] == "invalid_request"
    assert result["resource"] == "unknown"
    assert len(str(result)) < 512


def test_deep_mongo_filter_fails_before_api(monkeypatch):
    nested: dict[str, object] = {"state": "finished"}
    for _ in range(13):
        nested = {"$and": [nested]}
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an over-deep filter"),
    )

    result = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="runs",
        filters=nested,
    )

    assert result["error"] == "invalid_request"
    assert "depth" in result["message"]


def test_sdk_fallback_cursor_request_ceiling_is_checked_before_api(monkeypatch):
    fingerprint = query_module._collection_cursor_fingerprint(
        entity_name="entity",
        project_name="project",
        resource="runs",
        filters=None,
        order="-created_at",
        report_name=None,
        include=frozenset(),
        summary_keys=None,
        config_keys=None,
    )
    cursor = query_module._encode_query_cursor(kind="sdk_offset", position=500, fingerprint=fingerprint)
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created beyond the fallback request ceiling"),
    )

    result = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="runs",
        limit=50,
        cursor=cursor,
    )

    assert result["error"] == "invalid_cursor"
    assert "compatibility window" in result["message"]


def test_sdk_fallback_cursor_is_bound_to_original_query(monkeypatch):
    fingerprint = query_module._collection_cursor_fingerprint(
        entity_name="entity",
        project_name="project",
        resource="runs",
        filters=None,
        order="-created_at",
        report_name=None,
        include=frozenset(),
        summary_keys=None,
        config_keys=None,
    )
    cursor = query_module._encode_query_cursor(kind="sdk_offset", position=1, fingerprint=fingerprint)
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for a mismatched continuation"),
    )

    result = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="runs",
        filters={"state": "finished"},
        cursor=cursor,
    )

    assert result["error"] == "invalid_cursor"
    assert "does not match" in result["message"]


@pytest.mark.parametrize(
    "changed",
    [
        {"entity_name": "other-entity"},
        {"project_name": "other-project"},
        {"filters": {"state": "running"}},
        {"order": "+created_at"},
        {"include": ["summary", "config", "system_metrics"]},
        {"summary_keys": ["loss"]},
        {"config_keys": ["batch_size"]},
    ],
    ids=["entity", "project", "filters", "order", "include", "summary-keys", "config-keys"],
)
def test_projected_cursor_is_query_bound_before_api_construction(monkeypatch, changed):
    cursor = _projected_cursor_for()
    request = {
        "entity_name": "entity",
        "project_name": "project",
        "resource": "runs",
        "filters": {"state": "finished"},
        "order": "-created_at",
        "include": ["summary", "config"],
        "summary_keys": ["accuracy"],
        "config_keys": ["learning_rate"],
        "limit": 1,
        "cursor": cursor,
    }
    request.update(changed)
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for a mismatched continuation"),
    )

    result = query_module.query_wandb(**request)

    assert result["error"] == "invalid_cursor"
    assert "does not match" in result["message"]


def test_projected_cursor_is_bound_to_resource_and_report_selector_before_api(monkeypatch):
    collection_cursor = query_module._encode_query_cursor(
        kind="projected",
        position="private-backend-position",
        fingerprint=query_module._collection_cursor_fingerprint(
            entity_name="entity",
            project_name="project",
            resource="runs",
            filters=None,
            order="-created_at",
            report_name=None,
            include=frozenset(),
            summary_keys=None,
            config_keys=None,
        ),
    )
    report_cursor = _projected_cursor_for(
        resource="reports",
        filters=None,
        report_name="Quarterly report",
        include=frozenset(),
        summary_keys=None,
        config_keys=None,
    )
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for a mismatched continuation"),
    )

    cross_resource = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="sweeps",
        cursor=collection_cursor,
    )
    changed_selector = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="reports",
        report_name="Different report",
        cursor=report_cursor,
    )

    assert cross_resource["error"] == "invalid_cursor"
    assert changed_selector["error"] == "invalid_cursor"
    assert "does not match" in cross_resource["message"]
    assert "does not match" in changed_selector["message"]


@pytest.mark.parametrize(
    "cursor",
    [
        "private-backend-position",
        "mcp-query-v1:not-base64!",
        query_module._encode_query_cursor(kind="projected", position="position", fingerprint="wrong-query"),
    ],
    ids=["raw-backend", "malformed-envelope", "wrong-fingerprint"],
)
def test_malformed_or_unbound_typed_cursors_fail_before_api(monkeypatch, cursor):
    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an invalid continuation"),
    )

    result = query_module.query_wandb("entity", "project", "runs", cursor=cursor)

    assert result["error"] == "invalid_cursor"


def test_filtered_report_cursor_is_validated_before_api_construction(monkeypatch):
    def handler(query, variables):
        if variables["name"] is not None:
            return {
                "project": {
                    "allViews": {
                        "edges": [],
                        "pageInfo": {"endCursor": None, "hasNextPage": False},
                    }
                }
            }
        return {
            "project": {
                "allViews": {
                    "edges": [],
                    "pageInfo": {"endCursor": "display-page-1", "hasNextPage": True},
                }
            }
        }

    api = SelectiveApi(handler)
    _install_api(monkeypatch, api)
    first = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="reports",
        report_name="Release report",
        limit=1,
    )
    assert first["next_cursor"].startswith("mcp-query-v1:")

    monkeypatch.setattr(
        query_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an invalid filtered report cursor"),
    )
    mismatch = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="reports",
        report_name="Different report",
        limit=1,
        cursor=first["next_cursor"],
    )
    malformed = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="reports",
        report_name="Release report",
        limit=1,
        cursor="display-page-1",
    )
    missing_report_name = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="reports",
        limit=1,
        cursor=first["next_cursor"],
    )
    cross_resource = query_module.query_wandb(
        entity_name="entity",
        project_name="project",
        resource="runs",
        limit=1,
        cursor=first["next_cursor"],
    )

    assert mismatch["error"] == "invalid_cursor"
    assert "does not match" in mismatch["message"]
    assert malformed["error"] == "invalid_cursor"
    assert "not a valid" in malformed["message"]
    assert missing_report_name["error"] == "invalid_cursor"
    assert "does not match" in missing_report_name["message"]
    assert cross_resource["error"] == "invalid_cursor"
    assert "does not match" in cross_resource["message"]
