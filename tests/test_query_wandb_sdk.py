"""Contract tests for the SDK-backed query_wandb_tool implementation."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from wandb.apis.public import Api, Project

from wandb_mcp_server.mcp_tools import query_wandb as sdk_query
from wandb_mcp_server.server import create_mcp_server
from wandb_mcp_server.wandb_selective_reads import ProjectedResourcePage, ProjectedRunPage, SelectiveReadUnavailable


@contextmanager
def _tracking(*args, **kwargs):
    yield SimpleNamespace(mark_error=lambda error: None)


class FakeApi:
    def __init__(self):
        self.viewer = SimpleNamespace(username="viewer")
        self.calls = []
        self.project_result = SimpleNamespace(
            id="project-id",
            name="project",
            entity="entity",
            description="Project description",
            visibility="private",
            created_at="2026-01-01",
            updated_at="2026-02-01",
            tags=["production"],
            url="https://wandb.ai/entity/project/workspace",
            sweeps=lambda per_page: iter([]),
        )

    def project(self, name, entity=None):
        self.calls.append(("project", name, entity))
        return self.project_result

    def run(self, path):
        self.calls.append(("run", path))
        return _run("run-1")

    def runs(self, path, **kwargs):
        self.calls.append(("runs", path, kwargs))
        return iter([_run("run-1"), _run("run-2"), _run("run-3")])

    def sweep(self, path):
        self.calls.append(("sweep", path))
        return _sweep("sweep-1")

    def reports(self, path, **kwargs):
        self.calls.append(("reports", path, kwargs))
        return iter([_report("report-1"), _report("report-2")])


def _run(run_id):
    return SimpleNamespace(
        id=run_id,
        name=f"display-{run_id}",
        state="finished",
        entity="entity",
        project="project",
        url=f"https://wandb.ai/entity/project/runs/{run_id}",
        created_at="2026-01-01",
        heartbeat_at="2026-01-02",
        duration=12.5,
        group="baseline",
        job_type="train",
        tags=["test"],
        user=SimpleNamespace(username="alice", name="Alice", email="alice@example.com"),
        summary={"accuracy": 0.9},
        config={"learning_rate": 0.01},
        system_metrics={"cpu": 20},
        sweep=_sweep("sweep-1"),
    )


def _sweep(sweep_id):
    return SimpleNamespace(
        id=sweep_id,
        name=f"display-{sweep_id}",
        state="FINISHED",
        entity="entity",
        project="project",
        expected_run_count=10,
        url=f"https://wandb.ai/entity/project/sweeps/{sweep_id}",
        config={"method": "bayes"},
    )


def _report(report_id):
    return SimpleNamespace(
        id=report_id,
        name=f"internal-{report_id}",
        display_name=f"Display {report_id}",
        description="A report",
        user={"username": "alice"},
        created_at="2026-01-01",
        updated_at="2026-02-01",
        url=f"https://wandb.ai/entity/project/reports/{report_id}",
        spec={"panelGroups": []},
    )


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(sdk_query.WandBApiManager, "get_api", lambda: api)
    monkeypatch.setattr(sdk_query, "track_tool_execution", _tracking)
    return api


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"entity_name": "", "project_name": "project", "resource": "runs"}, "entity_name"),
        ({"entity_name": "entity", "project_name": "", "resource": "runs"}, "project_name"),
        ({"entity_name": "entity", "project_name": "project", "resource": "unknown"}, "unsupported resource"),
        ({"entity_name": "entity", "project_name": "project", "resource": []}, "unsupported resource"),
        ({"entity_name": "entity", "project_name": "project", "resource": "run"}, "run_id is required"),
        (
            {"entity_name": "entity", "project_name": "project", "resource": "sweep"},
            "sweep_id is required",
        ),
        (
            {"entity_name": "entity", "project_name": "project", "resource": "project", "filters": {}},
            "filters are supported only",
        ),
        (
            {"entity_name": "entity", "project_name": "project", "resource": "reports", "include": ["config"]},
            "unsupported include",
        ),
        ({"entity_name": "entity", "project_name": "project", "resource": "runs", "limit": 0}, "limit"),
        (
            {"entity_name": "entity", "project_name": "project", "resource": "runs", "response_mode": []},
            "response_mode",
        ),
        (
            {"entity_name": "entity", "project_name": "project", "resource": "project", "cursor": "cursor-1"},
            "cursor is supported only",
        ),
        (
            {
                "entity_name": "entity",
                "project_name": "project",
                "resource": "runs",
                "cursor": "cursor-1",
                "response_mode": "count",
            },
            "cursor is not supported",
        ),
    ],
)
def test_invalid_requests_do_not_create_api(monkeypatch, kwargs, message):
    monkeypatch.setattr(
        sdk_query.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for invalid requests"),
    )

    result = sdk_query.query_wandb(**kwargs)

    expected_error = "invalid_cursor" if "cursor" in kwargs else "invalid_request"
    assert result["error"] == expected_error
    assert message in result["message"]


def test_project_uses_fixed_projection_and_stable_single_envelope(fake_api, monkeypatch):
    projection_calls = []

    def projected(api, *, entity, project):
        projection_calls.append((api, entity, project))
        return {
            "id": "project-id",
            "name": project,
            "entity": entity,
            "description": "Project description",
            "run_count": 41,
        }

    monkeypatch.setattr(sdk_query, "fetch_project_metadata", projected)

    result = sdk_query.query_wandb("entity", "project", "project")

    assert projection_calls == [(fake_api, "entity", "project")]
    assert fake_api.calls == []
    assert result["source"] == "wandb_selective_read"
    assert result["resource"] == "project"
    assert result["item"]["id"] == "project-id"
    assert result["item"]["description"] == "Project description"
    assert result["item"]["run_count"] == 41
    assert result["item"]["url"] == "https://wandb.ai/entity/project"
    assert result["truncation"] == {"applied": False}


def test_project_projection_falls_back_to_two_bounded_public_sdk_reads(fake_api, monkeypatch):
    monkeypatch.setattr(
        sdk_query,
        "fetch_project_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )
    fake_api.project_result.description = None

    class CountedRuns:
        def __len__(self):
            return 3

    def counted_runs(path, **kwargs):
        fake_api.calls.append(("runs", path, kwargs))
        return CountedRuns()

    fake_api.runs = counted_runs

    result = sdk_query.query_wandb("entity", "project", "project")

    assert fake_api.calls == [
        ("project", "project", "entity"),
        (
            "runs",
            "entity/project",
            {"per_page": 1, "include_sweeps": False, "lazy": True},
        ),
    ]
    assert result["source"] == "wandb_sdk"
    assert result["item"]["description"] is None
    assert result["item"]["run_count"] == 3
    assert "description may be unavailable" in result["compatibility_caveat"]


def test_project_projection_preserves_actionable_upstream_errors_without_sdk_fallback(fake_api, monkeypatch):
    class UnauthorizedError(RuntimeError):
        status_code = 401

    def unauthorized(*args, **kwargs):
        raise UnauthorizedError("secret upstream response")

    monkeypatch.setattr(sdk_query, "fetch_project_metadata", unauthorized)

    result = sdk_query.query_wandb("entity", "project", "project")

    assert result["error"] == "authentication_failed"
    assert result["message"] == "W&B authentication failed"
    assert fake_api.calls == []


def test_run_includes_summary_by_default_and_requested_details(fake_api):
    result = sdk_query.query_wandb(
        "entity",
        "project",
        "run",
        run_id="run-1",
        include=["config", "system_metrics", "sweep"],
    )

    assert fake_api.calls == [("run", "entity/project/run-1")]
    assert result["item"]["summary"] == {"accuracy": 0.9}
    assert result["item"]["config"] == {"learning_rate": 0.01}
    assert result["item"]["system_metrics"] == {"cpu": 20}
    assert result["item"]["sweep"]["id"] == "sweep-1"


def test_runs_sdk_fallback_passes_filters_order_and_bounds_collection(fake_api, monkeypatch):
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )
    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        filters={"summary_metrics.accuracy": {"$gt": 0.8}},
        order="-summary_metrics.accuracy",
        limit=2,
    )

    call = fake_api.calls[0]
    assert call[0:2] == ("runs", "entity/project")
    assert call[2] == {
        "filters": {"summary_metrics.accuracy": {"$gt": 0.8}},
        "order": "-summary_metrics.accuracy",
        "per_page": 2,
        "include_sweeps": False,
        "lazy": True,
    }
    assert result["count"] == 2
    assert result["truncated"] is True
    assert [item["id"] for item in result["items"]] == ["run-1", "run-2"]
    assert all("summary" not in item for item in result["items"])


def test_run_collection_can_select_fields_without_full_summary(fake_api, monkeypatch):
    fake_api.runs = lambda *args, **kwargs: pytest.fail("projected reads must not hydrate SDK runs")
    projected_calls = []

    def projected(*args, **kwargs):
        projected_calls.append(kwargs)
        return ProjectedRunPage(
            items=[
                {
                    "id": "run-1",
                    "entity": "entity",
                    "project": "project",
                    "summary": {"accuracy": 0.9, "loss": 0.2},
                    "config": {"learning_rate": 0.01},
                }
            ],
            total_count=24_000,
            has_more=True,
            requests=1,
        )

    monkeypatch.setattr(sdk_query, "fetch_projected_runs", projected)

    filters = {
        "$and": [
            {"displayName": {"$regex": "cruise"}},
            {"$or": [{"sweep": "sweep-1"}, {"name": "arbitrary-run-id"}]},
        ]
    }
    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        filters=filters,
        order="-summary_metrics.accuracy",
        limit=50,
        summary_keys=["accuracy", "loss"],
        config_keys=["learning_rate"],
    )

    assert result["items"][0]["summary"] == {"accuracy": 0.9, "loss": 0.2}
    assert result["items"][0]["config"] == {"learning_rate": 0.01}
    assert result["total_count"] == 24_000
    assert result["returned_count"] == 1
    assert result["source"] == "wandb_selective_read"
    assert projected_calls[0]["filters"] == filters
    assert projected_calls[0]["order"] == "-summary_metrics.accuracy"
    assert projected_calls[0]["summary_keys"] == ["accuracy", "loss"]
    assert projected_calls[0]["config_keys"] == ["learning_rate"]


@pytest.mark.asyncio
async def test_public_mcp_schema_dispatches_targeted_summary_keys(fake_api, monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = create_mcp_server("stdio")
    query_tool = next(tool for tool in await server.list_tools() if tool.name == "query_wandb_tool")

    assert {
        "summary_keys",
        "config_keys",
        "response_mode",
        "cursor",
    } <= query_tool.inputSchema["properties"].keys()

    await server.call_tool(
        "query_wandb_tool",
        {
            "entity_name": "entity",
            "project_name": "project",
            "resource": "runs",
            "limit": 1,
            "summary_keys": ["accuracy"],
            "config_keys": ["learning_rate"],
        },
    )

    assert fake_api.calls[0][0:2] == ("runs", "entity/project")


@pytest.mark.parametrize(
    ("example_name", "arguments", "expected_call", "expected_shape"),
    [
        (
            "MinimalRunIdVsDisplayName-run-id",
            {"resource": "run", "run_id": "run-1"},
            ("run", "entity/project/run-1"),
            "item",
        ),
        (
            "MinimalRunIdVsDisplayName-display-name",
            {"resource": "runs", "filters": {"displayName": {"$eq": "display-run-1"}}, "limit": 1},
            ("runs", "entity/project"),
            "items",
        ),
        (
            "GetProjectInfo",
            {"resource": "project"},
            None,
            "item",
        ),
        (
            "GetSortedRuns",
            {
                "resource": "runs",
                "order": "+summary_metrics.accuracy",
                "summary_keys": ["accuracy"],
                "limit": 1,
            },
            ("runs", "entity/project"),
            "items",
        ),
        (
            "GetFilteredRuns",
            {
                "resource": "runs",
                "filters": {"state": "finished", "summary_metrics.accuracy": {"$gt": 0.8}},
                "order": "-summary_metrics.accuracy",
                "summary_keys": ["accuracy"],
                "limit": 1,
            },
            ("runs", "entity/project"),
            "items",
        ),
        (
            "GetRunByDisplayName",
            {
                "resource": "runs",
                "filters": {"displayName": {"$eq": "display-run-1"}},
                "summary_keys": ["accuracy"],
                "limit": 1,
            },
            ("runs", "entity/project"),
            "items",
        ),
    ],
)
@pytest.mark.asyncio
async def test_former_graphql_examples_succeed_through_public_mcp_boundary(
    fake_api,
    monkeypatch,
    example_name,
    arguments,
    expected_call,
    expected_shape,
):
    """Keep every former named GraphQL example working through typed MCP input."""
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    project_projection_calls = []
    if arguments["resource"] == "project":

        def projected(api, *, entity, project):
            project_projection_calls.append((api, entity, project))
            return {
                "id": "project-id",
                "name": project,
                "entity": entity,
                "description": "Project description",
                "run_count": 41,
            }

        monkeypatch.setattr(sdk_query, "fetch_project_metadata", projected)
    server = create_mcp_server("stdio")
    result = await server.call_tool(
        "query_wandb_tool",
        {"entity_name": "entity", "project_name": "project", **arguments},
    )

    assert isinstance(result, tuple), example_name
    payload = result[1]["result"]
    assert payload["resource"] == arguments["resource"]
    assert payload["source"] in {"wandb_sdk", "wandb_selective_read"}
    assert expected_shape in payload
    if expected_shape == "item":
        assert payload["item"]["id"] in {"project-id", "run-1"}
    else:
        assert payload["items"][0]["id"] == "run-1"
    if example_name == "GetProjectInfo":
        assert payload["item"]["description"] == "Project description"
        assert payload["item"]["run_count"] == 41
        assert project_projection_calls == [(fake_api, "entity", "project")]
    if arguments.get("summary_keys"):
        assert payload["items"][0]["summary"] == {"accuracy": 0.9}

    if expected_call is not None:
        call = fake_api.calls[-1]
        assert call[: len(expected_call)] == expected_call
    if arguments["resource"] == "runs":
        assert call[2]["filters"] == arguments.get("filters")
        assert call[2]["order"] == arguments.get("order", "-created_at")


def test_count_mode_uses_public_sdk_without_iterating(fake_api):
    class CountOnlyRuns:
        def __len__(self):
            return 42

        def __iter__(self):
            raise AssertionError("count mode must not iterate runs")

    fake_api.runs = lambda path, **kwargs: CountOnlyRuns()

    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        filters={"state": "finished"},
        response_mode="count",
    )

    assert result["total_count"] == 42
    assert result["response_mode"] == "count"
    assert result["project_exhaustive"] is True


@pytest.mark.asyncio
async def test_public_query_tools_run_sync_sdk_work_off_event_loop(fake_api, monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-graphql-compat")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-write")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    server = create_mcp_server("stdio")

    for name in ("query_wandb_tool", "query_wandb_graphql_tool"):
        registered = server._tool_manager.get_tool(name).fn
        assert inspect.iscoroutinefunction(registered)
        assert not inspect.iscoroutinefunction(inspect.unwrap(registered))


@pytest.mark.asyncio
async def test_public_mcp_dispatch_enforces_hosted_summary_limit(monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")
    monkeypatch.setattr(sdk_query, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(sdk_query, "MCP_MAX_FULL_DETAIL_ITEMS", 3)
    monkeypatch.setattr(
        sdk_query.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an unbounded hosted request"),
    )
    server = create_mcp_server("http")

    result = await server.call_tool(
        "query_wandb_tool",
        {
            "entity_name": "entity",
            "project_name": "project",
            "resource": "runs",
            "limit": 50,
            "include": ["summary"],
        },
    )

    assert "invalid_request" in str(result)
    assert "limit&lt;=3" in str(result) or "limit<=3" in str(result)


def test_hosted_full_collection_summary_is_rejected_before_api(monkeypatch):
    monkeypatch.setattr(sdk_query, "MCP_WORKLOAD_PROFILE", "dedicated")
    monkeypatch.setattr(sdk_query, "MCP_MAX_FULL_DETAIL_ITEMS", 10)
    monkeypatch.setattr(
        sdk_query.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for an unbounded hosted request"),
    )

    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        limit=50,
        include=["summary"],
    )

    assert result["error"] == "invalid_request"
    assert "limit<=10" in result["message"]


def test_collection_limit_is_clamped_to_deployment_ceiling(fake_api, monkeypatch):
    monkeypatch.setattr(sdk_query, "MCP_MAX_WANDB_QUERY_ITEMS", 1)

    result = sdk_query.query_wandb("entity", "project", "runs", limit=50)

    assert result["limit"] == 1
    assert result["count"] == 1
    assert result["truncated"] is True


def test_response_budget_drops_optional_fields_before_items(fake_api, monkeypatch):
    monkeypatch.setattr(sdk_query, "MAX_RESPONSE_TOKENS", 500)
    runs = [_run("run-1"), _run("run-2")]
    for run in runs:
        run.config = {"large": "x" * 4000}
    fake_api.runs = lambda path, **kwargs: iter(runs)

    result = sdk_query.query_wandb("entity", "project", "runs", limit=2, include=["config"])

    assert result["truncation"]["applied"] is True
    assert "config" in result["truncation"]["omitted_fields"]
    assert all("config" not in item for item in result["items"])


def test_single_sweep_uses_sdk_and_collection_uses_projection_without_fanout(fake_api, monkeypatch):
    single = sdk_query.query_wandb("entity", "project", "sweep", sweep_id="sweep-1", include=["config"])
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_sweeps",
        lambda *args, **kwargs: ProjectedResourcePage(
            items=[
                {
                    "id": "sweep-1",
                    "name": "display-sweep-1",
                    "entity": "entity",
                    "project": "project",
                    "config": {"method": "bayes"},
                }
            ],
            total_count=2,
            has_more=True,
            next_cursor="sweep-cursor",
            requests=1,
        ),
    )
    collection = sdk_query.query_wandb("entity", "project", "sweeps", limit=1, include=["config"])

    assert single["item"]["config"] == {"method": "bayes"}
    assert ("sweep", "entity/project/sweep-1") in fake_api.calls
    assert ("project", "project", "entity") not in fake_api.calls
    assert collection["items"][0]["id"] == "sweep-1"
    assert collection["truncated"] is True


def test_collection_fallbacks_never_reintroduce_n_plus_one_or_spec_overfetch(fake_api, monkeypatch):
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_runs",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_sweeps",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_reports",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )

    run_sweeps = sdk_query.query_wandb("entity", "project", "runs", limit=1, include=["sweep"])
    sweeps = sdk_query.query_wandb("entity", "project", "sweeps", limit=1)
    reports = sdk_query.query_wandb("entity", "project", "reports", limit=1)

    assert run_sweeps["error"] == "selective_read_unavailable"
    assert sweeps["error"] == "selective_read_unavailable"
    assert reports["error"] == "selective_read_unavailable"
    assert fake_api.calls == []


def test_explicit_report_spec_sdk_fallback_cursor_continues(fake_api, monkeypatch):
    monkeypatch.setattr(
        sdk_query,
        "fetch_projected_reports",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("projection unavailable")),
    )

    first = sdk_query.query_wandb(
        "entity",
        "project",
        "reports",
        report_name="Quarterly Review",
        limit=1,
        include=["spec"],
    )
    second = sdk_query.query_wandb(
        "entity",
        "project",
        "reports",
        report_name="Quarterly Review",
        limit=1,
        include=["spec"],
        cursor=first["next_cursor"],
    )

    assert first["items"][0]["id"] == "report-1"
    assert first["next_cursor"].startswith("mcp-query-v1:")
    assert second["items"][0]["id"] == "report-2"
    assert second["total_count"] == 2
    assert second["has_more"] is False
    assert second["next_cursor"] is None
    assert fake_api.calls == [
        ("reports", "entity/project", {"name": "Quarterly Review", "per_page": 2}),
        ("reports", "entity/project", {"name": "Quarterly Review", "per_page": 2}),
    ]


def test_reports_use_sdk_name_filter_and_optional_spec(fake_api):
    result = sdk_query.query_wandb(
        "entity",
        "project",
        "reports",
        report_name="Quarterly Review",
        limit=1,
        include=["spec"],
    )

    assert fake_api.calls == [("reports", "entity/project", {"name": "Quarterly Review", "per_page": 2})]
    assert result["items"][0]["spec"] == {"panelGroups": []}
    assert result["truncated"] is True


def test_missing_resource_returns_structured_error(fake_api):
    def missing(path):
        raise ValueError("run not found")

    fake_api.run = missing
    result = sdk_query.query_wandb("entity", "project", "run", run_id="missing")

    assert result["error"] == "resource_not_found"
    assert result["source"] == "wandb_sdk"


def test_malformed_collection_filter_returns_sdk_query_error(fake_api):
    def invalid_filter(path, **kwargs):
        raise ValueError("invalid filters")

    fake_api.runs = invalid_filter
    result = sdk_query.query_wandb("entity", "project", "runs", filters={"$bad": True})

    assert result["error"] == "sdk_query_failed"
    assert result["message"] == "W&B query failed (ValueError)"


def test_sdk_client_initialization_failure_returns_structured_error(monkeypatch):
    def fail_initialization():
        raise RuntimeError("temporary initialization failure")

    monkeypatch.setattr(
        sdk_query.WandBApiManager,
        "get_api",
        fail_initialization,
    )
    monkeypatch.setattr(sdk_query, "track_tool_execution", _tracking)

    result = sdk_query.query_wandb("entity", "project", "project")

    assert result["error"] == "sdk_query_failed"
    assert result["message"] == "W&B query failed (RuntimeError)"


def test_supported_sdk_exposes_every_public_query_operation():
    for method_name in ("project", "run", "runs", "sweep", "reports"):
        assert callable(getattr(Api, method_name))
    assert callable(Project.sweeps)


def test_default_query_module_has_no_raw_graphql_transport_usage():
    source = Path(sdk_query.__file__).read_text()

    assert "execute_graphql" not in source
    assert "_service_api" not in source
    assert "from graphql" not in source
    assert "import graphql" not in source


def test_server_has_no_top_level_raw_graphql_import():
    server_source = (Path(sdk_query.__file__).parents[1] / "server.py").read_text()
    import_section = server_source.split("def register_tools", maxsplit=1)[0]

    assert "query_wandb_gql" not in import_section
    assert "query_wandb_graphql" not in import_section
