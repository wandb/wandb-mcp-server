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
    ],
)
def test_invalid_requests_do_not_create_api(monkeypatch, kwargs, message):
    monkeypatch.setattr(
        sdk_query.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be created for invalid requests"),
    )

    result = sdk_query.query_wandb(**kwargs)

    assert result["error"] == "invalid_request"
    assert message in result["message"]


def test_project_uses_public_sdk_and_stable_single_envelope(fake_api):
    result = sdk_query.query_wandb("entity", "project", "project")

    assert fake_api.calls == [("project", "project", "entity")]
    assert result["source"] == "wandb_sdk"
    assert result["resource"] == "project"
    assert result["item"]["id"] == "project-id"
    assert result["truncation"] == {"applied": False}


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


def test_runs_pass_sdk_filters_order_and_bound_collection(fake_api):
    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        filters={"summary_metrics.accuracy": {"$gt": 0.8}},
        order="-summary_metrics.accuracy",
        limit=2,
        include=["sweep"],
    )

    call = fake_api.calls[0]
    assert call[0:2] == ("runs", "entity/project")
    assert call[2] == {
        "filters": {"summary_metrics.accuracy": {"$gt": 0.8}},
        "order": "-summary_metrics.accuracy",
        "per_page": 3,
        "include_sweeps": True,
        "lazy": True,
    }
    assert result["count"] == 2
    assert result["truncated"] is True
    assert [item["id"] for item in result["items"]] == ["run-1", "run-2"]
    assert all("summary" not in item for item in result["items"])


def test_run_collection_can_select_summary_keys_without_full_summary(fake_api):
    run = _run("run-1")
    run.summary = {"accuracy": 0.9, "loss": 0.2, **{f"metric_{index}": index for index in range(24_000)}}
    fake_api.runs = lambda path, **kwargs: iter([run])

    result = sdk_query.query_wandb(
        "entity",
        "project",
        "runs",
        limit=1,
        summary_keys=["accuracy", "loss"],
    )

    assert result["items"][0]["summary"] == {"accuracy": 0.9, "loss": 0.2}


@pytest.mark.asyncio
async def test_public_mcp_schema_dispatches_targeted_summary_keys(fake_api, monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = create_mcp_server("stdio")
    query_tool = next(tool for tool in await server.list_tools() if tool.name == "query_wandb_tool")

    assert "summary_keys" in query_tool.inputSchema["properties"]

    await server.call_tool(
        "query_wandb_tool",
        {
            "entity_name": "entity",
            "project_name": "project",
            "resource": "runs",
            "limit": 1,
            "summary_keys": ["accuracy"],
        },
    )

    assert fake_api.calls[0][0:2] == ("runs", "entity/project")


@pytest.mark.asyncio
async def test_public_query_tools_run_sync_sdk_work_off_event_loop(fake_api, monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr("wandb_mcp_server.config.WANDB_MCP_ENABLE_RAW_GRAPHQL", True)
    server = create_mcp_server("stdio")

    for name in ("query_wandb_tool", "query_wandb_graphql_tool"):
        registered = server._tool_manager.get_tool(name).fn
        assert inspect.iscoroutinefunction(registered)
        assert not inspect.iscoroutinefunction(inspect.unwrap(registered))


@pytest.mark.asyncio
async def test_public_mcp_dispatch_enforces_hosted_summary_limit(monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setattr(sdk_query, "MCP_HOSTED_MODE", True)
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
    monkeypatch.setattr(sdk_query, "MCP_HOSTED_MODE", True)
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
    assert "limit<=3" in result["message"]


def test_collection_limit_is_clamped_to_deployment_ceiling(fake_api, monkeypatch):
    monkeypatch.setattr(sdk_query, "MCP_MAX_WANDB_QUERY_ITEMS", 1)

    result = sdk_query.query_wandb("entity", "project", "runs", limit=50)

    assert result["limit"] == 1
    assert result["count"] == 1
    assert result["truncated"] is True


def test_response_budget_drops_optional_fields_before_items(fake_api, monkeypatch):
    monkeypatch.setattr(sdk_query, "MAX_RESPONSE_TOKENS", 250)
    runs = [_run("run-1"), _run("run-2")]
    for run in runs:
        run.config = {"large": "x" * 4000}
    fake_api.runs = lambda path, **kwargs: iter(runs)

    result = sdk_query.query_wandb("entity", "project", "runs", limit=2, include=["config"])

    assert result["truncation"]["applied"] is True
    assert "config" in result["truncation"]["omitted_fields"]
    assert all("config" not in item for item in result["items"])


def test_single_and_collection_sweeps_use_public_sdk(fake_api):
    single = sdk_query.query_wandb("entity", "project", "sweep", sweep_id="sweep-1", include=["config"])

    fake_api.project_result.sweeps = lambda per_page: iter([_sweep("sweep-1"), _sweep("sweep-2")])
    collection = sdk_query.query_wandb("entity", "project", "sweeps", limit=1, include=["config"])

    assert single["item"]["config"] == {"method": "bayes"}
    assert ("sweep", "entity/project/sweep-1") in fake_api.calls
    assert ("project", "project", "entity") in fake_api.calls
    assert collection["items"][0]["id"] == "sweep-1"
    assert collection["truncated"] is True


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
    assert result["message"] == "invalid filters"


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
    assert result["message"] == "temporary initialization failure"


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
