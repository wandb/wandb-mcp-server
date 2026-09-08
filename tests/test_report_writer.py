"""Regression tests for bounded W&B report persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.reports.v2.interface as wr_interface
from mcp.server.fastmcp.exceptions import ToolError
from wandb.apis.public.api import Api
from wandb.apis.public.projects import Project
from wandb.sdk.lib.service.service_connection import WandbApiFailedError

import wandb_mcp_server.mcp_tools.create_report as create_report_module
import wandb_mcp_server.wandb_urls as wandb_urls
from wandb_mcp_server.api_client import (
    WandBReportCreationFailed,
    WandBServerBusy,
    WandBWriteOutcomeUnknown,
)
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.mcp_tools.create_report import create_report
from wandb_mcp_server.wandb_report_writer import (
    WandBReportWriteError,
    save_report_bounded,
)

_API_KEY = "test-api-key-that-must-not-escape"
_INTERNAL_URL = "http://wandb-api.test.svc:8081"
_PUBLIC_URL = "https://customer.wandb.io"


def test_supported_sdk_and_workspaces_report_write_contract_is_available() -> None:
    assert callable(Api.project)
    assert callable(Api.create_project)
    assert callable(wr_interface.execute_graphql)
    assert callable(wr_interface.internal._generate_name)
    assert "mutation upsertView" in wr_interface.gql.upsert_view


class _FakeServiceApi:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.app_url = _INTERNAL_URL
        self.calls: list[tuple[str, dict]] = []
        self._result = result or {"upsertView": {"view": {"id": "report-id"}}}
        self._error = error

    def execute_graphql(self, query, variables=None):
        self.calls.append((query, dict(variables or {})))
        if self._error is not None:
            raise self._error
        return self._result


class _FakeApi:
    def __init__(
        self,
        *,
        result=None,
        error: Exception | None = None,
        project_error: Exception | None = None,
    ) -> None:
        self._service_api = _FakeServiceApi(result=result, error=error)
        self.project_calls: list[tuple[str, str]] = []
        self.create_project_calls: list[tuple[str, str]] = []
        self._project_error = project_error

    def project(self, name: str, entity: str):
        self.project_calls.append((name, entity))
        if self._project_error is None:
            return SimpleNamespace(id="project-id")

        error = self._project_error

        class _MissingOrFailedProject:
            @property
            def id(self):
                raise error

        return _MissingOrFailedProject()

    def create_project(self, name: str, entity: str) -> None:
        self.create_project_calls.append((name, entity))

    def projects(self, entity: str):
        raise AssertionError(f"unbounded projects enumeration for {entity}")


class _ProjectLookupService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def execute_graphql(self, query, variables=None, *, parse=None, **kwargs):
        self.calls.append((query, dict(variables or {})))
        if self.error is not None:
            raise self.error
        result = {"project": None}
        return parse(json.dumps(result)) if callable(parse) else result


class _PublicProjectLookupApi(_FakeApi):
    """Fake API using W&B 0.28's real lazy public Project object."""

    def __init__(self, *, lookup_error: Exception | None = None) -> None:
        super().__init__()
        self._lookup_service = _ProjectLookupService(error=lookup_error)

    def project(self, name: str, entity: str):
        self.project_calls.append((name, entity))
        return Project(self._lookup_service, entity, name, {})


def test_existing_project_uses_point_lookup_and_one_fixed_report_upsert(monkeypatch) -> None:
    api = _FakeApi()
    report = wr.Report(
        entity="test-entity",
        project="test-project",
        title="Bounded Report",
        description="description",
        blocks=[wr.P("body")],
    )
    monkeypatch.setattr(wr_interface.internal, "_generate_name", lambda: "generated-name")

    saved = save_report_bounded(report, api)

    assert saved is report
    assert report.id == "report-id"
    assert api.project_calls == [("test-project", "test-entity")]
    assert api.create_project_calls == []
    assert len(api._service_api.calls) == 1
    document, variables = api._service_api.calls[0]
    assert document == wr_interface.gql.upsert_view
    assert "mutation upsertView" in document
    assert variables["id"] is None
    assert variables["name"] == "generated-name"
    assert variables["entityName"] == "test-entity"
    assert variables["projectName"] == "test-project"
    assert variables["displayName"] == "Bounded Report"
    assert variables["description"] == "description"
    assert variables["type"] == "runs"
    assert '"blocks"' in variables["spec"]


def test_missing_project_is_created_after_bounded_point_lookup() -> None:
    api = _PublicProjectLookupApi()
    report = wr.Report(entity="entity", project="project", title="Report")

    save_report_bounded(report, api)

    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == [("project", "entity")]
    assert len(api._service_api.calls) == 1


def test_project_lookup_failure_does_not_attempt_project_creation() -> None:
    api = _PublicProjectLookupApi(
        lookup_error=WandbApiFailedError("permission denied"),
    )
    report = wr.Report(entity="entity", project="project", title="Report")

    with pytest.raises(ValueError, match="Unable to fetch project ID"):
        save_report_bounded(report, api)

    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert api._service_api.calls == []
    assert len(api._lookup_service.calls) == 1


def test_upsert_failure_is_not_retried() -> None:
    upstream_error = RuntimeError("upstream failed")
    api = _FakeApi(error=upstream_error)
    report = wr.Report(entity="entity", project="project", title="Report")

    with pytest.raises(RuntimeError, match="upstream failed"):
        save_report_bounded(report, api)

    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert len(api._service_api.calls) == 1


def test_upsert_timeout_has_unknown_outcome_and_is_not_retried() -> None:
    api = _FakeApi(error=TimeoutError("read timed out after dispatch"))
    report = wr.Report(entity="entity", project="project", title="Report")

    with pytest.raises(WandBWriteOutcomeUnknown):
        save_report_bounded(report, api)

    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert len(api._service_api.calls) == 1


def test_project_lookup_timeout_is_not_a_write_outcome_unknown() -> None:
    api = _FakeApi(project_error=TimeoutError("project lookup timed out"))
    report = wr.Report(entity="entity", project="project", title="Report")

    with pytest.raises(TimeoutError, match="project lookup timed out"):
        save_report_bounded(report, api)

    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert api._service_api.calls == []


def test_concurrent_reports_keep_api_and_target_state_isolated() -> None:
    api_a = _FakeApi(result={"upsertView": {"view": {"id": "report-a"}}})
    api_b = _FakeApi(result={"upsertView": {"view": {"id": "report-b"}}})
    report_a = wr.Report(entity="entity-a", project="project-a", title="Report A")
    report_b = wr.Report(entity="entity-b", project="project-b", title="Report B")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(save_report_bounded, report_a, api_a)
        future_b = executor.submit(save_report_bounded, report_b, api_b)
        assert future_a.result() is report_a
        assert future_b.result() is report_b

    assert report_a.id == "report-a"
    assert report_b.id == "report-b"
    assert api_a.project_calls == [("project-a", "entity-a")]
    assert api_b.project_calls == [("project-b", "entity-b")]
    assert api_a._service_api.calls[0][1]["entityName"] == "entity-a"
    assert api_a._service_api.calls[0][1]["projectName"] == "project-a"
    assert api_b._service_api.calls[0][1]["entityName"] == "entity-b"
    assert api_b._service_api.calls[0][1]["projectName"] == "project-b"


@pytest.mark.parametrize(
    "result",
    [
        None,
        {},
        {"upsertView": None},
        {"upsertView": {"view": {}}},
        {"upsertView": {"view": {"id": ""}}},
    ],
)
def test_bounded_save_rejects_invalid_upstream_response_without_echoing_it(result) -> None:
    api = _FakeApi(result=result)
    # Preserve an explicit false-y response in the fake.
    api._service_api._result = result
    report = wr.Report(entity="entity", project="project", title="Report")

    with pytest.raises(
        WandBReportWriteError,
        match="W&B returned an invalid report upsert response",
    ) as exc_info:
        save_report_bounded(report, api)

    assert repr(result) not in str(exc_info.value)
    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert len(api._service_api.calls) == 1


def test_create_report_returns_public_url_and_never_enumerates_projects(monkeypatch) -> None:
    api = _FakeApi()
    monkeypatch.setattr(wandb_urls, "WANDB_API_BASE_URL", _INTERNAL_URL)
    monkeypatch.setattr(wandb_urls, "WANDB_BASE_URL", _PUBLIC_URL)

    with (
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api_key",
            return_value=_API_KEY,
        ),
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api",
            return_value=api,
        ) as get_api,
    ):
        result = create_report(
            entity_name="entity",
            project_name="project",
            title="Customer Report",
            markdown_report_text="# Customer Report\n\nSafe body.",
        )

    assert result == {"url": "https://customer.wandb.io/entity/project/reports/Customer-Report--report-id"}
    assert api.project_calls == [("project", "entity")]
    assert api.create_project_calls == []
    assert len(api._service_api.calls) == 1
    assert get_api.call_count == 2
    assert get_api.call_args_list[0].args == (_API_KEY,)
    assert get_api.call_args_list[1].args == ()


@pytest.mark.asyncio
async def test_public_boundary_maps_upsert_timeout_to_non_retryable_unknown(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    api = _FakeApi(error=TimeoutError("upsert response timed out"))
    server = InstrumentedFastMCP("report-write-outcome-test")

    @server.tool(name="create_wandb_report_tool")
    def create_timed_out_report() -> dict[str, str]:
        return create_report("entity", "project", "Report")

    with (
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api_key",
            return_value=_API_KEY,
        ),
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api",
            return_value=api,
        ),
        pytest.raises(ToolError) as exc_info,
    ):
        await server.call_tool("create_wandb_report_tool", {})

    rendered = str(exc_info.value)
    assert '"error": "outcome_unknown"' in rendered
    assert '"retryable": false' in rendered
    assert "retry_after" not in rendered
    assert len(api._service_api.calls) == 1


def test_create_report_sanitizes_upstream_failures(monkeypatch) -> None:
    entity = "private-report-entity-canary"
    project = "private-report-project-canary"
    title = "private-report-title-canary"
    upstream_error = RuntimeError(
        f"POST {_INTERNAL_URL}/graphql failed for {entity}/{project}/{title}: Authorization=Bearer {_API_KEY}"
    )
    api = _FakeApi(error=upstream_error)
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", _INTERNAL_URL)

    with (
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api_key",
            return_value=_API_KEY,
        ),
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api",
            return_value=api,
        ),
        patch.object(create_report_module.logger, "error") as error_log,
        pytest.raises(WandBReportCreationFailed) as exc_info,
    ):
        create_report(entity, project, title)

    external_error = str(exc_info.value)
    assert _API_KEY not in external_error
    assert _INTERNAL_URL not in external_error
    assert "wandb-api.test.svc" not in external_error
    assert external_error == "The W&B report could not be created."
    error_log.assert_called_once()
    log_template, logged_error_type = error_log.call_args.args
    assert log_template == "Report creation failed after a bounded W&B write (error_type=%s)"
    assert logged_error_type == "RuntimeError"
    rendered_log_args = " ".join(map(str, error_log.call_args.args))
    assert _API_KEY not in rendered_log_args
    assert _INTERNAL_URL not in rendered_log_args
    for canary in (entity, project, title):
        assert canary not in rendered_log_args
    assert len(api._service_api.calls) == 1


@pytest.mark.asyncio
async def test_public_boundary_maps_report_failure_to_stable_bounded_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    upstream_error = RuntimeError(f"POST {_INTERNAL_URL}/graphql failed with token={_API_KEY}")
    api = _FakeApi(error=upstream_error)
    server = InstrumentedFastMCP("report-write-failure-test")

    @server.tool(name="create_wandb_report_tool")
    def create_failed_report() -> dict[str, str]:
        return create_report("entity", "project", "Report")

    with (
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api_key",
            return_value=_API_KEY,
        ),
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api",
            return_value=api,
        ),
        pytest.raises(ToolError) as exc_info,
    ):
        await server.call_tool("create_wandb_report_tool", {})

    rendered = str(exc_info.value)
    assert '"error": "report_creation_failed"' in rendered
    assert '"retryable": false' in rendered
    assert _API_KEY not in rendered
    assert _INTERNAL_URL not in rendered
    assert len(rendered) < 300
    assert len(api._service_api.calls) == 1


def test_create_report_preserves_retryable_backpressure() -> None:
    overload = RuntimeError("HTTP 429: too many requests")
    overload.response = SimpleNamespace(
        status_code=429,
        headers={"Retry-After": "2"},
        reason="Too Many Requests",
        text="",
    )
    api = _FakeApi(error=overload)

    with (
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api_key",
            return_value=_API_KEY,
        ),
        patch(
            "wandb_mcp_server.api_client.WandBApiManager.get_api",
            return_value=api,
        ),
        pytest.raises(WandBServerBusy) as exc_info,
    ):
        create_report("entity", "project", "Report")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_ms == 2_000
