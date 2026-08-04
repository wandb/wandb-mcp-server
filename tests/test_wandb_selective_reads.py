"""Contract tests for application-owned selective W&B reads."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from wandb_mcp_server.wandb_graphql import validate_read_only_graphql
from wandb_mcp_server.wandb_selective_reads import (
    ARTIFACT_INVENTORY_QUERY,
    METRIC_VALUE_STEPS_QUERY,
    PROJECTED_RUNS_QUERY,
    PROJECTED_RUN_QUERY,
    PROJECTED_REPORTS_QUERY,
    PROJECTED_SWEEPS_QUERY,
    PROJECT_COUNTS_QUERY,
    PROJECT_FIELDS_QUERY,
    PROJECT_METADATA_QUERY,
    REGISTRY_ARTIFACT_VERSIONS_QUERY,
    SAMPLED_HISTORY_SERIES_QUERY,
    ProjectedReportCursorError,
    SelectiveReadUnavailable,
    fetch_registry_artifact_versions,
    fetch_project_counts,
    fetch_project_fields,
    fetch_project_metadata,
    fetch_metric_value_steps,
    fetch_projected_reports,
    fetch_projected_run,
    fetch_projected_runs,
    fetch_projected_sweeps,
    fetch_sampled_history_series,
)


class FakeServiceApi:
    def __init__(self):
        self.calls = []

    def execute_graphql(self, query, variables=None):
        variables = variables or {}
        self.calls.append((query, variables))
        if "MCPProjectMetadata" in query:
            return {
                "project": {
                    "id": "project-id",
                    "name": variables["project"],
                    "entityName": variables["entity"],
                    "description": "Customer project",
                    "runCount": 41,
                }
            }
        if "MCPProjectedRuns" in query:
            summary = {key: 0.9 if key == "accuracy" else 0.2 for key in variables["summaryKeys"]}
            config = {key: {"value": 0.01} for key in variables["configKeys"]}
            return {
                "project": {
                    "runCount": 24_000,
                    "runs": {
                        "edges": [
                            {
                                "cursor": "cursor-1",
                                "node": {
                                    "id": "graphql-id",
                                    "name": "run-1",
                                    "displayName": "Run 1",
                                    "state": "finished",
                                    "createdAt": "2026-01-01",
                                    "heartbeatAt": "2026-01-02",
                                    "computeSeconds": 10,
                                    "historyLineCount": 1_000,
                                    "group": "baseline",
                                    "jobType": "train",
                                    "tags": ["production"],
                                    "user": {"username": "alice", "name": "Alice"},
                                    "summaryMetrics": json.dumps(summary),
                                    "config": json.dumps(config),
                                },
                            }
                        ],
                        "pageInfo": {"endCursor": "cursor-1", "hasNextPage": False},
                    },
                }
            }
        if "MCPProjectedRun" in query:
            return {
                "project": {
                    "run": {
                        "id": "graphql-id",
                        "name": variables["run"],
                        "displayName": "Single Run",
                        "state": "finished",
                        "summaryMetrics": json.dumps({"loss": 0.2}),
                        "config": json.dumps({"model.name": {"value": "small"}}),
                    }
                }
            }
        if "MCPProjectFields" in query:
            return {
                "project": {
                    "fields": {
                        "edges": [
                            {
                                "cursor": "field-1",
                                "node": {
                                    "path": "summary_metrics.validation/loss",
                                    "type": "number",
                                },
                            }
                        ],
                        "pageInfo": {"endCursor": "field-1", "hasNextPage": False},
                    }
                }
            }
        if "MCPProjectCounts" in query:
            return {
                "project": {
                    "total": 100,
                    "finished": 80,
                    "failed": 5,
                    "crashed": 3,
                    "running": 12,
                }
            }
        if "MCPMetricValueSteps" in query:
            return {"project": {"run": {"stepsForMetricValues": [42, None]}}}
        if "MCPRegistryArtifactVersions" in query:
            return {
                "organization": {
                    "orgEntity": {
                        "artifactMemberships": {
                            "edges": [
                                {
                                    "cursor": "version-1",
                                    "node": {
                                        "versionIndex": 7,
                                        "aliases": [{"alias": "production"}],
                                        "artifactCollection": {"name": "my-model"},
                                        "artifact": {
                                            "id": "artifact-7",
                                            "state": "COMMITTED",
                                            "description": "candidate",
                                            "size": 123,
                                            "fileCount": 2,
                                            "createdAt": "2026-01-01T00:00:00Z",
                                            "updatedAt": "2026-01-02T00:00:00Z",
                                            "digest": "digest-7",
                                            "tags": [{"name": "approved"}],
                                        },
                                    },
                                }
                            ],
                            "pageInfo": {"endCursor": "version-1", "hasNextPage": False},
                        }
                    }
                }
            }
        raise AssertionError("unexpected query")


class FakeApi:
    def __init__(self):
        self._service_api = FakeServiceApi()


def _projected_collection_response(query: str, *, cursor: str, has_more: bool = True):
    connection = {
        "edges": [],
        "pageInfo": {"endCursor": cursor, "hasNextPage": has_more},
    }
    if query == PROJECTED_RUNS_QUERY:
        return {"project": {"runCount": 50, "runs": connection}}
    if query == PROJECTED_SWEEPS_QUERY:
        return {"project": {"totalSweeps": 50, "sweeps": connection}}
    if query == PROJECTED_REPORTS_QUERY:
        return {"project": {"allViews": connection}}
    raise AssertionError("unexpected query")


def _fetch_projected_collection(
    kind: str,
    api,
    *,
    cursor: str | None = None,
    limit: int = 50,
    page_size: int = 20,
):
    common = {
        "api": api,
        "entity": "entity",
        "project": "project",
        "limit": limit,
        "page_size": page_size,
        "cursor": cursor,
    }
    if kind == "runs":
        return fetch_projected_runs(
            **common,
            filters=None,
            order="-created_at",
        )
    if kind == "sweeps":
        return fetch_projected_sweeps(**common)
    if kind == "reports":
        return fetch_projected_reports(**common, report_name=None)
    raise AssertionError("unexpected collection")


def test_every_application_owned_document_is_query_only():
    for document in (
        PROJECTED_RUNS_QUERY,
        PROJECTED_RUN_QUERY,
        PROJECTED_REPORTS_QUERY,
        PROJECTED_SWEEPS_QUERY,
        PROJECT_COUNTS_QUERY,
        PROJECT_FIELDS_QUERY,
        PROJECT_METADATA_QUERY,
        ARTIFACT_INVENTORY_QUERY,
        METRIC_VALUE_STEPS_QUERY,
        SAMPLED_HISTORY_SERIES_QUERY,
        REGISTRY_ARTIFACT_VERSIONS_QUERY,
    ):
        validate_read_only_graphql(document)


def test_project_metadata_is_one_fixed_query_only_request():
    api = FakeApi()

    result = fetch_project_metadata(api, entity="entity", project="project")

    assert result == {
        "id": "project-id",
        "name": "project",
        "entity": "entity",
        "description": "Customer project",
        "run_count": 41,
    }
    assert api._service_api.calls == [(PROJECT_METADATA_QUERY, {"entity": "entity", "project": "project"})]


def test_project_metadata_wraps_transport_incompatibility_for_sdk_fallback():
    class IncompatibleServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            self.calls += 1
            raise RuntimeError("field unavailable")

    api = FakeApi()
    api._service_api = IncompatibleServiceApi()

    with pytest.raises(SelectiveReadUnavailable, match="project metadata query unavailable: RuntimeError"):
        fetch_project_metadata(api, entity="entity", project="project")

    assert api._service_api.calls == 1


def test_projected_collection_fetches_only_requested_fields_in_one_request():
    api = FakeApi()

    result = fetch_projected_runs(
        api,
        entity="entity",
        project="project",
        filters={"state": "finished"},
        order="-summary_metrics.accuracy",
        limit=50,
        page_size=51,
        summary_keys=["accuracy", "loss"],
        config_keys=["learning_rate"],
    )

    assert result.total_count == 24_000
    assert result.requests == 1
    assert result.items[0]["summary"] == {"accuracy": 0.9, "loss": 0.2}
    assert result.items[0]["config"] == {"learning_rate": 0.01}
    assert len(api._service_api.calls) == 1
    variables = api._service_api.calls[0][1]
    assert variables["summaryKeys"] == ["accuracy", "loss"]
    assert variables["configKeys"] == ["learning_rate"]
    assert json.loads(variables["filters"]) == {"state": "finished"}


def test_report_filter_prefers_exact_internal_name_without_spec_overfetch():
    class InternalNameServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            variables = dict(variables or {})
            self.calls.append((query, variables))
            return {
                "project": {
                    "allViews": {
                        "edges": [
                            {
                                "cursor": "internal-cursor",
                                "node": {
                                    "id": "report-id",
                                    "name": "internal-report-name",
                                    "displayName": "Customer-visible title",
                                    "spec": '{"should_not": "be retained"}',
                                },
                            }
                        ],
                        "pageInfo": {"endCursor": "internal-cursor", "hasNextPage": False},
                    }
                }
            }

    api = type("Api", (), {"_service_api": InternalNameServiceApi()})()

    result = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="internal-report-name",
        limit=5,
        page_size=20,
    )

    assert result.requests == 1
    assert result.items == [
        {
            "id": "report-id",
            "name": "internal-report-name",
            "display_name": "Customer-visible title",
            "description": None,
            "user": None,
            "created_at": None,
            "updated_at": None,
        }
    ]
    assert result.has_more is False
    assert result.next_cursor is None
    variables = api._service_api.calls[0][1]
    assert variables["name"] == "internal-report-name"
    assert variables["includeSpec"] is False


def test_report_internal_name_continuation_preserves_the_filtered_connection():
    class DuplicateInternalNameServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            variables = dict(variables or {})
            self.calls.append((query, variables))
            after = variables["after"]
            index = 1 if after is None else 2
            if after is not None:
                assert after == "internal-page-1"
            return {
                "project": {
                    "allViews": {
                        "edges": [
                            {
                                "cursor": f"internal-page-{index}",
                                "node": {
                                    "id": f"report-{index}",
                                    "name": "duplicate-internal-name",
                                    "displayName": f"Report {index}",
                                },
                            }
                        ],
                        "pageInfo": {
                            "endCursor": f"internal-page-{index}",
                            "hasNextPage": index == 1,
                        },
                    }
                }
            }

    api = type("Api", (), {"_service_api": DuplicateInternalNameServiceApi()})()

    first = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="duplicate-internal-name",
        limit=1,
        page_size=1,
    )
    assert [item["id"] for item in first.items] == ["report-1"]
    assert first.requests == 1
    assert first.has_more is True
    assert first.next_cursor.startswith("mcp-report-v1:")

    second = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="duplicate-internal-name",
        limit=1,
        page_size=1,
        cursor=first.next_cursor,
    )
    assert [item["id"] for item in second.items] == ["report-2"]
    assert second.requests == 1
    assert second.has_more is False
    assert second.next_cursor is None
    assert len(api._service_api.calls) == 2
    continuation_variables = api._service_api.calls[1][1]
    assert continuation_variables["name"] == "duplicate-internal-name"
    assert continuation_variables["after"] == "internal-page-1"


def test_report_filter_falls_back_to_exact_display_title_in_two_bounded_requests():
    class DisplayTitleServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            variables = dict(variables or {})
            self.calls.append((query, variables))
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
                        "edges": [
                            {
                                "cursor": "contains-1",
                                "node": {
                                    "id": "near-match",
                                    "name": "generated-near-match",
                                    "displayName": "Release report (copy)",
                                },
                            },
                            {
                                "cursor": "contains-2",
                                "node": {
                                    "id": "exact-match",
                                    "name": "generated-exact-match",
                                    "displayName": "Release report",
                                    "spec": '{"should_not": "be retained"}',
                                },
                            },
                        ],
                        "pageInfo": {"endCursor": "contains-2", "hasNextPage": False},
                    }
                }
            }

    api = type("Api", (), {"_service_api": DisplayTitleServiceApi()})()

    result = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="Release report",
        limit=5,
        page_size=20,
    )

    assert result.requests == 2
    assert [item["id"] for item in result.items] == ["exact-match"]
    assert "spec" not in result.items[0]
    assert result.total_count == 1
    assert result.has_more is False
    assert len(api._service_api.calls) == 2
    internal_variables = api._service_api.calls[0][1]
    fallback_variables = api._service_api.calls[1][1]
    assert internal_variables["name"] == "Release report"
    assert fallback_variables["name"] is None
    assert fallback_variables["first"] == 5
    assert fallback_variables["includeSpec"] is False
    assert "displayNameContains" not in PROJECTED_REPORTS_QUERY


def test_report_display_title_filter_excludes_nonexact_matches_and_continues_from_backend_cursor():
    class PaginatedDisplayTitleServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            variables = dict(variables or {})
            self.calls.append((query, variables))
            if variables["name"] is not None:
                return {
                    "project": {
                        "allViews": {
                            "edges": [],
                            "pageInfo": {"endCursor": None, "hasNextPage": False},
                        }
                    }
                }
            if variables["after"] is None:
                return {
                    "project": {
                        "allViews": {
                            "edges": [
                                {
                                    "cursor": "display-page-1",
                                    "node": {
                                        "id": "near-match",
                                        "name": "generated-near-match",
                                        "displayName": "Release report (copy)",
                                    },
                                }
                            ],
                            "pageInfo": {"endCursor": "display-page-1", "hasNextPage": True},
                        }
                    }
                }
            assert variables["after"] == "display-page-1"
            return {
                "project": {
                    "allViews": {
                        "edges": [
                            {
                                "cursor": "display-page-2",
                                "node": {
                                    "id": "exact-match",
                                    "name": "generated-exact-match",
                                    "displayName": "Release report",
                                },
                            }
                        ],
                        "pageInfo": {"endCursor": "display-page-2", "hasNextPage": False},
                    }
                }
            }

    api = type("Api", (), {"_service_api": PaginatedDisplayTitleServiceApi()})()

    first = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="Release report",
        limit=1,
        page_size=1,
    )
    assert first.items == []
    assert first.requests == 2
    assert first.has_more is True
    assert first.next_cursor.startswith("mcp-report-v1:")
    assert len(api._service_api.calls) == 2

    second = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="Release report",
        limit=1,
        page_size=1,
        cursor=first.next_cursor,
    )
    assert [item["id"] for item in second.items] == ["exact-match"]
    assert second.requests == 1
    assert second.has_more is False
    assert second.next_cursor is None
    assert len(api._service_api.calls) == 3
    continuation_variables = api._service_api.calls[2][1]
    assert continuation_variables["name"] is None
    assert continuation_variables["after"] == "display-page-1"


def test_filtered_report_cursor_rejects_malformed_or_mismatched_queries_without_transport():
    class CursorServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            variables = dict(variables or {})
            self.calls.append((query, variables))
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

    api = type("Api", (), {"_service_api": CursorServiceApi()})()
    first = fetch_projected_reports(
        api,
        entity="entity",
        project="project",
        report_name="Release report",
        limit=1,
        page_size=1,
    )
    assert first.next_cursor.startswith("mcp-report-v1:")
    calls_before_rejection = len(api._service_api.calls)

    with pytest.raises(ProjectedReportCursorError, match="does not match"):
        fetch_projected_reports(
            api,
            entity="entity",
            project="project",
            report_name="Different report",
            limit=1,
            page_size=1,
            cursor=first.next_cursor,
        )
    with pytest.raises(ProjectedReportCursorError, match="not a valid"):
        fetch_projected_reports(
            api,
            entity="entity",
            project="project",
            report_name="Release report",
            limit=1,
            page_size=1,
            cursor="display-page-1",
        )
    malformed_mode_payload = (
        base64.urlsafe_b64encode(json.dumps({"after": "display-page-1", "mode": "other", "query": "invalid"}).encode())
        .decode()
        .rstrip("=")
    )
    with pytest.raises(ProjectedReportCursorError, match="not a valid"):
        fetch_projected_reports(
            api,
            entity="entity",
            project="project",
            report_name="Release report",
            limit=1,
            page_size=1,
            cursor=f"mcp-report-v1:{malformed_mode_payload}",
        )
    assert len(api._service_api.calls) == calls_before_rejection


@pytest.mark.parametrize("kind", ["runs", "sweeps", "reports"])
def test_projected_collections_reject_a_non_advancing_cursor(kind):
    class RepeatingServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            self.calls += 1
            return _projected_collection_response(query, cursor="same-cursor")

    api = type("Api", (), {"_service_api": RepeatingServiceApi()})()

    with pytest.raises(SelectiveReadUnavailable, match="repeated a continuation cursor"):
        _fetch_projected_collection(kind, api, cursor="same-cursor")

    assert api._service_api.calls == 1


@pytest.mark.parametrize("kind", ["runs", "sweeps", "reports"])
def test_projected_collections_stop_at_the_page_request_ceiling(kind):
    class AdvancingEmptyServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            self.calls += 1
            return _projected_collection_response(query, cursor=f"cursor-{self.calls}")

    api = type("Api", (), {"_service_api": AdvancingEmptyServiceApi()})()

    result = _fetch_projected_collection(kind, api)

    # ceil(50 / 20) plus one short-page allowance, independently capped in
    # production, prevents an empty but cursor-advancing backend from spinning.
    assert result.requests == 4
    assert api._service_api.calls == 4
    assert result.items == []
    assert result.has_more is True
    assert result.next_cursor == "cursor-4"


def test_projected_page_requests_have_an_absolute_ceiling():
    class AdvancingEmptyServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            self.calls += 1
            return _projected_collection_response(query, cursor=f"cursor-{self.calls}")

    api = type("Api", (), {"_service_api": AdvancingEmptyServiceApi()})()

    result = _fetch_projected_collection("runs", api, limit=50, page_size=1)

    assert result.requests == 10
    assert api._service_api.calls == 10
    assert result.has_more is True
    assert result.next_cursor == "cursor-10"


def test_projected_collections_reject_a_previously_seen_cursor():
    class CyclingServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            cursors = ("cursor-1", "cursor-2", "cursor-1")
            cursor = cursors[self.calls]
            self.calls += 1
            return _projected_collection_response(query, cursor=cursor)

    api = type("Api", (), {"_service_api": CyclingServiceApi()})()

    with pytest.raises(SelectiveReadUnavailable, match="repeated a continuation cursor"):
        _fetch_projected_collection("reports", api, limit=50, page_size=1)

    assert api._service_api.calls == 3


@pytest.mark.parametrize("kind", ["runs", "sweeps", "reports"])
def test_projected_collections_reject_missing_continuation_cursor(kind):
    class MissingCursorServiceApi:
        def execute_graphql(self, query, variables=None):
            return _projected_collection_response(query, cursor="")

    api = type("Api", (), {"_service_api": MissingCursorServiceApi()})()

    with pytest.raises(SelectiveReadUnavailable, match="without a continuation cursor"):
        _fetch_projected_collection(kind, api)


@pytest.mark.parametrize("kind", ["runs", "sweeps", "reports"])
def test_projected_collections_resume_after_last_retained_edge_when_backend_overreturns(kind):
    class OverReturningServiceApi:
        def execute_graphql(self, query, variables=None):
            after = (variables or {}).get("after")
            start = 0 if after is None else int(after.removeprefix("cursor-"))
            edges = []
            for index in range(start + 1, 4):
                if kind == "runs":
                    node = {"name": f"run-{index}"}
                elif kind == "sweeps":
                    node = {"name": f"sweep-{index}"}
                else:
                    node = {"id": f"report-{index}", "name": f"report-{index}"}
                edges.append({"cursor": f"cursor-{index}", "node": node})
            connection = {
                "edges": edges,
                "pageInfo": {"endCursor": "cursor-3", "hasNextPage": False},
            }
            if kind == "runs":
                return {"project": {"runCount": 3, "runs": connection}}
            if kind == "sweeps":
                return {"project": {"totalSweeps": 3, "sweeps": connection}}
            return {"project": {"allViews": connection}}

    api = type("Api", (), {"_service_api": OverReturningServiceApi()})()

    first = _fetch_projected_collection(kind, api, limit=1, page_size=1)
    second = _fetch_projected_collection(kind, api, cursor=first.next_cursor, limit=1, page_size=1)

    expected_prefix = {"runs": "run", "sweeps": "sweep", "reports": "report"}[kind]
    assert first.items[0]["id"] == f"{expected_prefix}-1"
    assert first.has_more is True
    assert first.next_cursor == "cursor-1"
    assert second.items[0]["id"] == f"{expected_prefix}-2"
    assert second.has_more is True
    assert second.next_cursor == "cursor-2"


def test_single_projected_run_supports_nested_and_literal_config_keys():
    api = FakeApi()

    result = fetch_projected_run(
        api,
        entity="entity",
        project="project",
        run_id="run-1",
        summary_keys=["loss"],
        config_keys=["model.name"],
    )

    assert result is not None
    assert result["id"] == "run-1"
    assert result["summary"] == {"loss": 0.2}
    assert result["config"] == {"model.name": "small"}


def test_project_field_index_and_counts_are_server_side():
    api = FakeApi()

    fields = fetch_project_fields(
        api,
        entity="entity",
        project="project",
        limit=500,
        pattern="validation",
    )
    counts = fetch_project_counts(api, entity="entity", project="project")

    assert fields.items == [{"path": "summary_metrics.validation/loss", "type": "number"}]
    assert fields.has_more is False
    assert counts == {
        "all": 100,
        "finished": 80,
        "failed": 5,
        "crashed": 3,
        "running": 12,
    }
    assert api._service_api.calls[0][1]["pattern"] == "validation"


def test_metric_value_lookup_returns_candidate_steps_for_caller_verification():
    api = FakeApi()

    steps = fetch_metric_value_steps(
        api,
        entity="entity",
        project="project",
        run_id="run-1",
        metric="validation/step",
        values=[1000.0, 2000.0],
    )

    assert steps == [42, None]
    query, variables = api._service_api.calls[0]
    assert "stepsForMetricValues" in query
    assert variables["metric"] == "validation/step"
    assert variables["values"] == [1000.0, 2000.0]


class SampledHistoryServiceApi:
    """Small ServiceApi fake that preserves the real sampledHistory envelope."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def execute_graphql(self, query, variables=None):
        self.calls.append((query, dict(variables or {})))
        return {"project": {"run": {"sampledHistory": self.payload}}}


def test_sampled_history_fetches_disjoint_series_in_one_fixed_query():
    service = SampledHistoryServiceApi(
        [
            [{"_step": 0, "loss": 1.0}, {"_step": 2, "loss": 0.5}],
            [{"_step": 1, "eval/loss": 0.9}, {"_step": 3, "eval/loss": 0.4}],
        ]
    )
    api = type("Api", (), {"_service_api": service})()

    batch = fetch_sampled_history_series(
        api,
        entity="entity",
        project="project",
        run_id="run-1",
        keys=["loss", "eval/loss"],
        x_axis="_step",
        samples=500,
    )

    assert batch.keys == ["loss", "eval/loss"]
    assert batch.series == service.payload
    assert batch.rows_received == 4
    assert batch.samples_per_series == 500
    assert batch.requests == 1
    assert len(service.calls) == 1
    query, variables = service.calls[0]
    assert query == SAMPLED_HISTORY_SERIES_QUERY
    assert variables["entity"] == "entity"
    assert variables["project"] == "project"
    assert variables["run"] == "run-1"
    assert [json.loads(spec) for spec in variables["specs"]] == [
        {"keys": ["_step", "loss"], "samples": 500},
        {"keys": ["_step", "eval/loss"], "samples": 500},
    ]


def test_sampled_history_deduplicates_keys_and_bounds_total_input_rows():
    service = SampledHistoryServiceApi([[], [], []])
    api = type("Api", (), {"_service_api": service})()

    batch = fetch_sampled_history_series(
        api,
        entity="entity",
        project="project",
        run_id="run-1",
        keys=["loss", "accuracy", "loss", "eval/loss"],
        x_axis="epoch",
        samples=10_000,
    )

    assert batch.keys == ["loss", "accuracy", "eval/loss"]
    assert batch.samples_per_series == 3_333
    assert batch.samples_per_series * len(batch.keys) <= 10_000
    assert len(service.calls) == 1
    specs = [json.loads(spec) for spec in service.calls[0][1]["specs"]]
    assert [spec["keys"] for spec in specs] == [
        ["epoch", "_step", "loss"],
        ["epoch", "_step", "accuracy"],
        ["epoch", "_step", "eval/loss"],
    ]
    assert {spec["samples"] for spec in specs} == {3_333}


def test_sampled_history_counts_raw_rows_but_drops_unobserved_values():
    service = SampledHistoryServiceApi(
        [
            [
                {"_step": 0, "loss": None},
                {"_step": 1, "loss": float("nan")},
                {"_step": 2, "loss": float("inf")},
                {"_step": 3, "loss": 0.5},
            ],
            [{"_step": 0, "eval/loss": 0.4}],
        ]
    )
    api = type("Api", (), {"_service_api": service})()

    batch = fetch_sampled_history_series(
        api,
        entity="entity",
        project="project",
        run_id="run-1",
        keys=["loss", "eval/loss"],
        x_axis="_step",
        samples=4,
    )

    assert batch.rows_received == 5
    assert batch.series == [
        [{"_step": 3, "loss": 0.5}],
        [{"_step": 0, "eval/loss": 0.4}],
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "invalid series count"),
        ([[]], "invalid series count"),
        ([{}, []], "invalid series"),
        ([[{"_step": 0}], ["not-a-row"]], "invalid series"),
    ],
)
def test_sampled_history_rejects_malformed_batch_responses(payload, message):
    service = SampledHistoryServiceApi(payload)
    api = type("Api", (), {"_service_api": service})()

    with pytest.raises(SelectiveReadUnavailable, match=message):
        fetch_sampled_history_series(
            api,
            entity="entity",
            project="project",
            run_id="run-1",
            keys=["loss", "eval/loss"],
            x_axis="_step",
            samples=500,
        )

    assert len(service.calls) == 1


def test_sampled_history_rejects_missing_run_without_retrying_per_key():
    class MissingRunServiceApi:
        def __init__(self):
            self.calls = 0

        def execute_graphql(self, query, variables=None):
            self.calls += 1
            return {"project": {"run": None}}

    service = MissingRunServiceApi()
    api = type("Api", (), {"_service_api": service})()

    with pytest.raises(ValueError, match="run was not found"):
        fetch_sampled_history_series(
            api,
            entity="entity",
            project="project",
            run_id="missing",
            keys=["loss", "eval/loss"],
            x_axis="_step",
            samples=500,
        )

    assert service.calls == 1


def test_sampled_history_rejects_backend_over_return_without_retrying():
    service = SampledHistoryServiceApi(
        [
            [{"_step": step, "loss": float(step)} for step in range(3)],
            [{"_step": 0, "eval/loss": 1.0}],
        ]
    )
    api = type("Api", (), {"_service_api": service})()

    with pytest.raises(SelectiveReadUnavailable, match="per-series row limit"):
        fetch_sampled_history_series(
            api,
            entity="entity",
            project="project",
            run_id="run-1",
            keys=["loss", "eval/loss"],
            x_axis="_step",
            samples=2,
        )

    assert len(service.calls) == 1


@pytest.mark.parametrize("status_code", [401, 403, 404, 429, 503])
def test_sampled_history_preserves_actionable_upstream_statuses(status_code):
    error = RuntimeError(f"upstream status {status_code}")
    error.status_code = status_code

    class RaisingServiceApi:
        def execute_graphql(self, query, variables=None):
            raise error

    api = type("Api", (), {"_service_api": RaisingServiceApi()})()

    with pytest.raises(RuntimeError) as caught:
        fetch_sampled_history_series(
            api,
            entity="entity",
            project="project",
            run_id="run-1",
            keys=["loss", "eval/loss"],
            x_axis="_step",
            samples=2,
        )

    assert caught.value is error


@pytest.mark.parametrize("error", [TimeoutError("timed out"), asyncio.CancelledError()])
def test_sampled_history_preserves_timeout_and_cancellation(error):
    class RaisingServiceApi:
        def execute_graphql(self, query, variables=None):
            raise error

    api = type("Api", (), {"_service_api": RaisingServiceApi()})()

    with pytest.raises(type(error)) as caught:
        fetch_sampled_history_series(
            api,
            entity="entity",
            project="project",
            run_id="run-1",
            keys=["loss", "eval/loss"],
            x_axis="_step",
            samples=2,
        )

    assert caught.value is error


def test_registry_artifact_versions_use_fixed_ordered_query():
    api = FakeApi()

    result = fetch_registry_artifact_versions(
        api,
        organization="my-org",
        registry_name="models",
        collection_name="my-model",
        order="-createdAt",
        scan_limit=10,
    )

    assert result.requests == 1
    assert result.has_more is False
    assert result.items == [
        {
            "version": "v7",
            "name": "my-model",
            "aliases": ["production"],
            "tags": ["approved"],
            "state": "COMMITTED",
            "size": 123,
            "file_count": 2,
            "description": "candidate",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "digest": "digest-7",
        }
    ]
    _, variables = api._service_api.calls[0]
    assert json.loads(variables["registryFilter"]) == {"name": "wandb-registry-models"}
    assert json.loads(variables["collectionFilter"]) == {"name": "my-model"}
    assert variables["order"] == "-createdAt"
