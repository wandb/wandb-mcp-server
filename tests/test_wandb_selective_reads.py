"""Contract tests for application-owned selective W&B reads."""

from __future__ import annotations

import json

from wandb_mcp_server.wandb_graphql import validate_read_only_graphql
from wandb_mcp_server.wandb_selective_reads import (
    ARTIFACT_INVENTORY_QUERY,
    METRIC_VALUE_STEPS_QUERY,
    PROJECTED_RUNS_QUERY,
    PROJECTED_RUN_QUERY,
    PROJECT_COUNTS_QUERY,
    PROJECT_FIELDS_QUERY,
    fetch_project_counts,
    fetch_project_fields,
    fetch_metric_value_steps,
    fetch_projected_run,
    fetch_projected_runs,
)


class FakeServiceApi:
    def __init__(self):
        self.calls = []

    def execute_graphql(self, query, variables=None):
        variables = variables or {}
        self.calls.append((query, variables))
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
        raise AssertionError("unexpected query")


class FakeApi:
    def __init__(self):
        self._service_api = FakeServiceApi()


def test_every_application_owned_document_is_query_only():
    for document in (
        PROJECTED_RUNS_QUERY,
        PROJECTED_RUN_QUERY,
        PROJECT_COUNTS_QUERY,
        PROJECT_FIELDS_QUERY,
        ARTIFACT_INVENTORY_QUERY,
        METRIC_VALUE_STEPS_QUERY,
    ):
        validate_read_only_graphql(document)


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
