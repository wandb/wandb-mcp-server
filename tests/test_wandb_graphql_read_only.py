"""Read-only contract tests for query_wandb_tool GraphQL execution."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from graphql import parse

import wandb_mcp_server.api_client as api_client
from wandb_mcp_server.mcp_tools import query_wandb_gql as gql_tool
from wandb_mcp_server.wandb_graphql import (
    GraphQLReadOnlyViolation,
    execute_graphql,
    validate_read_only_graphql,
)


@pytest.mark.parametrize(
    ("document", "operation_types"),
    [
        ("mutation Rename { renameProject(input: {}) { project { id } } }", ["mutation"]),
        ("mutation { deleteRun(input: {}) { success } }", ["mutation"]),
        ("subscription Updates { projectUpdated { id } }", ["subscription"]),
        (
            "query Viewer { viewer { id } } mutation Rename { renameProject(input: {}) { project { id } } }",
            ["mutation"],
        ),
    ],
)
def test_non_query_operations_are_rejected_before_api_creation(monkeypatch, document, operation_types):
    api_created = False

    def fail_if_called():
        nonlocal api_created
        api_created = True
        raise AssertionError("get_wandb_api must not be called for rejected GraphQL")

    monkeypatch.setattr(api_client, "get_wandb_api", fail_if_called)

    result = gql_tool.query_paginated_wandb_gql(document)

    assert api_created is False
    assert result == {
        "errors": [
            {
                "error": "read_only_violation",
                "message": (
                    "query_wandb_graphql_tool accepts GraphQL query operations only; "
                    f"rejected operation type(s): {', '.join(operation_types)}."
                ),
                "operation_types": operation_types,
            }
        ]
    }


def test_transport_revalidates_before_service_api_execution():
    executions = []
    service_api = SimpleNamespace(
        execute_graphql=lambda query, variables=None: executions.append((query, variables)) or {}
    )
    api = SimpleNamespace(_service_api=service_api)

    with pytest.raises(GraphQLReadOnlyViolation):
        execute_graphql(api, "mutation { deleteRun(input: {}) { success } }")

    assert executions == []


def test_transport_revalidates_before_sdk_compatibility_check():
    executions = []
    client = SimpleNamespace(
        execute=lambda document, variable_values=None: executions.append((document, variable_values)) or {}
    )
    api = SimpleNamespace(client=client)

    with pytest.raises(GraphQLReadOnlyViolation):
        execute_graphql(api, "subscription { projectUpdated { id } }")

    assert executions == []


def test_query_words_that_look_like_mutations_are_allowed(monkeypatch):
    executions = []
    service_api = SimpleNamespace(
        execute_graphql=lambda query, variables=None: executions.append((query, variables))
        or {"mutation": {"id": "viewer-id"}}
    )
    fake_api = SimpleNamespace(_service_api=service_api, viewer=SimpleNamespace(username="tester"))
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: fake_api)

    @contextmanager
    def fake_tracking(*args, **kwargs):
        yield SimpleNamespace(mark_error=lambda error: None)

    monkeypatch.setattr(gql_tool, "track_tool_execution", fake_tracking)
    document = """
    # The word mutation in a comment is harmless.
    query MutationReport {
      mutation: viewer { id }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(document, max_items=1, items_per_page=1)

    assert result == {"mutation": {"id": "viewer-id"}}
    assert len(executions) == 1


@pytest.mark.parametrize(
    "document",
    [
        "{ __typename }",
        "query Introspection { __schema { queryType { name } } }",
        "query Viewer { viewer { ...ViewerFields } } fragment ViewerFields on User { id }",
    ],
)
def test_read_only_query_shapes_are_allowed(document):
    parsed = validate_read_only_graphql(document)
    assert parsed == parse(document)


def test_application_graphql_transport_is_confined_to_approved_modules():
    package_root = Path(__file__).parents[1] / "src" / "wandb_mcp_server"
    approved = {
        package_root / "wandb_graphql.py",
        package_root / "wandb_report_writer.py",
        package_root / "wandb_selective_reads.py",
        package_root / "mcp_tools" / "query_wandb_gql.py",
    }
    violations = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if path not in approved and "execute_graphql" in path.read_text()
    ]

    assert violations == []
