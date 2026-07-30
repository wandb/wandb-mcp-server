"""Security and pagination tests for the opt-in raw GraphQL compatibility tool."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import wandb_mcp_server.api_client as api_client
from wandb_mcp_server.mcp_tools import query_wandb_gql as gql_tool


@contextmanager
def _tracking(*args, **kwargs):
    yield SimpleNamespace(mark_error=lambda error: None)


class Service:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def execute_graphql(self, query, variables=None):
        self.calls.append((query, dict(variables or {})))
        if self.responses:
            return self.responses.pop(0)
        return {"viewer": {"id": "viewer"}}


def _install(monkeypatch, service):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: SimpleNamespace(_service_api=service))
    monkeypatch.setattr(gql_tool, "track_tool_execution", _tracking)


def test_connection_hidden_in_fragment_is_bounded_by_actual_variable(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"cursor": "c1", "node": {"id": "r1"}}],
                        "pageInfo": {"hasNextPage": False, "endCursor": "c1"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)
    query = """
    query Runs($weirdBatch: Int!, $entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) { ...RunConnection }
    }
    fragment RunConnection on Project {
      runs(first: $weirdBatch) { ...ConnectionFields }
    }
    fragment ConnectionFields on RunConnection {
      edges { node { id } cursor }
      pageInfo { hasNextPage endCursor }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        {"entity": "e", "project": "p", "weirdBatch": 100_000},
        max_items=10,
        items_per_page=7,
    )

    assert service.calls[0][1]["weirdBatch"] == 7
    assert result["extensions"]["wandb_mcp"]["returned_count"] == 1


def test_inline_fragment_second_connection_is_rejected_before_api(monkeypatch):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))
    query = """
    query Multi {
      viewer {
        ... on User {
          projects(first: 2) {
            edges { node { id } }
            pageInfo { hasNextPage endCursor }
          }
          teams(first: 2) {
            edges { node { id } }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(query)

    assert result["errors"][0]["error"] == "query_too_complex"


@pytest.mark.parametrize(
    "query",
    [
        "query One { viewer { id } } query Two { viewer { name } }",
        "fragment Only on User { id }",
    ],
)
def test_exactly_one_query_operation_is_required_before_api(monkeypatch, query):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))

    result = gql_tool.query_paginated_wandb_gql(query)

    assert result["errors"][0]["error"] == "query_too_complex"


@pytest.mark.parametrize(
    ("max_items", "items_per_page"),
    [(0, 20), (-1, 20), (10, 0), (10, -1), (True, 20)],
)
def test_positive_integer_limits_are_required_before_api(monkeypatch, max_items, items_per_page):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))

    result = gql_tool.query_paginated_wandb_gql(
        "query Viewer { viewer { id } }",
        max_items=max_items,
        items_per_page=items_per_page,
    )

    assert result["errors"][0]["error"] == "invalid_request"


def test_document_size_depth_field_and_fragment_limits_reject_before_api(monkeypatch):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))
    oversized = "query Q { viewer { id } }" + (" " * gql_tool.MAX_GRAPHQL_DOCUMENT_BYTES)
    too_deep = "query Q { " + "a { " * 13 + "id" + " }" * 13 + " }"
    too_many_fields = "query Q { viewer { " + " ".join(f"f{i}" for i in range(201)) + " } }"
    fragments = "\n".join(f"fragment F{i} on User {{ id }}" for i in range(33))
    too_many_fragments = f"query Q {{ viewer {{ ...F0 }} }}\n{fragments}"

    for document in (oversized, too_deep, too_many_fields, too_many_fragments):
        result = gql_tool.query_paginated_wandb_gql(document)
        assert result["errors"][0]["error"] == "query_too_complex"


def test_limit_stop_preserves_truthful_page_info_and_cursor(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [
                            {"cursor": "c1", "node": {"id": "r1"}},
                            {"cursor": "c2", "node": {"id": "r2"}},
                        ],
                        "pageInfo": {"hasNextPage": True, "endCursor": "c2"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)
    query = """
    query Runs($first: Int!) {
      project(name: "p", entityName: "e") {
        runs(first: $first) {
          edges { cursor node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(query, {"first": 100}, max_items=1, items_per_page=20)
    connection = result["project"]["runs"]

    assert len(connection["edges"]) == 1
    assert connection["pageInfo"] == {"hasNextPage": True, "endCursor": "c1"}
    assert result["extensions"]["wandb_mcp"]["next_cursor"] == "c1"


def test_raw_query_and_variables_are_not_forwarded_to_analytics(monkeypatch):
    captured = {}

    @contextmanager
    def capture(*args, **kwargs):
        captured["parameters"] = args[2]
        yield SimpleNamespace(mark_error=lambda error: None)

    service = Service()
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: SimpleNamespace(_service_api=service))
    monkeypatch.setattr(gql_tool, "track_tool_execution", capture)

    gql_tool.query_paginated_wandb_gql(
        "query Secret($token: String!) { viewer { id } }",
        {"token": "secret-canary"},
    )

    assert "query" not in captured["parameters"]
    assert "variables" not in captured["parameters"]
    assert "secret-canary" not in str(captured)


def test_oversized_non_connection_response_returns_bounded_error(monkeypatch):
    service = Service([{"viewer": {"custom": "x" * 20_000}}])
    _install(monkeypatch, service)
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 100)

    result = gql_tool.query_paginated_wandb_gql("query Viewer { viewer { custom } }")

    assert result["errors"][0]["error"] == "response_too_large"
