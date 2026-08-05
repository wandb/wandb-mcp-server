"""Security and pagination tests for the opt-in raw GraphQL compatibility tool."""

from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

import wandb_mcp_server.api_client as api_client
from wandb_mcp_server.mcp_tools import query_wandb_gql as gql_tool
from wandb_mcp_server.trace_utils import count_tokens
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge


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


def test_first_argument_is_bounded_even_without_page_info_selection(monkeypatch):
    service = Service([{"project": {"runs": {"edges": [{"node": {"id": "r1"}}]}}}])
    _install(monkeypatch, service)
    query = """
    query Runs($unusualLimit: Int!) {
      project(name: "p", entityName: "e") {
        runs(first: $unusualLimit) {
          edges { node { id } }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        {"unusualLimit": 1_000_000},
        max_items=10,
        items_per_page=7,
    )

    assert service.calls[0][1]["unusualLimit"] == 7
    assert result["project"]["runs"]["edges"] == [{"node": {"id": "r1"}}]


@pytest.mark.parametrize("page_info", [None, "not-a-page-info-object"])
def test_local_item_limit_is_enforced_without_usable_page_info(monkeypatch, page_info):
    connection = {
        "edges": [{"cursor": f"c{index}", "node": {"id": f"r{index}"}} for index in range(1, 6)],
    }
    if page_info is not None:
        connection["pageInfo"] = page_info
    service = Service([{"project": {"runs": connection}}])
    _install(monkeypatch, service)
    query = """
    query Runs($batch: Int!) {
      project(name: "p", entityName: "e") {
        runs(first: $batch) {
          edges { cursor node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        {"batch": 100},
        max_items=2,
        items_per_page=2,
    )

    assert [edge["cursor"] for edge in result["project"]["runs"]["edges"]] == ["c1", "c2"]
    assert result["extensions"]["wandb_mcp"] == {
        "returned_count": 2,
        "has_more": True,
        "next_cursor": "c2",
        "truncated_by_response_budget": False,
        "limit_applied": 2,
    }


def test_local_item_limit_without_page_info_fails_when_cursor_was_not_selected(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"node": {"id": f"r{index}"}} for index in range(3)],
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!) {
          project(name: "p", entityName: "e") {
            runs(first: $batch) { edges { node { id } } }
          }
        }
        """,
        {"batch": 100},
        max_items=1,
        items_per_page=1,
    )

    assert result["errors"][0]["error"] == "pagination_cursor_unavailable"


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


def test_fragment_fanout_hits_expansion_budget_before_api(monkeypatch):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))
    fragments = []
    for index in range(20):
        body = "id" if index == 19 else f"...F{index + 1} ...F{index + 1}"
        fragments.append(f"fragment F{index} on User {{ {body} }}")
    query = "query Q { viewer { ...F0 } }\n" + "\n".join(fragments)

    result = gql_tool.query_paginated_wandb_gql(query)

    assert result["errors"][0]["error"] == "query_too_complex"
    assert "expanded selections" in result["errors"][0]["message"]


def test_malformed_forward_cursor_is_rejected_before_api(monkeypatch):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs {
          project(name: "p", entityName: "e") {
            runs(first: 1, after: 123) {
              edges { cursor node { id } }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
        """
    )

    assert result["errors"][0]["error"] == "invalid_request"
    assert "after must be a string literal, variable, or null" in result["errors"][0]["message"]


def test_variable_size_depth_and_node_limits_reject_before_api(monkeypatch):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: pytest.fail("must reject before API construction"))
    deeply_nested = "leaf"
    for _ in range(gql_tool.MAX_GRAPHQL_VARIABLE_DEPTH + 2):
        deeply_nested = {"nested": deeply_nested}
    cases = [
        {"value": "x" * gql_tool.MAX_GRAPHQL_DOCUMENT_BYTES},
        {"value": deeply_nested},
        {"value": [None] * gql_tool.MAX_GRAPHQL_VARIABLE_NODES},
    ]

    for variables in cases:
        result = gql_tool.query_paginated_wandb_gql(
            "query Viewer { viewer { id } }",
            variables,
        )
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


def test_backward_limit_stop_requires_retained_edge_cursor(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"node": {"id": f"r{index}"}} for index in range(5)],
                        "pageInfo": {"hasPreviousPage": False, "startCursor": "c0"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!) {
          project(name: "p", entityName: "e") {
            runs(last: $batch) {
              edges { node { id } }
              pageInfo { hasPreviousPage startCursor }
            }
          }
        }
        """,
        {"batch": 100},
        max_items=2,
        items_per_page=2,
    )

    assert result["errors"][0]["error"] == "pagination_cursor_unavailable"


def test_backward_limit_stop_uses_retained_boundary_cursor(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"cursor": f"c{index}", "node": {"id": f"r{index}"}} for index in range(5)],
                        "pageInfo": {"hasPreviousPage": False, "startCursor": "c0"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!) {
          project(name: "p", entityName: "e") {
            runs(last: $batch) {
              edges { cursor node { id } }
              pageInfo { hasPreviousPage startCursor }
            }
          }
        }
        """,
        {"batch": 100},
        max_items=2,
        items_per_page=2,
    )

    connection = result["project"]["runs"]
    assert [edge["cursor"] for edge in connection["edges"]] == ["c3", "c4"]
    assert connection["pageInfo"] == {"hasPreviousPage": True, "startCursor": "c3"}
    assert result["extensions"]["wandb_mcp"]["next_cursor"] == "c3"


def test_backward_page_cursor_must_advance_from_supplied_before(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"cursor": "same-cursor", "node": {"id": "r1"}}],
                        "pageInfo": {"hasPreviousPage": True, "startCursor": "same-cursor"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!, $before: String) {
          project(name: "p", entityName: "e") {
            runs(last: $batch, before: $before) {
              edges { cursor node { id } }
              pageInfo { hasPreviousPage startCursor }
            }
          }
        }
        """,
        {"batch": 1, "before": "same-cursor"},
        max_items=2,
        items_per_page=1,
    )

    assert result["errors"][0]["error"] == "pagination_cursor_non_advancing"
    assert len(service.calls) == 1


@pytest.mark.parametrize(
    "page_cursors",
    [
        ("page-1", "page-1"),
        ("page-1", "page-2", "page-1"),
    ],
)
def test_forward_pagination_rejects_non_advancing_and_cyclic_page_cursors(monkeypatch, page_cursors):
    responses = [
        {
            "project": {
                "runs": {
                    "edges": [{"cursor": f"edge-{index}", "node": {"id": f"r{index}"}}],
                    "pageInfo": {"hasNextPage": True, "endCursor": cursor},
                }
            }
        }
        for index, cursor in enumerate(page_cursors, start=1)
    ]
    service = Service(responses)
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!, $after: String) {
          project(name: "p", entityName: "e") {
            runs(first: $batch, after: $after) {
              edges { cursor node { id } }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
        """,
        {"batch": 1, "after": None},
        max_items=4,
        items_per_page=1,
    )

    assert result["errors"][0]["error"] == "pagination_cursor_non_advancing"
    assert len(service.calls) == len(page_cursors)


def test_initial_page_cursor_must_advance_from_supplied_after(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"cursor": "edge-1", "node": {"id": "r1"}}],
                        "pageInfo": {"hasNextPage": True, "endCursor": "same-cursor"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)

    result = gql_tool.query_paginated_wandb_gql(
        """
        query Runs($batch: Int!, $after: String) {
          project(name: "p", entityName: "e") {
            runs(first: $batch, after: $after) {
              edges { cursor node { id } }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
        """,
        {"batch": 1, "after": "same-cursor"},
        max_items=2,
        items_per_page=1,
    )

    assert result["errors"][0]["error"] == "pagination_cursor_non_advancing"
    assert len(service.calls) == 1


def test_paginated_response_aliases_preserve_truthful_cursor(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "aliasedRuns": {
                        "aliasedEdges": [
                            {"edgeCursor": "c1", "aliasedNode": {"nodeId": "r1"}},
                            {"edgeCursor": "c2", "aliasedNode": {"nodeId": "r2"}},
                        ],
                        "aliasedPageInfo": {"more": True, "pageEnd": "c2"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)
    query = """
    query Runs($batch: Int!) {
      project(name: "p", entityName: "e") {
        aliasedRuns: runs(first: $batch) {
          aliasedEdges: edges {
            edgeCursor: cursor
            aliasedNode: node { nodeId: id }
          }
          aliasedPageInfo: pageInfo {
            more: hasNextPage
            pageEnd: endCursor
          }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        {"batch": 100},
        max_items=1,
        items_per_page=20,
    )

    connection = result["project"]["aliasedRuns"]
    assert connection["aliasedEdges"] == [{"edgeCursor": "c1", "aliasedNode": {"nodeId": "r1"}}]
    assert connection["aliasedPageInfo"] == {"more": True, "pageEnd": "c1"}
    assert result["extensions"]["wandb_mcp"]["next_cursor"] == "c1"
    assert "edges" not in connection
    assert "pageInfo" not in connection


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


def test_transport_size_limit_returns_non_retryable_bounded_error(monkeypatch):
    service = Service()
    _install(monkeypatch, service)
    monkeypatch.setattr(
        gql_tool,
        "execute_graphql",
        lambda *args, **kwargs: (_ for _ in ()).throw(GraphQLResponseTooLarge("bounded")),
    )

    result = gql_tool.query_paginated_wandb_gql("query Viewer { viewer { custom } }")

    assert result == {
        "errors": [
            {
                "error": "response_too_large",
                "message": "The W&B GraphQL response exceeded the safety limit; request fewer items or fields",
            }
        ]
    }


def test_response_budget_uses_real_token_count_for_unicode(monkeypatch):
    service = Service([{"viewer": {"custom": "漢" * 200}}])
    _install(monkeypatch, service)
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 100)

    result = gql_tool.query_paginated_wandb_gql("query Viewer { viewer { custom } }")

    assert result["errors"][0]["error"] == "response_too_large"


def test_budget_truncation_keeps_a_truthful_resume_cursor(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [
                            {"cursor": "c1", "node": {"id": "r1", "custom": "漢" * 200}},
                            {"cursor": "c2", "node": {"id": "r2", "custom": "漢" * 200}},
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": "c2"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)
    budget = 650
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", budget)
    query = """
    query Runs($first: Int!) {
      project(name: "p", entityName: "e") {
        runs(first: $first) {
          edges { cursor node { id custom } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        {"first": 2},
        max_items=2,
        items_per_page=2,
    )

    connection = result["project"]["runs"]
    assert [edge["cursor"] for edge in connection["edges"]] == ["c1"]
    assert connection["pageInfo"] == {"hasNextPage": True, "endCursor": "c1"}
    assert result["extensions"]["wandb_mcp"] == {
        "returned_count": 1,
        "has_more": True,
        "next_cursor": "c1",
        "truncated_by_response_budget": True,
    }
    assert count_tokens(json.dumps(result, ensure_ascii=False)) <= budget


def test_budget_truncation_without_a_retained_cursor_fails_safely(monkeypatch):
    service = Service(
        [
            {
                "project": {
                    "runs": {
                        "edges": [{"node": {"id": "r1", "custom": "漢" * 500}}],
                        "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                    }
                }
            }
        ]
    )
    _install(monkeypatch, service)
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 100)
    query = """
    query Runs($first: Int!) {
      project(name: "p", entityName: "e") {
        runs(first: $first) {
          edges { node { id custom } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(query, {"first": 1})

    assert result["errors"][0]["error"] == "response_too_large"
