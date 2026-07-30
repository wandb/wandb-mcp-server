from contextlib import contextmanager
from types import SimpleNamespace

import wandb_mcp_server.api_client as api_client
import wandb_mcp_server.config as cfg
from wandb_mcp_server.mcp_tools import query_wandb_gql as gql_tool


class DummyToolContext:
    def __init__(self):
        self.errors = []

    def mark_error(self, error):
        self.errors.append(error)


class FakeClient:
    def __init__(self, response=None):
        self.response = response or {
            "project": {
                "runs": {
                    "edges": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        self.executions = []

    def execute_graphql(self, query, variables=None):
        self.executions.append((query, dict(variables or {})))
        return self.response


class FakeApi:
    def __init__(self, client):
        self._service_api = client
        self.viewer = SimpleNamespace(username="tester")


def _install_fake_api(monkeypatch, client):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: FakeApi(client))

    @contextmanager
    def fake_track_tool_execution(*args, **kwargs):
        yield DummyToolContext()

    monkeypatch.setattr(gql_tool, "track_tool_execution", fake_track_tool_execution)


def _enable_hosted(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    monkeypatch.setattr(cfg, "MCP_MAX_GQL_ITEMS", 100)
    monkeypatch.setattr(cfg, "MCP_MAX_GQL_ITEMS_PER_PAGE", 50)


def test_malformed_natural_language_query_fails_before_execute(monkeypatch):
    _enable_hosted(monkeypatch)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)

    result = gql_tool.query_paginated_wandb_gql(
        "Show the latest runs in this project with summary metrics",
        variables={"entity": "entity", "project": "project"},
    )

    assert "Syntax Error" in result["errors"][0]["message"]
    assert client.executions == []


def test_scalar_selection_error_is_not_rewritten(monkeypatch):
    _enable_hosted(monkeypatch)
    upstream_error = {
        "errors": [
            {
                "message": ('Field "tags" must not have a selection since type "[String!]" has no subfields.'),
            }
        ]
    }
    client = FakeClient(response=upstream_error)
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 10) {
          edges {
            node {
              id
              tags { name }
            }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
    )

    executed_query, _ = client.executions[0]
    assert "tags {" in executed_query
    assert "first: 10" in executed_query
    assert result == upstream_error


def test_sampled_history_scalar_error_is_not_rewritten(monkeypatch):
    _enable_hosted(monkeypatch)
    upstream_error = {
        "errors": [
            {
                "message": ('Field "sampledHistory" must not have a selection since type "[JSON!]!" has no subfields.'),
            }
        ]
    }
    client = FakeClient(response=upstream_error)
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 10) {
          edges {
            node {
              id
              sampledHistory { step }
            }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
    )

    executed_query, _ = client.executions[0]
    assert "sampledHistory {" in executed_query
    assert "first: 10" in executed_query
    assert result == upstream_error


def test_ambiguous_connection_without_first_is_rejected_before_execute(
    monkeypatch,
):
    _enable_hosted(monkeypatch)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query ProjectViews($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        views {
          edges { node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
    )

    assert result["errors"][0]["error"] == "query_too_complex"
    assert "must include first" in result["errors"][0]["message"]
    assert client.executions == []


def test_existing_first_on_ambiguous_connection_is_not_changed(monkeypatch):
    _enable_hosted(monkeypatch)
    upstream_error = {
        "errors": [
            {
                "message": ('Unknown argument "first" on field "views" of type "Project".'),
            }
        ]
    }
    client = FakeClient(response=upstream_error)
    _install_fake_api(monkeypatch, client)
    query = """
    query ProjectViews($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        views(first: 10) {
          edges { node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
    )

    executed_query, _ = client.executions[0]
    assert "views(first: 10, after: $__mcp_after)" in executed_query
    assert result == upstream_error


def test_valid_run_query_still_clamps_literal_first(monkeypatch):
    _enable_hosted(monkeypatch)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 100000) {
          edges { node { id name } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
        max_items=500,
        items_per_page=500,
    )

    executed_query, variables = client.executions[0]
    assert "runs(first: 50, after: $__mcp_after)" in executed_query
    assert "100000" not in executed_query
    assert variables["__mcp_after"] is None
    assert "errors" not in result


def test_variable_default_first_is_clamped_before_execute(monkeypatch):
    _enable_hosted(monkeypatch)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!, $first: Int = 100000) {
      project(name: $project, entityName: $entity) {
        runs(first: $first) {
          edges { node { id name } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "entity", "project": "project"},
        max_items=500,
        items_per_page=500,
    )

    executed_query, variables = client.executions[0]
    assert "$first: Int = 100000" in executed_query
    assert variables["first"] == 50
    assert "errors" not in result
