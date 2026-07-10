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


class FakeServiceApi:
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


class FakeServiceBackedApi:
    def __init__(self, service_api):
        self._service_api = service_api
        self.viewer = SimpleNamespace(username="tester")


def _install_fake_api(monkeypatch, client):
    monkeypatch.setattr(api_client, "get_wandb_api", lambda: FakeApi(client))

    @contextmanager
    def fake_track_tool_execution(*args, **kwargs):
        yield DummyToolContext()

    monkeypatch.setattr(gql_tool, "track_tool_execution", fake_track_tool_execution)


def _install_fake_service_api(monkeypatch, service_api):
    monkeypatch.setattr(
        api_client,
        "get_wandb_api",
        lambda: FakeServiceBackedApi(service_api),
    )

    @contextmanager
    def fake_track_tool_execution(*args, **kwargs):
        yield DummyToolContext()

    monkeypatch.setattr(gql_tool, "track_tool_execution", fake_track_tool_execution)


def _run_query():
    return """
    query Runs($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 100000) {
          edges { node { id name } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """


def test_hosted_literal_first_is_rewritten_before_execute(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    monkeypatch.setattr(cfg, "MCP_MAX_GQL_ITEMS_PER_PAGE", 50)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)

    result = gql_tool.query_paginated_wandb_gql(
        _run_query(),
        variables={"entity": "e", "project": "p"},
        max_items=100,
        items_per_page=100,
    )

    executed_query, variables = client.executions[0]
    assert "first: 50" in executed_query
    assert "100000" not in executed_query
    assert variables["limit"] == 50
    assert "errors" not in result


def test_query_executes_with_service_api_without_client(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", False)
    service_api = FakeServiceApi()
    _install_fake_service_api(monkeypatch, service_api)

    result = gql_tool.query_paginated_wandb_gql(
        _run_query(),
        variables={"entity": "e", "project": "p"},
        max_items=100,
        items_per_page=100,
    )

    executed_query, variables = service_api.executions[0]
    assert "first: 100000" in executed_query
    assert variables["limit"] == 100
    assert "errors" not in result


def test_hosted_limit_variable_is_clamped_before_execute(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    monkeypatch.setattr(cfg, "MCP_MAX_GQL_ITEMS_PER_PAGE", 50)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!, $first: Int) {
      project(name: $project, entityName: $entity) {
        runs(first: $first) {
          edges { node { id name } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "e", "project": "p", "first": 100000},
        max_items=100,
        items_per_page=100,
    )

    assert client.executions[0][1]["first"] == 50


def test_hosted_multi_connection_query_is_rejected_before_execute(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Multi($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 10) {
          edges { node { id } }
          pageInfo { hasNextPage endCursor }
        }
        sweeps(first: 10) {
          edges { node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "e", "project": "p"},
    )

    assert result["errors"][0]["error"] == "query_too_complex"
    assert client.executions == []


def test_hosted_nested_connection_query_is_rejected_before_execute(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Nested($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 10) {
          edges {
            node {
              id
              loggedArtifacts(first: 10) {
                edges { node { id } }
                pageInfo { hasNextPage endCursor }
              }
            }
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "e", "project": "p"},
    )

    assert result["errors"][0]["error"] == "query_too_complex"
    assert "Nested paginated collection" in result["errors"][0]["details"][0]
    assert client.executions == []


def test_hosted_last_pagination_is_rejected_before_execute(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", True)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)
    query = """
    query Runs($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        runs(last: 10) {
          edges { node { id } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """

    result = gql_tool.query_paginated_wandb_gql(
        query,
        variables={"entity": "e", "project": "p"},
    )

    assert result["errors"][0]["error"] == "query_too_complex"
    assert client.executions == []


def test_non_hosted_query_passes_literal_first_unchanged(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_HOSTED_MODE", False)
    client = FakeClient()
    _install_fake_api(monkeypatch, client)

    gql_tool.query_paginated_wandb_gql(
        _run_query(),
        variables={"entity": "e", "project": "p"},
        max_items=100,
        items_per_page=100,
    )

    executed_query, variables = client.executions[0]
    assert "first: 100000" in executed_query
    assert variables["limit"] == 100
