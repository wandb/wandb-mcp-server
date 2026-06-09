"""Tests for W&B SDK GraphQL transport compatibility."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

from wandb_mcp_server.wandb_graphql import execute_graphql


def test_query_tool_has_no_top_level_wandb_gql_import():
    module_path = Path(__file__).parents[1] / "src" / "wandb_mcp_server" / "mcp_tools" / "query_wandb_gql.py"
    tree = ast.parse(module_path.read_text())

    assert not any(isinstance(node, ast.ImportFrom) and node.module == "wandb_gql" for node in tree.body)


def test_execute_graphql_prefers_service_api_without_client():
    class ServiceApi:
        def __init__(self):
            self.calls = []

        def execute_graphql(self, query, variables=None):
            self.calls.append((query, variables))
            return {"ok": True}

    class Api:
        def __init__(self):
            self._service_api = ServiceApi()
            self.viewer = object()

    api = Api()
    result = execute_graphql(api, "query Test { viewer { id } }", {"x": 1})

    assert result == {"ok": True}
    assert api._service_api.calls == [("query Test { viewer { id } }", {"x": 1})]


def test_execute_graphql_lazily_falls_back_to_wandb_gql(monkeypatch):
    class Client:
        def __init__(self):
            self.calls = []

        def execute(self, query, *, variable_values):
            self.calls.append((query, variable_values))
            return {"ok": True}

    class Api:
        def __init__(self):
            self.client = Client()
            self.viewer = object()

    fake_wandb_gql = types.SimpleNamespace(gql=lambda query: f"parsed:{query}")
    monkeypatch.setitem(sys.modules, "wandb_gql", fake_wandb_gql)

    api = Api()
    result = execute_graphql(api, "query Test { viewer { id } }", {"x": 1})

    assert result == {"ok": True}
    assert api.client.calls == [("parsed:query Test { viewer { id } }", {"x": 1})]


def test_execute_graphql_uses_direct_bearer_request(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            calls.append(("raise_for_status",))

        def json(self):
            return {"data": {"viewer": {"username": "test"}}}

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    monkeypatch.setattr("wandb_mcp_server.wandb_graphql.requests.post", fake_post)

    result = execute_graphql(
        object(),
        "query Test { viewer { username } }",
        {"x": 1},
        api_key="wandb_v1_test",
    )

    assert result == {"data": {"viewer": {"username": "test"}}}
    url, headers, payload, timeout = calls[0]
    assert url == "https://api.wandb.ai/graphql"
    assert headers["Authorization"] == "Bearer wandb_v1_test"
    assert payload == {
        "query": "query Test { viewer { username } }",
        "variables": {"x": 1},
    }
    assert timeout == 30
    assert calls[1] == ("raise_for_status",)
