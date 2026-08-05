"""Tests for W&B SDK GraphQL transport compatibility."""

from __future__ import annotations

import ast
from importlib.metadata import version
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace

from packaging.version import Version
import pytest
from wandb.apis.public.service_api import ServiceApi

from wandb_mcp_server import wandb_graphql
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge, execute_graphql


def test_query_path_has_no_wandb_gql_or_legacy_client_usage():
    package_path = Path(__file__).parents[1] / "src" / "wandb_mcp_server"
    query_module = package_path / "mcp_tools" / "query_wandb_gql.py"
    transport_module = package_path / "wandb_graphql.py"
    tree = ast.parse(query_module.read_text())
    transport_source = transport_module.read_text()

    assert not any(isinstance(node, ast.ImportFrom) and node.module == "wandb_gql" for node in tree.body)
    assert "wandb_gql" not in transport_source
    assert "api.client" not in transport_source


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


def test_supported_sdk_exposes_service_graphql_transport():
    assert callable(ServiceApi.execute_graphql)
    assert callable(ServiceApi.send_api_request)


def test_execute_graphql_reports_sdk_compatibility_error_without_service_transport():
    with pytest.raises(RuntimeError, match=r"requires wandb>=0\.28\.0 with ServiceApi GraphQL support"):
        execute_graphql(object(), "query Test { viewer { id } }")


def test_execute_graphql_rejects_oversized_decoded_response(monkeypatch):
    class ServiceApi:
        def execute_graphql(self, query, variables=None):
            return {"viewer": {"payload": "x" * 2_000_000}}

    api = type("Api", (), {"_service_api": ServiceApi()})()
    monkeypatch.setattr(wandb_graphql, "_MAX_DECODED_GRAPHQL_RESPONSE_BYTES", 1024)

    with pytest.raises(GraphQLResponseTooLarge, match="decoded response safety limit"):
        execute_graphql(api, "query Test { viewer { id } }")


@pytest.mark.parametrize(
    "payload",
    [
        {"payload": [""] * 100_000},
        {"payload": [[] for _ in range(100_000)]},
    ],
)
def test_execute_graphql_counts_empty_value_and_container_amplification(monkeypatch, payload):
    class ServiceApi:
        def execute_graphql(self, query, variables=None):
            return payload

    api = type("Api", (), {"_service_api": ServiceApi()})()
    monkeypatch.setattr(wandb_graphql, "_MAX_DECODED_GRAPHQL_RESPONSE_BYTES", 1024)

    with pytest.raises(GraphQLResponseTooLarge, match="decoded response safety limit"):
        execute_graphql(api, "query Test { viewer { id } }")


def test_execute_graphql_bounds_service_json_before_decoding(monkeypatch):
    class ServiceApi:
        def __init__(self):
            self.request = None
            self.timeout = None

        def send_api_request(self, request, timeout=None):
            self.request = request
            self.timeout = timeout
            # Deliberately invalid JSON: the size guard must reject it before
            # json.loads can observe the malformed body.
            return SimpleNamespace(graphql_response=SimpleNamespace(data_json="[" + " " * 2_000_000))

        def execute_graphql(self, query, variables=None):
            raise AssertionError("bounded transport must not call execute_graphql")

    service = ServiceApi()
    api = type("Api", (), {"_service_api": service})()
    monkeypatch.setattr(wandb_graphql, "_MAX_DECODED_GRAPHQL_RESPONSE_BYTES", 1024)

    with pytest.raises(GraphQLResponseTooLarge, match="decoded response safety limit"):
        execute_graphql(api, "query Test($x: Int!) { viewer { id } }", {"x": 1})

    assert service.request.graphql_request.query.startswith("query Test")
    assert json.loads(service.request.graphql_request.variables_json) == {"x": 1}
    assert service.timeout == wandb_graphql.MCP_WANDB_REQUEST_TIMEOUT_SECONDS


def test_lockfile_pins_supported_wandb_versions():
    lock = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())
    locked = {package["name"]: package["version"] for package in lock["package"]}

    assert locked["wandb"] == "0.28.0"
    assert locked["wandb-workspaces"] == "0.4.4"


def test_installed_wandb_versions_meet_supported_floor():
    assert Version(version("wandb")) >= Version("0.28.0")
    assert Version(version("wandb-workspaces")) >= Version("0.4.4")
