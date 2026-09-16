"""Legacy request shapes must work through the public MCP entry point."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import Draft202012Validator
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from wandb_mcp_server import api_client
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.server import create_mcp_server, register_tools


QUERY = """query Project($entity: String!, $project: String!) {
  selected: project(name: $project, entityName: $entity) { id name }
}"""
VARIABLES = {"entity": "synthetic-entity", "project": "synthetic-project", "unused": "fixture"}
EXPECTED = {"selected": {"id": "global-project-id", "name": "synthetic-project"}}


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-write")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    monkeypatch.setenv("WF_TRACE_SERVER_URL", "https://trace.wandb.ai")
    calls = []

    def execute_graphql(query, variables=None):
        calls.append((query, variables))
        return EXPECTED

    monkeypatch.setattr(
        api_client,
        "get_wandb_api",
        lambda: SimpleNamespace(_service_api=SimpleNamespace(execute_graphql=execute_graphql)),
    )
    return calls


def _payload(result):
    if isinstance(result, tuple):
        return json.loads(result[0][0].text)
    if hasattr(result, "content"):
        return json.loads(result.content[0].text)
    return json.loads(result[0].text)


@pytest.mark.asyncio
@pytest.mark.parametrize("workload", ["shared", "dedicated", "local"])
@pytest.mark.parametrize("access", ["read-only", "read-write"])
async def test_observed_legacy_shape_dispatches_without_modern_fields(backend, monkeypatch, workload, access):
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", workload)
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", access)
    server = create_mcp_server("stdio")
    try:
        result = await server.call_tool("query_wandb_tool", {"query": QUERY, "variables": VARIABLES})
        assert _payload(result) == EXPECTED
        assert len(backend) == 1
        assert backend[0][1] == VARIABLES
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_advertised_schema_accepts_both_interfaces_and_rejects_ambiguous_inputs(backend):
    server = create_mcp_server("stdio")
    try:
        tool = next(tool for tool in await server.list_tools() if tool.name == "query_wandb_tool")
        validator = Draft202012Validator(tool.inputSchema)
        validator.validate({"query": QUERY, "variables": VARIABLES})
        validator.validate({"entity_name": "e", "project_name": "p", "resource": "runs"})
        for arguments in (
            {},
            {"variables": VARIABLES},
            {"query": ""},
            {"query": None},
            {"query": QUERY, "limit": 50},
            {"query": QUERY, "entity_name": None},
            {"entity_name": "e", "project_name": "p", "resource": "runs", "max_items": 100},
        ):
            assert list(validator.iter_errors(arguments)), arguments
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"variables": VARIABLES},
        {"max_items": 100},
        {"query": ""},
        {"query": "  "},
        {"query": None},
        {"query": QUERY, "limit": 50},
        {"query": QUERY, "response_mode": "items"},
        {"query": QUERY, "entity_name": None},
    ],
)
async def test_invalid_interface_fails_before_backend_even_for_explicit_defaults(backend, arguments):
    server = create_mcp_server("stdio")
    try:
        with pytest.raises(Exception, match="query|interface|GraphQL|structured"):
            await server.call_tool("query_wandb_tool", arguments)
        assert backend == []
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_official_http_client_legacy_success_and_mutation_error(backend):
    server = InstrumentedFastMCP("compat-http", stateless_http=True, json_response=True)
    register_tools(server)
    app = server.streamable_http_app()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        initialized = await client.initialize()
                        assert initialized.protocolVersion == "2025-11-25"
                        result = await client.call_tool("query_wandb_tool", {"query": QUERY, "variables": VARIABLES})
                        assert result.isError is False
                        assert _payload(result) == EXPECTED
                        assert result.structuredContent == {"result": EXPECTED}
                        rejected = await client.call_tool(
                            "query_wandb_tool", {"query": "mutation Delete { deleteRun(input: {}) { success } }"}
                        )
                        assert rejected.isError is True
                        assert len(backend) == 1
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_official_http_client_partial_graphql_error_preserves_data_without_private_details(backend, monkeypatch):
    from wandb_mcp_server import analytics
    from wandb_mcp_server.config import MAX_RESPONSE_TOKENS
    from wandb_mcp_server.trace_utils import count_tokens_conservative

    canary = "private-query-and-upstream-error-canary"
    query = """query Runs($entity: String!, $project: String!, $after: String) {
      selected: project(entityName: $entity, name: $project) {
        runs(first: 1, after: $after) {
          edges { cursor node { id name } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }"""
    edge = {"cursor": "cursor-one", "node": {"id": "global-one", "name": "run-one"}}

    def execute(document, variables=None):
        backend.append((document, variables))
        if len(backend) == 1:
            return {
                "selected": {
                    "runs": {
                        "edges": [edge],
                        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-one"},
                    }
                }
            }
        return {"errors": [{"message": canary, "path": [canary], "extensions": {"detail": canary}}]}

    monkeypatch.setattr(
        api_client, "get_wandb_api", lambda: SimpleNamespace(_service_api=SimpleNamespace(execute_graphql=execute))
    )
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    tracker = analytics.AnalyticsTracker()
    events = []
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(event))
    monkeypatch.setattr(analytics, "get_analytics_tracker", lambda: tracker)
    server = InstrumentedFastMCP("partial-compat-http", stateless_http=True, json_response=True)
    register_tools(server)
    app = server.streamable_http_app()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        assert (await client.initialize()).protocolVersion == "2025-11-25"
                        tools = (await client.list_tools()).tools
                        schema = next(tool for tool in tools if tool.name == "query_wandb_tool").inputSchema
                        validator = Draft202012Validator(schema)
                        validator.validate({"query": query, "variables": VARIABLES})
                        validator.validate({"entity_name": "e", "project_name": "p", "resource": "project"})
                        assert list(validator.iter_errors({"query": query, "limit": 50}))
                        result = await client.call_tool(
                            "query_wandb_tool",
                            {
                                "query": query,
                                "variables": {**VARIABLES, "unused": canary, "after": None},
                                "max_items": 2,
                                "items_per_page": 1,
                            },
                        )
                        payload = _payload(result)
                        assert result.isError is True
                        assert result.structuredContent == {"result": payload}
                        assert payload["selected"]["runs"]["edges"] == [edge]
                        assert payload["errors"][0]["error"] == "upstream_error"
                        pagination = payload["extensions"]["wandb_mcp"]
                        assert pagination["returned_count"] == 1
                        assert pagination["has_more"] is True
                        assert pagination["next_cursor"] == "cursor-one"
                        assert count_tokens_conservative(result.content[0].text) <= MAX_RESPONSE_TOKENS
                        assert canary not in result.model_dump_json()
        assert len(backend) == 2
        assert [variables["after"] for _, variables in backend] == [None, "cursor-one"]
        assert len(events) == 1
        assert events[0]["event_type"] == "tool_call"
        assert events[0]["tool_name"] == "query_wandb_tool"
        assert events[0]["success"] is False
        assert events[0]["error_diagnostics"]["category"] == "upstream_error"
        assert canary not in json.dumps(events)
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_structured_defaults_and_response_are_unchanged(backend, monkeypatch):
    import wandb_mcp_server.server as server_module

    calls = []
    expected = {"item": {"id": "fixture"}, "source": "wandb_sdk"}

    def typed(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(server_module, "query_wandb", typed)
    server = create_mcp_server("stdio")
    try:
        result = await server.call_tool(
            "query_wandb_tool", {"entity_name": "e", "project_name": "p", "resource": "project"}
        )
        assert _payload(result) == expected
        assert calls[0]["order"] == "-created_at"
        assert calls[0]["limit"] == 50
        assert calls[0]["response_mode"] == "items"
        assert "query" not in calls[0]
        assert backend == []
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_legacy_alias_pagination_and_json_fields_survive_public_dispatch(backend, monkeypatch):
    pages = []

    def execute(query, variables=None):
        after = variables.get("after")
        pages.append(after)
        number = 1 if after is None else 2
        return {
            "selected": {
                "runs": {
                    "edges": [
                        {
                            "cursor": f"c{number}",
                            "node": {
                                "id": f"global-id-{number}",
                                "name": f"short-id-{number}",
                                "summaryMetrics": '{"loss": 0.2}',
                                "config": '{"batch": {"value": 8}}',
                            },
                        }
                    ],
                    "pageInfo": {"hasNextPage": number == 1, "endCursor": f"c{number}"},
                }
            }
        }

    monkeypatch.setattr(
        api_client, "get_wandb_api", lambda: SimpleNamespace(_service_api=SimpleNamespace(execute_graphql=execute))
    )
    query = """query Runs($entity: String!, $project: String!, $after: String) {
      selected: project(entityName: $entity, name: $project) {
        runs(first: 1, after: $after) { edges { cursor node { id name summaryMetrics config } }
          pageInfo { hasNextPage endCursor } }
      }
    }"""
    server = create_mcp_server("stdio")
    try:
        result = _payload(
            await server.call_tool(
                "query_wandb_tool",
                {
                    "query": query,
                    "variables": {"entity": "e", "project": "p", "after": None},
                    "max_items": 2,
                    "items_per_page": 1,
                },
            )
        )
        edges = result["selected"]["runs"]["edges"]
        assert pages == [None, "c1"]
        assert [edge["node"]["id"] for edge in edges] == ["global-id-1", "global-id-2"]
        assert [edge["node"]["name"] for edge in edges] == ["short-id-1", "short-id-2"]
        assert all(edge["node"]["summaryMetrics"] == '{"loss": 0.2}' for edge in edges)
        assert all(edge["node"]["config"] == '{"batch": {"value": 8}}' for edge in edges)
        assert result["extensions"]["wandb_mcp"]["returned_count"] == 2
        assert result["selected"]["runs"]["pageInfo"]["hasNextPage"] is False
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_concurrent_legacy_calls_keep_request_scoped_authentication(monkeypatch):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-only")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-only")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    api_client.WandBApiManager._clear_api_cache()
    barrier = threading.Barrier(2, timeout=5)
    seen = []

    def api(*, api_key, **kwargs):
        seen.append(api_key)

        def execute_graphql(query, variables=None):
            barrier.wait()
            return {"viewer": {"id": api_key.removeprefix("synthetic-key-")}}

        return SimpleNamespace(_service_api=SimpleNamespace(execute_graphql=execute_graphql))

    monkeypatch.setattr(api_client.wandb, "Api", api)
    server = create_mcp_server("stdio")

    async def invoke(actor):
        token = api_client.WandBApiManager.set_context_api_key("synthetic-key-" + actor)
        try:
            return _payload(await server.call_tool("query_wandb_tool", {"query": "{ viewer { id } }"}))
        finally:
            api_client.WandBApiManager.reset_context_api_key(token)

    try:
        assert await asyncio.gather(invoke("actor-a"), invoke("actor-b")) == [
            {"viewer": {"id": "actor-a"}},
            {"viewer": {"id": "actor-b"}},
        ]
        assert set(seen) == {"synthetic-key-actor-a", "synthetic-key-actor-b"}
        assert api_client.WandBApiManager.get_api_key() is None
    finally:
        server._sync_executor.shutdown(wait=True)
        api_client.WandBApiManager._clear_api_cache()
