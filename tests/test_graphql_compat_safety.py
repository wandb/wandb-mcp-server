"""Safety regressions for legacy GraphQL at the public MCP boundary."""

import asyncio
import gc
import json
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mcp import types

from wandb_mcp_server.admission import ToolDeadlineExceeded, current_tool_deadline, tool_cost
from wandb_mcp_server.error_diagnostics import ToolErrorDiagnostics, current_error_diagnostics
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.mcp_tools import query_wandb_gql as gql_tool

CANARY = "private-query-resource-and-credential-canary"
RUNS_QUERY = """
query Runs($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runs(first: 1) {
      edges { node { id name } cursor }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


@pytest.fixture
def boundary(monkeypatch):
    tracker = Mock()
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", lambda: tracker)
    server = InstrumentedFastMCP("graphql-safety")
    server._admission_controller = None
    yield server, tracker
    server._sync_executor.shutdown(wait=True)


@pytest.mark.parametrize("query", ["{ viewer { id } }", "", None, {}, 1])
def test_legacy_query_is_heavy_before_dispatch_even_with_light_sdk_arguments(query):
    assert tool_cost("query_wandb_tool", {"query": query}) == ("heavy", 4)
    assert tool_cost("query_wandb_tool", {"query": query, "resource": "project", "response_mode": "count"}) == (
        "heavy",
        4,
    )
    assert tool_cost("query_wandb_tool", {"resource": "project"}) == ("light", 1)


@pytest.mark.parametrize("field", ["variables", "max_items", "items_per_page"])
def test_legacy_fields_without_query_cannot_downgrade_admission(field):
    assert tool_cost("query_wandb_tool", {field: None, "response_mode": "count"}) == ("heavy", 4)


@pytest.mark.asyncio
async def test_legacy_query_acquires_heavy_capacity_before_execution(boundary):
    server, tracker = boundary
    lease = SimpleNamespace(queue_ms=0.0, release=AsyncMock())
    server._admission_controller = SimpleNamespace(acquire=AsyncMock(return_value=lease))

    @server.tool()
    def query_wandb_tool(query: str) -> dict:
        assert server._admission_controller.acquire.await_count == 1
        return {"project": {"name": "fixture"}}

    await server.call_tool("query_wandb_tool", {"query": "{ project { name } }"})
    assert server._admission_controller.acquire.call_args.kwargs["weight"] == 4
    assert tracker.track_tool_call.call_args.kwargs["params"]["cost_class"] == "heavy"
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["query_wandb_tool", "query_wandb_graphql_tool"])
@pytest.mark.parametrize("partial", [False, True])
async def test_graphql_errors_are_native_protocol_failures_with_safe_partial_data(boundary, tool_name, partial):
    server, tracker = boundary
    payload = {"errors": [{"message": CANARY, "path": [CANARY], "extensions": {"private": CANARY}}]}
    if partial:
        payload["project"] = {"name": "safe-fixture-result"}

    @server.tool(name=tool_name)
    def legacy(query: str) -> dict:
        return payload

    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=tool_name, arguments={"query": "{ viewer { id } }"})
    )
    response = (await server._mcp_server.request_handlers[types.CallToolRequest](request)).root

    assert isinstance(response, types.CallToolResult)
    assert response.isError is True
    assert response.structuredContent["errors"][0]["error"] == "upstream_error"
    if partial:
        assert response.structuredContent["project"] == payload["project"]
    assert json.loads(response.content[0].text) == response.structuredContent
    assert CANARY not in response.model_dump_json()
    tracker.track_tool_call.assert_called_once()
    event = tracker.track_tool_call.call_args.kwargs
    assert event["success"] is False
    assert event["error"] == "upstream_error: tool failed"
    assert event["error_diagnostics"]["category"] == "upstream_error"
    assert CANARY not in json.dumps(event["error_diagnostics"])


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["read_tool", "query_wandb_tool"])
async def test_non_graphql_errors_data_is_not_reclassified(boundary, name):
    server, tracker = boundary

    @server.tool(name=name)
    def read() -> dict:
        return {"errors": [{"message": "ordinary data"}]}

    result = await server.call_tool(name, {})
    assert not isinstance(result, types.CallToolResult)
    assert tracker.track_tool_call.call_args.kwargs["success"] is True


@pytest.mark.asyncio
async def test_graphql_empty_error_array_is_still_success(boundary):
    server, tracker = boundary

    @server.tool()
    def query_wandb_tool(query: str) -> dict:
        return {"project": {"name": "fixture"}, "errors": []}

    await server.call_tool("query_wandb_tool", {"query": "{ project { name } }"})
    assert tracker.track_tool_call.call_args.kwargs["success"] is True


@pytest.mark.asyncio
async def test_graphql_fixed_error_category_survives_native_wrapping(boundary):
    server, tracker = boundary

    @server.tool()
    def query_wandb_tool(query: str) -> dict:
        return {"errors": [{"error": "query_too_complex", "message": CANARY}]}

    result = await server.call_tool("query_wandb_tool", {"query": "{ viewer { id } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    event = tracker.track_tool_call.call_args.kwargs
    assert event["error"] == "query_too_complex: tool failed"
    assert event["error_diagnostics"]["category"] == "input_validation"
    assert CANARY not in result.model_dump_json()


@pytest.mark.asyncio
async def test_graphql_error_retains_the_fastmcp_typed_result_wrapper(boundary):
    server, tracker = boundary

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return {"project": {"name": "fixture"}, "errors": [{"message": CANARY}]}

    result = await server.call_tool("query_wandb_tool", {"query": "{ project { name } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    assert result.structuredContent == {"result": json.loads(result.content[0].text)}
    assert result.structuredContent["result"]["project"] == {"name": "fixture"}
    assert tracker.track_tool_call.call_args.kwargs["error"] == "upstream_error: tool failed"


@pytest.mark.asyncio
async def test_graphql_result_alias_with_nested_errors_is_not_a_protocol_error(boundary):
    server, tracker = boundary
    payload = {"result": {"errors": ["normal queried data"]}}

    @server.tool()
    def query_wandb_tool(query: str) -> dict[str, Any]:
        return payload

    result = await server.call_tool("query_wandb_tool", {"query": "{ result: project { errors } }"})
    assert not isinstance(result, types.CallToolResult)
    assert tracker.track_tool_call.call_args.kwargs["success"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["error", "errors", "result"])
@pytest.mark.parametrize("fragment", [False, True])
async def test_graphql_root_aliases_remain_successful_data(boundary, monkeypatch, alias, fragment):
    server, tracker = boundary
    payload = {alias: {"id": "fixture"}}
    query = (
        f"query Test {{ ...Alias }} fragment Alias on Query {{ {alias}: viewer {{ id }} }}"
        if fragment
        else f"{{ {alias}: viewer {{ id }} }}"
    )
    _install_backend(monkeypatch, [payload])

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return gql_tool.query_paginated_wandb_gql(query)

    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name="query_wandb_tool", arguments={"query": query})
    )
    result = (await server._mcp_server.request_handlers[types.CallToolRequest](request)).root
    assert result.isError is False
    assert json.loads(result.content[0].text) == payload
    assert result.structuredContent == {"result": payload}
    assert tracker.track_tool_call.call_args.kwargs["success"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["error", "errors", "result", "viewer"])
@pytest.mark.parametrize("field", ["description", "message"])
async def test_graphql_long_data_matches_text_and_structured_content_without_losing_redaction(
    boundary, monkeypatch, alias, field
):
    from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text

    server, tracker = boundary
    monkeypatch.setenv("SAFETY_TEST_API_KEY", CANARY)
    value = 'long result "with quotes" and a newline\n' * 180 + CANARY + " http://api.private.svc/graphql"
    payload = {alias: {field: value}}
    expected = {alias: {field: sanitize_sensitive_text(value)}}
    _install_backend(monkeypatch, [payload])

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return gql_tool.query_paginated_wandb_gql(query)

    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name="query_wandb_tool", arguments={"query": f"{{ {alias}: viewer {{ {field} }} }}"}
        )
    )
    result = (await server._mcp_server.request_handlers[types.CallToolRequest](request)).root
    assert result.isError is False
    assert json.loads(result.content[0].text) == expected
    assert result.structuredContent == {"result": expected}
    assert CANARY not in result.model_dump_json()
    assert "api.private.svc" not in result.model_dump_json()
    assert len(result.structuredContent["result"][alias][field]) > 4096
    tracker.track_tool_call.assert_called_once()
    assert tracker.track_tool_call.call_args.kwargs["success"] is True


@pytest.mark.asyncio
async def test_graphql_partial_error_preserves_long_data_named_message(boundary, monkeypatch):
    server, tracker = boundary
    value = "valid partial result " * 400
    payload = {"project": {"message": value}, "errors": [{"message": CANARY}]}

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return payload

    result = await server.call_tool("query_wandb_tool", {"query": "{ project { message } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    assert result.structuredContent["result"]["project"]["message"] == value
    assert json.loads(result.content[0].text) == result.structuredContent["result"]
    assert CANARY not in result.model_dump_json()
    assert tracker.track_tool_call.call_args.kwargs["success"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["viewer", "error", "errors", "result"])
async def test_graphql_redaction_expansion_cannot_exceed_success_response_budget(boundary, monkeypatch, alias):
    from wandb_mcp_server.trace_utils import count_tokens_conservative

    server, tracker = boundary
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 100)
    payload = {alias: {"description": "Bearer x " * 20}}
    assert count_tokens_conservative(json.dumps(payload, separators=(",", ":"))) <= 100
    backend, _ = _install_backend(monkeypatch, [payload])

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return gql_tool.query_paginated_wandb_gql(query)

    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name="query_wandb_tool", arguments={"query": f"{{ {alias}: viewer {{ description }} }}"}
        )
    )
    result = (await server._mcp_server.request_handlers[types.CallToolRequest](request)).root
    assert result.isError is True
    assert count_tokens_conservative(result.content[0].text) <= 100
    decoded = json.loads(result.content[0].text)
    assert decoded["errors"][0]["error"] == "response_too_large"
    assert result.structuredContent == {"result": decoded}
    assert "Bearer x" not in result.model_dump_json()
    backend.assert_called_once()
    tracker.track_tool_call.assert_called_once()
    event = tracker.track_tool_call.call_args.kwargs
    assert event["success"] is False
    assert event["error"] == "response_too_large: tool failed"
    assert event["error_diagnostics"]["category"] == "response_too_large"


@pytest.mark.asyncio
async def test_graphql_root_errors_alias_does_not_hide_engine_failure(boundary, monkeypatch):
    server, tracker = boundary
    _install_backend(monkeypatch, [PermissionError(CANARY)])

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return gql_tool.query_paginated_wandb_gql(query)

    result = await server.call_tool("query_wandb_tool", {"query": "{ errors: viewer { id } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    assert tracker.track_tool_call.call_args.kwargs["error_diagnostics"]["category"] == "permission_denied"
    assert CANARY not in result.model_dump_json()


@pytest.mark.asyncio
async def test_abandoned_sync_worker_exception_is_consumed_without_logging_inputs():
    from wandb_mcp_server.instrumented_server import register_current_sync_future

    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    contexts = []
    loop.set_exception_handler(lambda loop, context: contexts.append(context))
    try:
        future = loop.create_future()
        register_current_sync_future(future)
        future.set_exception(RuntimeError(CANARY))
        await asyncio.sleep(0)
        del future
        gc.collect()
        await asyncio.sleep(0)
        assert contexts == []
    finally:
        loop.set_exception_handler(old_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", [{"query": "{ viewer { id } }", "variables": [CANARY]}, {"query": {CANARY: CANARY}}]
)
async def test_graphql_schema_errors_do_not_echo_inputs(boundary, arguments):
    server, tracker = boundary
    backend = Mock()

    @server.tool()
    def query_wandb_tool(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        backend()
        return {}

    request = types.CallToolRequest(params=types.CallToolRequestParams(name="query_wandb_tool", arguments=arguments))
    result = (await server._mcp_server.request_handlers[types.CallToolRequest](request)).root
    assert result.isError
    assert CANARY not in result.model_dump_json()
    backend.assert_not_called()
    tracker.track_tool_call.assert_called_once()
    assert tracker.track_tool_call.call_args.kwargs["error_diagnostics"]["category"] == "input_validation"


@pytest.mark.asyncio
@pytest.mark.parametrize("errors", [{"message": CANARY}, CANARY, True, [CANARY]])
async def test_graphql_malformed_errors_fail_safely(boundary, errors):
    server, tracker = boundary

    @server.tool()
    def query_wandb_tool(query: str) -> dict[str, Any]:
        return {"errors": errors}

    result = await server.call_tool("query_wandb_tool", {"query": "{ viewer { id } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    assert CANARY not in result.model_dump_json()
    assert tracker.track_tool_call.call_args.kwargs["success"] is False


def _install_backend(monkeypatch, execute):
    backend = Mock(side_effect=execute)
    api = object()
    get_api = Mock(return_value=api)
    monkeypatch.setattr("wandb_mcp_server.api_client.get_wandb_api", get_api)
    monkeypatch.setattr(gql_tool, "execute_graphql", backend)
    return backend, get_api


def _page(cursor, more):
    return {
        "project": {
            "runs": {
                "edges": [{"node": {"id": cursor, "name": cursor}, "cursor": cursor}],
                "pageInfo": {"hasNextPage": more, "endCursor": cursor},
            }
        }
    }


def test_expired_deadline_stops_before_graphql_api_construction(monkeypatch):
    backend, get_api = _install_backend(monkeypatch, lambda *a: {})
    token = current_tool_deadline.set(time.monotonic() - 1)
    try:
        with pytest.raises(ToolDeadlineExceeded):
            gql_tool.query_paginated_wandb_gql(RUNS_QUERY, {"entity": "fixture", "project": "fixture"})
    finally:
        current_tool_deadline.reset(token)
    backend.assert_not_called()
    get_api.assert_not_called()


def test_expired_deadline_stops_before_each_next_graphql_page(monkeypatch):
    def execute(*args):
        current_tool_deadline.set(time.monotonic() - 1)
        return _page("cursor-one", True)

    backend, _ = _install_backend(monkeypatch, execute)
    token = current_tool_deadline.set(time.monotonic() + 30)
    try:
        with pytest.raises(ToolDeadlineExceeded):
            gql_tool.query_paginated_wandb_gql(
                RUNS_QUERY, {"entity": "fixture", "project": "fixture"}, max_items=2, items_per_page=1
            )
    finally:
        current_tool_deadline.reset(token)
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_timed_out_worker_does_not_start_another_graphql_page(boundary, monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    server, tracker = boundary
    started, release = threading.Event(), threading.Event()

    def execute(*args):
        started.set()
        assert release.wait(2)
        return _page("cursor-one", True)

    backend, _ = _install_backend(monkeypatch, execute)
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.02)

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return gql_tool.query_paginated_wandb_gql(
            query, {"entity": "fixture", "project": "fixture"}, max_items=2, items_per_page=1
        )

    task = asyncio.create_task(server.call_tool("query_wandb_tool", {"query": RUNS_QUERY}))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        with pytest.raises(ToolError):
            await task
    finally:
        release.set()
        await asyncio.to_thread(server._sync_executor.shutdown, wait=True)
    assert backend.call_count == 1
    tracker.track_tool_call.assert_called_once()
    assert tracker.track_tool_call.call_args.kwargs["error_diagnostics"]["category"] == "tool_timeout"


def test_empty_graphql_errors_do_not_stop_pagination(monkeypatch):
    first_page = _page("cursor-one", True)
    first_page["errors"] = []
    backend, _ = _install_backend(monkeypatch, [first_page, _page("cursor-two", False)])
    result = gql_tool.query_paginated_wandb_gql(
        RUNS_QUERY, {"entity": "fixture", "project": "fixture"}, max_items=2, items_per_page=1
    )
    assert len(result["project"]["runs"]["edges"]) == 2
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_fixed_graphql_errors_still_respect_the_response_budget(boundary, monkeypatch):
    from wandb_mcp_server.trace_utils import count_tokens_conservative

    server, tracker = boundary
    monkeypatch.setattr("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 100)

    @server.tool()
    def query_wandb_tool(query: str) -> Dict[str, Any]:
        return {"project": {"description": "bounded " * 1000}, "errors": [{"message": "x"}]}

    result = await server.call_tool("query_wandb_tool", {"query": "{ project { description } }"})
    assert isinstance(result, types.CallToolResult)
    assert result.isError
    assert count_tokens_conservative(result.content[0].text) <= 100
    assert json.loads(result.content[0].text)["errors"][0]["error"] == "response_too_large"
    assert tracker.track_tool_call.call_args.kwargs["error"] == "response_too_large: tool failed"


@pytest.mark.parametrize("partial", [False, True])
def test_graphql_swallowed_http_exception_keeps_safe_typed_diagnostics(monkeypatch, partial):
    request = httpx.Request("POST", f"https://{CANARY}.invalid")
    exception = httpx.HTTPStatusError(CANARY, request=request, response=httpx.Response(403, request=request))
    replies = [_page("cursor-one", True), exception] if partial else [exception]
    backend, _ = _install_backend(monkeypatch, replies)
    state = ToolErrorDiagnostics(frozenset({"query", "variables", "max_items", "items_per_page"}))
    token = current_error_diagnostics.set(state)
    try:
        result = gql_tool.query_paginated_wandb_gql(
            RUNS_QUERY, {"entity": "fixture", "project": "fixture"}, max_items=2, items_per_page=1
        )
    finally:
        current_error_diagnostics.reset(token)
    assert state.value["category"] == "permission_denied"
    assert state.value["upstream_status"] == 403
    assert CANARY not in json.dumps(result)
    assert CANARY not in json.dumps(state.value)
    assert backend.call_count == (2 if partial else 1)


def test_graphql_syntax_error_does_not_echo_query_source(monkeypatch):
    backend, get_api = _install_backend(monkeypatch, lambda *a: {})
    state = ToolErrorDiagnostics(frozenset({"query"}))
    token = current_error_diagnostics.set(state)
    try:
        result = gql_tool.query_paginated_wandb_gql("query " + CANARY + " {")
    finally:
        current_error_diagnostics.reset(token)
    assert CANARY not in json.dumps(result)
    assert state.value["category"] == "input_validation"
    backend.assert_not_called()
    get_api.assert_not_called()


def test_graphql_dynamic_exception_class_is_not_echoed_or_logged(monkeypatch, caplog):
    exception = type(CANARY, (RuntimeError,), {})(CANARY)
    _install_backend(monkeypatch, [exception])
    result = gql_tool.query_paginated_wandb_gql("{ viewer { id } }")
    assert CANARY not in json.dumps(result)
    assert CANARY not in caplog.text


def test_malformed_second_page_is_an_explicit_partial_error(monkeypatch):
    _install_backend(monkeypatch, [_page("cursor-one", True), {"unexpected": "shape"}])
    result = gql_tool.query_paginated_wandb_gql(
        RUNS_QUERY, {"entity": "fixture", "project": "fixture"}, max_items=2, items_per_page=1
    )
    assert result["errors"][0]["error"] == "malformed_response"
    assert len(result["project"]["runs"]["edges"]) == 1
