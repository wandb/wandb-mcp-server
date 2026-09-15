"""Safe cause diagnostics at the real FastMCP dispatch boundary."""

import asyncio
import json
import threading
from typing import Literal
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from wandb_mcp_server.instrumented_server import InstrumentedFastMCP

CANARY = "private-customer-credential-host-and-query-canary"


@pytest.fixture
def boundary(monkeypatch):
    tracker = Mock()
    monkeypatch.setattr("wandb_mcp_server.analytics.get_analytics_tracker", lambda: tracker)
    server = InstrumentedFastMCP("safe-diagnostics")
    server._admission_controller = None
    yield server, tracker
    server._sync_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_wrapped_schema_validation_records_only_public_fields_and_codes(boundary):
    server, tracker = boundary
    backend = Mock()

    @server.tool()
    def query_wandb_tool(entity_name: str, resource: Literal["run", "project"], filters: dict[str, int]):
        backend()
        return {"ok": True}

    with pytest.raises(ToolError):
        await server.call_tool("query_wandb_tool", {"resource": CANARY, "filters": {CANARY: CANARY}, CANARY: CANARY})
    backend.assert_not_called()
    tracker.track_tool_call.assert_called_once()
    event = tracker.track_tool_call.call_args.kwargs
    diagnostic = event.get("error_diagnostics")
    assert diagnostic is not None
    assert diagnostic["category"] == "input_validation"
    assert diagnostic["exception_type"] == "ToolError"
    assert diagnostic["cause_type"] == "ValidationError"
    assert diagnostic["validation_fields"] == ["entity_name", "filters", "resource"]
    assert diagnostic["validation_codes"] == ["int_parsing", "literal_error", "missing"]
    assert CANARY not in json.dumps(diagnostic)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "category"),
    [
        (401, "authentication_failed"),
        (403, "permission_denied"),
        (404, "resource_not_found"),
        (429, "server_busy"),
        (503, "server_busy"),
    ],
)
async def test_wrapped_http_failure_keeps_status_without_response_or_url(boundary, status, category):
    server, tracker = boundary

    @server.tool()
    async def read_tool():
        request = httpx.Request("GET", f"https://{CANARY}.invalid/{CANARY}")
        raise httpx.HTTPStatusError(CANARY, request=request, response=httpx.Response(status, request=request))

    with pytest.raises(ToolError):
        await server.call_tool("read_tool", {})
    diagnostic = tracker.track_tool_call.call_args.kwargs.get("error_diagnostics")
    assert diagnostic is not None
    assert diagnostic["category"] == category
    assert diagnostic["upstream_status"] == status
    assert diagnostic["cause_type"] == "HTTPStatusError"
    assert CANARY not in json.dumps(diagnostic)


@pytest.mark.asyncio
async def test_original_swallowed_history_cause_survives_result_mapping(boundary, monkeypatch):
    from wandb_mcp_server.server import register_tools

    server, tracker = boundary

    def fail_history(**kwargs):
        raise ValueError(CANARY)

    monkeypatch.setattr("wandb_mcp_server.mcp_tools.run_history.get_run_history", fail_history)
    register_tools(server)
    await server.call_tool("get_run_history_tool", {"entity_name": CANARY, "project_name": CANARY, "run_id": CANARY})
    event = tracker.track_tool_call.call_args.kwargs
    assert event["error"] == "history_query_failed: tool failed"
    diagnostic = event.get("error_diagnostics")
    assert diagnostic is not None
    assert diagnostic["category"] == "invalid_value"
    assert diagnostic["exception_type"] == "ValueError"
    assert CANARY not in json.dumps(diagnostic)


@pytest.mark.asyncio
async def test_diagnostics_do_not_leak_between_concurrent_calls(boundary):
    server, tracker = boundary

    @server.tool()
    async def read_tool(resource: Literal["run"]):
        await asyncio.sleep(0)
        return {"ok": True}

    await asyncio.gather(
        server.call_tool("read_tool", {"resource": CANARY}),
        server.call_tool("read_tool", {"resource": "run"}),
        return_exceptions=True,
    )
    events = [call.kwargs for call in tracker.track_tool_call.call_args_list]
    assert len(events) == 2
    assert (
        next(e for e in events if e["success"] is False).get("error_diagnostics", {}).get("category")
        == "input_validation"
    )
    assert next(e for e in events if e["success"] is True).get("error_diagnostics") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["diagnose_run_tool", "compare_runs_tool", "query_wandb_tool"])
async def test_swallowed_selective_failures_keep_original_http_category(boundary, monkeypatch, name):
    from wandb_mcp_server.server import register_tools

    server, tracker = boundary
    request = httpx.Request("GET", f"https://{CANARY}.invalid")
    failure = httpx.HTTPStatusError(CANARY, request=request, response=httpx.Response(403, request=request))
    monkeypatch.setattr("wandb_mcp_server.api_client.WandBApiManager.get_api", lambda *a, **k: object())
    monkeypatch.setattr("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run", Mock(side_effect=failure))
    monkeypatch.setattr("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run", Mock(side_effect=failure))
    monkeypatch.setattr("wandb_mcp_server.mcp_tools.query_wandb.fetch_project_metadata", Mock(side_effect=failure))
    register_tools(server)
    args = {"entity_name": CANARY, "project_name": CANARY}
    if name == "query_wandb_tool":
        args["resource"] = "project"
    else:
        args.update(config_keys=[], summary_keys=[])
        args.update({"run_id": CANARY} if name == "diagnose_run_tool" else {"run_id_a": CANARY, "run_id_b": CANARY})
    await server.call_tool(name, args)
    event = tracker.track_tool_call.call_args.kwargs
    assert event["success"] is False
    diagnostic = event.get("error_diagnostics")
    assert diagnostic is not None
    assert diagnostic["category"] == "permission_denied"
    assert diagnostic["upstream_status"] == 403
    assert CANARY not in json.dumps(diagnostic)


def test_diagnostic_sanitizer_drops_untrusted_and_unbounded_values():
    from wandb_mcp_server.error_diagnostics import sanitize_error_diagnostics

    safe = sanitize_error_diagnostics(
        {
            "category": "input_validation",
            "exception_type": CANARY,
            "cause_type": "ValidationError",
            "validation_fields": [CANARY, "entity_name", "project_name", "filters"] * 50,
            "validation_codes": [CANARY, "missing", "literal_error"] * 50,
            "upstream_status": True,
            "message": CANARY,
            "input": CANARY,
            "traceback": CANARY,
        }
    )
    assert safe == {
        "category": "input_validation",
        "cause_type": "ValidationError",
        "validation_fields": ["entity_name", "filters", "project_name"],
        "validation_codes": ["literal_error", "missing"],
    }
    assert sanitize_error_diagnostics({"category": CANARY}) is None
    assert sanitize_error_diagnostics(CANARY) is None


def test_dynamic_exception_names_and_cyclic_causes_are_not_emitted():
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    dynamic = type(CANARY, (Exception,), {})
    error = dynamic(CANARY)
    error.__cause__ = error
    diagnostic = exception_diagnostics(error)
    assert diagnostic == {"category": "tool_error", "exception_type": "other_exception"}
    assert CANARY not in json.dumps(diagnostic)


def test_diagnostics_never_format_an_exception():
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    class BadString(ValueError):
        def __str__(self):
            raise AssertionError("diagnostics must not format exceptions")

    assert exception_diagnostics(BadString(CANARY))["category"] == "invalid_value"


@pytest.mark.parametrize("attribute", ["status_code", "status", "http_status", "http_status_code"])
def test_legacy_nested_service_status_is_categorical_and_cycle_safe(attribute):
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    class BadString(Exception):
        def __str__(self):
            raise AssertionError("diagnostics must not format exceptions")

    original = BadString(CANARY)
    setattr(original, attribute, 403)
    wrapper = ToolError(CANARY)
    wrapper.exc = original
    original.exc = wrapper
    diagnostic = exception_diagnostics(wrapper)
    assert diagnostic["category"] == "permission_denied"
    assert diagnostic["upstream_status"] == 403
    assert CANARY not in json.dumps(diagnostic)


def test_explicit_cause_wins_over_unrelated_exception_context():
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    request = httpx.Request("GET", f"https://{CANARY}.invalid")
    error = ToolError(CANARY)
    error.__cause__ = httpx.HTTPStatusError(CANARY, request=request, response=httpx.Response(403, request=request))
    error.__context__ = TimeoutError(CANARY)
    diagnostic = exception_diagnostics(error)
    assert diagnostic["category"] == "permission_denied"
    assert diagnostic["cause_type"] == "HTTPStatusError"


def test_large_validation_error_does_not_materialize_error_details():
    from pydantic import ValidationError
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    class OversizedValidation(ValidationError):
        def errors(self, **kwargs):
            raise AssertionError("large validation detail must never be materialized")

    error = OversizedValidation.from_exception_data(
        CANARY, [{"type": "int_parsing", "loc": ("filters", f"{CANARY}-{i}"), "input": CANARY} for i in range(33)]
    )
    diagnostic = exception_diagnostics(error, public_fields=frozenset({"filters"}))
    assert diagnostic == {"category": "input_validation", "exception_type": "ValidationError"}


def test_real_wandb_service_response_and_busy_types():
    from wandb.proto import wandb_api_pb2
    from wandb.sdk.lib.service.service_connection import WandbApiFailedError
    from wandb_mcp_server.api_client import WandBServerBusy
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    error = WandbApiFailedError(CANARY, response=wandb_api_pb2.ApiErrorResponse(http_status=403))
    assert exception_diagnostics(error) == {
        "category": "permission_denied",
        "exception_type": "WandbApiFailedError",
        "upstream_status": 403,
    }
    assert exception_diagnostics(WandBServerBusy(status_code=429, retry_after_ms=1000))["category"] == "server_busy"


@pytest.mark.asyncio
async def test_late_worker_cannot_change_emitted_or_next_call_diagnostics(boundary, monkeypatch):
    from wandb_mcp_server.error_diagnostics import current_error_diagnostics, record_exception_diagnostics

    server, tracker = boundary
    monkeypatch.setattr("wandb_mcp_server.instrumented_server.MCP_TOOL_TIMEOUT_SECONDS", 0.02)
    release = threading.Event()
    completed = threading.Event()
    states = []

    @server.tool()
    def slow_tool():
        states.append(current_error_diagnostics.get())
        release.wait(1)
        record_exception_diagnostics(PermissionError(CANARY))
        completed.set()
        return "late"

    @server.tool()
    async def healthy_tool():
        return "ok"

    try:
        with pytest.raises(ToolError):
            await server.call_tool("slow_tool", {})
        first = tracker.track_tool_call.call_args.kwargs
        assert first["error_diagnostics"]["category"] == "tool_timeout"
        assert states[0].active is False
        before = json.dumps(first["error_diagnostics"], sort_keys=True)
        release.set()
        await asyncio.to_thread(completed.wait, 1)
        await server.call_tool("healthy_tool", {})
        assert tracker.track_tool_call.call_count == 2
        assert tracker.track_tool_call.call_args.kwargs.get("error_diagnostics") is None
        assert json.dumps(first["error_diagnostics"], sort_keys=True) == before
        assert current_error_diagnostics.get() is None
    finally:
        release.set()


@pytest.mark.asyncio
async def test_diagnostic_introspection_failure_preserves_original_failure_and_cleanup(boundary, monkeypatch):
    from wandb_mcp_server.error_diagnostics import current_error_diagnostics

    server, tracker = boundary
    monkeypatch.setattr(
        "wandb_mcp_server.error_diagnostics.exception_diagnostics", Mock(side_effect=RuntimeError(CANARY))
    )

    @server.tool()
    async def failing_tool():
        raise ValueError("original failure")

    with pytest.raises(ToolError, match="original failure"):
        await server.call_tool("failing_tool", {})
    assert tracker.track_tool_call.call_args.kwargs["error_diagnostics"] == {"category": "tool_error"}
    assert current_error_diagnostics.get() is None


@pytest.mark.asyncio
async def test_dynamic_base_exception_name_is_not_used_as_canonical_error(boundary):
    server, tracker = boundary
    dynamic_name = "PrivateCustomerCredentialCanary"
    dynamic_error = type(dynamic_name, (BaseException,), {})

    @server.tool()
    async def failing_tool():
        raise dynamic_error(CANARY)

    with pytest.raises(dynamic_error):
        await server.call_tool("failing_tool", {})
    event = tracker.track_tool_call.call_args.kwargs
    assert CANARY not in event["error"]
    assert dynamic_name not in event["error"]
    assert CANARY not in json.dumps(event["error_diagnostics"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "field", "value"),
    [
        ("get_run_history_tool", "samples", 0),
        ("get_run_history_tool", "keys", [CANARY] * 1000),
        ("get_run_history_tool", "min_step", -1),
        ("compare_runs_tool", "history_samples", 0),
        ("compare_runs_tool", "history_keys", [CANARY] * 1000),
        ("diagnose_run_tool", "samples", 0),
        ("diagnose_run_tool", "config_keys", [CANARY] * 1000),
        ("diagnose_run_tool", "x_axis", ""),
        ("query_wandb_tool", "limit", 0),
    ],
)
async def test_manual_validation_identifies_public_field_without_backend_work(
    boundary, monkeypatch, tool_name, field, value
):
    from wandb_mcp_server.server import register_tools

    server, tracker = boundary
    backend = Mock(side_effect=AssertionError("invalid input must not reach W&B"))
    monkeypatch.setattr("wandb_mcp_server.api_client.WandBApiManager.get_api", backend)
    register_tools(server)
    arguments = {"entity_name": CANARY, "project_name": CANARY}
    if tool_name == "compare_runs_tool":
        arguments.update(run_id_a=CANARY, run_id_b=CANARY)
    elif tool_name == "query_wandb_tool":
        arguments.update(resource="project")
    else:
        arguments.update(run_id=CANARY)
    arguments[field] = value
    try:
        await server.call_tool(tool_name, arguments)
    except ToolError:
        pass
    backend.assert_not_called()
    diagnostic = tracker.track_tool_call.call_args.kwargs["error_diagnostics"]
    assert diagnostic["category"] == "input_validation"
    assert diagnostic["validation_fields"] == [field]
    assert diagnostic["validation_codes"]
    assert CANARY not in json.dumps(diagnostic)


def test_unknown_exception_descriptors_are_not_executed_for_diagnostics():
    from wandb_mcp_server.error_diagnostics import exception_diagnostics

    accesses = []

    class DescriptorError(Exception):
        @property
        def response(self):
            accesses.append("response")
            raise AssertionError("diagnostic code must not invoke custom descriptors")

        @property
        def exc(self):
            accesses.append("exc")
            return None

    diagnostic = exception_diagnostics(DescriptorError(CANARY))
    assert diagnostic == {"category": "tool_error", "exception_type": "other_exception"}
    assert accesses == []


@pytest.mark.asyncio
async def test_admission_rejection_is_categorized_without_executing_tool(boundary):
    from wandb_mcp_server.admission import AdmissionRejected
    from wandb_mcp_server.error_diagnostics import current_error_diagnostics

    server, tracker = boundary
    backend = Mock()
    server._admission_controller = Mock(acquire=AsyncMock(side_effect=AdmissionRejected(CANARY)))

    @server.tool()
    async def read_tool():
        backend()
        return "ok"

    with pytest.raises(ToolError):
        await server.call_tool("read_tool", {})
    backend.assert_not_called()
    event = tracker.track_tool_call.call_args.kwargs
    assert event["error_diagnostics"]["category"] == "server_busy"
    assert event["error_diagnostics"]["cause_type"] == "AdmissionRejected"
    assert current_error_diagnostics.get() is None
