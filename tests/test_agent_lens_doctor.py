"""Unit tests for the Agent Lens instrumentation doctor MCP tool."""

import asyncio
import json
import threading
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests
from mcp.server.fastmcp.exceptions import ToolError

from tests.test_agent_lens import BASE_URL, ENTITY, PROJECT, WINDOW, _FakeResponse, _mocked, _ok, _sync_call
from wandb_mcp_server.admission import WeightedAdmissionController
from wandb_mcp_server.api_client import WandBServerBusy
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.mcp_tools import agent_lens_doctor as doctor_mod
from wandb_mcp_server.mcp_tools import agent_lens_reach as reach_mod

SPAN_PROBE_KWARGS = {
    "started_after": WINDOW["start_at"],
    "started_before": WINDOW["end_at"],
    "sort_by": "started_at",
    "sort_direction": "desc",
    "limit": 1,
}


def diagnose_instrumentation(*args, **kwargs):
    return _sync_call(doctor_mod.diagnose_instrumentation, *args, **kwargs)


def _rule(code, severity, failing, applicable, *, groups=0, spans=0, rule_type="span"):
    return {
        "code": code,
        "severity": severity,
        "rule_type": rule_type,
        "title": f"{code} title",
        "impact": f"{code} impact",
        "fix": f"{code} fix",
        "applicable_count": applicable,
        "failing_count": failing,
        "examples": [{"trace_id": f"{code}-t{i}", "conversation_id": f"{code}-c{i}"} for i in range(min(failing, 3))],
        "failing_spans": [{"trace_id": "t", "span_id": f"s{i}", "conversation_id": "c"} for i in range(spans)],
        "breakdown": {
            "group_type": "span",
            "group_count": groups,
            "failing_group_count": groups,
            "failing_groups": [{"name": f"g{i}", "failing_count": 1, "applicable_count": 2} for i in range(groups)],
        },
    }


def _health(issues, *, spans=40, traces=4, unsettled=0):
    return _ok({"span_count": spans, "trace_count": traces, "unsettled_trace_count": unsettled, "issues": issues})


@contextmanager
def _trace_server(*pages, calls=0):
    """Stand in for trace-server reads: each of `pages` answers one agent-span query, in order."""
    query = MagicMock(side_effect=[json.dumps({"spans": spans}) for spans in pages])
    count = MagicMock(return_value=calls)
    with patch.object(reach_mod, "query_agent_spans", query), patch.object(reach_mod, "count_traces", count):
        yield query, count


@pytest.mark.parametrize(
    ("unsettled", "verdict", "window"),
    [
        (0, "clean", WINDOW),
        (3, "settling", WINDOW),
        pytest.param(
            0,
            "clean",
            {"start_at": "2026-11-20T12:00:00.100123Z", "end_at": "2026-11-20T12:00:00.900456Z"},
            id="subsecond-window",
        ),
    ],
)
def test_doctor_reports_clean_only_once_every_trace_has_settled(unsettled, verdict, window):
    passing = [_rule("span_no_operation_name", "must", 0, 40), _rule("trace_no_root", "should", 0, 4)]
    with _mocked(_health(passing, unsettled=unsettled)) as session, _trace_server() as (query, count):
        result = json.loads(diagnose_instrumentation(ENTITY, project_name=PROJECT, **window))

    assert session.last["method"] == "GET"
    assert session.last["url"] == f"{BASE_URL}/api/instrumentation/health"
    assert session.last["params"] == window
    assert result["verdict"] == verdict
    assert result["issues"] == []
    assert result["failing_rule_counts"] == {"must": 0, "should": 0}
    assert result["passing_rule_count"] == 2
    assert result["scope"] == {"entity_name": ENTITY, "project_name": PROJECT, **window}
    assert ("note" in result) is bool(unsettled)
    query.assert_not_called()
    count.assert_not_called()


def test_doctor_ranks_failing_rules_must_first_then_by_share_for_the_latest_conversation():
    issues = [
        _rule("should_half", "should", 5, 10, groups=1),
        _rule("must_tenth", "must", 2, 20, groups=5, spans=30),
        _rule("passing", "must", 0, 10),
        _rule("inapplicable", "should", 0, 0, rule_type="trace"),
        _rule("must_all", "must", 1, 1, groups=2, rule_type="trace"),
    ]
    newest = [{"conversation_id": "conv-9", "trace_id": "t9"}]
    latest = {"project_name": PROJECT, "conversation_id": "latest", **WINDOW}
    with _mocked(_health(issues)) as session, _trace_server(newest) as (query, count):
        result = json.loads(diagnose_instrumentation(ENTITY, **latest))
    with _mocked(_health(issues)), _trace_server(newest):
        detailed = json.loads(diagnose_instrumentation(ENTITY, include_failing_spans=True, **latest))

    query.assert_called_once_with(ENTITY, PROJECT, **SPAN_PROBE_KWARGS)
    assert session.last["params"] == {**WINDOW, "conversation_id": "conv-9"}
    assert result["scope"]["conversation_id"] == "conv-9"
    assert result["verdict"] == "failing"
    assert [issue["code"] for issue in result["issues"]] == ["must_all", "must_tenth", "should_half"]
    assert result["failing_rule_counts"] == {"must": 2, "should": 1}
    assert result["passing_rule_count"] == 1
    second = result["issues"][1]
    assert second["fix"] == "must_tenth fix"
    assert (second["failing_count"], second["applicable_count"], second["group_type"]) == (2, 20, "span")
    assert [group["name"] for group in second["top_failing_groups"]] == ["g0", "g1", "g2"]
    assert second["examples"] == [
        {"trace_id": "must_tenth-t0", "conversation_id": "must_tenth-c0"},
        {"trace_id": "must_tenth-t1", "conversation_id": "must_tenth-c1"},
    ]
    assert "failing_spans" not in second
    assert len(detailed["issues"][1]["failing_spans"]) == 20
    count.assert_not_called()


@pytest.mark.parametrize(
    ("status", "kwargs", "pages", "calls", "expected", "checks_listed", "reach_checked"),
    [
        (200, {}, ([],), 12, "/agents/otel/v1/traces", False, True),
        (
            200,
            {},
            ([{"trace_id": "x"}],),
            12,
            "none is eligible for instrumentation checks. Evaluation traces and Weave Signals spans are excluded.",
            False,
            False,
        ),
        (404, {"trace_id": "t1"}, ([],), 0, "Nothing reached", True, True),
        # A filter miss on a project that has agent spans is never blamed on the exporter, even with calls present.
        (404, {"conversation_id": "c1"}, ([{"trace_id": "x"}],), 12, "Check that id", False, False),
        (404, {"trace_id": "t1"}, ([{"trace_id": "x"}],), 12, 'conversation_id="latest"', False, False),
        (
            200,
            {"conversation_id": "latest"},
            ([{"trace_id": "x"}],),
            0,
            "the newest agent span carries no conversation id",
            False,
            False,
        ),
    ],
)
def test_doctor_never_reports_zero_spans_as_healthy(
    status, kwargs, pages, calls, expected, checks_listed, reach_checked
):
    response = _health([], spans=0, traces=0) if status == 200 else _FakeResponse({"detail": "no spans"}, 404)
    with _mocked(response), _trace_server(*pages, calls=calls) as (query, count):
        result = json.loads(diagnose_instrumentation(ENTITY, project_name=PROJECT, **kwargs, **WINDOW))

    query.assert_called_once_with(ENTITY, PROJECT, **SPAN_PROBE_KWARGS)
    assert result["verdict"] == "no_spans"
    assert result["span_count"] == 0
    assert expected in result["diagnosis"]
    assert "issues" not in result
    assert ("checks" in result) is checks_listed
    assert count.called is reach_checked
    assert ("calls_in_window" in result) is reach_checked
    if reach_checked:
        assert count.call_args.args[2] == {"time_range": {"start": WINDOW["start_at"], "end": WINDOW["end_at"]}}
        assert result["calls_in_window"] == calls


@pytest.mark.asyncio
@pytest.mark.parametrize(("probe", "result"), [("query_agent_spans", '{"spans": []}'), ("count_traces", 0)])
async def test_cancelled_doctor_keeps_admission_until_reach_probe_finishes(
    monkeypatch: pytest.MonkeyPatch, probe: str, result: str | int
) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("doctor-cancel-test")
    admission = WeightedAdmissionController(actor_capacity=4, process_capacity=4, wait_timeout_seconds=0.01)
    server._admission_controller = admission
    tool = "diagnose_agent_lens_instrumentation_tool"
    server.add_tool(doctor_mod.diagnose_instrumentation, name=tool)
    arguments = {"entity_name": ENTITY, "project_name": PROJECT, **WINDOW}
    started = threading.Event()
    finish = threading.Event()

    def blocking_probe(*args: object, **kwargs: object) -> str | int:
        started.set()
        finish.wait()

        return result

    with _mocked(_health([], spans=0, traces=0)), _trace_server([]), patch.object(reach_mod, probe, blocking_probe):
        first = asyncio.create_task(server.call_tool(tool, arguments))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            assert admission.process_in_use == 4
            with pytest.raises(ToolError, match="server_busy"):
                await server.call_tool(tool, arguments)
        finally:
            finish.set()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await asyncio.gather(*server._deferred_release_tasks)
            server.shutdown_sync_executor()

    assert admission.process_in_use == 0


@pytest.mark.parametrize(
    ("selectors", "expected_params"),
    [
        ({}, {**WINDOW, "start_at": "2026-09-07T00:00:00Z"}),
        ({"trace_id": "t1"}, {**WINDOW, "trace_id": "t1"}),
    ],
    ids=["project", "filtered"],
)
def test_doctor_sends_default_window(selectors: dict[str, str], expected_params: dict[str, str]):
    with _mocked(_health([])) as session, patch.object(doctor_mod, "datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime.fromisoformat(WINDOW["end_at"])
        diagnose_instrumentation(ENTITY, project_name=PROJECT, **selectors)

    assert session.last["params"] == expected_params


@pytest.mark.parametrize(
    ("conversation_id", "trace_id", "expected_filters"),
    [
        ("latest", "trace-1", None),
        ("", " \t", {}),
        (" \t", "", {}),
        ("latest", " \t", {"conversation_id": "conv-1"}),
        ("conv-1", "trace-1", {"conversation_id": "conv-1", "trace_id": "trace-1"}),
    ],
)
def test_doctor_validates_and_normalizes_selectors(
    conversation_id: str, trace_id: str, expected_filters: dict[str, str] | None
) -> None:
    newest = [{"conversation_id": "conv-1"}]
    with (
        _mocked(_health([])) as session,
        _trace_server(newest) as (query, count),
        patch.object(doctor_mod, "datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = datetime.fromisoformat(WINDOW["end_at"])
        result = json.loads(
            diagnose_instrumentation(ENTITY, project_name=PROJECT, conversation_id=conversation_id, trace_id=trace_id)
        )

    count.assert_not_called()
    if expected_filters is None:
        assert result["error"] == "agent_lens_invalid_request"
        assert session.calls == []
        query.assert_not_called()
        return

    window = WINDOW if expected_filters else {**WINDOW, "start_at": "2026-09-07T00:00:00Z"}
    expected_params = {**window, **expected_filters}
    assert session.last["params"] == expected_params
    assert result["scope"] == {"entity_name": ENTITY, "project_name": PROJECT, **expected_params}
    assert result["verdict"] == "clean"


def test_doctor_latest_never_checks_an_older_conversation():
    spans = [{"conversation_id": ""}, {"conversation_id": "older"}]

    def query_agent_spans(*_args: str, query: object = None, limit: int = 1, **_kwargs: object) -> str:
        matching_spans = spans if query is None else [span for span in spans if span["conversation_id"]]
        response = json.dumps({"spans": matching_spans[:limit]})

        return response

    with _mocked(_health([])) as session, patch.object(reach_mod, "query_agent_spans", query_agent_spans):
        result = json.loads(diagnose_instrumentation(ENTITY, project_name=PROJECT, conversation_id="latest", **WINDOW))

    assert result["verdict"] == "no_spans"
    assert session.calls == []


@pytest.mark.parametrize("spans", [None, [{"conversation_id": None}]])
def test_doctor_reports_invalid_latest_span_data(spans: list[dict[str, str | None]] | None):
    with _mocked(_health([])) as session, _trace_server(spans) as (_query, count):
        result = json.loads(diagnose_instrumentation(ENTITY, project_name=PROJECT, conversation_id="latest", **WINDOW))

    assert result == {
        "error": "agents_query_failed",
        "message": "The Agents API returned an unexpected span response.",
    }
    assert session.calls == []
    count.assert_not_called()


@pytest.mark.parametrize("status_code", [429, 503])
def test_doctor_count_probe_surfaces_the_same_overload_as_the_span_probe(status_code: int):
    busy_response = requests.Response()
    busy_response.status_code = status_code
    busy_response.reason = "Service overloaded"
    busy_response.headers["Retry-After"] = "7"
    empty_spans = _FakeResponse({"spans": []})

    with (
        _mocked(_health([], spans=0, traces=0)),
        patch.object(requests.Session, "post", side_effect=[busy_response, empty_spans, busy_response]),
    ):
        with pytest.raises(WandBServerBusy) as span_error:
            diagnose_instrumentation(ENTITY, project_name=PROJECT, **WINDOW)
        with pytest.raises(WandBServerBusy) as count_error:
            diagnose_instrumentation(ENTITY, project_name=PROJECT, **WINDOW)

    assert count_error.value.status_code == span_error.value.status_code == status_code
    assert count_error.value.as_dict() == span_error.value.as_dict()
    assert count_error.value.retry_after_ms == 7_000
