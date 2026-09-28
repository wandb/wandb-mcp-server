"""Unit tests for the Agent Lens Insights and conversation-tag MCP tools.

The HTTP boundary is mocked at the requests.Session level so we can assert
request shaping, auth, response passthrough, truncation, and error mapping
without a live Agent Lens. Canned response shapes mirror the Huma output
structs in internal/api/insights.go and internal/api/conversation_tags.go of
github.com/wandb/agent-lens.
"""

import json
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts import agent_lens_smoke
from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.api_client import WandBApiManager, WandBServerBusy
from wandb_mcp_server.mcp_tools import agent_lens as agent_lens_mod
from wandb_mcp_server.mcp_tools.agent_lens import (
    get_category_breakdowns,
    get_clustering_status,
    get_conversation_tags,
    get_insights_coverage,
    get_tag_distribution,
    list_category_example_turns,
    list_conversation_tag_names,
    list_matching_turns,
    list_tagged_conversations,
)
from wandb_mcp_server.trace_utils import count_tokens_conservative

BASE_URL = "https://agent-lens.example.com"
ENTITY = "acme"
PROJECT = "support-bot"
WINDOW = {"start_at": "2026-09-01T00:00:00Z", "end_at": "2026-09-08T00:00:00Z"}


class _FakeResponse:
    def __init__(self, json_data, status_code=200, *, raw=None, headers=None, chunks=None):
        self._json = json_data
        self.status_code = status_code
        self._raw = raw if raw is not None else json.dumps(json_data).encode("utf-8")
        self._chunks = chunks
        self.headers = dict(headers or {})
        self.headers.setdefault(
            "Content-Length", str(sum(len(chunk) for chunk in chunks) if chunks else len(self._raw))
        )
        self.closed = False

    def json(self):
        if self._json is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._json

    def iter_content(self, chunk_size, decode_unicode=False):
        del decode_unicode
        if self._chunks is not None:
            yield from self._chunks
            return
        for offset in range(0, len(self._raw), chunk_size):
            yield self._raw[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class _FakeSession:
    """Records requests and returns a canned response."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    @property
    def last(self):
        return self.calls[-1]


class _BodyPoisonResponse(_FakeResponse):
    @property
    def text(self):
        raise AssertionError("overload classification must not read the streamed response body")


class _BlockingResponse(_FakeResponse):
    """A stream that releases only when the deadline closer closes it."""

    def __init__(self):
        super().__init__({})
        self._released = threading.Event()

    def iter_content(self, chunk_size, decode_unicode=False):
        del chunk_size, decode_unicode
        self._released.wait(timeout=2)
        if self.closed:
            raise requests.ConnectionError("stream closed at deadline")
        yield self._raw

    def close(self):
        self.closed = True
        self._released.set()


@contextmanager
def _mocked(response, api_key="test-key", base_url=BASE_URL):
    """Point the module at a fake session, a fake key, and a fixed origin."""
    session = _FakeSession(response)
    with (
        patch.object(agent_lens_mod, "get_no_retry_session", return_value=session),
        patch.object(WandBApiManager, "get_api_key", staticmethod(lambda: api_key)),
        patch.object(agent_lens_mod, "resolve_agent_lens_base_url", return_value=base_url),
        patch.object(agent_lens_mod, "track_tool_execution", _noop_tracker),
    ):
        yield session


@contextmanager
def _noop_tracker(*_args, **_kwargs):
    ctx = MagicMock()
    yield ctx


def _ok(payload):
    return _FakeResponse({"data": payload})


# ----- request shaping -----


def test_get_requests_carry_bearer_auth_and_project_headers():
    with _mocked(_ok({"first_week": "2026-08-03", "latest_week": "2026-09-14"})) as session:
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))

    call = session.last
    assert call["method"] == "GET"
    assert call["url"] == f"{BASE_URL}/api/insights/latest-week"
    # Agent Lens parses a bearer token, not the trace server's basic auth.
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["headers"]["Accept-Encoding"] == "identity"
    assert call["headers"]["X-Wandb-Entity"] == ENTITY
    assert call["headers"]["X-Wandb-Project"] == PROJECT
    assert call["data"] is None
    assert call["allow_redirects"] is False
    assert call["stream"] is True
    assert result["data"]["latest_week"] == "2026-09-14"


def test_clustering_status_targets_its_own_path():
    with _mocked(_ok([{"signature_type": "intent", "completed_at": "2026-09-14T02:00:00Z"}])) as session:
        get_clustering_status(ENTITY, PROJECT)
    assert session.last["url"] == f"{BASE_URL}/api/insights/clustering-status"


def test_category_breakdowns_sends_the_window_as_query_parameters():
    with _mocked(_ok([{"category": "billing"}])) as session:
        get_category_breakdowns(ENTITY, PROJECT, **WINDOW)
    assert session.last["params"] == WINDOW


def test_example_turns_percent_encodes_the_category_in_the_path():
    with _mocked(_FakeResponse({"data": [], "next_cursor": None})) as session:
        list_category_example_turns(ENTITY, PROJECT, "failure", "billing/refund disputes", **WINDOW, limit=25)
    # A raw slash would otherwise change which endpoint is addressed.
    assert "/insights/failure/categories/billing%2Frefund%20disputes/example-turns" in session.last["url"]
    assert session.last["params"]["limit"] == 25


def test_example_turns_sends_cluster_ids_under_the_repeated_key():
    with _mocked(_FakeResponse({"data": [], "next_cursor": None})) as session:
        list_category_example_turns(ENTITY, PROJECT, "intent", "billing", **WINDOW, cluster_ids=["c1", "c2"])
    assert session.last["params"]["cluster_ids[]"] == ["c1", "c2"]


def test_example_turns_remove_message_bodies_but_keep_reviewed_metadata():
    upstream = {
        "data": [
            {
                "conversation_id": "conv-1",
                "trace_id": "trace-1",
                "agent_message": "private agent response",
                "message": "private user message",
                "failure_reason": "missing context",
                "duration_ms": 123,
                "total_tokens": 42,
                "cost_usd": 0.01,
            }
        ],
        "next_cursor": "cursor-2",
    }
    with _mocked(_FakeResponse(upstream)):
        result = json.loads(
            list_category_example_turns(ENTITY, PROJECT, "failure", "missing-context", **WINDOW, limit=25)
        )
    row = result["data"][0]
    assert "agent_message" not in row
    assert "message" not in row
    assert row == {
        "conversation_id": "conv-1",
        "trace_id": "trace-1",
        "failure_reason": "missing context",
        "duration_ms": 123,
        "total_tokens": 42,
        "cost_usd": 0.01,
    }
    assert result["next_cursor"] == "cursor-2"


def test_post_reads_send_a_json_body():
    with _mocked(_ok([])) as session:
        get_conversation_tags(ENTITY, PROJECT, ["conv-1", "conv-2"])
    call = session.last
    assert call["method"] == "POST"
    assert call["url"] == f"{BASE_URL}/api/conversation-tags/query"
    assert call["headers"]["Content-Type"] == "application/json"
    assert json.loads(call["data"]) == {"conversation_ids": ["conv-1", "conv-2"]}


def test_tagged_conversations_sends_the_tag_filter():
    with _mocked(_ok(["conv-1"])) as session:
        list_tagged_conversations(ENTITY, PROJECT, ["escalated"])
    assert json.loads(session.last["data"]) == {"tags": ["escalated"]}


def test_tag_distribution_sends_epoch_bounds_and_bucket_width():
    with _mocked(_ok({"buckets": []})) as session:
        get_tag_distribution(ENTITY, PROJECT, after_ms=1000, before_ms=2000, time_bucket_seconds=3600)
    assert json.loads(session.last["data"]) == {
        "after_ms": 1000,
        "before_ms": 2000,
        "time_bucket_seconds": 3600,
    }


def test_tag_names_is_a_get_without_a_body():
    with _mocked(_ok(["escalated", "resolved"])) as session:
        list_conversation_tag_names(ENTITY, PROJECT)
    assert session.last["method"] == "GET"
    assert session.last["data"] is None


def test_matching_turns_drops_unset_filters():
    with _mocked(_ok([])) as session:
        list_matching_turns(ENTITY, PROJECT, **WINDOW, intent_category="billing")
    params = session.last["params"]
    assert params["intent_category"] == "billing"
    assert "failure_category" not in params
    assert "cluster_id" not in params


# ----- local argument validation -----


@pytest.mark.parametrize(
    "start_at,end_at,expected",
    [
        ("2026-09-08T00:00:00Z", "2026-09-01T00:00:00Z", "nonempty time range"),
        ("2026-09-01T00:00:00Z", "2026-09-01T00:00:00Z", "nonempty time range"),
        ("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z", "must not exceed 30 days"),
        ("not-a-date", "2026-09-01T00:00:00Z", "RFC 3339"),
    ],
)
def test_invalid_windows_are_rejected_without_a_request(start_at, end_at, expected):
    with _mocked(_ok([])) as session:
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, start_at, end_at))
    assert session.calls == []
    assert result["error"] == "agent_lens_invalid_request"
    assert expected in result["message"]


def test_matching_turns_requires_at_least_one_filter():
    with _mocked(_ok([])) as session:
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW))
    assert session.calls == []
    assert "at least one of" in result["message"]


def test_cluster_id_requires_its_kind():
    with _mocked(_ok([])) as session:
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW, cluster_id="c1"))
    assert session.calls == []
    assert "must be supplied together" in result["message"]


def test_cluster_kind_requires_its_id():
    with _mocked(_ok([])) as session:
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW, cluster_kind="intent"))
    assert session.calls == []
    assert "must be supplied together" in result["message"]


def test_cluster_pair_is_only_a_category_refinement():
    with _mocked(_ok([])) as session:
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW, cluster_id="c1", cluster_kind="intent"))
    assert session.calls == []
    assert "intent_category or failure_category" in result["message"]


def test_category_can_be_refined_by_a_cluster_pair():
    with _mocked(_ok([])) as session:
        list_matching_turns(
            ENTITY,
            PROJECT,
            **WINDOW,
            intent_category="billing",
            cluster_id="c1",
            cluster_kind="intent",
        )
    assert session.last["params"]["intent_category"] == "billing"
    assert session.last["params"]["cluster_id"] == "c1"
    assert session.last["params"]["cluster_kind"] == "intent"


def test_signature_type_is_constrained_to_the_server_enum():
    with _mocked(_ok([])) as session:
        result = json.loads(list_category_example_turns(ENTITY, PROJECT, "sentiment", "billing", **WINDOW))
    assert session.calls == []
    assert result["error"] == "agent_lens_invalid_request"


def test_oversized_conversation_id_list_is_rejected_locally():
    with _mocked(_ok([])) as session:
        result = json.loads(get_conversation_tags(ENTITY, PROJECT, ["c"] * 5001))
    assert session.calls == []
    assert "at most 5000" in result["message"]


def test_empty_tag_list_is_rejected_locally():
    with _mocked(_ok([])) as session:
        result = json.loads(list_tagged_conversations(ENTITY, PROJECT, []))
    assert session.calls == []
    assert "at least one value" in result["message"]


def test_distribution_bounds_must_be_ordered():
    with _mocked(_ok({})) as session:
        result = json.loads(
            get_tag_distribution(ENTITY, PROJECT, after_ms=5000, before_ms=1000, time_bucket_seconds=60)
        )
    assert session.calls == []
    assert "greater than after_ms" in result["message"]


# ----- error mapping -----


def test_missing_api_key_reports_auth_required_without_calling_out():
    with _mocked(_ok([]), api_key=None) as session:
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert session.calls == []
    assert result["error"] == "auth_required"


def test_unconfigured_origin_reports_a_configuration_error():
    session = _FakeSession(_ok([]))
    with (
        patch.object(agent_lens_mod, "get_no_retry_session", return_value=session),
        patch.object(WandBApiManager, "get_api_key", staticmethod(lambda: "test-key")),
        patch.object(
            agent_lens_mod,
            "resolve_agent_lens_base_url",
            side_effect=ValueError("The Agent Lens tool profile requires an explicit AGENT_LENS_BASE_URL"),
        ),
    ):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert session.calls == []
    assert result["error"] == "agent_lens_not_configured"


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_map_to_a_forbidden_error(status):
    with _mocked(_FakeResponse({}, status_code=status)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_forbidden"
    assert result["status_code"] == status
    assert ENTITY not in result["message"]
    assert PROJECT not in result["message"]


def test_404_reports_an_unavailable_endpoint():
    with _mocked(_FakeResponse({}, status_code=404)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_unavailable"


def test_422_passes_through_the_validation_detail():
    with _mocked(_FakeResponse({"detail": "end_at must be after start_at"}, status_code=422)):
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, **WINDOW))
    assert result["error"] == "agent_lens_invalid_request"
    assert "end_at must be after start_at" in result["message"]


def test_422_sanitizes_validation_detail():
    with _mocked(
        _FakeResponse(
            {"detail": "authorization=Bearer test-key at service.svc.cluster.local"},
            status_code=422,
        )
    ):
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, **WINDOW))
    assert "test-key" not in result["message"]
    assert "service.svc" not in result["message"]


def test_422_stream_timeout_is_sanitized_and_closes_response():
    response = _FakeResponse({}, status_code=422)
    response.iter_content = MagicMock(side_effect=requests.Timeout("private validation timeout"))
    with _mocked(response):
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, **WINDOW))
    assert result["error"] == "tool_timeout"
    assert "private validation" not in json.dumps(result)
    assert response.closed is True


def test_422_stream_request_failure_is_sanitized_and_closes_response():
    response = _FakeResponse({}, status_code=422)
    response.iter_content = MagicMock(side_effect=requests.ConnectionError("private-host.svc disconnected"))
    with _mocked(response):
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, **WINDOW))
    assert result == {"error": "agent_lens_query_failed", "message": "The Agent Lens request failed."}
    assert response.closed is True


def test_unexpected_status_maps_to_a_generic_failure():
    with _mocked(_FakeResponse({}, status_code=500)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"
    assert result["status_code"] == 500


@pytest.mark.parametrize("status", [429, 503])
def test_overload_raises_bounded_server_busy_without_reading_body(status):
    response = _BodyPoisonResponse({}, status_code=status, headers={"Retry-After": "999999"})
    with _mocked(response):
        with pytest.raises(WandBServerBusy) as raised:
            get_insights_coverage(ENTITY, PROJECT)
    assert raised.value.status_code == status
    assert raised.value.retry_after_ms == 60_000
    assert response.closed is True


def test_transport_failure_does_not_leak_the_exception_text():
    with _mocked(RuntimeError("connect to agent-lens.internal:8080 refused")):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"
    assert "agent-lens.internal" not in json.dumps(result)


def test_invalid_json_body_maps_to_a_generic_failure():
    with _mocked(_FakeResponse(None, raw=b"not-json")):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_finite_json_is_rejected_as_malformed(constant):
    with _mocked(_FakeResponse({}, raw=b'{"data":' + constant + b"}")):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result == {
        "error": "agent_lens_query_failed",
        "message": "The Agent Lens API returned an invalid response.",
    }


def test_stream_timeout_is_sanitized_and_closes_response():
    response = _FakeResponse({})
    response.iter_content = MagicMock(side_effect=requests.Timeout("secret upstream timeout"))
    with _mocked(response):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "tool_timeout"
    assert "secret upstream" not in json.dumps(result)
    assert response.closed is True


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("private-host.svc disconnected"),
        requests.exceptions.ChunkedEncodingError("credential=secret chunk failed"),
        requests.RequestException("private request failure"),
    ],
)
def test_stream_request_failures_are_sanitized_and_close_response(error):
    response = _FakeResponse({})
    response.iter_content = MagicMock(side_effect=error)
    with _mocked(response):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result == {"error": "agent_lens_query_failed", "message": "The Agent Lens request failed."}
    assert response.closed is True


def test_redirect_is_not_followed():
    with _mocked(_FakeResponse({}, status_code=302, headers={"Location": "https://other.example"})) as session:
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert len(session.calls) == 1
    assert session.last["allow_redirects"] is False
    assert result["error"] == "agent_lens_query_failed"
    assert result["status_code"] == 302


def test_response_is_bounded_before_json_decode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_lens_mod, "MAX_ACCUMULATED_BYTES", 64)
    response = _FakeResponse({}, raw=b"{" + b"x" * 128, headers={"Content-Length": "129"})
    response.json = MagicMock(side_effect=AssertionError("must not decode an oversized response"))
    with _mocked(response):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    response.json.assert_not_called()
    assert result["error"] == "agent_lens_query_failed"
    assert "download limit" in result["message"]


def test_chunked_response_is_bounded_before_json_decode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_lens_mod, "MAX_ACCUMULATED_BYTES", 64)
    response = _FakeResponse({}, headers={}, chunks=[b"{" + b"x" * 40, b"y" * 40])
    response.headers.pop("Content-Length")
    with _mocked(response):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"
    assert "download limit" in result["message"]


def test_active_tool_deadline_bounds_request_timeout():
    token = current_tool_deadline.set(time.monotonic() + 0.5)
    try:
        with _mocked(_ok({"latest_week": "2026-09-14"})) as session:
            get_insights_coverage(ENTITY, PROJECT)
    finally:
        current_tool_deadline.reset(token)
    assert 0 < session.last["timeout"] <= 0.5


def test_expired_tool_deadline_stops_before_backend_request():
    token = current_tool_deadline.set(time.monotonic() - 1)
    try:
        with _mocked(_ok({})) as session:
            result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    finally:
        current_tool_deadline.reset(token)
    assert session.calls == []
    assert result["error"] == "tool_timeout"


@pytest.mark.parametrize("active_deadline_seconds", [None, 5.0])
def test_absolute_request_cap_closes_a_blocked_stream_without_or_before_tool_deadline(
    monkeypatch: pytest.MonkeyPatch,
    active_deadline_seconds: float | None,
):
    monkeypatch.setattr(agent_lens_mod, "_REQUEST_TIMEOUT_SECONDS", 0.05)
    response = _BlockingResponse()
    deadline = None if active_deadline_seconds is None else time.monotonic() + active_deadline_seconds
    token = current_tool_deadline.set(deadline)
    started = time.monotonic()
    try:
        with _mocked(response):
            result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    finally:
        current_tool_deadline.reset(token)
    assert result["error"] == "tool_timeout"
    assert time.monotonic() - started < 1
    assert response.closed is True


def test_request_credentials_are_isolated_between_callers():
    with _mocked(_ok({}), api_key="actor-one-key") as first:
        get_insights_coverage(ENTITY, PROJECT)
    with _mocked(_ok({}), api_key="actor-two-key") as second:
        get_insights_coverage(ENTITY, PROJECT)
    assert first.last["headers"]["Authorization"] == "Bearer actor-one-key"
    assert second.last["headers"]["Authorization"] == "Bearer actor-two-key"


# ----- truncation -----


def test_oversized_list_response_is_trimmed_and_annotated():
    rows = [{"conversation_id": f"c{i}", "trace_id": "t" * 200} for i in range(4000)]
    with _mocked(_ok(rows)):
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW, intent_category="billing"))
    assert result["_truncation"]["applied"] is True
    assert result["_truncation"]["field"] == "data"
    assert result["_truncation"]["original"] == 4000
    assert 0 < len(result["data"]) < 4000


def test_truncation_notice_is_included_in_final_token_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(agent_lens_mod, "MAX_RESPONSE_TOKENS", 300)
    rows = [{"conversation_id": f"c{i}", "trace_id": "t" * 100} for i in range(100)]
    with _mocked(_ok(rows)):
        serialized = list_matching_turns(ENTITY, PROJECT, **WINDOW, intent_category="billing")
    result = json.loads(serialized)
    assert result["_truncation"]["applied"] is True
    assert count_tokens_conservative(serialized) <= 300


def test_paginated_page_fails_instead_of_returning_cursor_after_dropping_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(agent_lens_mod, "MAX_RESPONSE_TOKENS", 300)
    rows = [{"conversation_id": f"c{i}", "trace_id": "t" * 100} for i in range(100)]
    with _mocked(_FakeResponse({"data": rows, "next_cursor": "would-skip-dropped-rows"})):
        serialized = list_category_example_turns(
            ENTITY,
            PROJECT,
            "intent",
            "billing",
            **WINDOW,
            limit=50,
        )
    result = json.loads(serialized)
    assert result["error"] == "agent_lens_response_too_large"
    assert "lower limit" in result["message"]
    assert "next_cursor" not in result
    assert "data" not in result
    assert count_tokens_conservative(serialized) <= 300


def test_oversized_distribution_trims_the_nested_buckets():
    buckets = [{"time_bucket_start_ms": i, "tag_counts": {f"tag-{n}": n for n in range(50)}} for i in range(4000)]
    payload = {"time_bucket_seconds": 60, "after_ms": 0, "before_ms": 1, "buckets": buckets}
    with _mocked(_ok(payload)):
        result = json.loads(get_tag_distribution(ENTITY, PROJECT, after_ms=0, before_ms=1, time_bucket_seconds=60))
    assert result["_truncation"]["field"] == "buckets"
    assert 0 < len(result["data"]["buckets"]) < 4000
    # Sibling fields must survive so the caller can still read the bucket width.
    assert result["data"]["time_bucket_seconds"] == 60


def test_small_response_is_returned_untouched():
    with _mocked(_ok(["escalated"])):
        result = json.loads(list_conversation_tag_names(ENTITY, PROJECT))
    assert result == {"data": ["escalated"]}


# ----- live-smoke qualification harness -----


def _run_smoke_with_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    empty_tool: str | None = None,
) -> tuple[int, list[str]]:
    responses = {
        "get_insights_coverage": {"latest_week": "2026-09-01"},
        "get_clustering_status": [{"signature_type": "intent"}],
        "get_category_breakdowns": [{"category": "action_request", "failure_breakdowns": []}],
        "list_category_example_turns": [{"conversation_id": "conv-1", "trace_id": "trace-1"}],
        "list_matching_turns": [{"conversation_id": "conv-1", "trace_id": "trace-1"}],
        "list_conversation_tag_names": ["reviewed"],
        "get_conversation_tags": [{"conversation_id": "conv-1", "tag": "reviewed"}],
        "list_tagged_conversations": ["conv-1"],
        "get_tag_distribution": {"buckets": [{"tag_counts": {"reviewed": 1}}]},
    }
    calls: list[str] = []

    def fake(name: str):
        def invoke(*_args, **_kwargs):
            calls.append(name)
            data = [] if name == empty_tool else responses[name]
            return json.dumps({"data": data})

        return invoke

    for name in responses:
        monkeypatch.setattr(agent_lens_smoke, name, fake(name))
    monkeypatch.setattr(agent_lens_smoke, "_seed_api_key", lambda: "test-key")
    monkeypatch.setenv("AGENT_LENS_BASE_URL", BASE_URL)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_lens_smoke.py",
            "--entity",
            ENTITY,
            "--project",
            PROJECT,
            "--start-at",
            WINDOW["start_at"],
            "--end-at",
            WINDOW["end_at"],
            "--signature-type",
            "intent",
            "--category-id",
            "action_request",
            "--tag-name",
            "reviewed",
            "--conversation-id",
            "conv-1",
        ],
    )
    return agent_lens_smoke.main(), calls


def test_live_smoke_requires_matching_data_from_all_nine_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    result, calls = _run_smoke_with_fixtures(monkeypatch)
    output = capsys.readouterr().out
    assert result == 0
    assert len(calls) == len(set(calls)) == 9
    assert "9/9 checks passed" in output
    assert "skip" not in output.lower()


def test_live_smoke_empty_fixture_fails_but_still_exercises_all_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    result, calls = _run_smoke_with_fixtures(monkeypatch, empty_tool="get_clustering_status")
    output = capsys.readouterr().out
    assert result == 1
    assert len(calls) == len(set(calls)) == 9
    assert "8/9 checks passed" in output
    assert "skip" not in output.lower()
