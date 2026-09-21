"""Unit tests for the Agent Lens Insights and conversation-tag MCP tools.

The HTTP boundary is mocked at the requests.Session level so we can assert
request shaping, auth, response passthrough, truncation, and error mapping
without a live Agent Lens. Canned response shapes mirror the Huma output
structs in internal/api/insights.go and internal/api/conversation_tags.go of
github.com/wandb/agent-lens.
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.api_client import WandBApiManager
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

BASE_URL = "https://agent-lens.example.com"
ENTITY = "acme"
PROJECT = "support-bot"
WINDOW = {"start_at": "2026-09-01T00:00:00Z", "end_at": "2026-09-08T00:00:00Z"}


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.headers = {}

    def json(self):
        if self._json is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._json


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
    assert call["headers"]["X-Wandb-Entity"] == ENTITY
    assert call["headers"]["X-Wandb-Project"] == PROJECT
    assert call["data"] is None
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
    assert "cluster_kind is required" in result["message"]


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
    assert f"{ENTITY}/{PROJECT}" in result["message"]


def test_404_reports_an_unavailable_endpoint():
    with _mocked(_FakeResponse({}, status_code=404)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_unavailable"


def test_422_passes_through_the_validation_detail():
    with _mocked(_FakeResponse({"detail": "end_at must be after start_at"}, status_code=422)):
        result = json.loads(get_category_breakdowns(ENTITY, PROJECT, **WINDOW))
    assert result["error"] == "agent_lens_invalid_request"
    assert "end_at must be after start_at" in result["message"]


def test_unexpected_status_maps_to_a_generic_failure():
    with _mocked(_FakeResponse({}, status_code=500)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"
    assert result["status_code"] == 500


def test_transport_failure_does_not_leak_the_exception_text():
    with _mocked(RuntimeError("connect to agent-lens.internal:8080 refused")):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"
    assert "agent-lens.internal" not in json.dumps(result)


def test_invalid_json_body_maps_to_a_generic_failure():
    with _mocked(_FakeResponse(None)):
        result = json.loads(get_insights_coverage(ENTITY, PROJECT))
    assert result["error"] == "agent_lens_query_failed"


# ----- truncation -----


def test_oversized_list_response_is_trimmed_and_annotated():
    rows = [{"conversation_id": f"c{i}", "trace_id": "t" * 200} for i in range(4000)]
    with _mocked(_ok(rows)):
        result = json.loads(list_matching_turns(ENTITY, PROJECT, **WINDOW, intent_category="billing"))
    assert result["_truncation"]["applied"] is True
    assert result["_truncation"]["field"] == "data"
    assert result["_truncation"]["original"] == 4000
    assert 0 < len(result["data"]) < 4000


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
