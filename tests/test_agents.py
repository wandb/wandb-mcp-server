"""Unit tests for the Weave Agents (OTel) MCP tools.

The HTTP boundary (the trace server's /agents/* endpoints) is mocked at the
requests.Session level so we can assert request shaping, response passthrough,
truncation, and error mapping without a live backend. Canned response shapes
mirror the wandb/weave Node SDK cassettes.
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools import agents as agents_mod
from wandb_mcp_server.mcp_tools.agents import (
    _normalize_stats_metrics,
    get_agent_conversation,
    get_agent_span_stats,
    get_agent_trace,
    list_agent_custom_attributes,
    list_agent_versions,
    list_agents,
    query_agent_spans,
    search_agents,
)


class _FakeResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.text = text

    def json(self):
        if self._json is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._json


class _FakeSession:
    """Records POSTs and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return self.response


@contextmanager
def _patched(json_data, status_code=200, text=""):
    """Patch auth + the agents HTTP session; yield the fake session for asserts."""
    session = _FakeSession(_FakeResponse(json_data, status_code=status_code, text=text))
    with (
        patch.object(WandBApiManager, "get_api_key", return_value="test-key"),
        patch.object(WandBApiManager, "get_api", return_value=MagicMock(viewer=MagicMock())),
        patch("wandb_mcp_server.mcp_tools.agents.get_retry_session", return_value=session),
    ):
        yield session


def _body(session):
    return json.loads(session.calls[0]["data"])


def _url(session):
    return session.calls[0]["url"]


class TestListAgents:
    def test_request_and_response_passthrough(self):
        canned = {
            "agents": [
                {
                    "project_id": "e/p",
                    "agent_name": "my-cool-agent",
                    "invocation_count": 33,
                    "span_count": 33,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_duration_ms": 4,
                    "error_count": 0,
                    "first_seen": "2026-06-15T20:37:39",
                    "last_seen": "2026-06-15T20:49:24",
                }
            ],
            "total_count": 1,
        }
        with _patched(canned) as session:
            out = list_agents("e", "p")

        data = json.loads(out)
        assert data["total_count"] == 1
        assert data["agents"][0]["agent_name"] == "my-cool-agent"
        assert _url(session).endswith("/agents/query")
        body = _body(session)
        assert body["project_id"] == "e/p"
        assert body["include_costs"] is False
        # agent_name not provided -> filters dropped (None), server applies default
        assert "filters" not in body

    def test_agent_name_and_sort_shape_request(self):
        with _patched({"agents": [], "total_count": 0}) as session:
            list_agents("e", "p", agent_name="bot", sort_by="last_seen", limit=5)
        body = _body(session)
        assert body["filters"] == {"agent_name": "bot"}
        assert body["sort_by"] == [{"field": "last_seen", "direction": "desc"}]
        assert body["limit"] == 5


class TestListAgentVersions:
    def test_request(self):
        with _patched({"versions": [], "total_count": 0}) as session:
            list_agent_versions("e", "p", agent_name="bot")
        assert _url(session).endswith("/agents/agent-versions/query")
        assert _body(session)["agent_name"] == "bot"


class TestQueryAgentSpans:
    def test_agent_name_becomes_query_expr(self):
        with _patched({"spans": [], "total_count": 0}) as session:
            query_agent_spans("e", "p", agent_name="bot")
        assert _url(session).endswith("/agents/spans/query")
        body = _body(session)
        assert body["query"]["$expr"]["$eq"] == [{"$getField": "agent_name"}, {"$literal": "bot"}]
        # default keeps payloads lean
        assert body["include_details"] is False

    def test_agent_name_and_query_combined_with_and(self):
        user_query = {"$expr": {"$eq": [{"$getField": "status_code"}, {"$literal": "ERROR"}]}}
        with _patched({"spans": [], "total_count": 0}) as session:
            query_agent_spans("e", "p", agent_name="bot", query=user_query)
        expr = _body(session)["query"]["$expr"]
        assert "$and" in expr
        assert len(expr["$and"]) == 2


class TestSpanStats:
    def test_metric_normalization_and_request(self):
        with _patched({"start": "x", "end": "y", "timezone": "UTC", "columns": [], "rows": []}) as session:
            get_agent_span_stats(
                "e",
                "p",
                start="2026-06-01T00:00:00Z",
                metrics=["input_tokens", "is_error", "total_cost_usd"],
                group_by=["agent_name"],
                granularity_seconds=3600,
            )
        assert _url(session).endswith("/agents/spans/stats")
        body = _body(session)
        assert body["granularity"] == 3600
        assert body["group_by"] == [{"source": "field", "key": "agent_name"}]
        by_alias = {m["alias"]: m for m in body["metrics"]}
        # raw field -> field source, number, requested aggregation
        assert by_alias["input_tokens"]["value"] == {"source": "field", "key": "input_tokens"}
        assert by_alias["input_tokens"]["aggregations"] == ["sum"]
        # boolean derived -> count_true
        assert by_alias["is_error"]["value_type"] == "boolean"
        assert by_alias["is_error"]["aggregations"] == ["count_true"]
        assert by_alias["is_error"]["value"]["source"] == "derived"
        # numeric derived
        assert by_alias["total_cost_usd"]["value"] == {"source": "derived", "key": "total_cost_usd"}

    def test_default_metrics(self):
        with _patched({"rows": []}) as session:
            get_agent_span_stats("e", "p", start="2026-06-01T00:00:00Z")
        aliases = {m["alias"] for m in _body(session)["metrics"]}
        assert aliases == {"input_tokens", "output_tokens"}


class TestSearchAndChat:
    def test_search_request(self):
        canned = {"results": [{"conversation_id": "c1", "matched_messages": []}], "total_conversations": 1}
        with _patched(canned) as session:
            out = search_agents("e", "p", query="Liverpool", limit=2)
        assert _url(session).endswith("/agents/search")
        body = _body(session)
        assert body["query"] == "Liverpool"
        assert body["limit"] == 2
        assert json.loads(out)["total_conversations"] == 1

    def test_trace_chat_request(self):
        with _patched({"trace_id": "t1", "messages": []}) as session:
            get_agent_trace("e", "p", trace_id="t1")
        assert _url(session).endswith("/agents/traces/chat")
        assert _body(session)["trace_id"] == "t1"

    def test_conversation_chat_request(self):
        with _patched({"conversation_id": "c1", "turns": []}) as session:
            get_agent_conversation("e", "p", conversation_id="c1")
        assert _url(session).endswith("/agents/conversations/chat")
        assert _body(session)["conversation_id"] == "c1"


class TestCustomAttrs:
    def test_request(self):
        with _patched({"attributes": [], "has_more": False}) as session:
            list_agent_custom_attributes("e", "p")
        assert _url(session).endswith("/agents/spans/custom-attrs/schema")
        assert _body(session)["project_id"] == "e/p"


class TestErrorHandling:
    def test_missing_api_key(self):
        with patch.object(WandBApiManager, "get_api_key", return_value=None):
            out = list_agents("e", "p")
        assert json.loads(out)["error"] == "auth_required"

    def test_404_maps_to_unavailable(self):
        with _patched(None, status_code=404, text="not found"):
            out = list_agents("e", "p")
        data = json.loads(out)
        assert data["error"] == "agents_api_unavailable"
        assert data["status_code"] == 404

    def test_500_maps_to_query_failed(self):
        with _patched(None, status_code=500, text="boom"):
            out = list_agents("e", "p")
        assert json.loads(out)["error"] == "agents_query_failed"


class TestTruncation:
    def test_large_span_response_is_truncated(self, monkeypatch):
        monkeypatch.setattr(agents_mod, "MAX_RESPONSE_TOKENS", 50)
        spans = [{"span_id": f"s{i}", "content": "x" * 200} for i in range(20)]
        with _patched({"spans": spans, "total_count": 20}):
            out = query_agent_spans("e", "p")
        data = json.loads(out)
        assert data["_truncation"]["applied"] is True
        assert data["_truncation"]["field"] == "spans"
        assert len(data["spans"]) < 20


class TestNormalizeStatsMetrics:
    def test_raw_field_string(self):
        specs = _normalize_stats_metrics(["input_tokens"], "avg")
        assert specs == [
            {
                "alias": "input_tokens",
                "value_type": "number",
                "aggregations": ["avg"],
                "value": {"source": "field", "key": "input_tokens"},
            }
        ]

    def test_boolean_derived_uses_count_true(self):
        (spec,) = _normalize_stats_metrics(["is_invocation"], "sum")
        assert spec["value_type"] == "boolean"
        assert spec["aggregations"] == ["count_true"]
        assert spec["value"] == {"source": "derived", "key": "is_invocation"}

    def test_full_dict_passthrough(self):
        raw = {
            "alias": "p95_latency",
            "value_type": "number",
            "aggregations": ["avg"],
            "percentiles": [95.0],
            "value": {"source": "derived", "key": "duration_ms"},
        }
        (spec,) = _normalize_stats_metrics([raw], "sum")
        assert spec == raw

    def test_dict_shorthand_builds_value(self):
        (spec,) = _normalize_stats_metrics([{"field": "output_tokens"}], "max")
        assert spec["value"] == {"source": "field", "key": "output_tokens"}
        assert spec["aggregations"] == ["max"]
        assert spec["alias"] == "output_tokens"

    def test_invalid_entry_raises(self):
        with pytest.raises(ValueError):
            _normalize_stats_metrics([123], "sum")
