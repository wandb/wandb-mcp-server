"""
Unit tests for the Weave API client, processors, query builder, and service.

Ported from wandb-mcp-server-internal. All tests use mocks --
no real API keys or network calls are needed.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from wandb_mcp_server.weave_api.client import WeaveApiClient
from wandb_mcp_server.weave_api.models import (
    FilterOperator,
    QueryResult,
)
from wandb_mcp_server.weave_api.processors import TraceProcessor
from wandb_mcp_server.weave_api.query_builder import QueryBuilder


# ---------------------------------------------------------------------------
# TraceProcessor tests
# ---------------------------------------------------------------------------


class TestTraceProcessor(unittest.TestCase):
    """Tests for the TraceProcessor class."""

    def test_truncate_value_with_string(self):
        value = "a" * 300
        result = TraceProcessor.truncate_value(value, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_truncate_value_with_dict(self):
        value = {"key1": "a" * 300, "key2": "b" * 50}
        result = TraceProcessor.truncate_value(value, max_length=100)
        assert len(result["key1"]) == 103
        assert result["key2"] == "b" * 50

    def test_truncate_value_with_zero_length(self):
        value = {"key1": "value1", "key2": 123, "key3": ["item1", "item2"]}
        result = TraceProcessor.truncate_value(value, max_length=0)
        assert result == {}

    def test_truncate_value_with_none(self):
        assert TraceProcessor.truncate_value(None) is None

    def test_truncate_value_with_complex_type(self):
        complex_object = {"__type__": "ComplexObject", "data": "a" * 200}
        result = TraceProcessor.truncate_value(complex_object, max_length=50)
        assert result == {"type": "ComplexObject"}

    def test_count_tokens(self):
        text = "This is a test of the token counter."
        result = TraceProcessor.count_tokens(text)
        assert result > 0

    def test_count_tokens_fallback(self):
        with patch("tiktoken.get_encoding", side_effect=Exception("Tiktoken error")):
            result = TraceProcessor.count_tokens("This is a test of the token counter.")
            assert result == 8  # word count fallback

    def test_process_traces(self):
        now = datetime.now()
        traces = [
            {
                "id": "trace1",
                "project_id": "entity/project",
                "op_name": "weave:///entity/project/op/test:123",
                "trace_id": "trace1",
                "started_at": now.isoformat(),
                "inputs": {"text": "a" * 300},
                "output": "b" * 300,
                "status": "success",
            },
            {
                "id": "trace2",
                "project_id": "entity/project",
                "op_name": "weave:///entity/project/op/other:456",
                "trace_id": "trace2",
                "started_at": now.isoformat(),
                "inputs": {"text": "c" * 300},
                "output": "d" * 300,
                "status": "error",
            },
        ]

        result = TraceProcessor.process_traces(traces, truncate_length=100)
        assert isinstance(result, QueryResult)
        assert result.metadata.total_traces == 2
        assert result.metadata.status_summary == {"success": 1, "error": 1, "other": 0}
        assert len(result.traces) == 2

    def test_process_traces_metadata_only(self):
        traces = [
            {
                "id": "t1",
                "project_id": "e/p",
                "op_name": "weave:///e/p/op/x:1",
                "trace_id": "t1",
                "started_at": datetime.now().isoformat(),
                "status": "success",
            }
        ]
        result = TraceProcessor.process_traces(traces, metadata_only=True)
        assert result.metadata.total_traces == 1
        assert result.traces is None

    def test_extract_op_name_distribution(self):
        traces = [
            {"op_name": "weave:///entity/project/op/test:123"},
            {"op_name": "weave:///entity/project/op/test:456"},
            {"op_name": "weave:///entity/project/op/other:789"},
            {"op_name": "invalid_op_name"},
        ]
        distribution = TraceProcessor.extract_op_name_distribution(traces)
        assert distribution == {"test": 2, "other": 1}

    def test_get_time_range(self):
        now = datetime.now()
        earlier = now - timedelta(hours=1)
        later = now + timedelta(hours=1)
        traces = [
            {"started_at": earlier.isoformat(), "ended_at": now.isoformat()},
            {"started_at": now.isoformat(), "ended_at": later.isoformat()},
        ]
        time_range = TraceProcessor.get_time_range(traces)
        assert time_range["earliest"] == earlier.isoformat()
        assert time_range["latest"] == later.isoformat()
        assert TraceProcessor.get_time_range([]) == {"earliest": None, "latest": None}

    def test_get_cost(self):
        trace = {
            "costs": {
                "gpt-4": {"prompt_tokens_total_cost": 0.5, "completion_tokens_total_cost": 1.0, "total_cost": 1.5},
                "gpt-3.5-turbo": {
                    "prompt_tokens_total_cost": 0.1,
                    "completion_tokens_total_cost": 0.2,
                    "total_cost": 0.3,
                },
            }
        }
        assert TraceProcessor.get_cost(trace, "total_cost") == 1.8
        assert TraceProcessor.get_cost(trace, "completion_cost") == 1.2
        assert TraceProcessor.get_cost(trace, "prompt_cost") == 0.6
        assert TraceProcessor.get_cost(trace, "invalid_cost") == 0.0
        assert TraceProcessor.get_cost({}, "total_cost") == 0.0


# ---------------------------------------------------------------------------
# QueryBuilder tests
# ---------------------------------------------------------------------------


class TestQueryBuilder(unittest.TestCase):
    """Tests for the QueryBuilder class."""

    def test_datetime_to_timestamp(self):
        assert QueryBuilder.datetime_to_timestamp("2021-01-01T00:00:00Z") == 1609459200
        assert QueryBuilder.datetime_to_timestamp("2021-01-01T00:00:00+00:00") == 1609459200
        assert QueryBuilder.datetime_to_timestamp("invalid_datetime") == 0
        assert QueryBuilder.datetime_to_timestamp("") == 0

    def test_separate_filters(self):
        filters = {"trace_roots_only": True, "op_name": "test_op", "status": "success", "latency": {"$gt": 1000}}
        direct, complex_f = QueryBuilder.separate_filters(filters)
        assert direct == {"trace_roots_only": True, "op_names": ["test_op"]}
        assert complex_f == {"status": "success", "latency": {"$gt": 1000}}

    def test_separate_filters_with_trace_id(self):
        direct, complex_f = QueryBuilder.separate_filters({"trace_id": "123"})
        assert direct == {"trace_ids": ["123"]}

    def test_separate_filters_with_call_ids_list(self):
        direct, _ = QueryBuilder.separate_filters({"call_ids": ["123", "456"]})
        assert direct == {"call_ids": ["123", "456"]}

    def test_separate_filters_with_call_ids_string(self):
        direct, _ = QueryBuilder.separate_filters({"call_ids": "123"})
        assert direct == {"call_ids": ["123"]}

    def test_prepare_query_params(self):
        params = {
            "entity_name": "test_entity",
            "project_name": "test_project",
            "filters": {"trace_roots_only": True, "op_name": "test_op", "status": "success"},
            "sort_by": "started_at",
            "sort_direction": "desc",
            "limit": 10,
            "offset": 0,
            "include_costs": True,
            "include_feedback": True,
            "columns": ["id", "op_name", "status"],
        }
        result = QueryBuilder.prepare_query_params(params)
        assert result["project_id"] == "test_entity/test_project"
        assert result["filter"] == {"trace_roots_only": True, "op_names": ["test_op"]}
        assert "query" in result
        assert result["sort_by"] == [{"field": "started_at", "direction": "desc"}]
        assert result["limit"] == 10

    def test_create_contains_operation(self):
        op = QueryBuilder.create_contains_operation("op_name", "test")
        d = op.model_dump(by_alias=True)
        assert d["$contains"]["substr"]["$literal"] == "test"
        assert d["$contains"]["input"]["$getField"] == "op_name"

    def test_create_comparison_operation(self):
        eq_op = QueryBuilder.create_comparison_operation("field", FilterOperator.EQUALS, "value")
        assert eq_op.model_dump(by_alias=True)["$eq"][0]["$getField"] == "field"

        gt_op = QueryBuilder.create_comparison_operation("field", FilterOperator.GREATER_THAN, 100)
        assert gt_op.model_dump(by_alias=True)["$gt"][1]["$literal"] == 100

        lt_op = QueryBuilder.create_comparison_operation("field", FilterOperator.LESS_THAN, 100)
        assert "$not" in lt_op.model_dump(by_alias=True)

        assert QueryBuilder.create_comparison_operation("field", "invalid_operator", "v") is None

    def test_in_operation(self):
        """$in filter produces an InOperation with correct field and values."""
        filters = {"$in": {"summary.weave.status": ["error", "running"]}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        dumped = query.model_dump(by_alias=True)
        assert "$in" in str(dumped)

    def test_or_operation(self):
        """$or combines sub-filters with OrOperation."""
        filters = {"$or": [{"status": "error"}, {"status": "running"}]}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        dumped = query.model_dump(by_alias=True)
        expr = dumped.get("$expr", {})
        assert "$or" in str(expr) or "$and" in str(expr)

    def test_or_with_single_clause_unwraps(self):
        """$or with one clause should simplify to just that clause."""
        filters = {"$or": [{"status": "error"}]}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        dumped = query.model_dump(by_alias=True)
        assert "$or" not in str(dumped.get("$expr", {}))

    def test_in_and_regular_filters_combine(self):
        """$in coexists with regular filters under $and."""
        filters = {"has_exception": True, "$in": {"summary.weave.status": ["error", "running"]}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        dumped = query.model_dump(by_alias=True)
        assert "$and" in str(dumped) or "$in" in str(dumped)


# ---------------------------------------------------------------------------
# WeaveApiClient tests
# ---------------------------------------------------------------------------


class TestWeaveApiClient(unittest.TestCase):
    """Tests for the WeaveApiClient class."""

    def test_init_with_explicit_key(self):
        client = WeaveApiClient(api_key="test_key_12345")
        assert client.api_key == "test_key_12345"
        assert client.retries == 3
        assert client.timeout == WeaveApiClient.DEFAULT_TIMEOUT

    def test_init_without_key_raises(self):
        with pytest.raises(ValueError, match="API key not provided"):
            WeaveApiClient(api_key=None)

    def test_get_auth_headers(self):
        client = WeaveApiClient(api_key="test_key")
        headers = client._get_auth_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/jsonl"
        assert "Basic " in headers["Authorization"]

    @patch("requests.Session.post")
    def test_query_traces_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"id": "1", "op_name": "test"}',
            b'{"id": "2", "op_name": "test2"}',
        ]
        mock_post.return_value = mock_response

        client = WeaveApiClient(api_key="test_key")
        results = list(client.query_traces({"project_id": "entity/project"}))
        assert len(results) == 2
        assert results[0]["id"] == "1"

    @patch("requests.Session.post")
    def test_query_traces_error_response(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        client = WeaveApiClient(api_key="test_key")
        with pytest.raises(Exception, match="Error 400"):
            list(client.query_traces({"project_id": "entity/project"}))

    @patch("requests.Session.post")
    def test_query_traces_network_error(self, mock_post):
        mock_post.side_effect = requests.RequestException("Network error")
        client = WeaveApiClient(api_key="test_key")
        with pytest.raises(Exception, match="Failed to query Weave traces"):
            list(client.query_traces({"project_id": "entity/project"}))

    def test_retry_adapter_mounted(self):
        """Verify retry adapter is mounted on the session for https and http."""
        client = WeaveApiClient(api_key="test_key")
        https_adapter = client.session.get_adapter("https://trace.wandb.ai")
        http_adapter = client.session.get_adapter("http://localhost:8080")
        assert https_adapter.max_retries.total == 3
        assert 429 in https_adapter.max_retries.status_forcelist
        assert 503 in https_adapter.max_retries.status_forcelist
        assert http_adapter.max_retries.total == 3

    def test_custom_retries_and_timeout(self):
        """Verify custom retries and timeout are applied."""
        client = WeaveApiClient(api_key="test_key", retries=5, timeout=60)
        assert client.timeout == 60
        adapter = client.session.get_adapter("https://trace.wandb.ai")
        assert adapter.max_retries.total == 5

    def test_no_retry_on_client_error_status(self):
        """Status codes like 400 should not be in the retry list."""
        client = WeaveApiClient(api_key="test_key")
        adapter = client.session.get_adapter("https://trace.wandb.ai")
        assert 400 not in adapter.max_retries.status_forcelist
        assert 401 not in adapter.max_retries.status_forcelist
        assert 404 not in adapter.max_retries.status_forcelist


if __name__ == "__main__":
    unittest.main()
