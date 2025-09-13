"""
Tests for the Weave API client.

This module provides tests for the Weave API client implementation.
"""

import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from wandb_mcp_server.weave_api.client import WeaveApiClient
from wandb_mcp_server.weave_api.models import (
    FilterOperator,
    QueryFilter,
    QueryParams,
    QueryResult,
    TraceMetadata,
    WeaveTrace,
)
from wandb_mcp_server.weave_api.processors import TraceProcessor
from wandb_mcp_server.weave_api.query_builder import QueryBuilder
from wandb_mcp_server.weave_api.service import TraceService


class TestTraceProcessor(unittest.TestCase):
    """Tests for the TraceProcessor class."""

    def test_truncate_value_with_string(self):
        """Test truncating a string value."""
        value = "a" * 300
        result = TraceProcessor.truncate_value(value, max_length=100)
        assert len(result) == 103  # 100 + 3 for "..."
        assert result.endswith("...")

    def test_truncate_value_with_dict(self):
        """Test truncating a dictionary with string values."""
        value = {"key1": "a" * 300, "key2": "b" * 50}
        result = TraceProcessor.truncate_value(value, max_length=100)
        assert len(result["key1"]) == 103
        assert result["key1"].endswith("...")
        assert len(result["key2"]) == 50
        assert not result["key2"].endswith("...")

    def test_truncate_value_with_zero_length(self):
        """Test truncating with zero length."""
        value = {"key1": "value1", "key2": 123, "key3": ["item1", "item2"]}
        result = TraceProcessor.truncate_value(value, max_length=0)
        assert result == {}

    def test_truncate_value_with_complex_types(self):
        """Test truncating complex types."""
        # Test with None
        assert TraceProcessor.truncate_value(None) is None

        # Test with complex object reference - need to use max_length >= 50 to get the type representation
        complex_object = {"__type__": "ComplexObject", "data": "a" * 200}
        result = TraceProcessor.truncate_value(complex_object, max_length=50)
        assert result == {
            "type": "ComplexObject"
        }  # The processor detects this special case

        # Test with very small max_length for complex object
        result = TraceProcessor.truncate_value(complex_object, max_length=5)
        assert result == {}  # With very small max_length, it returns empty dict

        # Test with list of strings
        value = ["a" * 300, "b" * 50]
        result = TraceProcessor.truncate_value(value, max_length=100)
        assert len(result[0]) == 103
        assert result[0].endswith("...")
        assert len(result[1]) == 50

        # Test with non-serializable object
        class CustomClass:
            def __str__(self):
                return "CustomObject" * 50

        custom_obj = CustomClass()
        result = TraceProcessor.truncate_value(custom_obj, max_length=20)
        assert len(result) == 23
        assert result.endswith("...")

    def test_count_tokens(self):
        """Test counting tokens in a string."""
        text = "This is a test of the token counter."
        result = TraceProcessor.count_tokens(text)
        assert result > 0

    def test_count_tokens_fallback(self):
        """Test token counting fallback when tiktoken fails."""
        with patch("tiktoken.get_encoding", side_effect=Exception("Tiktoken error")):
            text = "This is a test of the token counter."
            result = TraceProcessor.count_tokens(text)
            # Fallback method should count words
            assert result == 8

    def test_process_traces(self):
        """Test processing traces."""
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

        # Test with normal truncation
        result = TraceProcessor.process_traces(traces, truncate_length=100)
        assert isinstance(result, QueryResult)
        assert isinstance(result.metadata, TraceMetadata)
        assert result.metadata.total_traces == 2
        assert result.metadata.status_summary == {"success": 1, "error": 1, "other": 0}
        assert len(result.traces) == 2

        # Now that traces are WeaveTrace objects, we need to access their attributes properly
        assert len(result.traces[0].inputs["text"]) == 103
        assert result.traces[0].inputs["text"].endswith("...")

        # Test with metadata only
        result = TraceProcessor.process_traces(traces, metadata_only=True)
        assert isinstance(result, QueryResult)
        assert isinstance(result.metadata, TraceMetadata)
        assert result.metadata.total_traces == 2
        assert result.traces is None

        # Test with full data
        result = TraceProcessor.process_traces(traces, return_full_data=True)
        assert isinstance(result, QueryResult)
        assert len(result.traces) == 2
        assert len(result.traces[0].inputs["text"]) == 300

    def test_extract_op_name_distribution(self):
        """Test extracting operation name distribution."""
        traces = [
            {
                "op_name": "weave:///entity/project/op/test:123",
            },
            {
                "op_name": "weave:///entity/project/op/test:456",
            },
            {
                "op_name": "weave:///entity/project/op/other:789",
            },
            {
                "op_name": "invalid_op_name",
            },
        ]

        distribution = TraceProcessor.extract_op_name_distribution(traces)
        assert distribution == {"test": 2, "other": 1}

    def test_get_time_range(self):
        """Test getting time range from traces."""
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

        # Test with empty traces
        assert TraceProcessor.get_time_range([]) == {"earliest": None, "latest": None}

        # Test with missing timestamps
        assert TraceProcessor.get_time_range([{}]) == {"earliest": None, "latest": None}

    def test_get_cost(self):
        """Test extracting cost information from a trace."""
        trace = {
            "costs": {
                "gpt-4": {
                    "prompt_tokens_total_cost": 0.5,
                    "completion_tokens_total_cost": 1.0,
                    "total_cost": 1.5,
                },
                "gpt-3.5-turbo": {
                    "prompt_tokens_total_cost": 0.1,
                    "completion_tokens_total_cost": 0.2,
                    "total_cost": 0.3,
                },
            }
        }

        # Test total cost
        assert TraceProcessor.get_cost(trace, "total_cost") == 1.8

        # Test completion cost
        assert TraceProcessor.get_cost(trace, "completion_cost") == 1.2

        # Test prompt cost
        assert TraceProcessor.get_cost(trace, "prompt_cost") == 0.6

        # Test with invalid cost type
        assert TraceProcessor.get_cost(trace, "invalid_cost") == 0.0

        # Test with no costs
        assert TraceProcessor.get_cost({}, "total_cost") == 0.0

        # Test with non-numeric cost values
        trace_with_invalid = {
            "costs": {
                "invalid": {
                    "prompt_tokens_total_cost": "not_a_number",
                    "completion_tokens_total_cost": None,
                }
            }
        }
        assert TraceProcessor.get_cost(trace_with_invalid, "total_cost") == 0.0


class TestQueryBuilder(unittest.TestCase):
    """Tests for the QueryBuilder class."""

    def test_datetime_to_timestamp(self):
        """Test converting datetime string to timestamp."""
        dt_str = "2021-01-01T00:00:00Z"
        result = QueryBuilder.datetime_to_timestamp(dt_str)
        assert result == 1609459200

        # Test with +00:00 format
        dt_str = "2021-01-01T00:00:00+00:00"
        result = QueryBuilder.datetime_to_timestamp(dt_str)
        assert result == 1609459200

        # Test with invalid format
        dt_str = "invalid_datetime"
        result = QueryBuilder.datetime_to_timestamp(dt_str)
        assert result == 0

        # Test with empty string
        result = QueryBuilder.datetime_to_timestamp("")
        assert result == 0

    def test_separate_filters(self):
        """Test separating filters into direct and complex."""
        filters = {
            "trace_roots_only": True,
            "op_name": "test_op",
            "status": "success",
            "latency": {"$gt": 1000},
        }
        direct, complex = QueryBuilder.separate_filters(filters)
        assert direct == {"trace_roots_only": True, "op_names": ["test_op"]}
        assert complex == {"status": "success", "latency": {"$gt": 1000}}

        # Test with pattern in op_name
        filters = {
            "op_name": "test_op*",
            "trace_roots_only": True,
        }
        direct, complex = QueryBuilder.separate_filters(filters)
        assert "op_names" not in direct
        assert complex == {"op_name": "test_op*"}

        # Test with trace_id
        filters = {
            "trace_id": "123",
        }
        direct, complex = QueryBuilder.separate_filters(filters)
        assert direct == {"trace_ids": ["123"]}
        assert complex == {}

        # Test with call_ids
        filters = {
            "call_ids": ["123", "456"],
        }
        direct, complex = QueryBuilder.separate_filters(filters)
        assert direct == {"call_ids": ["123", "456"]}
        assert complex == {}

        # Test with call_ids as string
        filters = {
            "call_ids": "123",
        }
        direct, complex = QueryBuilder.separate_filters(filters)
        assert direct == {"call_ids": ["123"]}
        assert complex == {}

        # Test with empty filters
        direct, complex = QueryBuilder.separate_filters({})
        assert direct == {}
        assert complex == {}

    def test_prepare_query_params(self):
        """Test preparing query parameters."""
        params = {
            "entity_name": "test_entity",
            "project_name": "test_project",
            "filters": {
                "trace_roots_only": True,
                "op_name": "test_op",
                "status": "success",
            },
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
        assert "query" in result  # Should have a query for the status filter
        assert result["sort_by"] == [{"field": "started_at", "direction": "desc"}]
        assert result["limit"] == 10
        assert result["include_costs"] is True
        assert result["include_feedback"] is True
        # Status is a synthetic field, so the API columns should include both "summary"
        # (to get the data) and the original columns
        assert set(result["columns"]) == {"id", "op_name", "summary"}
        assert "_synthetic_fields" in result
        assert "status" in result["_synthetic_fields"]

        # Test with QueryParams object
        query_params = QueryParams(
            entity_name="test_entity",
            project_name="test_project",
            filters=QueryFilter(
                trace_roots_only=True,
                op_name="test_op",
                status="success",
            ),
            sort_by="started_at",
            sort_direction="desc",
            limit=10,
            offset=0,
            include_costs=True,
            include_feedback=True,
            columns=["id", "op_name", "status"],
        )
        result = QueryBuilder.prepare_query_params(query_params)
        assert result["project_id"] == "test_entity/test_project"
        # Check the same _synthetic_fields handling
        assert set(result["columns"]) == {"id", "op_name", "summary"}
        assert "_synthetic_fields" in result
        assert "status" in result["_synthetic_fields"]

        # Test with cost field sorting (which should not be sent to server)
        params = {
            "entity_name": "test_entity",
            "project_name": "test_project",
            "sort_by": "total_cost",
            "sort_direction": "desc",
        }
        result = QueryBuilder.prepare_query_params(params)
        assert "sort_by" not in result

    def test_create_contains_operation(self):
        """Test creating a contains operation."""
        operation = QueryBuilder.create_contains_operation("op_name", "test")
        operation_dict = operation.model_dump(by_alias=True)
        assert operation_dict["$contains"]["substr"]["$literal"] == "test"
        assert operation_dict["$contains"]["input"]["$getField"] == "op_name"
        assert operation_dict["$contains"]["case_insensitive"] is True

        # Test with case sensitivity
        operation = QueryBuilder.create_contains_operation(
            "op_name", "test", case_insensitive=False
        )
        operation_dict = operation.model_dump(by_alias=True)
        assert operation_dict["$contains"]["case_insensitive"] is False

    def test_create_comparison_operation(self):
        """Test creating comparison operations."""
        # Test equals
        eq_op = QueryBuilder.create_comparison_operation(
            "field", FilterOperator.EQUALS, "value"
        )
        eq_dict = eq_op.model_dump(by_alias=True)
        assert eq_dict["$eq"][0]["$getField"] == "field"
        assert eq_dict["$eq"][1]["$literal"] == "value"

        # Test greater than
        gt_op = QueryBuilder.create_comparison_operation(
            "field", FilterOperator.GREATER_THAN, 100
        )
        gt_dict = gt_op.model_dump(by_alias=True)
        assert gt_dict["$gt"][0]["$getField"] == "field"
        assert gt_dict["$gt"][1]["$literal"] == 100

        # Test less than (implemented as not(gte))
        lt_op = QueryBuilder.create_comparison_operation(
            "field", FilterOperator.LESS_THAN, 100
        )
        lt_dict = lt_op.model_dump(by_alias=True)
        assert lt_dict["$not"][0]["$gte"][0]["$getField"] == "field"
        assert lt_dict["$not"][0]["$gte"][1]["$literal"] == 100

        # Test with attribute field (should use conversion)
        attr_op = QueryBuilder.create_comparison_operation(
            "attributes.count", FilterOperator.GREATER_THAN, 100
        )
        attr_dict = attr_op.model_dump(by_alias=True)
        assert "$convert" in attr_dict["$gt"][0]
        assert attr_dict["$gt"][0]["$convert"]["to"] == "double"

        # Test with invalid operator
        invalid_op = QueryBuilder.create_comparison_operation(
            "field", "invalid_operator", "value"
        )
        assert invalid_op is None


class TestWeaveApiClient(unittest.TestCase):
    """Tests for the WeaveApiClient class."""

    def setUp(self):
        """Set up test environment."""
        self.api_key = "fake_api_key"
        self.client = WeaveApiClient(api_key=self.api_key)

    def test_init_with_defaults(self):
        """Test initializing client with defaults."""
        with patch.dict(os.environ, {"WANDB_API_KEY": "env_api_key"}):
            client = WeaveApiClient()
            assert client.api_key == "env_api_key"
            assert client.server_url == "https://trace.wandb.ai"
            assert client.retries == 3
            assert client.timeout == 10

    def test_init_with_missing_api_key(self):
        """Test initializing client with missing API key."""
        with patch.dict(os.environ, {"WANDB_API_KEY": ""}):
            with pytest.raises(ValueError, match="API key not found"):
                WeaveApiClient()

    def test_get_auth_headers(self):
        """Test getting authentication headers."""
        headers = self.client._get_auth_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/jsonl"
        assert "Basic " in headers["Authorization"]

    @patch("requests.Session.post")
    def test_query_traces_success(self, mock_post):
        """Test successful querying of traces."""
        # Set up mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"id": "1", "op_name": "test"}',
            b'{"id": "2", "op_name": "test2"}',
        ]
        mock_post.return_value = mock_response

        # Call the client
        query_params = {"project_id": "entity/project"}
        results = list(self.client.query_traces(query_params))

        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[1]["id"] == "2"

        # Verify request
        mock_post.assert_called_once()
        # Analyze the args directly since we know the structure
        args, kwargs = mock_post.call_args

        # The first positional arg should be the URL
        expected_url = f"{self.client.server_url}/calls/stream_query"
        assert mock_post.call_args[0][0] == expected_url

        # Check the other expected keyword arguments
        assert "headers" in kwargs
        assert kwargs["headers"]["Authorization"].startswith("Basic ")
        assert kwargs["data"] == json.dumps(query_params)
        assert kwargs["timeout"] == 10
        assert kwargs["stream"] is True

    @patch("requests.Session.post")
    def test_query_traces_error_response(self, mock_post):
        """Test error response from Weave API."""
        # Set up mock response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        # Call the client and expect exception
        with pytest.raises(Exception, match="Error 400: Bad request"):
            list(self.client.query_traces({"project_id": "entity/project"}))

    @patch("requests.Session.post")
    def test_query_traces_network_error(self, mock_post):
        """Test network error during request."""
        # Set up mock to raise exception
        mock_post.side_effect = requests.RequestException("Network error")

        # Call the client and expect exception
        with pytest.raises(
            Exception, match="Failed to query Weave traces due to network error"
        ):
            list(self.client.query_traces({"project_id": "entity/project"}))

    @patch("requests.Session.post")
    def test_query_traces_json_error(self, mock_post):
        """Test JSON parsing error."""
        # Set up mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"id": "1", "op_name": "test"}',
            b"invalid_json",
        ]
        mock_post.return_value = mock_response

        # Call the client and expect exception
        with pytest.raises(Exception, match="Failed to parse Weave API response"):
            list(self.client.query_traces({"project_id": "entity/project"}))


class TestTraceService(unittest.TestCase):
    """Tests for the TraceService class."""

    def setUp(self):
        """Set up test environment."""
        self.service = TraceService()

    @patch("wandb_mcp_server.weave_api.client.WeaveApiClient.query_traces")
    def test_query_traces(self, mock_query_traces):
        """Test querying traces."""
        now = datetime.now()
        # Set up mock response
        mock_traces = [
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
        ]
        mock_query_traces.return_value = mock_traces

        # Test basic query
        result = self.service.query_traces(
            entity_name="test_entity",
            project_name="test_project",
            filters={"status": "success"},
            limit=10,
        )

        # Verify result
        assert isinstance(result, QueryResult)
        assert result.metadata.total_traces == 1
        assert len(result.traces) == 1

        # Verify request construction
        mock_query_traces.assert_called_once()
        call_args = mock_query_traces.call_args[0][0]
        assert call_args["project_id"] == "test_entity/test_project"
        assert "query" in call_args  # Should have a query for the status filter

    @patch("wandb_mcp_server.weave_api.client.WeaveApiClient.query_traces")
    def test_query_traces_with_cost_sorting(self, mock_query_traces):
        """Test querying traces with cost-based sorting."""
        now = datetime.now()
        # Set up mock response
        mock_traces = [
            {
                "id": "trace1",
                "project_id": "entity/project",
                "op_name": "test_op",
                "trace_id": "trace1",
                "started_at": now.isoformat(),
                "costs": {"model1": {"total_cost": 2.0}},
            },
            {
                "id": "trace2",
                "project_id": "entity/project",
                "op_name": "test_op",
                "trace_id": "trace2",
                "started_at": now.isoformat(),
                "costs": {"model1": {"total_cost": 1.0}},
            },
        ]
        mock_query_traces.return_value = mock_traces

        # Test with cost sorting
        result = self.service.query_traces(
            entity_name="test_entity",
            project_name="test_project",
            sort_by="total_cost",
            sort_direction="desc",
        )

        # Verify sorting (now the traces are WeaveTrace objects with dict access)
        assert result.traces[0].id == "trace1"  # Higher cost first
        assert result.traces[1].id == "trace2"

        # Verify no limit was sent to the server
        call_args = mock_query_traces.call_args[0][0]
        assert "limit" not in call_args

    @patch("wandb_mcp_server.weave_api.client.WeaveApiClient.query_traces")
    def test_query_traces_with_synthesized_fields(self, mock_query_traces):
        """Test querying traces with synthesized fields."""
        now = datetime.now()
        # Set up mock response
        mock_traces = [
            {
                "id": "trace1",
                "project_id": "entity/project",
                "op_name": "test_op",
                "trace_id": "trace1",
                "started_at": now.isoformat(),
                "summary": {"weave": {"status": "success", "latency_ms": 500}},
            },
        ]
        mock_query_traces.return_value = mock_traces

        # Test with synthesized fields
        result = self.service.query_traces(
            entity_name="test_entity",
            project_name="test_project",
            columns=["id", "status", "latency_ms"],
        )

        # Verify synthesized fields (now the traces are WeaveTrace objects)
        assert result.traces[0].status == "success"
        assert result.traces[0].latency_ms == 500

    @patch("wandb_mcp_server.weave_api.service.TraceService.query_traces")
    def test_query_paginated_traces(self, mock_query_traces):
        """Test paginated querying of traces."""
        now = datetime.now()
        # Create complete trace objects for the mocks
        trace1 = WeaveTrace(
            id="1",
            project_id="entity/project",
            op_name="test_op1",
            trace_id="trace1",
            started_at=now,
        )
        trace2 = WeaveTrace(
            id="2",
            project_id="entity/project",
            op_name="test_op2",
            trace_id="trace2",
            started_at=now,
        )
        trace3 = WeaveTrace(
            id="3",
            project_id="entity/project",
            op_name="test_op3",
            trace_id="trace3",
            started_at=now,
        )

        # Set up mock to return different results for each call
        mock_query_traces.side_effect = [
            QueryResult(
                metadata=TraceMetadata(total_traces=2),
                traces=[trace1, trace2],
            ),
            QueryResult(metadata=TraceMetadata(total_traces=1), traces=[trace3]),
            QueryResult(metadata=TraceMetadata(total_traces=0), traces=[]),
        ]

        # Test paginated query
        result = self.service.query_paginated_traces(
            entity_name="test_entity",
            project_name="test_project",
            chunk_size=2,
            target_limit=5,
        )

        # Verify all traces were collected
        assert len(result.traces) == 3
        assert [t.id for t in result.traces] == ["1", "2", "3"]

        # The implementation should make 2 calls - until it gets an empty result or reaches the target limit
        # 1st call: returns ["1", "2"]
        # 2nd call: returns ["3"]
        # No need for 3rd call as we already have 3 results, which is less than target_limit=5
        assert mock_query_traces.call_count == 2

    @patch("wandb_mcp_server.weave_api.client.WeaveApiClient.query_traces")
    def test_query_for_cost_sorting(self, mock_query_traces):
        """Test two-stage cost-based sorting."""
        now = datetime.now()
        # Set up mock to return different results for each call
        mock_query_traces.side_effect = [
            # First pass returns costs
            [
                {
                    "id": "1",
                    "project_id": "entity/project",
                    "op_name": "test_op",
                    "trace_id": "trace1",
                    "started_at": now.isoformat(),
                    "costs": {"model": {"total_cost": 3.0}},
                },
                {
                    "id": "2",
                    "project_id": "entity/project",
                    "op_name": "test_op",
                    "trace_id": "trace2",
                    "started_at": now.isoformat(),
                    "costs": {"model": {"total_cost": 1.0}},
                },
                {
                    "id": "3",
                    "project_id": "entity/project",
                    "op_name": "test_op",
                    "trace_id": "trace3",
                    "started_at": now.isoformat(),
                    "costs": {"model": {"total_cost": 2.0}},
                },
            ],
            # Second pass returns details - needs to return only the requested IDs
            [
                {
                    "id": "1",
                    "project_id": "entity/project",
                    "op_name": "test_op",
                    "trace_id": "trace1",
                    "started_at": now.isoformat(),
                    "data": "details1",
                },
                {
                    "id": "3",
                    "project_id": "entity/project",
                    "op_name": "test_op",
                    "trace_id": "trace3",
                    "started_at": now.isoformat(),
                    "data": "details3",
                },
            ],
        ]

        # Call the method
        result = self.service._query_for_cost_sorting(
            entity_name="test_entity",
            project_name="test_project",
            sort_by="total_cost",
            sort_direction="desc",
            target_limit=2,  # Only get top 2
        )

        # Verify sorted results
        assert len(result) == 2
        assert result[0]["id"] == "1"  # Highest cost
        assert result[1]["id"] == "3"  # Second highest
        assert "data" in result[0]  # Contains details

        # Verify second call used call_ids
        second_call_args = mock_query_traces.call_args_list[1][0][0]
        assert "call_ids" in second_call_args["filter"]
        assert set(second_call_args["filter"]["call_ids"]) == {"1", "3"}


class TestIntegration:
    """Integration tests for the Weave API components."""

    @patch("wandb_mcp_server.weave_api.client.WeaveApiClient.query_traces")
    def test_complete_flow(self, mock_query_traces):
        """Test the complete flow from service to client to processor."""
        now = datetime.now()
        # Set up mock response
        mock_traces = [
            {
                "id": "trace1",
                "project_id": "entity/project",
                "op_name": "weave:///entity/project/op/test:123",
                "trace_id": "trace1",
                "started_at": now.isoformat(),
                "inputs": {"text": "a" * 300},
                "output": "b" * 300,
                "summary": {"weave": {"status": "success", "latency_ms": 500}},
                "costs": {
                    "model1": {
                        "total_cost": 1.0,
                        "prompt_tokens_total_cost": 0.3,
                        "completion_tokens_total_cost": 0.7,
                    }
                },
            },
        ]
        mock_query_traces.return_value = mock_traces

        # Create service and query
        service = TraceService()
        result = service.query_traces(
            entity_name="test_entity",
            project_name="test_project",
            filters={"op_name_contains": "test", "trace_roots_only": True},
            columns=["id", "op_name", "status", "latency_ms", "costs"],
            truncate_length=50,
        )

        # Verify request construction
        call_args = mock_query_traces.call_args[0][0]
        assert call_args["project_id"] == "test_entity/test_project"
        assert "filter" in call_args and "trace_roots_only" in call_args["filter"]
        assert "query" in call_args  # Should have a query for op_name_contains

        # Verify result processing
        assert isinstance(result, QueryResult)
        assert result.metadata.total_traces == 1
        assert len(result.traces) == 1

        # Verify synthesized fields (now the traces are WeaveTrace objects)
        assert result.traces[0].status == "success"
        assert result.traces[0].latency_ms == 500

        # Verify truncation (now the traces are WeaveTrace objects)
        assert len(result.traces[0].inputs["text"]) == 53  # 50 + "..."


if __name__ == "__main__":
    unittest.main()
