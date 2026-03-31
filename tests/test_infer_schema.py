"""Tests for the infer_trace_schema_tool.

Uses spec=TraceService on mocks so that wrong keyword arguments
(like the target_limit bug) fail at test time instead of being
silently accepted by MagicMock.
"""

import json
from unittest.mock import MagicMock, patch

from wandb_mcp_server.mcp_tools.infer_schema import (
    INFER_TRACE_SCHEMA_TOOL_DESCRIPTION,
    _flatten_dict,
    _infer_type,
    infer_trace_schema,
)
from wandb_mcp_server.weave_api.service import TraceService


class TestFlattenDict:
    def test_flat_dict(self):
        d = {"a": 1, "b": "hello"}
        assert _flatten_dict(d) == {"a": 1, "b": "hello"}

    def test_nested_dict(self):
        d = {"a": {"b": {"c": 1}}}
        assert _flatten_dict(d) == {"a.b.c": 1}

    def test_deeply_nested_stops_at_large_dicts(self):
        large = {f"k{i}": i for i in range(15)}
        d = {"top": large}
        result = _flatten_dict(d)
        assert "top" in result
        assert isinstance(result["top"], dict)

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_non_dict_input(self):
        assert _flatten_dict("string") == {}
        assert _flatten_dict(42) == {}
        assert _flatten_dict(None) == {}


class TestInferType:
    def test_none(self):
        assert _infer_type(None) == "null"

    def test_bool(self):
        assert _infer_type(True) == "boolean"
        assert _infer_type(False) == "boolean"

    def test_int(self):
        assert _infer_type(42) == "int"

    def test_float(self):
        assert _infer_type(3.14) == "float"

    def test_string(self):
        assert _infer_type("hello") == "string"

    def test_datetime_string(self):
        assert _infer_type("2026-03-06T12:00:00Z") == "datetime"

    def test_datetime_with_timezone(self):
        assert _infer_type("2026-03-06T12:00:00+00:00") == "datetime"

    def test_long_string_with_hyphens_not_datetime(self):
        assert _infer_type("my-long-experiment-name-v2") == "string"

    def test_url_not_datetime(self):
        assert _infer_type("https://wandb.ai/entity/project") == "string"

    def test_list(self):
        assert _infer_type([1, 2, 3]) == "array"

    def test_dict(self):
        assert _infer_type({"a": 1}) == "object"


class TestInferTraceSchema:
    """Tests use spec=TraceService on the mock so wrong kwargs are caught."""

    def _make_mock_service(self, traces):
        mock_result = MagicMock()
        mock_result.traces = traces
        mock_service = MagicMock(spec=TraceService)
        mock_service.query_traces.return_value = mock_result
        return mock_service

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_basic_schema_inference(self, mock_count, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = [100, 25]

        mock_service = self._make_mock_service(
            [
                {"id": "t1", "op_name": "eval.run", "status": "success", "started_at": "2026-03-06T12:00:00Z"},
                {"id": "t2", "op_name": "eval.run", "status": "error", "started_at": "2026-03-06T13:00:00Z"},
            ]
        )

        with patch("wandb_mcp_server.mcp_tools.query_weave.get_trace_service", return_value=mock_service):
            result = json.loads(infer_trace_schema("entity", "project"))

        assert result["total_traces"] == 100
        assert result["root_traces"] == 25
        assert result["sample_size"] == 2

        field_paths = [f["path"] for f in result["fields"]]
        assert "id" in field_paths
        assert "op_name" in field_paths

        mock_service.query_traces.assert_called_once()
        call_kwargs = mock_service.query_traces.call_args[1]
        assert "limit" in call_kwargs, "Should use 'limit', not 'target_limit'"
        assert call_kwargs["limit"] == 20

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_empty_project(self, mock_count, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = [0, 0]

        mock_service = self._make_mock_service([])

        with patch("wandb_mcp_server.mcp_tools.query_weave.get_trace_service", return_value=mock_service):
            result = json.loads(infer_trace_schema("entity", "project"))

        assert result["total_traces"] == 0
        assert result["fields"] == []
        assert "note" in result

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_top_values_limited(self, mock_count, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = [50, 10]

        traces = [{"status": f"val{i % 7}"} for i in range(20)]
        mock_service = self._make_mock_service(traces)

        with patch("wandb_mcp_server.mcp_tools.query_weave.get_trace_service", return_value=mock_service):
            result = json.loads(infer_trace_schema("e", "p", top_n_values=3))

        status_field = next(f for f in result["fields"] if f["path"] == "status")
        assert len(status_field["top_values"]) <= 3

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_nested_fields_flattened(self, mock_count, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = [10, 5]

        mock_service = self._make_mock_service(
            [
                {"id": "t1", "summary": {"weave": {"status": "success", "latency_ms": 100}}},
            ]
        )

        with patch("wandb_mcp_server.mcp_tools.query_weave.get_trace_service", return_value=mock_service):
            result = json.loads(infer_trace_schema("e", "p"))

        field_paths = [f["path"] for f in result["fields"]]
        assert "summary.weave.status" in field_paths
        assert "summary.weave.latency_ms" in field_paths

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_query_failure_returns_error(self, mock_count, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = Exception("Network timeout")

        result = json.loads(infer_trace_schema("e", "p"))
        assert "error" in result

    @patch("wandb_mcp_server.mcp_tools.infer_schema.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.count_traces")
    def test_pydantic_trace_objects_handled(self, mock_count, mock_api_mgr):
        """Non-dict trace objects with model_dump() should be converted, not dropped."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_count.side_effect = [5, 2]

        pydantic_trace = MagicMock()
        pydantic_trace.model_dump.return_value = {"id": "t1", "op_name": "test", "status": "success"}
        mock_service = self._make_mock_service([pydantic_trace])

        with patch("wandb_mcp_server.mcp_tools.query_weave.get_trace_service", return_value=mock_service):
            result = json.loads(infer_trace_schema("e", "p"))

        field_paths = [f["path"] for f in result["fields"]]
        assert "id" in field_paths
        assert "op_name" in field_paths


class TestInferSchemaToolDescription:
    def test_has_when_to_use(self):
        assert "<when_to_use>" in INFER_TRACE_SCHEMA_TOOL_DESCRIPTION
        assert "</when_to_use>" in INFER_TRACE_SCHEMA_TOOL_DESCRIPTION
