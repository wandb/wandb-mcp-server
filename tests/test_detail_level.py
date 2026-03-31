"""Tests for the detail_level parameter on query_weave_traces_tool."""

from unittest.mock import AsyncMock, patch

import pytest

from wandb_mcp_server.mcp_tools.query_weave import QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.trace_utils import process_traces


class TestDetailLevelInDescription:
    def test_detail_level_documented(self):
        assert "detail_level" in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_schema_level_documented(self):
        assert '"schema"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_summary_level_documented(self):
        assert '"summary"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_full_level_documented(self):
        assert '"full"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_when_to_use_mentions_detail_level(self):
        start = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION.index("<when_to_use>")
        end = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION.index("</when_to_use>")
        when_to_use = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION[start:end]
        assert "detail_level" in when_to_use


class TestProcessTracesDetailLevel:
    """Tests for the standalone process_traces function in trace_utils.py."""

    def _make_traces(self):
        return [
            {
                "id": "t1",
                "trace_id": "tr1",
                "op_name": "weave:///entity/proj/op/my_op:abc",
                "started_at": "2026-03-06T12:00:00Z",
                "ended_at": "2026-03-06T12:01:00Z",
                "status": "success",
                "parent_id": None,
                "display_name": "my_op",
                "inputs": {"text": "hello world " * 100},
                "output": {"result": "response " * 100},
                "summary": {"weave": {"status": "success", "latency_ms": 500}},
            }
        ]

    def test_schema_level_strips_data(self):
        result = process_traces(self._make_traces(), detail_level="schema")
        traces = result["traces"]
        assert len(traces) == 1
        trace = traces[0]
        assert "op_name" in trace
        assert "trace_id" in trace
        assert "status" in trace
        assert "inputs" not in trace
        assert "output" not in trace

    def test_summary_level_truncates(self):
        result = process_traces(self._make_traces(), detail_level="summary", truncate_length=50)
        traces = result["traces"]
        assert len(traces) == 1
        trace = traces[0]
        assert "op_name" in trace

    def test_full_level_no_truncation(self):
        result = process_traces(self._make_traces(), detail_level="full")
        traces = result["traces"]
        assert len(traces) == 1
        trace = traces[0]
        if "inputs" in trace and isinstance(trace["inputs"], dict):
            text_val = trace["inputs"].get("text", "")
            assert len(text_val) > 200

    def test_schema_fields_only_structural(self):
        result = process_traces(self._make_traces(), detail_level="schema")
        trace = result["traces"][0]
        allowed = {"id", "trace_id", "op_name", "started_at", "ended_at", "status", "parent_id", "display_name"}
        for key in trace:
            assert key in allowed, f"Unexpected key '{key}' in schema-level trace"

    def test_process_traces_handles_pydantic_models(self):
        """process_traces should handle Pydantic models via model_dump() normalization."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.model_dump.return_value = {
            "id": "t1",
            "trace_id": "tr1",
            "op_name": "weave:///entity/proj/op/test:abc",
            "started_at": "2026-03-06T12:00:00Z",
            "ended_at": "2026-03-06T12:01:00Z",
            "status": "success",
            "parent_id": None,
            "display_name": "test",
            "inputs": {"text": "hello"},
            "output": {"result": "world"},
        }

        result = process_traces([mock_model], detail_level="schema")
        mock_model.model_dump.assert_called_once()
        trace = result["traces"][0]
        assert trace["id"] == "t1"
        assert "inputs" not in trace


class TestSchemaProjection:
    """Verify detail_level='schema' passes server-side column projection."""

    @pytest.mark.asyncio
    @patch("wandb_mcp_server.server.count_traces", return_value=5)
    @patch("wandb_mcp_server.server.query_paginated_weave_traces", new_callable=AsyncMock)
    async def test_schema_level_passes_columns_to_query(self, mock_query, mock_count):
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata

        mock_query.return_value = QueryResult(
            metadata=TraceMetadata(total_traces=0),
            traces=[],
        )
        from wandb_mcp_server.server import register_tools
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for name, fn in mcp._tool_manager._tools.items():
            if "query_weave_traces" in name:
                tool_fn = fn
                break
        assert tool_fn is not None

        with patch("wandb_mcp_server.api_client.WandBApiManager.get_api_key", return_value="fake_key"):
            await tool_fn.fn(
                entity_name="ent",
                project_name="proj",
                detail_level="schema",
            )

        call_kwargs = mock_query.call_args[1]
        expected_cols = ["id", "trace_id", "op_name", "started_at", "ended_at", "display_name", "parent_id", "summary"]
        assert call_kwargs["columns"] == expected_cols
        assert call_kwargs["include_costs"] is False
        assert call_kwargs["include_feedback"] is False

    @pytest.mark.asyncio
    @patch("wandb_mcp_server.server.count_traces", return_value=5)
    @patch("wandb_mcp_server.server.query_paginated_weave_traces", new_callable=AsyncMock)
    async def test_schema_level_user_columns_not_overridden(self, mock_query, mock_count):
        """If user passes explicit columns with detail_level='schema', user columns win."""
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata

        mock_query.return_value = QueryResult(
            metadata=TraceMetadata(total_traces=0),
            traces=[],
        )
        from wandb_mcp_server.server import register_tools
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test")
        register_tools(mcp)

        tool_fn = None
        for name, fn in mcp._tool_manager._tools.items():
            if "query_weave_traces" in name:
                tool_fn = fn
                break

        user_columns = ["id", "op_name", "inputs"]
        with patch("wandb_mcp_server.api_client.WandBApiManager.get_api_key", return_value="fake_key"):
            await tool_fn.fn(
                entity_name="ent",
                project_name="proj",
                columns=user_columns,
                detail_level="schema",
            )

        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["columns"] == user_columns
