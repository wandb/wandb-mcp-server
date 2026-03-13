"""Tests for v0.3.0 latency + score optimization changes."""

import json
from unittest.mock import patch

import pytest

from wandb_mcp_server.weave_api.models import TraceMetadata, QueryResult
from wandb_mcp_server.mcp_tools.query_weave import QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.count_traces import COUNT_WEAVE_TRACES_TOOL_DESCRIPTION


class TestTotalMatchingCountModel:
    """Verify TraceMetadata has total_matching_count field."""

    def test_default_value_is_zero(self):
        meta = TraceMetadata()
        assert meta.total_matching_count == 0

    def test_can_set_value(self):
        meta = TraceMetadata(total_matching_count=42)
        assert meta.total_matching_count == 42

    def test_serializes_in_json(self):
        meta = TraceMetadata(total_traces=10, total_matching_count=500)
        data = json.loads(meta.model_dump_json())
        assert data["total_matching_count"] == 500
        assert data["total_traces"] == 10

    def test_query_result_includes_field(self):
        result = QueryResult(
            metadata=TraceMetadata(total_traces=5, total_matching_count=100),
            traces=[],
        )
        data = json.loads(result.model_dump_json())
        assert data["metadata"]["total_matching_count"] == 100


class TestTotalMatchingCountInDescription:
    """Verify tool descriptions document total_matching_count."""

    def test_query_weave_mentions_total_matching_count(self):
        assert "total_matching_count" in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_query_weave_says_no_separate_count_needed(self):
        desc_lower = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION.lower()
        assert "do not need to call count_weave_traces_tool" in desc_lower

    def test_count_traces_no_longer_says_call_before(self):
        assert "Call before large" not in COUNT_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_count_traces_says_use_query_instead(self):
        assert "query_weave_traces_tool" in COUNT_WEAVE_TRACES_TOOL_DESCRIPTION


class TestTraceRootsOnlyFilterName:
    """Verify the filter typo is fixed in tool description examples."""

    def test_example_uses_correct_filter_name(self):
        assert "trace_roots_only" in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_example_does_not_use_wrong_filter_name_in_code(self):
        examples_start = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION.find("<examples>")
        if examples_start >= 0:
            examples_section = QUERY_WEAVE_TRACES_TOOL_DESCRIPTION[examples_start:]
            assert "root_traces_only" not in examples_section


class TestErrorFormatStandardization:
    """Verify tools return JSON errors, not plain strings."""

    @patch("wandb_mcp_server.mcp_tools.count_traces.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.count_traces.get_retry_session")
    def test_count_traces_error_returns_json(self, mock_session, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "test-key-1234567890123456789012345678"
        mock_session.return_value.post.side_effect = Exception("Connection refused")

        from wandb_mcp_server.mcp_tools.count_traces import count_traces

        with pytest.raises(Exception):
            count_traces(entity_name="test", project_name="test")
