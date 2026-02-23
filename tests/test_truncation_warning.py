"""Unit tests for truncation_warning field propagation and trace_utils deletion (MCP-6, MCP-10)."""

import importlib

import pytest

from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata


class TestTruncationWarning:

    def test_defaults_to_none(self):
        assert TraceMetadata(total_traces=5).truncation_warning is None

    def test_can_be_set(self):
        m = TraceMetadata(total_traces=10, truncation_warning="Truncated.")
        assert m.truncation_warning == "Truncated."

    def test_serialized_in_query_result(self):
        m = TraceMetadata(total_traces=3, truncation_warning="Level 2.")
        r = QueryResult(metadata=m)
        assert r.model_dump()["metadata"]["truncation_warning"] == "Level 2."

    def test_none_when_absent(self):
        r = QueryResult(metadata=TraceMetadata(total_traces=1))
        assert r.model_dump()["metadata"]["truncation_warning"] is None


class TestTraceUtilsDeletion:

    def test_trace_utils_not_importable(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("wandb_mcp_server.trace_utils")
