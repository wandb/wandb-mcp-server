"""Tests for resolve_trace_roots: TraceService method and standalone MCP tool."""

import json
from unittest.mock import MagicMock, patch

from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata


def _make_root_trace(trace_id, op_name, display_name=None):
    return {
        "id": f"root-{trace_id}",
        "trace_id": trace_id,
        "op_name": op_name,
        "display_name": display_name,
        "started_at": "2026-01-01T00:00:00",
        "parent_id": None,
    }


def _make_query_result(traces):
    return QueryResult(
        traces=traces,
        metadata=TraceMetadata(trace_count=len(traces)),
    )


class TestResolveTraceRoots:
    """resolve_trace_roots batches trace_id -> root span in one query."""

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_returns_root_map(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        service.query_traces = MagicMock(
            return_value=_make_query_result(
                [
                    _make_root_trace("t1", "session_a", "Session A"),
                    _make_root_trace("t2", "session_b", "Session B"),
                ]
            )
        )

        result = service.resolve_trace_roots("entity", "project", ["t1", "t2"])

        assert "t1" in result
        assert "t2" in result
        assert result["t1"]["op_name"] == "session_a"
        assert result["t2"]["display_name"] == "Session B"

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_empty_trace_ids(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        result = service.resolve_trace_roots("entity", "project", [])
        assert result == {}

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_deduplicates_trace_ids(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        service.query_traces = MagicMock(
            return_value=_make_query_result(
                [
                    _make_root_trace("t1", "session"),
                ]
            )
        )

        result = service.resolve_trace_roots("entity", "project", ["t1", "t1", "t1"])

        assert len(result) == 1
        assert "t1" in result
        service.query_traces.assert_called_once()
        call_kwargs = service.query_traces.call_args[1]
        assert len(call_kwargs["filters"]["trace_ids"]) == 1

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_missing_root_omitted(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        service.query_traces = MagicMock(
            return_value=_make_query_result(
                [
                    _make_root_trace("t1", "session"),
                ]
            )
        )

        result = service.resolve_trace_roots("entity", "project", ["t1", "t2", "t3"])

        assert "t1" in result
        assert "t2" not in result
        assert "t3" not in result

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_passes_trace_roots_only(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        service.query_traces = MagicMock(return_value=_make_query_result([]))

        service.resolve_trace_roots("entity", "project", ["t1"])

        call_kwargs = service.query_traces.call_args[1]
        assert call_kwargs["filters"]["trace_roots_only"] is True
        assert call_kwargs["filters"]["trace_ids"] == ["t1"]

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_query_params(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        service.query_traces = MagicMock(return_value=_make_query_result([]))

        service.resolve_trace_roots("entity", "project", ["t1"])

        call_kwargs = service.query_traces.call_args[1]
        assert call_kwargs["metadata_only"] is False
        assert call_kwargs["return_full_data"] is True
        assert "op_name" in call_kwargs["columns"]
        assert "trace_id" in call_kwargs["columns"]
        assert "display_name" in call_kwargs["columns"]
        assert "parent_id" in call_kwargs["columns"]

    @patch("wandb_mcp_server.weave_api.service.WeaveApiClient")
    def test_large_batch(self, mock_client_cls):
        from wandb_mcp_server.weave_api.service import TraceService

        service = TraceService(api_key="test", server_url="http://test")
        trace_ids = [f"t{i}" for i in range(200)]
        roots = [_make_root_trace(f"t{i}", f"session_{i}") for i in range(200)]
        service.query_traces = MagicMock(return_value=_make_query_result(roots))

        result = service.resolve_trace_roots("entity", "project", trace_ids)

        assert len(result) == 200
        call_kwargs = service.query_traces.call_args[1]
        assert call_kwargs["limit"] == 200


class TestResolveTraceRootsTool:
    """Tests for the standalone resolve_trace_roots MCP tool wrapper."""

    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.get_trace_service")
    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.WandBApiManager")
    def test_tool_returns_json_with_roots(self, mock_mgr, mock_get_svc):
        from wandb_mcp_server.mcp_tools.resolve_trace_roots import resolve_trace_roots

        mock_api = MagicMock()
        mock_api.viewer = {"username": "testuser"}
        mock_mgr.get_api.return_value = mock_api

        mock_svc = MagicMock()
        mock_svc.resolve_trace_roots.return_value = {
            "t1": _make_root_trace("t1", "chat_session", "Chat Session"),
            "t2": _make_root_trace("t2", "pipeline", "Pipeline Run"),
        }
        mock_get_svc.return_value = mock_svc

        result = json.loads(resolve_trace_roots("entity", "project", ["t1", "t2"]))

        assert result["resolved"] == 2
        assert result["total_requested"] == 2
        assert result["roots"]["t1"]["op_name"] == "chat_session"
        assert result["roots"]["t2"]["display_name"] == "Pipeline Run"

    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.get_trace_service")
    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.WandBApiManager")
    def test_tool_empty_trace_ids(self, mock_mgr, mock_get_svc):
        from wandb_mcp_server.mcp_tools.resolve_trace_roots import resolve_trace_roots

        mock_api = MagicMock()
        mock_api.viewer = {"username": "testuser"}
        mock_mgr.get_api.return_value = mock_api

        result = json.loads(resolve_trace_roots("entity", "project", []))

        assert result["roots"] == {}
        assert result["resolved"] == 0
        assert result["total_requested"] == 0
        mock_get_svc.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.get_trace_service")
    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.WandBApiManager")
    def test_tool_handles_service_error(self, mock_mgr, mock_get_svc):
        from wandb_mcp_server.mcp_tools.resolve_trace_roots import resolve_trace_roots

        mock_api = MagicMock()
        mock_api.viewer = {"username": "testuser"}
        mock_mgr.get_api.return_value = mock_api

        mock_svc = MagicMock()
        mock_svc.resolve_trace_roots.side_effect = ValueError("connection failed")
        mock_get_svc.return_value = mock_svc

        result = json.loads(resolve_trace_roots("entity", "project", ["t1"]))

        assert result["error"] == "resolve_failed"
        assert "connection failed" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.get_trace_service")
    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.WandBApiManager")
    def test_tool_deduplicates_in_response(self, mock_mgr, mock_get_svc):
        from wandb_mcp_server.mcp_tools.resolve_trace_roots import resolve_trace_roots

        mock_api = MagicMock()
        mock_api.viewer = {"username": "testuser"}
        mock_mgr.get_api.return_value = mock_api

        mock_svc = MagicMock()
        mock_svc.resolve_trace_roots.return_value = {
            "t1": _make_root_trace("t1", "session"),
        }
        mock_get_svc.return_value = mock_svc

        result = json.loads(resolve_trace_roots("entity", "project", ["t1", "t1", "t1"]))

        assert result["resolved"] == 1
        assert result["total_requested"] == 1

    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.get_trace_service")
    @patch("wandb_mcp_server.mcp_tools.resolve_trace_roots.WandBApiManager")
    def test_tool_response_shape(self, mock_mgr, mock_get_svc):
        """Verify the tool output contains only the expected root span fields."""
        from wandb_mcp_server.mcp_tools.resolve_trace_roots import resolve_trace_roots

        mock_api = MagicMock()
        mock_api.viewer = {"username": "testuser"}
        mock_mgr.get_api.return_value = mock_api

        full_root = _make_root_trace("t1", "chat", "Chat")
        full_root["extra_field"] = "should_not_appear"
        mock_svc = MagicMock()
        mock_svc.resolve_trace_roots.return_value = {"t1": full_root}
        mock_get_svc.return_value = mock_svc

        result = json.loads(resolve_trace_roots("entity", "project", ["t1"]))

        root_fields = set(result["roots"]["t1"].keys())
        assert root_fields == {"id", "trace_id", "op_name", "display_name", "started_at"}
