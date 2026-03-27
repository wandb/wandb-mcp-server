"""Tests for log_analysis_to_wandb tool."""

from unittest.mock import MagicMock, patch

import pytest


class TestLogAnalysis:
    """Test the log_analysis function."""

    @patch("wandb_mcp_server.mcp_tools.log_analysis.wandb")
    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_basic_table_logging(self, mock_api_manager, mock_wandb):
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = "test-key"
        mock_api_manager.get_api.return_value = MagicMock(viewer={"username": "test"})

        mock_run = MagicMock()
        mock_run.id = "abc123"
        mock_run.url = "https://wandb.ai/test/proj/runs/abc123"
        mock_wandb.init.return_value = mock_run

        result = log_analysis(
            entity_name="test",
            project_name="proj",
            analysis_name="test-analysis",
            data=[
                {"latency_ms": 100, "status": "success"},
                {"latency_ms": 200, "status": "error"},
            ],
        )

        assert result["run_id"] == "abc123"
        assert result["row_count"] == 2
        assert "latency_ms" in result["table_columns"]
        mock_wandb.log.assert_called_once()
        mock_run.finish.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.log_analysis.wandb")
    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_with_charts(self, mock_api_manager, mock_wandb):
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = "test-key"
        mock_api_manager.get_api.return_value = MagicMock(viewer={"username": "test"})

        mock_run = MagicMock()
        mock_run.id = "abc123"
        mock_run.url = "https://wandb.ai/test/proj/runs/abc123"
        mock_wandb.init.return_value = mock_run
        mock_wandb.plot.histogram.return_value = MagicMock()

        result = log_analysis(
            entity_name="test",
            project_name="proj",
            analysis_name="test-analysis",
            data=[{"latency_ms": 100}, {"latency_ms": 200}],
            charts=[{"type": "histogram", "column": "latency_ms", "title": "Latency"}],
        )

        assert "chart_0" in result["logged_keys"]
        mock_wandb.plot.histogram.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.log_analysis.wandb")
    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_with_scalars(self, mock_api_manager, mock_wandb):
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = "test-key"
        mock_api_manager.get_api.return_value = MagicMock(viewer={"username": "test"})

        mock_run = MagicMock()
        mock_run.id = "abc123"
        mock_run.url = "https://wandb.ai/test/proj/runs/abc123"
        mock_wandb.init.return_value = mock_run

        result = log_analysis(
            entity_name="test",
            project_name="proj",
            analysis_name="test-analysis",
            data=[{"val": 1}],
            scalars={"p50": 1.2, "p95": 4.5},
        )

        assert result["run_id"] == "abc123"
        log_call = mock_wandb.log.call_args[0][0]
        assert "p50" in log_call
        assert "p95" in log_call

    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_no_api_key_raises(self, mock_api_manager):
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = None

        with pytest.raises(Exception, match="No W&B API key"):
            log_analysis("e", "p", "name", [{"a": 1}])

    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_empty_data_raises(self, mock_api_manager):
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = "key"

        with pytest.raises(ValueError, match="non-empty"):
            log_analysis("e", "p", "name", [])

    @patch("wandb_mcp_server.mcp_tools.log_analysis.wandb")
    @patch("wandb_mcp_server.mcp_tools.log_analysis.WandBApiManager")
    def test_run_always_finished(self, mock_api_manager, mock_wandb):
        """Run.finish() should be called even if wandb.log raises."""
        from wandb_mcp_server.mcp_tools.log_analysis import log_analysis

        mock_api_manager.get_api_key.return_value = "test-key"
        mock_api_manager.get_api.return_value = MagicMock(viewer={"username": "test"})

        mock_run = MagicMock()
        mock_run.id = "abc"
        mock_run.url = "url"
        mock_wandb.init.return_value = mock_run
        mock_wandb.log.side_effect = RuntimeError("log failed")

        with pytest.raises(RuntimeError):
            log_analysis("e", "p", "name", [{"a": 1}])

        mock_run.finish.assert_called_once()
