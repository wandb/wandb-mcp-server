"""Tests for the panels parameter on create_wandb_report_tool."""

from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.create_report import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    _build_panel_blocks,
    create_report,
)


class TestCreateReportPanelsDescription:
    def test_panels_documented(self):
        assert "panels" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION

    def test_line_panel_documented(self):
        assert "line" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()

    def test_bar_panel_documented(self):
        assert "bar" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()

    def test_has_when_to_use(self):
        assert "<when_to_use>" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "</when_to_use>" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION


class TestBuildPanelBlocks:
    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_line_panel(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"}]
        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        mock_wr.LinePlot.assert_called_once_with(x="_step", y=["loss"], title="Loss")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_bar_panel(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.BarPlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "bar", "metrics": ["accuracy", "f1"], "title": "Metrics"}]
        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        mock_wr.BarPlot.assert_called_once_with(metrics=["accuracy", "f1"], title="Metrics")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_run_comparison_panel(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "run_comparison", "metrics": ["loss"], "run_ids": ["r1", "r2"], "title": "Compare"}]
        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        mock_wr.Runset.assert_called_once()
        mock_wr.LinePlot.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_empty_panels_list(self, mock_wr):
        blocks = _build_panel_blocks([], "entity", "project")
        assert blocks == []

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_unknown_panel_type(self, mock_wr):
        mock_wr.P = MagicMock()
        panels = [{"type": "unknown_chart", "title": "Bad"}]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert blocks == []

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_line_panel_empty_y_skipped(self, mock_wr):
        panels = [{"type": "line", "x": "_step", "y": [], "title": "Empty"}]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert blocks == []

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_bar_panel_empty_metrics_skipped(self, mock_wr):
        panels = [{"type": "bar", "metrics": [], "title": "Empty"}]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert blocks == []

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_panel_error_graceful(self, mock_wr):
        mock_wr.PanelGrid = MagicMock(side_effect=Exception("Panel construction failed"))
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()
        mock_wr.P = MagicMock(return_value="fallback")

        panels = [{"type": "line", "x": "_step", "y": ["loss"], "title": "Broken"}]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert len(blocks) == 1

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_multiple_panels(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.BarPlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [
            {"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"},
            {"type": "bar", "metrics": ["accuracy"], "title": "Accuracy"},
        ]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert len(blocks) == 2


class TestCreateReportWithPanels:
    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_panels_none_backward_compat(self, mock_api_mgr, mock_wr):
        """Report creation with panels=None should work same as before."""
        mock_api_mgr.get_api_key.return_value = "fake_key"
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")

        mock_report = MagicMock()
        mock_report.url = "https://wandb.ai/report/123"
        mock_wr.Report.return_value = mock_report
        mock_wr.P = MagicMock()
        mock_wr.H2 = MagicMock()

        result = create_report("entity", "project", "Test Report", panels=None)
        assert result["url"] == "https://wandb.ai/report/123"

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_run_comparison_without_run_ids(self, mock_wr):
        """run_comparison panel without run_ids should not set filters."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "run_comparison", "metrics": ["loss"], "title": "Compare"}]
        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        call_kwargs = mock_wr.Runset.call_args[1]
        assert "filters" not in call_kwargs

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_create_report_with_panels_and_markdown(self, mock_api_mgr, mock_wr):
        """Full flow: markdown content + panels should produce blocks in the
        correct order: security notice, content, H2 'Charts', panel grids."""
        mock_api_mgr.get_api_key.return_value = "fake_key"
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")

        mock_report = MagicMock()
        mock_report.url = "https://wandb.ai/report/456"
        mock_wr.Report.return_value = mock_report
        mock_wr.P = MagicMock(side_effect=lambda text: f"P:{text}")
        mock_wr.H2 = MagicMock(side_effect=lambda text: f"H2:{text}")
        mock_wr.PanelGrid = MagicMock(side_effect=lambda **kw: "PanelGrid")
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        result = create_report(
            "entity",
            "project",
            "Full Report",
            markdown_report_text="Hello world",
            panels=[{"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"}],
        )

        assert result["url"] == "https://wandb.ai/report/456"
        blocks = mock_report.blocks
        assert len(blocks) >= 3
        # First block is security notice
        assert "MCP Server" in str(blocks[0])
