"""Tests for report creation improvements: SVG, new panel types, analysis_run_id."""

from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.create_report import _build_panel_blocks


class TestSVGSupport:
    """Test SVG/HTML support in plots_html."""

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_svg_string_creates_image_block(self, mock_api, mock_wr):
        from wandb_mcp_server.mcp_tools.create_report import create_report

        mock_api.get_api_key.return_value = "key"
        mock_api.get_api.return_value = MagicMock(viewer={"username": "test"})
        mock_wr.Report.return_value = MagicMock(url="https://wandb.ai/report")
        mock_wr.Image = MagicMock()
        mock_wr.P = MagicMock()

        svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'

        create_report(
            entity_name="e",
            project_name="p",
            title="Test",
            plots_html=svg,
        )

        mock_wr.Image.assert_called_once()
        call_args = mock_wr.Image.call_args
        assert call_args[1]["url"].startswith("data:image/svg+xml;base64,")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_html_string_creates_markdown_block(self, mock_api, mock_wr):
        from wandb_mcp_server.mcp_tools.create_report import create_report

        mock_api.get_api_key.return_value = "key"
        mock_api.get_api.return_value = MagicMock(viewer={"username": "test"})
        mock_wr.Report.return_value = MagicMock(url="https://wandb.ai/report")
        mock_wr.MarkdownBlock = MagicMock()
        mock_wr.P = MagicMock()

        create_report(
            entity_name="e",
            project_name="p",
            title="Test",
            plots_html="<div>Hello</div>",
        )

        mock_wr.MarkdownBlock.assert_called()


class TestNewPanelTypes:
    """Test scatter, markdown_table, and markdown_panel panel types."""

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_scatter_panel(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.ScatterPlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "scatter", "x": "latency_ms", "y": "token_count", "title": "Scatter"}]
        blocks = _build_panel_blocks(panels, "e", "p")

        assert len(blocks) == 1
        mock_wr.ScatterPlot.assert_called_once_with(x="latency_ms", y="token_count", title="Scatter")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_markdown_table_panel(self, mock_wr):
        mock_wr.MarkdownBlock = MagicMock()

        panels = [
            {
                "type": "markdown_table",
                "title": "Stats",
                "headers": ["Metric", "Value"],
                "rows": [["p50", "1.2s"], ["p95", "4.5s"]],
            }
        ]
        blocks = _build_panel_blocks(panels, "e", "p")

        assert len(blocks) == 1
        call_text = mock_wr.MarkdownBlock.call_args[0][0]
        assert "| Metric | Value |" in call_text
        assert "| p50 | 1.2s |" in call_text

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_markdown_panel_in_grid(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.MarkdownPanel = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "markdown_panel", "markdown": "## Summary\np50 = 1.2s"}]
        blocks = _build_panel_blocks(panels, "e", "p")

        assert len(blocks) == 1
        mock_wr.MarkdownPanel.assert_called_once_with(markdown="## Summary\np50 = 1.2s")


class TestAnalysisRunId:
    """Test that analysis_run_id scopes Runset to a specific run."""

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_analysis_run_id_creates_filtered_runset(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [
            {
                "type": "line",
                "x": "_step",
                "y": ["p50"],
                "title": "Latency",
                "analysis_run_id": "abc123",
            }
        ]
        blocks = _build_panel_blocks(panels, "e", "p")

        assert len(blocks) == 1
        runset_call = mock_wr.Runset.call_args
        assert runset_call[1]["query"] == "abc123"
        assert "filters" not in runset_call[1]

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_no_analysis_run_id_uses_default_runset(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"}]
        result = _build_panel_blocks(panels, "e", "p")

        assert len(result) == 1
        runset_call = mock_wr.Runset.call_args
        assert "filters" not in runset_call[1]


class TestRunComparisonFix:
    """Test that run_comparison now attempts to filter by run_ids."""

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_run_comparison_with_run_ids_creates_filter(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [
            {
                "type": "run_comparison",
                "metrics": ["loss"],
                "run_ids": ["r1", "r2"],
                "title": "Compare",
            }
        ]
        blocks = _build_panel_blocks(panels, "e", "p")

        assert len(blocks) == 1
        filtered_runset_call = mock_wr.Runset.call_args_list[-1]
        assert filtered_runset_call[1]["query"] == "r1 r2"
        assert "filters" not in filtered_runset_call[1]
