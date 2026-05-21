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
        assert mock_wr.Runset.call_count == 2
        mock_wr.LinePlot.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_analysis_run_id_uses_query_not_filters(self, mock_wr):
        """Runset for analysis_run_id must use query= to avoid ast.Dict crash."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.BarPlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "bar", "metrics": ["p50"], "title": "Latency", "analysis_run_id": "abc123"}]
        _build_panel_blocks(panels, "entity", "project")

        runset_call = mock_wr.Runset.call_args
        assert runset_call[1].get("query") == "abc123"
        assert "filters" not in runset_call[1]

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_run_comparison_uses_query_not_filters(self, mock_wr):
        """run_comparison with run_ids must use query= to avoid ast.Dict crash."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "run_comparison", "metrics": ["loss"], "run_ids": ["r1", "r2"], "title": "Compare"}]
        _build_panel_blocks(panels, "entity", "project")

        comp_call = mock_wr.Runset.call_args_list[1]
        assert comp_call[1].get("query") == "r1 r2"
        assert "filters" not in comp_call[1]

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

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_combined_panel_types_all_produce_blocks(self, mock_wr):
        """E2E multi-tool scenario: markdown_table + analysis_run_id + line in one call."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.BarPlot = MagicMock()
        mock_wr.MarkdownBlock = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [
            {"type": "markdown_table", "headers": ["Op", "Count"], "rows": [["predict", "42"]], "title": "Ops"},
            {"type": "bar", "metrics": ["p50_ms"], "title": "Latency", "analysis_run_id": "abc123"},
            {"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"},
        ]
        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 3
        mock_wr.MarkdownBlock.assert_called_once()
        assert mock_wr.PanelGrid.call_count == 2

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_custom_chart_panel(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.CustomChart = MagicMock(return_value="CustomChart")
        mock_wr.Runset = MagicMock(return_value="Runset")

        panels = [
            {
                "type": "custom_chart",
                "title": "PR Curve",
                "query": {"summaryTable": {"tableKey": "pr_curve_table"}},
                "chart_name": "wandb/line/v0",
                "chart_fields": {"x": "recall", "y": "precision"},
                "chart_strings": {"title": "PR Curve"},
                "run_ids": ["abc123"],
                "hide_run_sets": True,
            }
        ]

        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        mock_wr.Runset.assert_called_once_with(entity="entity", project="project", query="abc123")
        mock_wr.CustomChart.assert_called_once_with(
            query={"summaryTable": {"tableKey": "pr_curve_table"}},
            chart_name="wandb/line/v0",
            chart_fields={"x": "recall", "y": "precision"},
            chart_strings={"title": "PR Curve"},
        )
        mock_wr.PanelGrid.assert_called_once_with(runsets=["Runset"], hide_run_sets=True, panels=["CustomChart"])

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_custom_chart_table_panel(self, mock_wr):
        chart = MagicMock()
        mock_wr.PanelGrid = MagicMock()
        mock_wr.CustomChart.from_table = MagicMock(return_value=chart)
        mock_wr.Runset = MagicMock(return_value="Runset")

        panels = [
            {
                "type": "custom_chart_table",
                "title": "PR Curve",
                "table_name": "pr_curve_table",
                "chart_name": "wandb/line/v0",
                "chart_fields": {"x": "recall", "y": "precision"},
                "chart_strings": {"title": "PR Curve"},
            }
        ]

        blocks = _build_panel_blocks(panels, "entity", "project")

        assert len(blocks) == 1
        mock_wr.CustomChart.from_table.assert_called_once_with(
            "pr_curve_table",
            chart_fields={"x": "recall", "y": "precision"},
            chart_strings={"title": "PR Curve"},
        )
        assert chart.chart_name == "wandb/line/v0"
        mock_wr.PanelGrid.assert_called_once_with(runsets=["Runset"], hide_run_sets=False, panels=[chart])

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_custom_chart_invalid_spec_falls_back(self, mock_wr):
        mock_wr.P = MagicMock(return_value="fallback")
        panels = [{"type": "custom_chart", "title": "Broken", "chart_name": "wandb/line/v0"}]

        blocks = _build_panel_blocks(panels, "entity", "project")

        assert blocks == ["fallback"]
        mock_wr.P.assert_called_once_with("*Panel 'Broken' could not be rendered.*")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_mixed_native_markdown_and_custom_chart_panels(self, mock_wr):
        mock_wr.PanelGrid = MagicMock(side_effect=lambda **kw: f"grid:{kw['panels'][0]}")
        mock_wr.LinePlot = MagicMock(return_value="line")
        mock_wr.CustomChart = MagicMock(return_value="custom")
        mock_wr.MarkdownBlock = MagicMock(return_value="markdown")
        mock_wr.Runset = MagicMock(return_value="Runset")

        panels = [
            {"type": "markdown_table", "headers": ["Metric", "Value"], "rows": [["p50", "1s"]]},
            {"type": "line", "x": "_step", "y": ["loss"], "title": "Loss"},
            {
                "type": "custom_chart",
                "query": {"summaryTable": {"tableKey": "curve"}},
                "chart_name": "wandb/line/v0",
                "chart_fields": {"x": "recall", "y": "precision"},
            },
        ]

        blocks = _build_panel_blocks(panels, "entity", "project")

        assert blocks == ["markdown", "grid:line", "grid:custom"]

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_markdown_table_with_unicode_titles(self, mock_wr):
        """Report panels with unicode characters in titles should not crash."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.MarkdownBlock = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [
            {
                "type": "markdown_table",
                "headers": ["Metric", "Value"],
                "rows": [["p50\u2014latency", "2.8s"], ["accuracy & recall", "95%"]],
                "title": "Email Agent \u2014 Metrics & Stats",
            },
        ]
        blocks = _build_panel_blocks(panels, "entity", "project")
        assert len(blocks) == 1
        call_arg = mock_wr.MarkdownBlock.call_args[0][0]
        assert "Email Agent" in call_arg
        assert "p50" in call_arg


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

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_create_report_with_custom_chart_table_block_order(self, mock_api_mgr, mock_wr):
        """Custom chart table panels keep the report block ordering contract."""
        mock_api_mgr.get_api_key.return_value = "fake_key"
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")

        mock_report = MagicMock()
        mock_report.url = "https://wandb.ai/report/789"
        mock_wr.Report.return_value = mock_report
        mock_wr.P = MagicMock(side_effect=lambda text: f"P:{text}")
        mock_wr.H2 = MagicMock(side_effect=lambda text: f"H2:{text}")
        mock_wr.PanelGrid = MagicMock(side_effect=lambda **kw: "PanelGrid")
        mock_wr.CustomChart.from_table = MagicMock(return_value=MagicMock())
        mock_wr.Runset = MagicMock()

        result = create_report(
            "entity",
            "project",
            "Custom Chart Report",
            markdown_report_text="Hello world",
            panels=[
                {
                    "type": "custom_chart_table",
                    "table_name": "pr_curve_table",
                    "chart_name": "wandb/line/v0",
                    "chart_fields": {"x": "recall", "y": "precision"},
                }
            ],
        )

        assert result["url"] == "https://wandb.ai/report/789"
        blocks = mock_report.blocks
        assert "MCP Server" in str(blocks[0])
        assert "H2:Charts" in blocks
        assert blocks[-1] == "PanelGrid"
