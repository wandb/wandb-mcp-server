"""Tests for the panels parameter on create_wandb_report_tool."""

import importlib
from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.create_report import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    _build_layout_blocks,
    _build_panel_blocks,
    create_report,
)

create_report_module = importlib.import_module("wandb_mcp_server.mcp_tools.create_report")


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
        mock_wr.Runset.assert_called_once_with(entity="entity", project="project", filters='name in ["r1", "r2"]')
        mock_wr.LinePlot.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_analysis_run_id_uses_deterministic_filter(self, mock_wr):
        """Runset for analysis_run_id uses persisted Reports v2 filters."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.BarPlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "bar", "metrics": ["p50"], "title": "Latency", "analysis_run_id": "abc123"}]
        _build_panel_blocks(panels, "entity", "project")

        runset_call = mock_wr.Runset.call_args
        assert runset_call[1].get("filters") == 'name == "abc123"'
        assert "query" not in runset_call[1]

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_run_comparison_uses_deterministic_filter(self, mock_wr):
        """run_comparison with run_ids uses a deterministic Reports v2 filter."""
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock()

        panels = [{"type": "run_comparison", "metrics": ["loss"], "run_ids": ["r1", "r2"], "title": "Compare"}]
        _build_panel_blocks(panels, "entity", "project")

        comp_call = mock_wr.Runset.call_args
        assert comp_call[1].get("filters") == 'name in ["r1", "r2"]'
        assert "query" not in comp_call[1]

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
        mock_wr.Runset.assert_called_once_with(entity="entity", project="project", filters='name == "abc123"')
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
    def test_explicit_filters_win_over_run_ids(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock(return_value="Runset")

        panels = [
            {
                "type": "line",
                "x": "_step",
                "y": ["loss"],
                "title": "Loss",
                "run_ids": ["ignored"],
                "filters": 'displayName == "Customer Run"',
            }
        ]

        _build_panel_blocks(panels, "entity", "project")

        mock_wr.Runset.assert_called_once_with(
            entity="entity",
            project="project",
            filters='displayName == "Customer Run"',
        )

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_explicit_runset_query_is_search_passthrough(self, mock_wr):
        mock_wr.PanelGrid = MagicMock()
        mock_wr.LinePlot = MagicMock()
        mock_wr.Runset = MagicMock(return_value="Runset")

        panels = [{"type": "line", "x": "_step", "y": ["loss"], "runset_query": "baseline"}]

        _build_panel_blocks(panels, "entity", "project")

        mock_wr.Runset.assert_called_once_with(entity="entity", project="project", query="baseline")

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_panel_grid_groups_multiple_custom_charts(self, mock_wr):
        mock_wr.H2 = MagicMock(side_effect=lambda text: f"H2:{text}")
        mock_wr.P = MagicMock(side_effect=lambda text: f"P:{text}")
        mock_wr.MarkdownBlock = MagicMock(side_effect=lambda text: f"Markdown:{text}")
        mock_wr.PanelGrid = MagicMock(return_value="PanelGrid")
        mock_wr.CustomChart = MagicMock(side_effect=lambda **kw: f"Custom:{kw['chart_strings']['title']}")
        mock_wr.Runset = MagicMock(return_value="Runset")

        blocks = _build_layout_blocks(
            [
                {"type": "heading", "level": 2, "text": "Average precision by class"},
                {"type": "markdown", "text": "These panels share the same filtered runset."},
                {
                    "type": "panel_grid",
                    "run_ids": ["run_a", "run_b"],
                    "panels": [
                        {
                            "type": "custom_chart",
                            "query": {"summaryTable": {"tableKey": "car_ap"}},
                            "chart_name": "cruise/bar_chart/v2",
                            "chart_fields": {"x": "threshold", "y": "ap"},
                            "chart_strings": {"title": "CAR AP"},
                        },
                        {
                            "type": "custom_chart",
                            "query": {"summaryTable": {"tableKey": "truck_ap"}},
                            "chart_name": "cruise/bar_chart/v2",
                            "chart_fields": {"x": "threshold", "y": "ap"},
                            "chart_strings": {"title": "TRUCK AP"},
                        },
                        {
                            "type": "custom_chart",
                            "query": {"summaryTable": {"tableKey": "motorcycle_ap"}},
                            "chart_name": "cruise/bar_chart/v2",
                            "chart_fields": {"x": "threshold", "y": "ap"},
                            "chart_strings": {"title": "MOTORCYCLE AP"},
                        },
                    ],
                },
            ],
            "entity",
            "project",
        )

        assert blocks == [
            "H2:Average precision by class",
            "P:These panels share the same filtered runset.",
            "PanelGrid",
        ]
        mock_wr.Runset.assert_called_once_with(entity="entity", project="project", filters='name in ["run_a", "run_b"]')
        mock_wr.PanelGrid.assert_called_once_with(
            runsets=["Runset"],
            hide_run_sets=False,
            panels=["Custom:CAR AP", "Custom:TRUCK AP", "Custom:MOTORCYCLE AP"],
        )

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

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_create_report_with_ordered_layout_does_not_add_charts_heading(self, mock_api_mgr, mock_wr):
        mock_api_mgr.get_api_key.return_value = "fake_key"
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")

        mock_report = MagicMock()
        mock_report.url = "https://wandb.ai/report/layout"
        mock_wr.Report.return_value = mock_report
        mock_wr.P = MagicMock(side_effect=lambda text: f"P:{text}")
        mock_wr.H2 = MagicMock(side_effect=lambda text: f"H2:{text}")
        mock_wr.MarkdownBlock = MagicMock(side_effect=lambda text: f"Markdown:{text}")
        mock_wr.PanelGrid = MagicMock(side_effect=lambda **kw: "PanelGrid")
        mock_wr.LinePlot = MagicMock(return_value="LinePlot")
        mock_wr.Runset = MagicMock(return_value="Runset")

        create_report(
            "entity",
            "project",
            "Layout Report",
            markdown_report_text="Hello world",
            panels=[
                {"type": "heading", "level": 2, "text": "Section"},
                {"type": "markdown", "text": "Section intro"},
                {"type": "panel_grid", "run_ids": ["run_a"], "panels": [{"type": "line", "y": ["loss"]}]},
            ],
        )

        blocks = mock_report.blocks
        assert "H2:Charts" not in blocks
        assert blocks[-3:] == ["H2:Section", "P:Section intro", "PanelGrid"]


class TestRealWorkspacesReportObjects:
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_agent_style_layout_serializes_panel_grid_and_runset_filter(self, mock_api_mgr):
        """Build the same report shape an agent should send and inspect Reports v2 objects."""
        mock_api_mgr.get_api_key.return_value = "fake_key"
        mock_api_mgr.get_api.return_value = MagicMock(viewer={"username": "test-user"})

        fake_api = MagicMock()
        fake_api.client.app_url = "https://wandb.ai"
        fake_api.client.execute.return_value = {"project": {"internalId": "project-internal-id"}}
        fake_api._service_api = MagicMock()
        fake_api._service_api.app_url = "https://wandb.ai"
        fake_api._service_api.execute_graphql.return_value = {"project": {"internalId": "project-internal-id"}}
        captured_reports = []

        def fake_save(report, *args, **kwargs):
            captured_reports.append(report)
            report.id = "report-id"

        with (
            patch.object(create_report_module.wr.Report, "save", fake_save),
            patch("wandb_workspaces.reports.v2.interface._get_api", return_value=fake_api),
        ):
            result = create_report(
                "entity",
                "project",
                "Agent Report",
                markdown_report_text="# Agent Report\n\n[TOC]",
                panels=[
                    {"type": "heading", "level": 2, "text": "Average precision by class"},
                    {"type": "markdown", "text": "These panels share the same filtered runset."},
                    {
                        "type": "panel_grid",
                        "run_ids": ["run_a", "run_b"],
                        "hide_run_sets": False,
                        "panels": [
                            {
                                "type": "custom_chart",
                                "query": {"summaryTable": {"tableKey": "car_ap"}},
                                "chart_name": "cruise/bar_chart/v2",
                                "chart_fields": {"x": "threshold", "y": "ap"},
                                "chart_strings": {"title": "CAR AP"},
                            },
                            {
                                "type": "custom_chart",
                                "query": {"summaryTable": {"tableKey": "truck_ap"}},
                                "chart_name": "cruise/bar_chart/v2",
                                "chart_fields": {"x": "threshold", "y": "ap"},
                                "chart_strings": {"title": "TRUCK AP"},
                            },
                        ],
                    },
                ],
            )
            report_model = captured_reports[0]._to_model()

        assert result["url"] == "https://wandb.ai/entity/project/reports/Agent-Report--report-id"
        panel_grids = [block for block in report_model.spec.blocks if getattr(block, "type", None) == "panel-grid"]
        assert len(panel_grids) == 1

        panel_grid = panel_grids[0]
        runset = panel_grid.metadata.run_sets[0]
        assert runset.search.query == ""
        if isinstance(runset.filters, dict):

            def iter_filter_nodes(node):
                if isinstance(node, dict):
                    yield node
                    for child in node.get("filters", []):
                        yield from iter_filter_nodes(child)

            assert any(
                node.get("key") == {"section": "run", "name": "name"}
                and node.get("op") == "IN"
                and node.get("value") == ["run_a", "run_b"]
                for node in iter_filter_nodes(runset.filters)
            )
        else:
            filters = runset.filters.filters[0].filters
            assert len(filters) == 1
            assert filters[0].key.name == "name"
            assert filters[0].op == "IN"
            assert filters[0].value == ["run_a", "run_b"]

        panels = panel_grid.metadata.panel_bank_section_config.panels
        assert len(panels) == 2
        assert [panel.config.panel_def_id for panel in panels] == [
            "cruise/bar_chart/v2",
            "cruise/bar_chart/v2",
        ]
        assert [panel.config.string_settings["title"] for panel in panels] == ["CAR AP", "TRUCK AP"]
