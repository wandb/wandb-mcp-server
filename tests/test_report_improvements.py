"""Tests for report creation improvements: SVG, new panel types, analysis_run_id."""

from unittest.mock import MagicMock, patch

import pytest

import wandb_mcp_server.mcp_tools.create_report as create_report_module
from wandb_mcp_server.mcp_tools.create_report import _build_panel_blocks


@pytest.fixture(autouse=True)
def _stub_bounded_report_save(monkeypatch):
    monkeypatch.setattr(
        create_report_module,
        "save_report_bounded",
        lambda report, api: report,
    )


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

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    @patch("wandb_mcp_server.api_client.WandBApiManager")
    def test_success_logs_omit_report_scope_title_and_content_label(
        self,
        mock_api,
        mock_wr,
    ):
        from wandb_mcp_server.mcp_tools.create_report import create_report

        entity = "private-report-entity-canary"
        project = "private-report-project-canary"
        title = "private-report-title-canary"
        label = "private-report-label-canary"
        mock_api.get_api_key.return_value = "key"
        mock_api.get_api.return_value = MagicMock(viewer={"username": "test"})
        mock_wr.Report.return_value = MagicMock(url="https://wandb.ai/report")
        mock_wr.MarkdownBlock = MagicMock()
        mock_wr.P = MagicMock()

        with patch.object(create_report_module.logger, "info") as info_log:
            create_report(
                entity_name=entity,
                project_name=project,
                title=title,
                plots_html={label: "<div>safe fixture body</div>"},
            )

        rendered = "\n".join(" ".join(map(str, call.args)) for call in info_log.call_args_list)
        assert "Added report content block (kind=html)" in rendered
        assert "Created W&B report (panels=%d blocks=%d)" in rendered
        for canary in (entity, project, title, label):
            assert canary not in rendered


class TestReportLogPrivacy:
    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_panel_failure_log_omits_scope_title_and_exception_text(
        self,
        mock_wr,
        monkeypatch,
    ):
        entity = "private-panel-entity-canary"
        project = "private-panel-project-canary"
        title = "private-panel-title-canary"
        mock_wr.P = MagicMock()
        monkeypatch.setattr(
            create_report_module,
            "_build_panel_block",
            MagicMock(side_effect=RuntimeError(f"failed for {entity}/{project}/{title}")),
        )

        with patch.object(create_report_module.logger, "warning") as warning_log:
            create_report_module._build_panel_blocks(
                [{"type": "line", "title": title}],
                entity,
                project,
            )

        warning_log.assert_called_once_with(
            "Failed to build report panel (error_type=%s)",
            "RuntimeError",
        )
        rendered = " ".join(map(str, warning_log.call_args.args))
        for canary in (entity, project, title):
            assert canary not in rendered

    @patch("wandb_mcp_server.mcp_tools.create_report.wr")
    def test_layout_failure_log_omits_scope_title_and_exception_text(
        self,
        mock_wr,
        monkeypatch,
    ):
        entity = "private-layout-entity-canary"
        project = "private-layout-project-canary"
        title = "private-layout-title-canary"
        mock_wr.P = MagicMock()
        monkeypatch.setattr(
            create_report_module,
            "_build_layout_block",
            MagicMock(side_effect=RuntimeError(f"failed for {entity}/{project}/{title}")),
        )

        with patch.object(create_report_module.logger, "warning") as warning_log:
            create_report_module._build_layout_blocks(
                [{"type": "heading", "text": title}],
                entity,
                project,
            )

        warning_log.assert_called_once_with(
            "Failed to build report layout block (error_type=%s)",
            "RuntimeError",
        )
        rendered = " ".join(map(str, warning_log.call_args.args))
        for canary in (entity, project, title):
            assert canary not in rendered


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
        assert runset_call[1]["filters"] == 'name == "abc123"'
        assert "query" not in runset_call[1]

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
        assert filtered_runset_call[1]["filters"] == 'name in ["r1", "r2"]'
        assert "query" not in filtered_runset_call[1]
