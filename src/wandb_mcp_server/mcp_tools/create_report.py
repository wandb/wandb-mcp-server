#!/usr/bin/env python
"""
SAFE VERSION - W&B Report creation with markdown-only output
This version eliminates the singleton contamination vulnerability and uses only markdown.
"""

from typing import Any, Dict, List, Optional, Union
import json
import re

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.reports.v2.interface as wr_interface

from wandb_mcp_server.api_client import (
    WandBReportCreationFailed,
    WandBWriteOutcomeUnknown,
    raise_for_wandb_server_busy,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_report_writer import save_report_bounded
from wandb_mcp_server.wandb_urls import publicize_wandb_url

logger = get_rich_logger(__name__)


# Patch wandb_workspaces._get_api once at module import to read from contextvar
# This is SAFE for concurrent requests - each request has its own contextvar value
def _get_api_from_context():
    """Patched _get_api that reads API key from request context."""
    from wandb_mcp_server.api_client import WandBApiManager

    # The manager reads the API key from a ContextVar and returns a client
    # configured with the resolved internal/public base URL and request timeout.
    return WandBApiManager.get_api()


# Patch once at import - concurrent-safe because it reads from contextvar
wr_interface._get_api = _get_api_from_context


CREATE_WANDB_REPORT_TOOL_DESCRIPTION = """Create a new Weights & Biases Report to document analysis and findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases.
Always provide the returned report link to the user.

<when_to_use>
Call this tool AFTER completing analysis to create a shareable report. Combine
markdown text (for narrative, tables, and findings) with optional panels (for
native charts, custom Vega charts, and W&B Table-backed charts) to produce a
polished deliverable. If you have metric data from get_run_history_tool, use
native panels to visualize it in the report. If chart data already exists in a
W&B Table or summary table, use custom_chart_table.
</when_to_use>

<markdown_generation_guide>
When generating the markdown_report_text parameter, structure your content using:

**Headers**: Organize content hierarchically
- # Main Title (H1)
- ## Section Title (H2)
- ### Subsection Title (H3)

**Paragraphs**: Write clear, informative text separated by blank lines

**Lists**: Present information clearly
- Bullet points: Use - or *
- Numbered lists: Use 1. 2. 3.

**Formatting**:
- **bold** for emphasis
- *italic* for subtle emphasis
- `inline code` for technical terms
- Links: [link text](url)

**Code blocks**: For code snippets or technical content
```language
code here
```

**Table of Contents**: Add [TOC] on its own line to auto-generate navigation

**Best Practices**:
- Start with a clear H1 title
- Use [TOC] after the title for easy navigation
- Structure content with logical sections (H2) and subsections (H3)
- Keep paragraphs concise and focused
- Use lists for multiple related items
- Include code blocks for technical examples
</markdown_generation_guide>

Args:
    entity_name: str, The W&B entity (team or username) - required
    project_name: str, The W&B project name - required
    title: str, Title of the W&B Report - required
    description: str, Optional brief description of the report
    markdown_report_text: str, Well-structured markdown content for the report body
    panels: list of dict, optional - Chart panels or ordered layout blocks to add after the markdown content.
        Each dict specifies a chart type and configuration:
        - {"type": "line", "x": "_step", "y": ["loss", "val_loss"], "title": "Training Loss"}
          Creates a LinePlot tracking metrics over steps.
        - {"type": "bar", "metrics": ["accuracy", "f1"], "title": "Metrics"}
          Creates a BarPlot comparing metrics across runs.
        - {"type": "run_comparison", "metrics": ["loss", "accuracy"], "run_ids": ["abc", "def"], "title": "Compare"}
          Creates a PanelGrid comparing specific runs on selected metrics.
        - {"type": "custom_chart", "title": "PR Curve", "query": {"summaryTable": {"tableKey": "pr_curve_table"}},
           "chart_name": "wandb/line/v0", "chart_fields": {"x": "recall", "y": "precision"},
           "chart_strings": {"title": "PR Curve"}, "run_ids": ["abc123"], "hide_run_sets": true}
          Creates a CustomChart from an explicit wandb-workspaces query.
        - {"type": "custom_chart_table", "title": "PR Curve", "table_name": "pr_curve_table",
           "chart_name": "wandb/line/v0", "chart_fields": {"x": "recall", "y": "precision"},
           "chart_strings": {"title": "PR Curve"}, "hide_run_sets": true}
          Creates a CustomChart from a W&B Table or summary table key via CustomChart.from_table().
        - {"type": "heading", "level": 2, "text": "Average precision by class"}
          Adds an H1/H2/H3 report heading at this position.
        - {"type": "markdown", "text": "These charts share the same filtered runset."}
          Adds markdown narrative at this position.
        - {"type": "panel_grid", "run_ids": ["run_a", "run_b"], "hide_run_sets": false,
           "panels": [{...chart panel...}, {...chart panel...}]}
          Creates one PanelGrid whose child panels share a single Runset.
        Use custom_chart_table for summary-table-backed charts. Use custom_chart with an explicit historyTable
        query for PR/ROC curves or other charts logged through run history.
        If panels only contains chart specs, the tool appends them under a Charts heading for backward compatibility.
        If panels contains heading, markdown, or panel_grid blocks, the list is treated as an ordered layout and no
        automatic Charts heading is added. If omitted, report is markdown-only.

<custom_chart_panel_guide>
Use native panel types for ordinary run metrics:
- line: metric history over _step
- bar: summary metric comparisons
- scatter: two summary/config fields

Use custom_chart_table when the source data already exists as a W&B Table saved
in run summary. This is the preferred path for final confusion matrices,
per-class AP tables, and other one-snapshot table-backed visualizations:
{
  "type": "custom_chart_table",
  "title": "Precision-Recall Curve",
  "table_name": "pr_curve_table",
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "recall", "y": "precision"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "hide_run_sets": true
}

Use custom_chart only when you know the exact wandb-workspaces query shape:
{
  "type": "custom_chart",
  "query": {"summaryTable": {"tableKey": "pr_curve_table"}},
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "recall", "y": "precision"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "run_ids": ["abc123"],
  "hide_run_sets": true
}

For PR curves, ROC curves, and other charts logged through run history, use an
explicit historyTable query. tableKey is the key passed to run.log(), not a
column name inside the table:
{
  "type": "custom_chart",
  "query": {"historyTable": {"tableKey": "pr_curve"}},
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "r", "y": "p", "color": "c"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "run_ids": ["abc123"],
  "hide_run_sets": true
}

If the chart data is computed inside MCP rather than already stored as a W&B
Table, call log_analysis_to_wandb first, then reference the logged run/table
from this report tool.
</custom_chart_panel_guide>

<report_layout_guide>
Use panel_grid when multiple panels should share one run selector / Runset. This is the correct structure for
reports such as H2 / markdown / panel-grid / H2 / markdown / panel-grid:
{
  "type": "panel_grid",
  "run_ids": ["run_a", "run_b"],
  "hide_run_sets": false,
  "panels": [
    {"type": "custom_chart", "query": {"summaryTable": {"tableKey": "car_ap"}},
     "chart_name": "cruise/bar_chart/v2", "chart_fields": {"x": "threshold", "y": "ap"},
     "chart_strings": {"title": "CAR AP"}},
    {"type": "custom_chart", "query": {"summaryTable": {"tableKey": "truck_ap"}},
     "chart_name": "cruise/bar_chart/v2", "chart_fields": {"x": "threshold", "y": "ap"},
     "chart_strings": {"title": "TRUCK AP"}}
  ]
}

Runset scoping:
- run_ids means W&B internal run keys (Python SDK run.id), not display names.
- run_ids are converted to deterministic Reports v2 filters, for example name in ["run_a", "run_b"].
- filters may be passed as a Reports v2 expression string and wins over generated run_ids filters.
- runset_query may be passed for explicit search behavior. Do not use custom_chart query for run filtering.
- chart_name maps to the Vega panelDefId such as wandb/line/v0 or cruise/bar_chart/v2. Put visible titles in chart_strings.
</report_layout_guide>

<manual_validation_recipe>
To validate a table-backed custom chart manually:
1. Pick a run that has a logged W&B Table key, such as a summary table or PR/ROC history table.
2. Call create_wandb_report_tool with custom_chart_table for summary tables or custom_chart with historyTable for PR/ROC curves.
3. Open the returned report URL and confirm the custom Vega chart renders.
4. If the chart does not render, verify the table_name and chart_fields match the table columns and UI chart config.
</manual_validation_recipe>

Returns:
    The URL to the created report

Example markdown structure:
```markdown
# Analysis Report Title

[TOC]

## Executive Summary
Brief overview of the analysis and key findings.

## Methodology
Description of the approach used in the analysis.

### Data Collection
- Source 1: Description
- Source 2: Description

### Analysis Techniques
Technical details about methods used.

## Results
Key findings from the analysis.

### Performance Metrics
- Accuracy: 95%
- Precision: 92%
- Recall: 89%

## Conclusions
Summary of insights and recommendations.
```
"""


def create_report(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: Optional[str] = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
    panels: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Create a W&B Report with markdown content and optional chart panels.

    Security: No singleton contamination, reads API key from contextvar.
    Thread Safety: Each request has its own contextvar value.
    """
    from wandb_mcp_server.api_client import WandBApiManager

    api_key = WandBApiManager.get_api_key()

    if not api_key:
        logger.warning("No API key available for W&B")
        raise Exception("No W&B API key available")

    with track_tool_execution(
        "create_report",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "title": title,
            "description": description,
            "has_panels": bool(panels),
        },
    ):
        try:
            report = wr.Report(
                entity=entity_name,
                project=project_name,
                title=title,
                description=description or "",
                width="fluid",
            )

            blocks = parse_markdown_to_blocks(markdown_report_text or "")

            if plots_html:
                import base64 as b64

                if isinstance(plots_html, str):
                    plots_html = {"chart": plots_html}
                for label, html_content in plots_html.items():
                    content = html_content.strip() if isinstance(html_content, str) else ""
                    if content.startswith("<svg") or content.startswith("data:image/svg"):
                        if content.startswith("<svg"):
                            data_uri = f"data:image/svg+xml;base64,{b64.b64encode(content.encode()).decode()}"
                        else:
                            data_uri = content
                        blocks.append(wr.Image(url=data_uri, caption=label))
                        logger.info("Added report content block (kind=svg)")
                    elif content:
                        blocks.append(wr.MarkdownBlock(content))
                        logger.info("Added report content block (kind=html)")

            security_notice = wr.P("*Report created via W&B MCP Server*")
            report.blocks = [security_notice] + blocks

            if panels:
                layout_mode = _is_layout_mode(panels)
                panel_blocks = (
                    _build_layout_blocks(panels, entity_name, project_name)
                    if layout_mode
                    else _build_panel_blocks(panels, entity_name, project_name)
                )
                if panel_blocks:
                    if not layout_mode:
                        report.blocks.append(wr.H2("Charts"))
                    report.blocks.extend(panel_blocks)

            api = WandBApiManager.get_api(api_key)
            save_report_bounded(report, api)

            logger.info(
                "Created W&B report (panels=%d blocks=%d)",
                len(panels or []),
                len(report.blocks),
            )

            return {"url": publicize_wandb_url(report.url)}

        except Exception as e:
            raise_for_wandb_server_busy(e)
            if isinstance(e, WandBWriteOutcomeUnknown):
                logger.error("W&B did not confirm the bounded report write")
                raise
            # Workspaces exceptions may interpolate report titles, scope, or
            # other customer values. Preserve only a bounded categorical type.
            logger.error(
                "Report creation failed after a bounded W&B write (error_type=%s)",
                type(e).__name__[:64],
            )
            raise WandBReportCreationFailed() from e


def _build_panel_blocks(
    panels: List[Dict[str, Any]],
    entity_name: str,
    project_name: str,
) -> List:
    """Convert panel dicts to wandb_workspaces report blocks."""
    blocks = []
    for panel_spec in panels:
        panel_type = panel_spec.get("type", "").lower()
        panel_title = panel_spec.get("title", "")

        try:
            block = _build_panel_block(panel_spec, entity_name, project_name)
            if block is not None:
                blocks.append(block)
            elif panel_type not in _KNOWN_PANEL_TYPES:
                logger.warning("Skipped unsupported report panel type")

        except Exception as e:
            logger.warning(
                "Failed to build report panel (error_type=%s)",
                type(e).__name__[:64],
            )
            blocks.append(wr.P(f"*Panel '{panel_title}' could not be rendered.*"))

    return blocks


_KNOWN_PANEL_TYPES = {
    "line",
    "bar",
    "scatter",
    "run_comparison",
    "markdown_table",
    "markdown_panel",
    "custom_chart",
    "custom_chart_table",
}

_LAYOUT_BLOCK_TYPES = {
    "heading",
    "markdown",
    "panel_grid",
}


def _build_panel_block(
    panel_spec: Dict[str, Any],
    entity_name: str,
    project_name: str,
):
    """Build one report block for a panel spec."""
    panel_type = panel_spec.get("type", "").lower()
    if panel_type == "markdown_table":
        return _build_markdown_table_block(panel_spec)
    panel = _build_panel_object(panel_spec)
    if panel is not None:
        return _build_panel_grid(panel_spec, _build_runset(panel_spec, entity_name, project_name), panel)
    return None


def _build_runset(
    panel_spec: Dict[str, Any],
    entity_name: str,
    project_name: str,
    *,
    include_run_ids: bool = False,
):
    """Build a Reports v2 Runset from MCP JSON-safe runset fields."""
    runset_spec = panel_spec.get("runset", {})
    if runset_spec is None:
        runset_spec = {}
    if not isinstance(runset_spec, dict):
        raise ValueError("runset must be an object")

    filters = panel_spec.get("filters", runset_spec.get("filters"))
    if filters is not None and not isinstance(filters, str):
        raise ValueError("filters must be a Reports v2 expression string")

    run_id = panel_spec.get("analysis_run_id")
    if not filters and run_id:
        filters = _run_ids_filter([run_id])

    run_ids = panel_spec.get("run_ids", runset_spec.get("run_ids", []))
    if not filters and run_ids:
        filters = _run_ids_filter(run_ids)

    query = panel_spec.get("runset_query", runset_spec.get("query", ""))
    if query is None:
        query = ""
    if not isinstance(query, str):
        raise ValueError("runset query must be a string")

    kwargs = {"entity": entity_name, "project": project_name}
    if query:
        kwargs["query"] = query
    if filters:
        kwargs["filters"] = filters
    return wr.Runset(**kwargs)


def _run_ids_filter(run_ids: List[str]) -> str:
    """Return a deterministic Reports v2 filter expression for run keys."""
    if isinstance(run_ids, str) or not isinstance(run_ids, list):
        raise ValueError("run_ids must be a list of W&B internal run keys")
    normalized = []
    for run_id in run_ids:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_ids must contain non-empty strings")
        normalized.append(run_id)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return f"name == {json.dumps(normalized[0])}"
    return f"name in {json.dumps(normalized)}"


def _build_panel_grid(panel_spec: Dict[str, Any], runset, panel):
    """Wrap one panel in a PanelGrid."""
    return wr.PanelGrid(
        runsets=[runset],
        hide_run_sets=bool(panel_spec.get("hide_run_sets", False)),
        panels=[panel],
    )


def _build_panel_object(panel_spec: Dict[str, Any]):
    """Build one wandb_workspaces panel object without wrapping it in a grid."""
    panel_type = panel_spec.get("type", "").lower()
    panel_title = panel_spec.get("title", "")

    if panel_type == "line":
        x_key = panel_spec.get("x", "_step")
        y_keys = panel_spec.get("y", [])
        if not y_keys:
            return None
        return wr.LinePlot(x=x_key, y=y_keys, title=panel_title)

    if panel_type == "bar":
        metrics = panel_spec.get("metrics", [])
        if not metrics:
            return None
        return wr.BarPlot(metrics=metrics, title=panel_title)

    if panel_type == "scatter":
        x_key = panel_spec.get("x", "")
        y_key = panel_spec.get("y", "")
        if not x_key or not y_key:
            return None
        return wr.ScatterPlot(x=x_key, y=y_key, title=panel_title)

    if panel_type == "run_comparison":
        metrics = panel_spec.get("metrics", [])
        if not metrics:
            return None
        return wr.LinePlot(x="_step", y=metrics, title=panel_title)

    if panel_type == "markdown_table":
        return None

    if panel_type == "markdown_panel":
        markdown = panel_spec.get("markdown", "")
        if not markdown:
            return None
        return wr.MarkdownPanel(markdown=markdown)

    if panel_type == "custom_chart":
        return _build_custom_chart_object(panel_spec)

    if panel_type == "custom_chart_table":
        return _build_custom_chart_from_table_object(panel_spec)

    return None


def _build_markdown_table_block(panel_spec: Dict[str, Any]):
    """Build a report-level MarkdownBlock table."""
    panel_title = panel_spec.get("title", "")
    headers = panel_spec.get("headers", [])
    rows = panel_spec.get("rows", [])
    if not headers or not rows:
        return None
    md = f"### {panel_title}\n\n" if panel_title else ""
    md += "| " + " | ".join(str(h) for h in headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(str(v) for v in row) + " |\n"
    return wr.MarkdownBlock(md)


def _build_custom_chart_object(panel_spec: Dict[str, Any]):
    """Build a CustomChart from an explicit workspaces query."""
    query = _required_dict(panel_spec, "query")
    chart_fields = _required_dict(panel_spec, "chart_fields")
    chart_strings = _optional_dict(panel_spec, "chart_strings")
    chart_name = _required_string(panel_spec, "chart_name")
    return wr.CustomChart(
        query=query,
        chart_name=chart_name,
        chart_fields=chart_fields,
        chart_strings=chart_strings,
    )


def _build_custom_chart_from_table_object(panel_spec: Dict[str, Any]):
    """Build a CustomChart backed by a W&B Table or summary table key."""
    table_name = _required_string(panel_spec, "table_name")
    chart_fields = _required_dict(panel_spec, "chart_fields")
    chart_strings = _optional_dict(panel_spec, "chart_strings")
    chart_name = panel_spec.get("chart_name") or ""
    chart = wr.CustomChart.from_table(
        table_name,
        chart_fields=chart_fields,
        chart_strings=chart_strings,
    )
    if chart_name:
        chart.chart_name = chart_name
    return chart


def _is_layout_mode(panels: List[Dict[str, Any]]) -> bool:
    """Return True when panels contains ordered report layout blocks."""
    return any(panel.get("type", "").lower() in _LAYOUT_BLOCK_TYPES for panel in panels)


def _build_layout_blocks(
    panels: List[Dict[str, Any]],
    entity_name: str,
    project_name: str,
) -> List:
    """Build ordered report blocks from layout-mode panel specs."""
    blocks = []
    for panel_spec in panels:
        panel_type = panel_spec.get("type", "").lower()
        panel_title = panel_spec.get("title") or panel_spec.get("text", "")
        try:
            block = _build_layout_block(panel_spec, entity_name, project_name)
            if block is None:
                if panel_type not in _KNOWN_PANEL_TYPES and panel_type not in _LAYOUT_BLOCK_TYPES:
                    logger.warning("Skipped unsupported report layout block type")
                continue
            if isinstance(block, list):
                blocks.extend(block)
            else:
                blocks.append(block)
        except Exception as e:
            logger.warning(
                "Failed to build report layout block (error_type=%s)",
                type(e).__name__[:64],
            )
            blocks.append(wr.P(f"*Panel '{panel_title}' could not be rendered.*"))
    return blocks


def _build_layout_block(
    panel_spec: Dict[str, Any],
    entity_name: str,
    project_name: str,
):
    """Build one ordered layout block."""
    panel_type = panel_spec.get("type", "").lower()
    if panel_type == "heading":
        text = _required_string(panel_spec, "text")
        level = panel_spec.get("level", 2)
        if level == 1:
            return wr.H1(text)
        if level == 3:
            return wr.H3(text)
        return wr.H2(text)
    if panel_type == "markdown":
        text = _required_string(panel_spec, "text")
        return parse_markdown_to_blocks(text)
    if panel_type == "panel_grid":
        return _build_panel_grid_block(panel_spec, entity_name, project_name)
    return _build_panel_block(panel_spec, entity_name, project_name)


def _build_panel_grid_block(
    panel_spec: Dict[str, Any],
    entity_name: str,
    project_name: str,
):
    """Build one PanelGrid with a shared Runset and multiple child panels."""
    child_specs = panel_spec.get("panels", [])
    if not isinstance(child_specs, list) or not child_specs:
        raise ValueError("panel_grid panels must be a non-empty list")
    panel_objects = []
    for child_spec in child_specs:
        if not isinstance(child_spec, dict):
            raise ValueError("panel_grid child panels must be objects")
        child = _build_panel_object(child_spec)
        if child is not None:
            panel_objects.append(child)
    if not panel_objects:
        return None
    return wr.PanelGrid(
        runsets=[_build_runset(panel_spec, entity_name, project_name)],
        hide_run_sets=bool(panel_spec.get("hide_run_sets", False)),
        panels=panel_objects,
    )


def _required_string(panel_spec: Dict[str, Any], key: str) -> str:
    value = panel_spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_dict(panel_spec: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = panel_spec.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{key} must be a non-empty object")
    return value


def _optional_dict(panel_spec: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = panel_spec.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def parse_markdown_to_blocks(
    text: str,
) -> List[Union[wr.H1, wr.H2, wr.H3, wr.P, wr.TableOfContents, wr.MarkdownBlock, wr.CodeBlock]]:
    """
    Parse markdown text into W&B report blocks.

    Supports the following W&B report blocks:
    - Headers: H1, H2, H3 (extracted for TOC support)
    - Table of Contents: TableOfContents (via [TOC] marker)
    - Code Blocks: CodeBlock (with language syntax highlighting)
    - Rich Markdown: MarkdownBlock (for tables, lists, blockquotes, etc.)
    - Paragraphs: P (for simple text)

    Strategy:
    - Extract top-level headers (H1, H2, H3) as separate blocks for TOC
    - Use CodeBlock for code with syntax highlighting
    - Use MarkdownBlock for complex markdown (tables, lists, etc.)
    - Use P for simple paragraphs
    """
    blocks = []
    lines = text.strip().split("\n") if text else []

    current_content = []
    in_code_block = False
    code_language = None
    code_block_content = []

    def flush_content():
        """Helper to flush accumulated content as MarkdownBlock or P"""
        if not current_content:
            return

        content_text = "\n".join(current_content).strip()
        if not content_text:
            return

        # Check if content has complex markdown (tables, lists, etc.)
        has_table = "|" in content_text and "---" in content_text
        has_list = re.search(r"^\s*[-*+]\s", content_text, re.MULTILINE)
        has_ordered_list = re.search(r"^\s*\d+\.\s", content_text, re.MULTILINE)
        has_blockquote = re.search(r"^\s*>\s", content_text, re.MULTILINE)
        has_inline_code = "`" in content_text
        has_bold_italic = re.search(r"[*_]{1,2}\w", content_text)

        # Use MarkdownBlock for rich content, P for simple paragraphs
        if has_table or has_list or has_ordered_list or has_blockquote:
            blocks.append(wr.MarkdownBlock(content_text))
        elif has_inline_code or has_bold_italic or len(content_text) > 200:
            # Use MarkdownBlock for formatted text or longer content
            blocks.append(wr.MarkdownBlock(content_text))
        else:
            # Simple paragraph
            blocks.append(wr.P(content_text))

        current_content.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.startswith("```"):
            if in_code_block:
                # End of code block
                flush_content()
                code_content = "\n".join(code_block_content)
                # wandb_workspaces only accepts these language tags
                _WR_SUPPORTED_LANGUAGES = {
                    "javascript",
                    "python",
                    "css",
                    "json",
                    "html",
                    "markdown",
                    "yaml",
                }
                _LANGUAGE_MAP = {
                    "typescript": "javascript",
                    "bash": "python",
                    "shell": "python",
                    "sh": "python",
                    "sql": None,
                    "go": None,
                    "rust": None,
                    "java": None,
                    "c": None,
                    "cpp": None,
                }
                mapped = code_language
                if code_language and code_language not in _WR_SUPPORTED_LANGUAGES:
                    mapped = _LANGUAGE_MAP.get(code_language)
                if mapped and mapped in _WR_SUPPORTED_LANGUAGES:
                    blocks.append(wr.CodeBlock(code=code_content, language=mapped))
                else:
                    blocks.append(wr.CodeBlock(code=code_content))
                code_block_content = []
                in_code_block = False
                code_language = None
            else:
                # Start of code block
                flush_content()
                in_code_block = True
                # Extract language identifier
                lang_match = re.match(r"```(\w+)", line)
                if lang_match:
                    code_language = lang_match.group(1).lower()
            i += 1
            continue

        # If in code block, accumulate lines
        if in_code_block:
            code_block_content.append(line)
            i += 1
            continue

        # Check for top-level headers (extract for TOC support)
        h1_match = re.match(r"^# (.+)$", line)
        h2_match = re.match(r"^## (.+)$", line)
        h3_match = re.match(r"^### (.+)$", line)

        # Check for Table of Contents marker
        is_toc = line.strip().lower() == "[toc]"

        if h1_match or h2_match or h3_match or is_toc:
            flush_content()

            if h1_match:
                blocks.append(wr.H1(h1_match.group(1)))
            elif h2_match:
                blocks.append(wr.H2(h2_match.group(1)))
            elif h3_match:
                blocks.append(wr.H3(h3_match.group(1)))
            elif is_toc:
                blocks.append(wr.TableOfContents())
        else:
            # Accumulate content
            current_content.append(line)

        i += 1

    # Flush any remaining content
    flush_content()

    # If no blocks were created, add a default paragraph
    if not blocks:
        blocks.append(wr.P("*Empty report*"))

    return blocks
