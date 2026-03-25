#!/usr/bin/env python
"""
SAFE VERSION - W&B Report creation with markdown-only output
This version eliminates the singleton contamination vulnerability and uses only markdown.
"""

from typing import Any, Dict, List, Optional, Union
import re

import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.reports.v2.interface as wr_interface

import wandb
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.config import WANDB_BASE_URL
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call

logger = get_rich_logger(__name__)


# Patch wandb_workspaces._get_api once at module import to read from contextvar
# This is SAFE for concurrent requests - each request has its own contextvar value
def _get_api_from_context():
    """Patched _get_api that reads API key from request context."""
    from wandb_mcp_server.api_client import WandBApiManager

    api_key = WandBApiManager.get_api_key()

    if not api_key:
        raise Exception("No W&B API key available in context")

    try:
        # Uses explicit api_key from contextvar, not singleton
        # and points to the configured base URL
        return wandb.Api(api_key=api_key, overrides={"base_url": WANDB_BASE_URL})
    except wandb.errors.UsageError as e:
        raise Exception("Not logged in to W&B, check API key") from e


# Patch once at import - concurrent-safe because it reads from contextvar
wr_interface._get_api = _get_api_from_context


CREATE_WANDB_REPORT_TOOL_DESCRIPTION = """Create a new Weights & Biases Report to document analysis and findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases.
Always provide the returned report link to the user.

<when_to_use>
Call this tool AFTER completing analysis to create a shareable report. Combine
markdown text (for narrative, tables, and findings) with optional panels (for
line/bar charts) to produce a polished deliverable. If you have metric data
from get_run_history_tool, use panels to visualize it in the report.
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
    panels: list of dict, optional - Chart panels to add after the markdown content.
        Each dict specifies a chart type and configuration:
        - {"type": "line", "x": "_step", "y": ["loss", "val_loss"], "title": "Training Loss"}
          Creates a LinePlot tracking metrics over steps.
        - {"type": "bar", "metrics": ["accuracy", "f1"], "title": "Metrics"}
          Creates a BarPlot comparing metrics across runs.
        - {"type": "run_comparison", "metrics": ["loss", "accuracy"], "run_ids": ["abc", "def"], "title": "Compare"}
          Creates a PanelGrid comparing specific runs on selected metrics.
        Panels are additive to markdown content. If omitted, report is markdown-only.

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
    if plots_html:
        logger.info("Note: plots_html parameter provided but ignored in safe markdown-only mode")

    from wandb_mcp_server.api_client import WandBApiManager

    api_key = WandBApiManager.get_api_key()

    if not api_key:
        logger.warning("No API key available for W&B")
        raise Exception("No W&B API key available")

    try:
        api = WandBApiManager.get_api()
        log_tool_call(
            "create_report",
            api.viewer,
            {
                "entity_name": entity_name,
                "project_name": project_name,
                "title": title,
                "description": description,
                "has_panels": bool(panels),
            },
        )
    except Exception:
        logger.debug("analytics emit failed", exc_info=True)

    try:
        report = wr.Report(
            entity=entity_name,
            project=project_name,
            title=title,
            description=description or "",
            width="fluid",
        )

        blocks = parse_markdown_to_blocks(markdown_report_text or "")

        security_notice = wr.P("*Report created via W&B MCP Server*")
        report.blocks = [security_notice] + blocks

        if panels:
            panel_blocks = _build_panel_blocks(panels, entity_name, project_name)
            if panel_blocks:
                report.blocks.append(wr.H2("Charts"))
                report.blocks.extend(panel_blocks)

        report.save()

        logger.info(f"Created report: {title} (panels={len(panels or [])})")

        return {"url": report.url}

    except Exception as e:
        logger.error(f"Error creating report: {e}")
        raise Exception(f"Error creating report: {e}")


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
            if panel_type == "line":
                x_key = panel_spec.get("x", "_step")
                y_keys = panel_spec.get("y", [])
                if not y_keys:
                    continue
                pg = wr.PanelGrid(
                    runsets=[wr.Runset(entity=entity_name, project=project_name)],
                    panels=[wr.LinePlot(x=x_key, y=y_keys, title=panel_title)],
                )
                blocks.append(pg)

            elif panel_type == "bar":
                metrics = panel_spec.get("metrics", [])
                if not metrics:
                    continue
                pg = wr.PanelGrid(
                    runsets=[wr.Runset(entity=entity_name, project=project_name)],
                    panels=[wr.BarPlot(metrics=metrics, title=panel_title)],
                )
                blocks.append(pg)

            elif panel_type == "run_comparison":
                metrics = panel_spec.get("metrics", [])
                run_ids = panel_spec.get("run_ids", [])
                if not metrics:
                    continue
                runset_kwargs: Dict[str, Any] = {"entity": entity_name, "project": project_name}
                if run_ids:
                    # Runset.filters expects a filter expression string, not a dict.
                    # Use the report expr helpers so the panel actually selects the
                    # requested runs instead of merely labeling the runset.
                    runset_kwargs["filters"] = str(expr.Metric("name").isin(run_ids))
                chart_panels = [wr.LinePlot(x="_step", y=metrics, title=panel_title)]
                pg = wr.PanelGrid(
                    runsets=[wr.Runset(**runset_kwargs)],
                    panels=chart_panels,
                )
                blocks.append(pg)

            else:
                logger.warning(f"Unknown panel type: {panel_type}")

        except Exception as e:
            logger.warning(f"Failed to build panel '{panel_title}': {e}", exc_info=True)
            blocks.append(wr.P(f"*Panel '{panel_title}' could not be rendered.*"))

    return blocks


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
                # Create CodeBlock with language if specified
                if code_language and code_language in [
                    "python",
                    "javascript",
                    "typescript",
                    "css",
                    "json",
                    "html",
                    "markdown",
                    "yaml",
                    "bash",
                    "shell",
                ]:
                    blocks.append(wr.CodeBlock(code=code_content, language=code_language))
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
