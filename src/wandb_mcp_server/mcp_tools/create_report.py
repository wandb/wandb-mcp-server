#!/usr/bin/env python
"""
Weave MCP Server - Report creation and editing functionality for W&B reports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import wandb_workspaces.reports.v2 as wr
from dotenv import load_dotenv

import wandb
from wandb_mcp_server.utils import get_rich_logger

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logger = get_rich_logger(__name__)

CREATE_WANDB_REPORT_TOOL_DESCRIPTION = """Create a new Weights & Biases Report and add text and HTML-rendered charts. Useful to save/document analysis and other findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases. 

Always provide the returned report link to the user.

<plots_html_usage_guide>
- If the analsis has generated plots then they can be logged to a Weights & Biases report via converting them to html.
- All charts should be properly rendered in raw HTML, do not use any placeholders for any chart, render everything.
- All charts should be beautiful, tasteful and well proportioned.
- Plot html code should use SVG chart elements that should render properly in any modern browser.
- Include interactive hover effects where it makes sense.
- If the analysis contains multiple charts, break up the html into one section of html per chart.
- Ensure that the axis labels are properly set and aligned for each chart.
- Always use valid markdown for the report text.
</plots_html_usage_guide>

<plots_html_format_guide>
**IMPORTANT: plots_html Parameter Format**
- The plots_html parameter accepts either:
  1. A dictionary where keys are chart names and values are HTML strings: {"chart1": "<html>...</html>", "chart2": "<html>...</html>"}
  2. A single HTML string (will be automatically wrapped with key "chart")
- Do NOT pass raw HTML as a JSON string - pass it directly as an HTML string
- If you have multiple charts, use the dictionary format for better organization
- The tool will provide feedback about how your input was processed
</plots_html_format_guide>

Args:
    entity_name: str, The W&B entity (team or username) - required
    project_name: str, The W&B project name - required
    title: str, Title of the W&B Report - required
    description: str, Optional description of the W&B Report
    markdown_report_text: str, beuatifully formatted markdown text for the report body
    plots_html: str, Optional dict of plot name and html string of any charts created as part of an analysis

Returns:
    str, The url to the report

Example:
    ```python
    # Create a simple report
    report = create_report(
        entity_name="my-team",
        project_name="my-project",
        title="Model Analysis Report",
        description="Analysis of our latest model performance",
        markdown_report_text='''
            # Model Analysis Report
            [TOC]
            ## Performance Summary
            Our model achieved 95% accuracy on the test set.
            ### Key Metrics
            Precision: 0.92
            Recall: 0.89
        '''
    )
    ```
"""


def create_report(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: Optional[str] = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
) -> Dict[str, str]:
    """
    Create a new Weights & Biases Report and add text and charts. Useful to save/document analysis and other findings.

    Args:
        entity_name: The W&B entity (team or username)
        project_name: The W&B project name
        title: Title of the W&B Report
        description: Optional description of the W&B Report
        markdown_report_text: Optional markdown text for the report body
        plots_html: Optional dict of plot name and html string, or single HTML string

    Returns:
        Dict with 'url' and 'processing_details' keys
    """
    import json
    
    # Process plots_html and collect warnings
    processed_plots_html = None
    processing_warnings = []
    
    if isinstance(plots_html, str):
        try:
            # First try to parse as JSON (dictionary)
            processed_plots_html = json.loads(plots_html)
            processing_warnings.append("Successfully parsed plots_html as JSON dictionary")
        except json.JSONDecodeError:
            # If it's not valid JSON, treat as raw HTML and wrap in dictionary
            if plots_html.strip():  # Only if not empty
                processed_plots_html = {"chart": plots_html}
                processing_warnings.append("plots_html was not valid JSON, treated as raw HTML and wrapped with key 'chart'")
            else:
                processed_plots_html = None
                processing_warnings.append("plots_html was empty string, no charts will be included")
    elif isinstance(plots_html, dict):
        processed_plots_html = plots_html
        processing_warnings.append(f"Successfully processed plots_html dictionary with {len(plots_html)} chart(s)")
    elif plots_html is None:
        processing_warnings.append("No plots_html provided, report will contain only text content")
    else:
        processing_warnings.append(f"Unexpected plots_html type: {type(plots_html)}, no charts will be included")
        processed_plots_html = None

    try:
        wandb.init(
            entity=entity_name, project=project_name, job_type="mcp_report_creation"
        )

        # Initialize the report
        report = wr.Report(
            entity=entity_name,
            project=project_name,
            title=title,
            description=description or "",
            width="fluid",
        )

        # Log plots
        plots_dict = {}
        if processed_plots_html:
            for plot_name, html in processed_plots_html.items():
                wandb.log({plot_name: wandb.Html(html)})
                plots_dict[plot_name] = html

            pg = []
            for k, v in plots_dict.items():
                pg.append(
                    wr.PanelGrid(
                        panels=[
                            wr.MediaBrowser(
                                media_keys=[k],
                                num_columns=1,
                                layout=wr.Layout(w=20, h=20),  # , x=5, y=5)
                            ),
                        ]
                    )
                )
        else:
            pg = None

        blocks = parse_report_content_enhanced(markdown_report_text or "")

        # Add blocks if provided
        if pg:
            report.blocks = blocks + pg
        else:
            report.blocks = blocks

        logger.info(f"Report blocks: {report.blocks}")

        # Save the report
        report.save()
        wandb.finish()
        logger.info(f"Created report: {title}")
        
        return {
            "url": report.url,
            "processing_details": processing_warnings
        }

    except Exception as e:
        logger.error(f"Error creating report: {e}")
        # Include processing details in the error for better debugging
        error_msg = f"Error creating report: {e}"
        if processing_warnings:
            error_msg += f"\n\nProcessing details: {'; '.join(processing_warnings)}"
        raise Exception(error_msg)


def edit_report(
    report_url: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    blocks: Optional[List[Union[wr.H1, wr.H2, wr.H3, wr.PanelGrid]]] = None,
    append_blocks: bool = False,
) -> wr.Report:
    """
    Edit an existing W&B report.

    Args:
        report_url: The URL of the report to edit
        title: Optional new title for the report
        description: Optional new description for the report
        blocks: Optional list of blocks to update or append
        append_blocks: If True, append new blocks to existing ones. If False, replace existing blocks.
    """
    try:
        # Load the existing report
        report = wr.Report.from_url(report_url)

        # Update title if provided
        if title:
            report.title = title

        # Update description if provided
        if description:
            report.description = description

        # Update blocks if provided
        if blocks:
            if append_blocks:
                # Append new blocks to existing ones
                report.blocks = (report.blocks or []) + blocks
            else:
                # Replace existing blocks
                report.blocks = blocks

        # Save the changes
        report.save()

        logger.info(f"Updated report: {report.title}")
        return report

    except Exception as e:
        logger.error(f"Error editing report: {e}")
        raise


def parse_report_content_enhanced(
    text: str,
) -> List[Union[wr.H1, wr.H2, wr.H3, wr.P, wr.TableOfContents]]:
    """
    Parse markdown-like text into W&B report blocks with paragraph grouping.
    """
    blocks = []
    lines = text.strip().split("\n")

    current_paragraph = []

    for line in lines:
        # Check if this is a special line (header or TOC)
        h1_match = re.match(r"^# (.+)$", line)
        h2_match = re.match(r"^## (.+)$", line)
        h3_match = re.match(r"^### (.+)$", line)
        is_toc = line.strip().lower() == "[toc]"

        # If we hit a special line and have paragraph content, finalize the paragraph
        if (h1_match or h2_match or h3_match or is_toc) and current_paragraph:
            blocks.append(wr.P("\n".join(current_paragraph)))
            current_paragraph = []

        # Handle the current line
        if h1_match:
            blocks.append(wr.H1(h1_match.group(1)))
        elif h2_match:
            blocks.append(wr.H2(h2_match.group(1)))
        elif h3_match:
            blocks.append(wr.H3(h3_match.group(1)))
        elif is_toc:
            blocks.append(wr.TableOfContents())
        else:
            if line.strip():  # Only add non-empty lines
                current_paragraph.append(line)

    # Don't forget any remaining paragraph content
    if current_paragraph:
        blocks.append(wr.P("\n".join(current_paragraph)))

    return blocks
