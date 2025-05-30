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

def create_report(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: str = None,
    plots_html: Optional[Dict[str, str]] = None
) -> wr.Report:
    """
    Create a new Weights & Biases Report and add text and charts. Useful to save/document analysis and other findings.
    
    Args:
        entity_name: The W&B entity (team or username)
        project_name: The W&B project name
        title: Title of the W&B Report
        description: Optional description of the W&B Report
        blocks: Optional list of W&B Report blocks (headers, paragraphs and tables of contents etc.)
        plot_htmls: Optional dict of plot name and html string of any charts created as part of an analysis
    """
    try:
        wandb.init(
            entity=entity_name,
            project=project_name,
            job_type="mcp_report_creation"
        )
        
        # Initialize the report
        report = wr.Report(
            entity=entity_name,
            project=project_name,
            title=title,
            description=description or "",
            width='fluid'
        )

        # Log plots 
        plots_dict = {}
        if plots_html:
            for plot_name, html in plots_html.items():
                wandb.log({plot_name: wandb.Html(html)})
                plots_dict[plot_name] = html
            wandb.finish()

            pg = []
            for k,v in plots_dict.items():
                pg.append(wr.PanelGrid(
                    panels=[
                        wr.MediaBrowser(
                            media_keys=[k],
                            num_columns=1,
                            layout=wr.Layout(w=20, h=20) #, x=5, y=5)
                            ),
                    ]
                ))
        else:
            pg = None
        
        blocks = parse_report_content_enhanced(markdown_report_text)

        # Add blocks if provided
        if pg:
            report.blocks = blocks + pg
        else:
            report.blocks = blocks

        logger.info(f"Report blocks: {report.blocks}")
            
        # Save the report
        report.save()
        logger.info(f"Created report: {title}")
        return report.url
        
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        raise

def edit_report(
    report_url: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    blocks: Optional[List[Union[wr.H1, wr.H2, wr.H3, wr.PanelGrid]]] = None,
    append_blocks: bool = False
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

def parse_report_content_enhanced(text: str) -> List[Union[wr.H1, wr.H2, wr.H3, wr.P, wr.TableOfContents]]:
    """
    Parse markdown-like text into W&B report blocks with paragraph grouping.
    """
    blocks = []
    lines = text.strip().split('\n')
    
    current_paragraph = []
    
    for line in lines:
        # Check if this is a special line (header or TOC)
        h1_match = re.match(r'^# (.+)$', line)
        h2_match = re.match(r'^## (.+)$', line)
        h3_match = re.match(r'^### (.+)$', line)
        is_toc = line.strip().lower() == '[toc]'
        
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
