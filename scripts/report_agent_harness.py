#!/usr/bin/env python3
"""Seed a complex W&B project and validate agent-style report creation.

This is a live harness for WB-34806. It creates a synthetic project with
multiple runs, metric histories, and table-backed AP curves, then uses a small
deterministic "agent" to emit the same `create_wandb_report_tool` payload shape
an MCP client should use for a multi-section report.

Requirements:
    WANDB_API_KEY must be set in the environment or in `.env`.

Usage:
    uv run python scripts/report_agent_harness.py
    uv run python scripts/report_agent_harness.py --env-file ../wandb-mcp-server/.env
    uv run python scripts/report_agent_harness.py --panel-def-id cruise/bar_chart/v2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
from dotenv import load_dotenv

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.create_report import create_report

OBJECT_CLASSES = ("CAR", "TRUCK", "MOTORCYCLE")
TABLE_KEYS = {
    "CAR": "car_ap_curve",
    "TRUCK": "truck_ap_curve",
    "MOTORCYCLE": "motorcycle_ap_curve",
}


@dataclass(frozen=True)
class HarnessConfig:
    """Runtime configuration for the live report harness."""

    entity: str
    project: str
    run_count: int
    panel_def_id: str
    timeout_s: int
    stamp: str


@dataclass(frozen=True)
class SeededRun:
    """Synthetic run metadata needed by the agent harness."""

    run_id: str
    display_name: str
    model_family: str


class BasicReportAgent:
    """Deterministic agent that emits a complex MCP report layout payload."""

    def __init__(self, *, panel_def_id: str) -> None:
        self.panel_def_id = panel_def_id

    def build_panels(self, seeded_runs: list[SeededRun]) -> list[dict[str, Any]]:
        """Build ordered report layout blocks from seeded run metadata."""
        run_ids = [run.run_id for run in seeded_runs]
        return [
            {
                "type": "heading",
                "level": 2,
                "text": "Scenario summary",
            },
            {
                "type": "markdown",
                "text": (
                    "This synthetic report mimics an agent assembling a customer evaluation report "
                    "with narrative text, shared runsets, native metric panels, and custom Vega panels."
                ),
            },
            {
                "type": "markdown_table",
                "title": "Seeded runs",
                "headers": ["Run ID", "Display name", "Model family"],
                "rows": [[run.run_id, run.display_name, run.model_family] for run in seeded_runs],
            },
            {
                "type": "heading",
                "level": 2,
                "text": "Training and evaluation metrics",
            },
            {
                "type": "markdown",
                "text": "These native panels share one deterministic runset filter across all seeded runs.",
            },
            {
                "type": "panel_grid",
                "run_ids": run_ids,
                "hide_run_sets": False,
                "panels": [
                    {
                        "type": "line",
                        "x": "_step",
                        "y": ["train/loss", "eval/mAP"],
                        "title": "Loss and mAP over steps",
                    },
                    {
                        "type": "bar",
                        "metrics": ["summary/final_map", "summary/p95_latency_ms"],
                        "title": "Final quality and latency",
                    },
                    {
                        "type": "scatter",
                        "x": "summary/p95_latency_ms",
                        "y": "summary/final_map",
                        "title": "Latency vs mAP",
                    },
                ],
            },
            {
                "type": "heading",
                "level": 2,
                "text": "Average precision by class",
            },
            {
                "type": "markdown",
                "text": "Each custom Vega panel reads a table key from the same shared runset.",
            },
            {
                "type": "panel_grid",
                "run_ids": run_ids,
                "hide_run_sets": False,
                "panels": [self._class_ap_panel(class_name) for class_name in OBJECT_CLASSES],
            },
        ]

    def _class_ap_panel(self, class_name: str) -> dict[str, Any]:
        """Build a table-backed custom chart panel for one object class."""
        return {
            "type": "custom_chart_table",
            "table_name": TABLE_KEYS[class_name],
            "chart_name": self.panel_def_id,
            "chart_fields": {"x": "threshold", "y": "ap"},
            "chart_strings": {"title": f"{class_name} AP by confidence threshold"},
        }


def _env_first(*names: str, default: str) -> str:
    """Return the first non-empty environment variable value."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _parse_args() -> HarnessConfig:
    """Parse CLI arguments and environment fallbacks."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--env-file",
        default=os.getenv("MCP_REPORT_HARNESS_ENV_FILE", ""),
        help="Optional .env file to load before reading W&B settings.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    load_dotenv()
    if pre_args.env_file and not load_dotenv(pre_args.env_file, override=True):
        pre_parser.error(f"Could not load --env-file {pre_args.env_file!r}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument(
        "--entity",
        default=_env_first(
            "MCP_REPORT_TEST_ENTITY",
            "MCP_LOGS_WANDB_ENTITY",
            "WANDB_ENTITY",
            default="wandb-applied-ai-team",
        ),
        help="W&B entity to seed.",
    )
    parser.add_argument(
        "--project",
        default=_env_first(
            "MCP_REPORT_HARNESS_PROJECT",
            "MCP_REPORT_TEST_PROJECT",
            "MCP_LOGS_WANDB_PROJECT",
            "WANDB_PROJECT",
            default="wandb-mcp-report-agent-harness",
        ),
        help="W&B project to seed.",
    )
    parser.add_argument("--run-count", type=int, default=3, help="Number of synthetic runs to create.")
    parser.add_argument(
        "--panel-def-id",
        default=os.getenv("MCP_REPORT_HARNESS_PANEL_DEF_ID", "wandb/line/v0"),
        help="Vega panelDefId to persist for custom charts.",
    )
    parser.add_argument("--timeout-s", type=int, default=90, help="Seconds to wait for table summaries.")
    args = parser.parse_args()
    if args.run_count < 2:
        parser.error("--run-count must be at least 2")
    return HarnessConfig(
        entity=args.entity,
        project=args.project,
        run_count=args.run_count,
        panel_def_id=args.panel_def_id,
        timeout_s=args.timeout_s,
        stamp=stamp,
    )


def _setup_wandb_context() -> str:
    """Load credentials for W&B SDK calls and MCP report creation."""
    api_key = os.getenv("WANDB_API_KEY", "")
    if not api_key:
        print("ERROR: WANDB_API_KEY is not set in the environment or .env.", file=sys.stderr)
        sys.exit(1)
    WandBApiManager.set_context_api_key(api_key)
    return api_key


def _seed_mock_project(config: HarnessConfig) -> list[SeededRun]:
    """Create synthetic runs with metrics and table-backed AP curves."""
    seeded_runs: list[SeededRun] = []
    model_families = ("baseline", "candidate", "experimental", "ablation")
    for run_index in range(config.run_count):
        model_family = model_families[run_index % len(model_families)]
        display_name = f"agent-harness-{model_family}-{config.stamp}"
        with wandb.init(
            entity=config.entity,
            project=config.project,
            name=display_name,
            job_type="mcp-report-agent-harness",
            tags=["mcp-generated", "report-layout-harness", config.stamp],
            config={
                "model_family": model_family,
                "dataset": "synthetic-urban-perception",
                "seed": 9000 + run_index,
            },
        ) as run:
            final_map = 0.64 + run_index * 0.045
            p95_latency = 42 - run_index * 3.5
            for step in range(12):
                progress = step / 11
                run.log(
                    {
                        "train/loss": round(1.25 - progress * (0.48 + run_index * 0.06), 4),
                        "eval/mAP": round(0.38 + progress * (final_map - 0.38), 4),
                        "summary/p95_latency_ms": round(p95_latency + (1 - progress) * 4.0, 4),
                    },
                    step=step,
                )

            for class_index, class_name in enumerate(OBJECT_CLASSES):
                run.log({TABLE_KEYS[class_name]: _ap_curve_table(run_index, class_index)})

            run.summary["summary/final_map"] = round(final_map, 4)
            run.summary["summary/p95_latency_ms"] = round(p95_latency, 4)
            run.summary["summary/report_harness_stamp"] = config.stamp
            seeded_runs.append(
                SeededRun(
                    run_id=run.id,
                    display_name=display_name,
                    model_family=model_family,
                )
            )
    return seeded_runs


def _ap_curve_table(run_index: int, class_index: int) -> wandb.Table:
    """Create one synthetic AP curve table."""
    rows = []
    class_name = OBJECT_CLASSES[class_index]
    for threshold_index in range(7):
        threshold = round(0.2 + threshold_index * 0.1, 2)
        ap = 0.48 + run_index * 0.05 + class_index * 0.025 + threshold_index * 0.018
        precision = 0.72 + run_index * 0.025 + threshold_index * 0.02
        recall = 0.92 - threshold_index * 0.055 + run_index * 0.01
        rows.append(
            [
                threshold,
                round(min(ap, 0.98), 4),
                round(min(precision, 0.99), 4),
                round(max(recall, 0.1), 4),
                class_name,
            ]
        )
    return wandb.Table(
        columns=["threshold", "ap", "precision", "recall", "class_name"],
        data=rows,
    )


def _wait_for_tables(api_key: str, config: HarnessConfig, seeded_runs: list[SeededRun]) -> None:
    """Wait until table-backed chart inputs are queryable from run summaries."""
    api = wandb.Api(api_key=api_key)
    deadline = time.monotonic() + config.timeout_s
    pending = {(run.run_id, table_key) for run in seeded_runs for table_key in TABLE_KEYS.values()}
    last_seen: dict[str, list[str]] = {}

    while pending and time.monotonic() < deadline:
        for run in seeded_runs:
            api_run = api.run(f"{config.entity}/{config.project}/{run.run_id}")
            summary = dict(api_run.summary)
            last_seen[run.run_id] = sorted(summary.keys())
            for table_key in TABLE_KEYS.values():
                table_ref = summary.get(table_key)
                if getattr(table_ref, "get", lambda *_: None)("_type") == "table-file":
                    pending.discard((run.run_id, table_key))
        if pending:
            time.sleep(2)

    if pending:
        raise AssertionError(
            f"Timed out waiting for table summaries. Pending={sorted(pending)} Last summary keys={last_seen}"
        )


def _create_agent_report(config: HarnessConfig, seeded_runs: list[SeededRun]) -> str:
    """Create a report using the same MCP implementation exposed to agents."""
    agent = BasicReportAgent(panel_def_id=config.panel_def_id)
    result = create_report(
        entity_name=config.entity,
        project_name=config.project,
        title=f"MCP agent report harness {config.stamp}",
        description="Synthetic validation report for structured MCP report assembly.",
        markdown_report_text=(
            "# MCP agent report harness\n\n"
            "[TOC]\n\n"
            "This report was generated by a deterministic local harness that mimics an agent using only MCP inputs."
        ),
        panels=agent.build_panels(seeded_runs),
    )
    return result["url"]


def _validate_report(report_url: str, config: HarnessConfig, seeded_runs: list[SeededRun]) -> None:
    """Reload the saved report and validate the Reports v2 structure."""
    report_model = wr.Report.from_url(report_url, as_model=True)
    panel_grids = [block for block in report_model.spec.blocks if getattr(block, "type", None) == "panel-grid"]
    if len(panel_grids) != 2:
        raise AssertionError(f"Expected 2 panel grids, found {len(panel_grids)}")

    run_ids = [run.run_id for run in seeded_runs]
    for index, grid in enumerate(panel_grids):
        if len(grid.metadata.run_sets) != 1:
            raise AssertionError(f"Grid {index} should have exactly one runset")
        runset_json = grid.metadata.run_sets[0].model_dump_json(by_alias=True, exclude_none=True)
        for run_id in run_ids:
            if run_id not in runset_json:
                raise AssertionError(f"Run ID {run_id} missing from grid {index} runset filter")
        if '"name"' not in runset_json:
            raise AssertionError(f"Grid {index} runset filter is not keyed by run name: {runset_json}")

    custom_grid = panel_grids[1]
    custom_panels = [
        panel
        for panel in custom_grid.metadata.panel_bank_section_config.panels
        if getattr(getattr(panel, "config", None), "panel_def_id", None)
    ]
    if len(custom_panels) != len(OBJECT_CLASSES):
        raise AssertionError(f"Expected {len(OBJECT_CLASSES)} custom panels, found {len(custom_panels)}")

    panel_def_ids = [panel.config.panel_def_id for panel in custom_panels]
    if panel_def_ids != [config.panel_def_id] * len(OBJECT_CLASSES):
        raise AssertionError(f"Unexpected panelDefIds: {panel_def_ids}")

    report_json = report_model.model_dump_json(by_alias=True, exclude_none=True)
    for table_key in TABLE_KEYS.values():
        if table_key not in report_json:
            raise AssertionError(f"Table key {table_key!r} missing from saved report viewspec")


def main() -> int:
    """Run the live harness and return a process exit code."""
    config = _parse_args()
    api_key = _setup_wandb_context()

    print(f"Seeding {config.run_count} runs in {config.entity}/{config.project}")
    seeded_runs = _seed_mock_project(config)
    print("Seeded run IDs:", ", ".join(run.run_id for run in seeded_runs))

    print("Waiting for AP curve tables to appear in run summaries")
    _wait_for_tables(api_key, config, seeded_runs)

    print("Creating report through create_report() with an agent-style layout payload")
    report_url = _create_agent_report(config, seeded_runs)
    print(f"Report URL: {report_url}")

    print("Reloading report and validating saved Reports v2 structure")
    _validate_report(report_url, config, seeded_runs)
    print("Harness validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
