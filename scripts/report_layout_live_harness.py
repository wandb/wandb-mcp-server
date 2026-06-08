#!/usr/bin/env python
"""Live harness for validating MCP report layout and custom Vega panels."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROJECT = "wandb-mcp-report-layout-live"
_SUMMARY_TABLE_KEYS = ("car_ap", "truck_ap")
_RENDERABLE_TABLE_KEYS = ("loss_curve", "accuracy_curve", "pr_curve_table")
_HISTORY_TABLE_KEY = "pr_curve"
_REPORT_TAG = "mcp-report-layout-live"
_DATA_MODES = ("api-shape", "renderable")


@dataclass(frozen=True)
class HarnessSettings:
    """Resolved live harness settings."""

    entity: str
    project: str
    api_key: str
    loaded_env_files: tuple[Path, ...]
    data_mode: str


@dataclass(frozen=True)
class HarnessResult:
    """Live harness output."""

    report_url: str
    run_ids: tuple[str, ...]
    entity: str
    project: str
    data_mode: str


def env_file_candidates(explicit_env_file: str | None = None) -> list[Path]:
    """Return `.env` files to load without exposing their contents."""
    candidates: list[Path] = []
    if explicit_env_file:
        candidates.append(Path(explicit_env_file).expanduser())
    candidates.extend(
        [
            Path.cwd() / ".env",
            _REPO_ROOT / ".env",
            _REPO_ROOT.parent / "WandBAgentFactory" / ".env",
            _REPO_ROOT.parent / "wandb-mcp-server-test" / ".env",
        ]
    )

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)
    return unique_candidates


def load_harness_env(explicit_env_file: str | None = None) -> tuple[Path, ...]:
    """Load local env files in a deterministic order.

    Existing shell variables win over `.env` values so callers can override
    secrets or test targets from the command line.
    """
    loaded: list[Path] = []
    for env_file in env_file_candidates(explicit_env_file):
        if env_file.exists():
            load_dotenv(env_file, override=False)
            loaded.append(env_file)
    return tuple(loaded)


def resolve_settings(
    *,
    explicit_env_file: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    data_mode: str | None = None,
) -> HarnessSettings:
    """Resolve live harness settings from args, `.env`, and W&B identity."""
    loaded_env_files = load_harness_env(explicit_env_file)
    api_key = os.getenv("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is required. Set it in the shell or in one of the loaded .env files.")

    resolved_entity = (
        entity
        or os.getenv("MCP_REPORT_TEST_ENTITY")
        or os.getenv("MCP_LOGS_WANDB_ENTITY")
        or os.getenv("WANDB_ENTITY")
        or _viewer_username(api_key)
    )
    if not resolved_entity:
        raise RuntimeError(
            "Could not resolve a W&B entity. Set MCP_REPORT_TEST_ENTITY, MCP_LOGS_WANDB_ENTITY, or WANDB_ENTITY."
        )

    resolved_project = (
        project
        or os.getenv("MCP_REPORT_TEST_PROJECT")
        or os.getenv("MCP_LOGS_WANDB_PROJECT")
        or os.getenv("WANDB_PROJECT")
        or _DEFAULT_PROJECT
    )
    resolved_data_mode = data_mode or os.getenv("MCP_REPORT_DATA_MODE") or "api-shape"
    if resolved_data_mode not in _DATA_MODES:
        raise RuntimeError(f"data_mode must be one of: {', '.join(_DATA_MODES)}")

    return HarnessSettings(
        entity=resolved_entity,
        project=resolved_project,
        api_key=api_key,
        loaded_env_files=loaded_env_files,
        data_mode=resolved_data_mode,
    )


def run_live_report_layout_verification(settings: HarnessSettings) -> HarnessResult:
    """Seed runs, create a report layout, reload it, and assert viewspec shape."""
    import wandb
    import wandb_workspaces.reports.v2 as wr

    from wandb_mcp_server.api_client import WandBApiManager
    from wandb_mcp_server.mcp_tools.create_report import create_report

    os.environ["WANDB_API_KEY"] = settings.api_key
    WandBApiManager.set_context_api_key(settings.api_key)

    run_ids = tuple(_seed_runs(settings.entity, settings.project))
    for run_id in run_ids:
        _wait_for_summary_tables(settings.entity, settings.project, run_id)

    report = create_report(
        entity_name=settings.entity,
        project_name=settings.project,
        title=f"MCP report layout {settings.data_mode} verification {int(time.time())}",
        description="Live MCP report layout harness verification.",
        markdown_report_text="# MCP report layout live verification\n\n[TOC]",
        panels=_report_layout_panels(run_ids, data_mode=settings.data_mode),
    )
    report_url = report["url"]

    report_model = wr.Report.from_url(report_url, as_model=True)
    _assert_report_model(report_model, run_ids, data_mode=settings.data_mode)

    wandb.termlog(f"MCP report layout harness URL: {report_url}")
    return HarnessResult(
        report_url=report_url,
        run_ids=run_ids,
        entity=settings.entity,
        project=settings.project,
        data_mode=settings.data_mode,
    )


def _viewer_username(api_key: str) -> str:
    import wandb

    api = wandb.Api(api_key=api_key)
    viewer = api.viewer
    return getattr(viewer, "username", "") or getattr(viewer, "entity", "")


def _seed_runs(entity: str, project: str) -> list[str]:
    import wandb

    run_ids: list[str] = []
    for idx, model_name in enumerate(("baseline", "candidate")):
        with wandb.init(
            entity=entity,
            project=project,
            name=f"{_REPORT_TAG}-{model_name}-{int(time.time())}",
            job_type="mcp-report-layout-live",
            tags=[_REPORT_TAG, model_name],
            config={"model_name": model_name, "seed": idx},
        ) as run:
            run.log(
                {
                    "score": 0.71 + idx * 0.08,
                    "loss": 0.42 - idx * 0.05,
                    "car_ap": _ap_table("car", idx),
                    "truck_ap": _ap_table("truck", idx),
                    _HISTORY_TABLE_KEY: _pr_curve_table(idx),
                    "loss_curve": _loss_curve_table(idx),
                    "accuracy_curve": _accuracy_curve_table(idx),
                    "pr_curve_table": _renderable_pr_curve_table(idx),
                }
            )
            run_ids.append(run.id)
    return run_ids


def _ap_table(label: str, run_offset: int):
    import wandb

    return wandb.Table(
        columns=["threshold", "ap", "class"],
        data=[
            [0.25, 0.70 + run_offset * 0.05, label],
            [0.50, 0.77 + run_offset * 0.05, label],
            [0.75, 0.74 + run_offset * 0.05, label],
        ],
    )


def _pr_curve_table(run_offset: int):
    import wandb

    return wandb.Table(
        columns=["r", "p", "c"],
        data=[
            [0.0, 1.00, f"class_{run_offset}"],
            [0.5, 0.84 + run_offset * 0.04, f"class_{run_offset}"],
            [1.0, 0.61 + run_offset * 0.04, f"class_{run_offset}"],
        ],
    )


def _loss_curve_table(run_offset: int):
    import wandb

    return wandb.Table(
        columns=["step", "loss"],
        data=[[step, round(0.95 / (step + 1) + run_offset * 0.04, 4)] for step in range(12)],
    )


def _accuracy_curve_table(run_offset: int):
    import wandb

    return wandb.Table(
        columns=["step", "accuracy"],
        data=[[step, round(0.55 + step * 0.035 + run_offset * 0.05, 4)] for step in range(12)],
    )


def _renderable_pr_curve_table(run_offset: int):
    import wandb

    return wandb.Table(
        columns=["recall", "precision"],
        data=[[round(step / 10, 2), round(1.0 - step * 0.035 + run_offset * 0.02, 4)] for step in range(11)],
    )


def _wait_for_summary_tables(
    entity: str,
    project: str,
    run_id: str,
    timeout_s: int = 90,
) -> None:
    import wandb

    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    deadline = time.monotonic() + timeout_s
    expected_keys = set(_SUMMARY_TABLE_KEYS) | set(_RENDERABLE_TABLE_KEYS) | {_HISTORY_TABLE_KEY}
    last_summary: dict[str, Any] = {}

    while time.monotonic() < deadline:
        api_run = api.run(f"{entity}/{project}/{run_id}")
        last_summary = dict(api_run.summary)
        if expected_keys.issubset(last_summary):
            return
        time.sleep(2)

    missing = sorted(expected_keys - set(last_summary))
    raise AssertionError(f"Timed out waiting for table keys on run {run_id}. Missing: {missing}")


def _report_layout_panels(
    run_ids: tuple[str, ...],
    *,
    data_mode: str = "api-shape",
) -> list[dict[str, Any]]:
    if data_mode == "renderable":
        return _renderable_report_layout_panels(run_ids)
    return [
        {"type": "heading", "level": 2, "text": "Average precision by class"},
        {
            "type": "markdown",
            "text": "These custom Vega panels share one deterministic runset.",
        },
        {
            "type": "panel_grid",
            "run_ids": list(run_ids),
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
        {"type": "heading", "level": 2, "text": "Precision-recall curve"},
        {
            "type": "markdown",
            "text": "This panel uses a historyTable query for run-history data.",
        },
        {
            "type": "panel_grid",
            "run_ids": [run_ids[0]],
            "hide_run_sets": True,
            "panels": [
                {
                    "type": "custom_chart",
                    "query": {"historyTable": {"tableKey": _HISTORY_TABLE_KEY}},
                    "chart_name": "wandb/line/v0",
                    "chart_fields": {"x": "r", "y": "p", "color": "c"},
                    "chart_strings": {"title": "Precision-Recall Curve"},
                }
            ],
        },
    ]


def _renderable_report_layout_panels(run_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"type": "heading", "level": 2, "text": "Renderable metric curves"},
        {
            "type": "markdown",
            "text": "These built-in W&B line charts use known-good summary tables.",
        },
        {
            "type": "panel_grid",
            "run_ids": list(run_ids),
            "hide_run_sets": False,
            "panels": [
                {
                    "type": "custom_chart_table",
                    "table_name": "loss_curve",
                    "chart_name": "wandb/line/v0",
                    "chart_fields": {"x": "step", "y": "loss"},
                    "chart_strings": {"title": "Loss curve"},
                },
                {
                    "type": "custom_chart_table",
                    "table_name": "accuracy_curve",
                    "chart_name": "wandb/line/v0",
                    "chart_fields": {"x": "step", "y": "accuracy"},
                    "chart_strings": {"title": "Accuracy curve"},
                },
            ],
        },
        {"type": "heading", "level": 2, "text": "Renderable PR curve"},
        {
            "type": "markdown",
            "text": "This built-in W&B line chart uses a summary-table PR curve.",
        },
        {
            "type": "panel_grid",
            "run_ids": [run_ids[0]],
            "hide_run_sets": True,
            "panels": [
                {
                    "type": "custom_chart_table",
                    "table_name": "pr_curve_table",
                    "chart_name": "wandb/line/v0",
                    "chart_fields": {"x": "recall", "y": "precision"},
                    "chart_strings": {"title": "Precision-Recall Curve"},
                }
            ],
        },
    ]


def _assert_report_model(
    report_model: Any,
    run_ids: tuple[str, ...],
    *,
    data_mode: str,
) -> None:
    report_json = report_model.model_dump_json(by_alias=True, exclude_none=True)
    assert '"text":"Charts"' not in report_json
    if data_mode == "renderable":
        assert "Renderable metric curves" in report_json
        assert "Renderable PR curve" in report_json
    else:
        assert "Average precision by class" in report_json
        assert "Precision-recall curve" in report_json

    panel_grids = [block for block in report_model.spec.blocks if getattr(block, "type", None) == "panel-grid"]
    assert len(panel_grids) == 2

    first_grid, second_grid = panel_grids
    assert first_grid.metadata.hide_run_sets is False
    assert second_grid.metadata.hide_run_sets is True
    _assert_run_filter(first_grid.metadata.run_sets[0], list(run_ids))
    _assert_run_filter(second_grid.metadata.run_sets[0], [run_ids[0]])

    if data_mode == "renderable":
        _assert_renderable_panel_grids(first_grid, second_grid)
    else:
        _assert_api_shape_panel_grids(first_grid, second_grid)


def _assert_api_shape_panel_grids(first_grid: Any, second_grid: Any) -> None:
    first_panels = first_grid.metadata.panel_bank_section_config.panels
    assert len(first_panels) == 2
    assert [panel.config.panel_def_id for panel in first_panels] == [
        "cruise/bar_chart/v2",
        "cruise/bar_chart/v2",
    ]
    assert [panel.config.field_settings for panel in first_panels] == [
        {"x": "threshold", "y": "ap"},
        {"x": "threshold", "y": "ap"},
    ]
    assert [panel.config.string_settings["title"] for panel in first_panels] == [
        "CAR AP",
        "TRUCK AP",
    ]
    first_panel_json = first_grid.model_dump_json(by_alias=True, exclude_none=True)
    assert "summaryTable" in first_panel_json
    assert "car_ap" in first_panel_json
    assert "truck_ap" in first_panel_json

    second_panels = second_grid.metadata.panel_bank_section_config.panels
    assert len(second_panels) == 1
    pr_panel = second_panels[0]
    assert pr_panel.config.panel_def_id == "wandb/line/v0"
    assert pr_panel.config.field_settings == {"x": "r", "y": "p", "color": "c"}
    assert pr_panel.config.string_settings["title"] == "Precision-Recall Curve"
    second_panel_json = second_grid.model_dump_json(
        by_alias=True,
        exclude_none=True,
    )
    assert "historyTable" in second_panel_json
    assert _HISTORY_TABLE_KEY in second_panel_json


def _assert_renderable_panel_grids(first_grid: Any, second_grid: Any) -> None:
    first_panels = first_grid.metadata.panel_bank_section_config.panels
    assert len(first_panels) == 2
    assert [panel.config.panel_def_id for panel in first_panels] == [
        "wandb/line/v0",
        "wandb/line/v0",
    ]
    assert [panel.config.field_settings for panel in first_panels] == [
        {"x": "step", "y": "loss"},
        {"x": "step", "y": "accuracy"},
    ]
    assert [panel.config.string_settings["title"] for panel in first_panels] == [
        "Loss curve",
        "Accuracy curve",
    ]
    first_panel_json = first_grid.model_dump_json(by_alias=True, exclude_none=True)
    assert "summaryTable" in first_panel_json
    assert "loss_curve" in first_panel_json
    assert "accuracy_curve" in first_panel_json

    second_panels = second_grid.metadata.panel_bank_section_config.panels
    assert len(second_panels) == 1
    pr_panel = second_panels[0]
    assert pr_panel.config.panel_def_id == "wandb/line/v0"
    assert pr_panel.config.field_settings == {"x": "recall", "y": "precision"}
    assert pr_panel.config.string_settings["title"] == "Precision-Recall Curve"
    second_panel_json = second_grid.model_dump_json(
        by_alias=True,
        exclude_none=True,
    )
    assert "summaryTable" in second_panel_json
    assert "pr_curve_table" in second_panel_json


def _assert_run_filter(runset: Any, expected_run_ids: list[str]) -> None:
    assert runset.search.query == ""
    nodes = list(_iter_filter_nodes(runset.filters))
    expected_op = "=" if len(expected_run_ids) == 1 else "IN"
    expected_value: str | list[str] = expected_run_ids[0] if len(expected_run_ids) == 1 else expected_run_ids
    assert any(
        node.get("key") == {"section": "run", "name": "name"}
        and node.get("op") == expected_op
        and node.get("value") == expected_value
        for node in nodes
    )


def _iter_filter_nodes(node: Any):
    if isinstance(node, dict):
        yield node
        for child in node.get("filters", []):
            yield from _iter_filter_nodes(child)
        return

    filters = getattr(node, "filters", None)
    if filters:
        for child in filters:
            yield from _iter_filter_nodes(child)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Optional explicit .env file path.")
    parser.add_argument("--entity", help="W&B entity for live verification.")
    parser.add_argument("--project", help="W&B project for live verification.")
    parser.add_argument(
        "--data-mode",
        choices=_DATA_MODES,
        default=None,
        help=(
            "api-shape validates customer-style custom query shapes; "
            "renderable uses known-good built-in W&B line charts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = resolve_settings(
        explicit_env_file=args.env_file,
        entity=args.entity,
        project=args.project,
        data_mode=args.data_mode,
    )
    loaded = ", ".join(str(path) for path in settings.loaded_env_files) or "none"
    print(f"Loaded env files: {loaded}")
    print(f"Using W&B target: {settings.entity}/{settings.project}")
    print(f"Using data mode: {settings.data_mode}")
    result = run_live_report_layout_verification(settings)
    print(f"Report URL: {result.report_url}")
    print(f"Seeded run IDs: {', '.join(result.run_ids)}")


if __name__ == "__main__":
    main()
