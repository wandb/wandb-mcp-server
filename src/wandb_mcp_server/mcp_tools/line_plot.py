"""Bounded history projection for the offline MCP line-plot app."""

from __future__ import annotations

from importlib.resources import files
import json
import math
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.config import (
    MAX_RESPONSE_TOKENS,
    MCP_MAX_HISTORY_KEYS,
    MCP_MAX_HISTORY_RANGE_STEPS,
    MCP_MAX_HISTORY_SAMPLES,
    MCP_WORKLOAD_PROFILE,
)
from wandb_mcp_server.error_diagnostics import ToolInputValidationError
from wandb_mcp_server.mcp_tools.run_history import get_run_history
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.wandb_selective_reads import MAX_SAFE_HISTORY_STEP
from wandb_mcp_server.wandb_urls import public_wandb_url

RESOURCE_URI = "ui://wandb/line-plot.html"
TOOL_NAME = "render_wandb_line_plot_tool"
LINE_PLOT_DESCRIPTION = """Render W&B run metrics as an interactive inline line plot with an Open in W&B link.

Reads 1–10 runs from one entity/project and 1–5 explicit numeric metric keys.
Supports legend toggles, hover values, drag zoom, double-click reset and light/dark
themes in MCP Apps hosts. Other hosts receive a text summary with per-run,
per-metric point counts, first/last/min/max values and W&B links.

x_axis defaults to _step; _timestamp and logged numeric custom axes are supported.
samples is a per-run merged-row budget (default 500), clamped to the deployment's
history sample cap. Sparse metrics are outer-unioned by step; custom-axis and
metric values must share a logged row. Non-finite/non-numeric values are omitted.
Optional min_step/max_step are inclusive non-negative history step bounds, subject
to deployment range caps; managed profiles require both bounds. Step-axis zoom
can requery bounded windows; other zooms are local. Sampling and truncation flags
describe coverage, not exhaustive history. Runs are read sequentially; individual
failures are reported without discarding successful runs. This tool is read-only.
"""


def register_line_plot_resource(mcp: FastMCP) -> None:
    ui: dict[str, Any] = {"csp": {"connectDomains": [], "resourceDomains": []}}
    domain = os.environ.get("WANDB_MCP_APP_DOMAIN")
    if domain:
        ui["domain"] = domain

    @mcp.resource(RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def line_plot_html() -> str:
        return files("wandb_mcp_server").joinpath("data/line-plot.html").read_text(encoding="utf-8")


def _text(name: str, value: Any, limit: int = 512) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ToolInputValidationError(f"{name} must be a non-empty string of at most {limit} characters", field=name)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ToolInputValidationError(f"{name} must contain valid UTF-8 text", field=name) from None


def _validate(
    entity: str,
    project: str,
    run_ids: list[str],
    keys: list[str],
    x_axis: str,
    samples: int,
    min_step: int | None,
    max_step: int | None,
    title: str | None,
) -> None:
    for name, value in (("entity_name", entity), ("project_name", project), ("x_axis", x_axis)):
        _text(name, value)
    for name, values, cap in (("run_ids", run_ids, 10), ("keys", keys, min(5, MCP_MAX_HISTORY_KEYS))):
        if not isinstance(values, list) or not 1 <= len(values) <= cap:
            raise ToolInputValidationError(f"{name} must contain 1–{cap} entries", field=name)
        for value in values:
            _text(name, value)
        if len(set(values)) != len(values):
            raise ToolInputValidationError(f"{name} must contain unique entries", field=name)
    if title is not None:
        _text("title", title, 200)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ToolInputValidationError("samples must be a positive integer", field="samples")
    for name, value in (("min_step", min_step), ("max_step", max_step)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_HISTORY_STEP
        ):
            raise ToolInputValidationError(f"{name} must be a non-negative safe history step", field=name)
    if min_step is not None and max_step is not None:
        if max_step < min_step or max_step - min_step + 1 > MCP_MAX_HISTORY_RANGE_STEPS:
            raise ToolInputValidationError(
                f"Inclusive step range must be ordered and at most {MCP_MAX_HISTORY_RANGE_STEPS} steps",
                field="max_step",
            )
    elif MCP_WORKLOAD_PROFILE != "local" and (min_step is not None or max_step is not None):
        raise ToolInputValidationError("Managed history ranges require both min_step and max_step", field="min_step")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _summary(payload: dict[str, Any]) -> str:
    lines = [payload["title"], f"Open in W&B: {payload['url']}"]
    for run in payload["runs"]:
        lines.append(f"{run['name']}: {run['url']}")
        if "error" in run:
            lines.append(f"  {run['error']}")
            continue
        for key, points in run["series"].items():
            values = [point[1] for point in points]
            stats = (
                f"first={values[0]:.6g}, last={values[-1]:.6g}, min={min(values):.6g}, max={max(values):.6g}"
                if values
                else "no finite numeric points"
            )
            lines.append(f"  {key}: {len(points)} points; {stats}")
        lines.append(f"  sampled={run['sampled']}, truncated={run['truncated']}")
    return "\n".join(lines)


def _result(payload: dict[str, Any]) -> CallToolResult:
    while True:
        summary = _summary(payload)
        serialized = json.dumps(
            {"content": summary, "structuredContent": payload}, allow_nan=False, separators=(",", ":")
        )
        if count_tokens_conservative(serialized) <= MAX_RESPONSE_TOKENS:
            return CallToolResult(
                content=[TextContent(type="text", text=summary)],
                structuredContent=payload,
                isError=all("error" in run for run in payload["runs"]),
            )
        reduced = False
        for run in payload["runs"]:
            for key, points in run.get("series", {}).items():
                if points:
                    run["series"][key] = (
                        points[::2] + (points[-1:] if len(points) % 2 == 0 else []) if len(points) > 2 else []
                    )
                    run["truncated"] = True
                    payload["truncated"] = True
                    reduced = True
        if not reduced:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="Line plot exceeds the response budget; request fewer runs or keys.")
                ],
                structuredContent={"error": "response_too_large"},
                isError=True,
            )


def render_line_plot(
    entity_name: str,
    project_name: str,
    run_ids: list[str],
    keys: list[str],
    x_axis: str = "_step",
    samples: int = 500,
    min_step: int | None = None,
    max_step: int | None = None,
    title: str | None = None,
) -> CallToolResult:
    try:
        _validate(entity_name, project_name, run_ids, keys, x_axis, samples, min_step, max_step, title)
    except ToolInputValidationError as exc:
        payload = {"error": "invalid_input", "message": str(exc), "field": exc.diagnostic_field}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))], structuredContent=payload, isError=True
        )
    arguments = dict(
        entity_name=entity_name,
        project_name=project_name,
        run_ids=run_ids,
        keys=keys,
        x_axis=x_axis,
        samples=min(samples, MCP_MAX_HISTORY_SAMPLES),
        min_step=min_step,
        max_step=max_step,
        title=title,
    )
    payload: dict[str, Any] = {
        "title": title or ", ".join(keys),
        "url": public_wandb_url(entity_name, project_name),
        "x_axis": x_axis,
        "arguments": arguments,
        "range_cap": MCP_MAX_HISTORY_RANGE_STEPS,
        "truncated": False,
        "runs": [],
    }
    for run_id in run_ids:
        run: dict[str, Any] = {
            "id": run_id,
            "name": run_id,
            "url": public_wandb_url(entity_name, project_name, "runs", run_id),
        }
        try:
            raise_if_tool_deadline_exceeded()
            history = json.loads(
                get_run_history(
                    entity_name=entity_name,
                    project_name=project_name,
                    run_id=run_id,
                    keys=keys,
                    x_axis=x_axis,
                    samples=arguments["samples"],
                    min_step=min_step,
                    max_step=max_step,
                )
            )
            if not isinstance(history, dict) or "error" in history or not isinstance(history.get("rows"), list):
                raise ValueError("History unavailable")
            run["name"] = str(history.get("run_name") or run_id)[:512]
            run["series"] = {
                key: sorted(
                    [
                        [row[x_axis], row[key]]
                        for row in history["rows"]
                        if isinstance(row, dict) and _finite(row.get(x_axis)) and _finite(row.get(key))
                    ],
                    key=lambda point: point[0],
                )
                for key in keys
            }
            run["sampled"] = bool(history.get("sampled", True))
            run["truncated"] = bool(
                history.get("truncated") or history.get("source_truncated") or history.get("source_values_truncated")
            )
            payload["truncated"] |= run["truncated"]
        except Exception:
            run["error"] = "history_fetch_failed"
        payload["runs"].append(run)
    return _result(payload)
