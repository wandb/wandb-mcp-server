"""Native panel configuration projected into bounded JSON-safe Graph series."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from importlib.resources import files
import json
import math
import os
import re
import statistics
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from wandb_mcp_server.admission import ToolDeadlineExceeded, raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import WandBApiManager, WandBServerBusy, raise_for_wandb_server_busy
from wandb_mcp_server.config import (
    MAX_RESPONSE_TOKENS,
    MCP_MAX_HISTORY_RANGE_STEPS,
    MCP_MAX_HISTORY_SAMPLES,
    MCP_WORKLOAD_PROFILE,
)
from wandb_mcp_server.error_diagnostics import ToolInputValidationError
from wandb_mcp_server.mcp_tools.line_plot_smoothing import smooth
from wandb_mcp_server.mcp_tools.run_history import get_run_history
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.wandb_selective_reads import (
    MAX_SAFE_HISTORY_STEP,
    SelectiveReadUnavailable,
    fetch_projected_run,
    fetch_projected_runs,
    fetch_viewer_default_entity,
)
from wandb_mcp_server.wandb_urls import public_wandb_url

RESOURCE_URI = "ui://wandb/line-plot.html"
TOOL_NAME = "render_wandb_line_plot_tool"
ACTIVE_RUN_STATES = frozenset({"running", "pending", "preempting"})
LINE_PLOT_DESCRIPTION = """Render W&B run metrics as an interactive MCP App line chart (`ui://wandb/line-plot.html`) plus compact summary stats.

Quick 1-call mode or full `RunsLinePlotConfig`:
- `project_name` is required. `entity_name` is optional (resolves default W&B entity when omitted).
- `run_ids` is optional (1–10 unique run IDs; when omitted, auto-selects newest runs logging the metric(s), capped by `limit`).
- Provide `metric` (1-metric shorthand, e.g. `"train/loss"`) or camelCase `config` matching W&B report spec panels (`query_wandb_tool` `include=["spec"]`) with `config.metrics` (1–5 unique keys).

Supported `config`: `chartTitle`, `metrics`, `xAxis` (`_step` default, `_timestamp`, `_runtime`, or numeric key), `xAxisMin/Max`, `yAxisMin/Max`, `xLogScale`, `yLogScale`, `xAxisTitle`, `yAxisTitle`, `smoothingWeight`, `smoothingType` (`none`, `exponential`, `exponentialTimeWeighted`, `gaussian`, `average`), `showOriginalAfterSmoothing`, `groupBy` (`group` or `config.<key>`/`<key>.value`), `aggregate`, `groupAgg` (`mean`, `min`, `max`, `median`, `sum`), `groupArea` (`minmax`, `stddev`, `stderr`, `none`), `legendPosition`, `showLegend`, `limit` (`1`–`10`), `overrideColors`, `overrideSeriesTitles`, `overrideLineWidths`, `overrideMarks`, `plotType="line"`. Unsupported data-changing panel options return `unsupported_config`; cosmetic-only options are listed in `coverage.ignoredConfig`.

When to use: call this tool when the user wants a visual plot or live chart; use `get_run_history_tool` or `compare_runs_tool` for exact numerical inspection without a chart. Interactive hosts support fullscreen/PiP expand, outward `_step` zoom, smoothing/log-Y/isolate controls, 10s active-run live refresh, and `ui/update-model-context` (`[MCP Line Plot Context]` in follow-up turns carries the open chart's `entity_name`, `project_name`, `run_ids`, and `visibleRange`).
"""
PALETTE = ["#00a99d", "#e87535", "#5c73d5", "#d85b8c", "#99a638", "#8e62bd", "#c49a24", "#359bc3", "#cc6657", "#6f9e73"]
SUPPORTED = set(
    """chartTitle metrics xAxis xAxisMin xAxisMax yAxisMin yAxisMax xLogScale yLogScale xAxisTitle yAxisTitle
smoothingWeight smoothingType showOriginalAfterSmoothing groupBy groupAgg groupArea aggregate legendPosition showLegend
limit overrideColors overrideSeriesTitles overrideLineWidths overrideMarks plotType""".split()
)
IGNORED = set(
    """fontSize legendFields legendTemplate colorRunNames highlightedCompanionRunOnly displayFullRunName
showXRange coordinateZooming singlePointRenderMode forceReactVis tooltipNumberOfRuns""".split()
)
DATA_OPTIONS = set(
    """expressions metricRegex useMetricRegex xExpression fullFidelityMode groupedFullFidelityMode
pointVisualizationMethod startingXAxis excludeOutliers metricLimit groupRunsLimit groupingDepth stabilizeRunGroups
isGrouped useRunsTableGroupingInPanels windowing aggregateMetrics colorEachMetricDifferently metricRegexMaxNumMatches
isMergedPanel useLocalSmoothing useGlobalSmoothingWeight deltaQueryTypeOverride xAxisFormat""".split()
)
MARKS = {"solid", "dashed", "dotted", "dotdash", "dotdotdash", "points"}


def register_line_plot_resource(mcp: FastMCP) -> None:
    if not hasattr(mcp, "resource"):
        return
    ui: dict[str, Any] = {
        "prefersBorder": True,
        "csp": {"connectDomains": [], "resourceDomains": ["data:"]},
    }
    if domain := os.environ.get("WANDB_MCP_APP_DOMAIN"):
        ui["domain"] = domain

    @mcp.resource(RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def line_plot_html() -> str:
        return files("wandb_mcp_server").joinpath("data/line-plot.html").read_text(encoding="utf-8")


def _invalid(message: str) -> None:
    raise ToolInputValidationError(message, field="config")


def _text(name: str, value: Any, limit: int = 512) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _invalid(f"{name} must be non-empty text of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _invalid(f"{name} must contain valid UTF-8")


def _finite(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _group_key(value: str) -> str | None:
    if value == "group":
        return None
    if value.startswith("config."):
        value = value[7:]
    elif ".value" not in value:
        _invalid("groupBy must be group or a config.key/key.value path")
    value = value.replace(".value", "", 1)
    if not value or any(not p for p in value.split(".")):
        _invalid("Invalid groupBy path")
    return value


def _normalize_config(
    config: dict[str, Any] | None,
    *,
    metric: str | None,
    limit: int | None,
) -> dict[str, Any]:
    if metric is not None:
        _text("metric", metric)
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 10):
        _invalid("limit must be an integer from 1 to 10")
    if config is None:
        if metric is None:
            _invalid("Provide metric or config.metrics")
        normalized: dict[str, Any] = {"metrics": [metric]}
    else:
        if not isinstance(config, dict) or any(not isinstance(k, str) for k in config):
            _invalid("config must be a native panel config object")
        try:
            if len(json.dumps(config, allow_nan=False).encode("utf-8")) > 32768:
                _invalid("config exceeds 32 KiB")
        except (TypeError, ValueError, UnicodeEncodeError):
            _invalid("config must be finite JSON")
        normalized = deepcopy(config)
        if metric is not None:
            if "metrics" in normalized and normalized["metrics"] != [metric]:
                _invalid("Pass either metric or config.metrics, not conflicting values")
            normalized["metrics"] = [metric]
    if limit is not None:
        if "limit" in normalized and normalized["limit"] != limit:
            _invalid("Pass either limit or config.limit, not conflicting values")
        normalized["limit"] = limit
    return normalized


def _validate(
    entity: str | None,
    project: str,
    run_ids: list[str] | None,
    config: dict[str, Any] | None,
    samples: int,
    min_step: int | None,
    max_step: int | None,
    *,
    metric: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    if entity is not None:
        _text("entity_name", entity)
    _text("project_name", project)
    raw_config = _normalize_config(config, metric=metric, limit=limit)
    unknown = set(raw_config) - SUPPORTED - IGNORED - DATA_OPTIONS
    if unknown:
        _invalid("Unknown config fields: " + ", ".join(sorted(unknown)))
    if run_ids is not None:
        if not isinstance(run_ids, list) or not 1 <= len(run_ids) <= 10:
            _invalid("run_ids must contain 1–10 entries")
        for value in run_ids:
            _text("run_ids", value)
        if len(set(run_ids)) != len(run_ids):
            _invalid("run_ids must contain unique entries")
    metrics = raw_config.get("metrics")
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= 5:
        _invalid("metrics must contain 1–5 entries")
    for value in metrics:
        _text("metrics", value)
    if len(set(metrics)) != len(metrics):
        _invalid("metrics must contain unique entries")
    if type(samples) is not int or samples < 1:
        _invalid("samples must be a positive integer")
    for name, value in (("min_step", min_step), ("max_step", max_step)):
        if value is not None and (type(value) is not int or not 0 <= value <= MAX_SAFE_HISTORY_STEP):
            _invalid(f"{name} must be a non-negative safe history step")
    if min_step is not None and max_step is not None:
        if max_step < min_step or max_step - min_step + 1 > MCP_MAX_HISTORY_RANGE_STEPS:
            _invalid(f"Inclusive range must be ordered and at most {MCP_MAX_HISTORY_RANGE_STEPS} steps")
    elif MCP_WORKLOAD_PROFILE != "local" and (min_step is not None or max_step is not None):
        _invalid("Managed history ranges require both min_step and max_step")
    cfg: dict[str, Any] = {
        "xAxis": "_step",
        "smoothingWeight": 0,
        "smoothingType": "exponential",
        "groupAgg": "mean",
        "groupArea": "minmax",
        "plotType": "line",
        "showLegend": True,
        "legendPosition": "north",
        **deepcopy(raw_config),
    }
    for key in ("xAxis", "chartTitle", "xAxisTitle", "yAxisTitle", "groupBy"):
        if key in cfg and not (key != "xAxis" and cfg[key] == ""):
            _text(key, cfg[key], 200 if key == "chartTitle" else 512)
    for key in ("xAxisMin", "xAxisMax", "yAxisMin", "yAxisMax"):
        if cfg.get(key) is not None and not _finite(cfg[key]):
            _invalid(f"{key} must be a finite number or null")
    for key in ("xLogScale", "yLogScale", "showOriginalAfterSmoothing", "aggregate", "showLegend"):
        if key in cfg and type(cfg[key]) is not bool:
            _invalid(f"{key} must be boolean")
    for axis in ("x", "y"):
        lo, hi = cfg.get(axis + "AxisMin"), cfg.get(axis + "AxisMax")
        if lo is not None and hi is not None and lo >= hi:
            _invalid("Display domains must be increasing")
        if cfg.get(axis + "LogScale") and any(v is not None and v <= 0 for v in (lo, hi)):
            _invalid("Log display domains must be positive")
    for key in ("smoothingType", "groupAgg", "groupArea", "legendPosition", "plotType"):
        if not isinstance(cfg[key], str):
            _invalid(f"{key} must be a string")
    if cfg["smoothingType"] not in {"none", "exponential", "exponentialTimeWeighted", "gaussian", "average"}:
        _invalid("Invalid smoothingType")
    cap = 100 if cfg["smoothingType"] == "gaussian" else 10000 if cfg["smoothingType"] == "average" else 1
    if not _finite(cfg["smoothingWeight"]) or not 0 <= cfg["smoothingWeight"] <= cap:
        _invalid(f"smoothingWeight must be between 0 and {cap}")
    if cfg["legendPosition"] not in {"north", "south", "east", "west"}:
        _invalid("Invalid legendPosition")
    if "limit" in cfg and (type(cfg["limit"]) is not int or not 1 <= cfg["limit"] <= 10):
        _invalid("limit must be an integer from 1 to 10")
    for key in ("overrideColors", "overrideSeriesTitles", "overrideLineWidths", "overrideMarks"):
        if key not in cfg:
            continue
        if not isinstance(cfg[key], dict) or len(cfg[key]) > 50:
            _invalid(f"{key} must be an object of at most 50 entries")
        for name, value in cfg[key].items():
            _text(key, name, 1025)
            if key == "overrideColors":
                if not isinstance(value, dict) or set(value) != {"color", "transparentColor"}:
                    _invalid("Colors require color and transparentColor")
                for color in value.values():
                    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{3,8}|rgba?\([\d.,%\s]+\)", color):
                        _invalid("Colors must be hex or rgb/rgba literals")
            elif key == "overrideLineWidths":
                if not _finite(value) or not 0 < value <= 20:
                    _invalid("Line widths must be positive and at most 20")
            elif key == "overrideMarks":
                if not isinstance(value, str) or value not in MARKS:
                    _invalid("Invalid line mark")
            else:
                _text(key, value)
    unsupported = set(raw_config) & DATA_OPTIONS
    if raw_config.get("pointVisualizationMethod") == "sampling":
        unsupported.discard("pointVisualizationMethod")
    for key, valid in (
        ("plotType", {"line"}),
        ("groupAgg", {"mean", "min", "max", "median", "sum"}),
        ("groupArea", {"minmax", "stddev", "stderr", "none"}),
    ):
        if cfg[key] not in valid:
            unsupported.add(key)
    if cfg["xAxis"] == "_absolute_runtime":
        unsupported.add("xAxis")
    if cfg.get("groupBy"):
        try:
            _group_key(cfg["groupBy"])
        except ToolInputValidationError:
            unsupported.add("groupBy")
    ignored = sorted(set(raw_config) & IGNORED)
    for key in ignored:
        del cfg[key]
    return raw_config, cfg, ignored, sorted(unsupported)


def _discover_runs(
    api: Any,
    *,
    entity_name: str,
    project_name: str,
    metrics: Sequence[str],
    limit: int,
    config_keys: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Select the newest runs that logged at least one requested metric."""
    candidate_filters: list[dict[str, Any] | None] = [
        {"$or": [{f"keys_info.keys.{metric}": {"$ne": None}} for metric in metrics]},
        {"$or": [{f"summary_metrics.{metric}": {"$exists": True}} for metric in metrics]},
        None,
    ]
    last_error: SelectiveReadUnavailable | None = None
    for mongo_filter in candidate_filters:
        raise_if_tool_deadline_exceeded()
        try:
            page = fetch_projected_runs(
                api,
                entity=entity_name,
                project=project_name,
                filters=mongo_filter,
                order="-createdAt",
                limit=limit,
                page_size=limit,
                summary_keys=(),
                config_keys=config_keys,
            )
            return page.items
        except SelectiveReadUnavailable as exc:
            raise_for_wandb_server_busy(exc)
            last_error = exc
    if last_error is not None:
        raise last_error
    return []


def _error(code: str, message: str, **details: Any) -> CallToolResult:
    payload = {"error": code, "message": message, **details}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], structuredContent=payload, isError=True
    )


def _summary(payload: dict[str, Any]) -> str:
    lines = [payload["title"], f"Open in W&B: {payload['url']}"]
    if (
        not payload["series"]
        and payload["coverage"].get("autoDiscoveredRuns")
        and not payload["coverage"]["perRunErrors"]
    ):
        metrics_label = ", ".join(payload["effectiveConfig"]["metrics"])
        entity_name = payload["arguments"]["entity_name"]
        project_name = payload["arguments"]["project_name"]
        lines.append(f"No runs found logging {metrics_label} in {entity_name}/{project_name}.")
    for series in payload["series"]:
        meta = payload["seriesMetadataById"][series["metadataId"]]
        values = [p["y"] for p in series["historySeries"]]
        stats = (
            f"first={values[0]:.6g}, last={values[-1]:.6g}, min={min(values):.6g}, max={max(values):.6g}"
            if values
            else "no finite numeric points"
        )
        lines.append(f"{meta['title']} ({series['role']}): {len(values)} points; {stats}; {meta['legendHref']}")
    for error in payload["coverage"]["perRunErrors"]:
        lines.append(f"{error['runId']}: {error['error']}; {error['url']}")
    active_count = sum(1 for info in payload.get("runsById", {}).values() if info.get("active"))
    suffix = f", active_runs={active_count}" if active_count else ""
    lines.append(f"sampled={payload['coverage']['sampled']}, truncated={payload['coverage']['truncated']}{suffix}")
    return "\n".join(lines)


def _downsample_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= 2:
        return []
    return points[::2] + (points[-1:] if len(points) % 2 == 0 else [])


def _result(payload: dict[str, Any], failed: bool = False) -> CallToolResult:
    for series in payload["series"]:
        points = series["historySeries"]
        series["historySeries"] = [p for p in points if all(_finite(v) for v in p.values())]
        payload["coverage"]["truncated"] |= len(points) != len(series["historySeries"])
        if "rawHistorySeries" in series:
            raw_points = series["rawHistorySeries"]
            series["rawHistorySeries"] = [p for p in raw_points if all(_finite(v) for v in p.values())]
            payload["coverage"]["truncated"] |= len(raw_points) != len(series["rawHistorySeries"])
    while True:
        summary = _summary(payload)
        serialized = json.dumps(
            {"content": summary, "structuredContent": payload}, allow_nan=False, separators=(",", ":")
        )
        if count_tokens_conservative(serialized) <= MAX_RESPONSE_TOKENS:
            return CallToolResult(
                content=[TextContent(type="text", text=summary)], structuredContent=payload, isError=failed
            )
        reduced = False
        for series in payload["series"]:
            points = series["historySeries"]
            if points:
                series["historySeries"] = _downsample_points(points)
                if "rawHistorySeries" in series and series["rawHistorySeries"]:
                    series["rawHistorySeries"] = _downsample_points(series["rawHistorySeries"])
                reduced = True
        if not reduced:
            return _error("response_too_large", "Line plot exceeds the response budget; request fewer runs or metrics.")
        payload["coverage"]["truncated"] = True


def render_line_plot(
    entity_name: str | None = None,
    project_name: str = "",
    run_ids: list[str] | None = None,
    config: dict[str, Any] | None = None,
    samples: int = 500,
    min_step: int | None = None,
    max_step: int | None = None,
    *,
    metric: str | None = None,
    limit: int | None = None,
) -> CallToolResult:
    try:
        raw_config, cfg, ignored, unsupported = _validate(
            entity_name,
            project_name,
            run_ids,
            config,
            samples,
            min_step,
            max_step,
            metric=metric,
            limit=limit,
        )
    except (ToolInputValidationError, RecursionError, ValueError) as exc:
        return _error(
            "invalid_input", str(exc) if isinstance(exc, ToolInputValidationError) else "Invalid config object"
        )
    if unsupported:
        return _error("unsupported_config", "Unsupported native data configuration", fields=unsupported)

    resolved_entity = entity_name
    if resolved_entity is None:
        try:
            resolved_entity = fetch_viewer_default_entity(WandBApiManager.get_api())
        except (ToolDeadlineExceeded, WandBServerBusy):
            raise
        except SelectiveReadUnavailable as exc:
            raise_for_wandb_server_busy(exc)
            return _error(
                "entity_lookup_failed",
                "Unable to resolve your default W&B entity. Pass entity_name explicitly.",
            )
        except Exception as exc:
            raise_for_wandb_server_busy(exc)
            return _error(
                "entity_lookup_failed",
                "Unable to resolve your default W&B entity. Pass entity_name explicitly.",
            )
        if not resolved_entity:
            return _error(
                "no_default_entity",
                "Your W&B account has no default entity. Pass entity_name explicitly.",
            )

    axis = cfg["xAxis"]
    grouped = bool(cfg.get("groupBy") or cfg.get("aggregate"))
    group_cfg_key = _group_key(cfg["groupBy"]) if cfg.get("groupBy") else None
    auto_discovered = run_ids is None
    projected_by_id: dict[str, dict[str, Any]] = {}

    if auto_discovered:
        try:
            discovered = _discover_runs(
                WandBApiManager.get_api(),
                entity_name=resolved_entity,
                project_name=project_name,
                metrics=cfg["metrics"],
                limit=cfg.get("limit", 10),
                config_keys=[group_cfg_key] if group_cfg_key else (),
            )
        except (ToolDeadlineExceeded, WandBServerBusy):
            raise
        except Exception as exc:
            raise_for_wandb_server_busy(exc)
            return _error(
                "project_unavailable",
                f"Unable to discover runs in {resolved_entity}/{project_name}.",
            )
        for item in discovered:
            rid = item.get("id")
            if isinstance(rid, str) and rid and rid not in projected_by_id:
                projected_by_id[rid] = item
                if len(projected_by_id) >= 10:
                    break
        resolved_run_ids = list(projected_by_id.keys())
    else:
        resolved_run_ids = list(run_ids or [])

    url = public_wandb_url(resolved_entity, project_name)
    workspace_url = public_wandb_url(resolved_entity, project_name, "workspace")
    payload: dict[str, Any] = {
        "schema": "wandb.graph/v1",
        "series": [],
        "seriesStylesById": {},
        "seriesMetadataById": {},
        "runsById": {},
        "hasActiveRuns": False,
        "axis": {
            "xKey": axis,
            "xIsDatetime": axis == "_timestamp",
            "xScale": "log" if cfg.get("xLogScale") else "linear",
            "yScale": "log" if cfg.get("yLogScale") else "linear",
            "xTitle": cfg.get("xAxisTitle", axis),
            "yTitle": cfg.get("yAxisTitle", ""),
        },
        "xDomain": [cfg.get("xAxisMin"), cfg.get("xAxisMax")],
        "yDomain": [cfg.get("yAxisMin"), cfg.get("yAxisMax")],
        "title": cfg.get("chartTitle") or ", ".join(cfg["metrics"]),
        "legend": {"show": cfg["showLegend"], "position": cfg["legendPosition"]},
        "effectiveConfig": cfg,
        "coverage": {
            "sampled": False,
            "truncated": False,
            "ignoredConfig": ignored,
            "perRunErrors": [],
            "groupingAlignment": "exact-match-union",
            "grouped": grouped,
            "autoDiscoveredRuns": auto_discovered,
        },
        "url": url,
        "workspaceUrl": workspace_url,
        "range_cap": MCP_MAX_HISTORY_RANGE_STEPS,
        "arguments": dict(
            entity_name=resolved_entity,
            project_name=project_name,
            run_ids=resolved_run_ids,
            config=deepcopy(raw_config),
            samples=min(samples, MCP_MAX_HISTORY_SAMPLES),
            min_step=min_step,
            max_step=max_step,
        ),
    }
    runs: list[dict[str, Any]] = []
    for run_id in resolved_run_ids:
        try:
            raise_if_tool_deadline_exceeded()
            snapshot = projected_by_id.get(run_id)
            group: Any = "All runs"
            if cfg.get("groupBy"):
                if snapshot is None:
                    snapshot = fetch_projected_run(
                        WandBApiManager.get_api(),
                        entity=resolved_entity,
                        project=project_name,
                        run_id=run_id,
                        config_keys=[group_cfg_key] if group_cfg_key else [],
                    )
                if snapshot is None:
                    raise ValueError("Run unavailable")
                group = snapshot.get("config", {}).get(group_cfg_key) if group_cfg_key else snapshot.get("group")
                if isinstance(group, (dict, list)) or (isinstance(group, float) and not _finite(group)):
                    raise ValueError("Non-scalar group value")
            raise_if_tool_deadline_exceeded()
            history = json.loads(
                get_run_history(
                    entity_name=resolved_entity,
                    project_name=project_name,
                    run_id=run_id,
                    keys=cfg["metrics"],
                    x_axis=axis,
                    samples=min(samples, MCP_MAX_HISTORY_SAMPLES),
                    min_step=min_step,
                    max_step=max_step,
                )
            )
            if not isinstance(history, dict) or "error" in history or not isinstance(history.get("rows"), list):
                raise ValueError("History unavailable")
            series_by_metric: dict[str, list[dict[str, float]]] = {}
            for metric_key in cfg["metrics"]:
                points_map = {
                    row[axis]: row[metric_key]
                    for row in history["rows"]
                    if isinstance(row, dict) and _finite(row.get(axis)) and _finite(row.get(metric_key))
                }
                series_by_metric[metric_key] = [
                    {"x": x * (1000 if axis == "_timestamp" else 1), "y": y}
                    for x, y in sorted(points_map.items())
                    if _finite(x * (1000 if axis == "_timestamp" else 1))
                ]
            display_name = str(
                history.get("run_name") or (snapshot.get("display_name") if snapshot else None) or run_id
            )[:512]
            raw_state = (snapshot.get("state") if snapshot else None) or history.get("run_state")
            run_state = raw_state.strip().lower() if isinstance(raw_state, str) and raw_state.strip() else "unknown"
            is_active = run_state in ACTIVE_RUN_STATES
            run_url = public_wandb_url(resolved_entity, project_name, "runs", run_id)
            payload["runsById"][run_id] = {
                "displayName": display_name,
                "state": run_state,
                "active": is_active,
                "group": group if grouped else None,
                "url": run_url,
            }
            payload["hasActiveRuns"] |= is_active
            runs.append(
                {
                    "id": run_id,
                    "name": display_name,
                    "state": run_state,
                    "group": group,
                    "series": series_by_metric,
                }
            )
            payload["coverage"]["sampled"] |= bool(history.get("sampled", True))
            payload["coverage"]["truncated"] |= bool(
                history.get("truncated") or history.get("source_truncated") or history.get("source_values_truncated")
            )
        except ToolDeadlineExceeded:
            return _error("tool_deadline_exceeded", "Line plot exceeded the tool execution deadline.")
        except WandBServerBusy:
            raise
        except Exception as exc:
            raise_for_wandb_server_busy(exc)
            payload["coverage"]["perRunErrors"].append(
                {
                    "runId": run_id,
                    "error": "history_fetch_failed",
                    "url": public_wandb_url(resolved_entity, project_name, "runs", run_id),
                }
            )

    sources: list[tuple[str, str, str, list[dict[str, float]], list[dict[str, float]], str]] = []
    if grouped:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            groups[json.dumps(run["group"], sort_keys=True)].append(run)
        for members in groups.values():
            label = str(members[0]["group"]) if members[0]["group"] is not None else "(missing)"
            for metric_key in cfg["metrics"]:
                buckets: dict[float, list[float]] = defaultdict(list)
                for run in members:
                    for p in run["series"][metric_key]:
                        buckets[p["x"]].append(p["y"])
                points, band = [], []
                agg = {"mean": statistics.mean, "min": min, "max": max, "median": statistics.median, "sum": sum}[
                    cfg["groupAgg"]
                ]
                for x, ys in sorted(buckets.items()):
                    points.append({"x": x, "y": agg(ys)})
                    if cfg["groupArea"] != "none":
                        try:
                            deviation = statistics.stdev(ys) if len(ys) > 1 and cfg["groupArea"] != "minmax" else 0
                        except OverflowError:
                            deviation = math.inf
                        if cfg["groupArea"] == "stderr":
                            deviation /= math.sqrt(len(ys))
                        lo, hi = (
                            (min(ys), max(ys))
                            if cfg["groupArea"] == "minmax"
                            else (statistics.mean(ys) - deviation, statistics.mean(ys) + deviation)
                        )
                        band.append({"x": x, "y": hi, "y0": lo})
                sources.append((label, label, metric_key, points, band, url))
    else:
        for run in runs:
            for metric_key, points in run["series"].items():
                sources.append(
                    (
                        run["id"],
                        run["name"],
                        metric_key,
                        points,
                        [],
                        public_wandb_url(resolved_entity, project_name, "runs", run["id"]),
                    )
                )

    identities = list(dict.fromkeys(source[0] for source in sources))
    selected = identities[: cfg.get("limit", 10)]
    payload["coverage"]["truncated"] |= len(selected) < len(identities)
    x_max = max((p["x"] for source in sources for p in source[3]), default=100)
    smoothing_active = bool(cfg["smoothingWeight"] and cfg["smoothingType"] != "none")
    for owner, name, metric_key, points, band, href in sources:
        if owner not in selected:
            continue
        color_key = owner + ":" + metric_key
        color = (
            cfg.get("overrideColors", {})
            .get(color_key, {})
            .get("color", PALETTE[identities.index(owner) % len(PALETTE)])
        )
        transparent = (
            cfg.get("overrideColors", {})
            .get(color_key, {})
            .get("transparentColor", color + "33" if color.startswith("#") and len(color) == 7 else color)
        )
        metadata_id = json.dumps([owner, metric_key], separators=(",", ":"))
        run_info = None if grouped else payload["runsById"].get(owner)
        payload["seriesMetadataById"][metadata_id] = {
            "displayName": name,
            "title": cfg.get("overrideSeriesTitles", {}).get(color_key, f"{name} · {metric_key}"),
            "runName": None if grouped else owner,
            "runState": run_info["state"] if run_info else None,
            "legendHref": href,
            "entityName": resolved_entity,
            "projectName": project_name,
        }
        smoothed = smooth(points, cfg["smoothingWeight"], cfg["smoothingType"], x_max)
        variants: list[tuple[str, str, list[dict[str, float]]]] = [("primary", "line", smoothed)]
        if smoothing_active and cfg.get("showOriginalAfterSmoothing"):
            variants.append(("original", "line", points))
        if band:
            variants.append(("band", "area", band))
        for role, kind, data in variants:
            sid = json.dumps([owner, metric_key, role], separators=(",", ":"))
            entry: dict[str, Any] = {
                "seriesId": sid,
                "metadataId": metadata_id,
                "role": role,
                "kind": kind,
                "metricName": metric_key,
                "historySeries": data,
            }
            if role == "primary" and smoothing_active and not cfg.get("showOriginalAfterSmoothing"):
                entry["rawHistorySeries"] = points
            if grouped:
                entry["groupKey"] = owner
            payload["series"].append(entry)
            payload["seriesStylesById"][sid] = {
                "color": transparent if role in {"original", "band"} else color,
                "lineWidth": cfg.get("overrideLineWidths", {}).get(color_key, 1.6),
                "mark": cfg.get("overrideMarks", {}).get(color_key, "solid"),
                "colorKey": color_key,
            }
    return _result(payload, failed=bool(resolved_run_ids) and not runs)
