"""Native panel configuration projected into bounded JSON-safe Graph series."""

from __future__ import annotations

from collections import defaultdict
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

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import WandBApiManager
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
from wandb_mcp_server.wandb_selective_reads import MAX_SAFE_HISTORY_STEP, fetch_projected_run
from wandb_mcp_server.wandb_urls import public_wandb_url

RESOURCE_URI = "ui://wandb/line-plot.html"
TOOL_NAME = "render_wandb_line_plot_tool"
LINE_PLOT_DESCRIPTION = """Render W&B run metrics inline using native W&B line plot panel config.

config uses camelCase RunsLinePlotConfig, the same config object as a report spec
panel returned by query_wandb_tool with include=["spec"]. Reads 1–10 unique
run_ids in one entity/project. config.metrics is required (1–5 unique metric
keys). xAxis defaults to _step; _timestamp, _runtime and numeric logged custom
keys are supported. This is a read-only sampled chart, not full fidelity.

Supported config: chartTitle, metrics, xAxis, xAxisMin/Max, yAxisMin/Max (display
domains), xLogScale, yLogScale, xAxisTitle, yAxisTitle, smoothingWeight,
smoothingType (none, exponential, exponentialTimeWeighted, gaussian, average),
showOriginalAfterSmoothing, groupBy, aggregate, groupAgg (mean, min, max, median,
sum), groupArea (minmax, stddev, stderr, none), legendPosition, showLegend,
limit, overrideColors, overrideSeriesTitles, overrideLineWidths, overrideMarks,
plotType="line". groupBy accepts group (the run group) or config.key/key.value
paths. Grouping uses exact-match samples on the union of x values, without
forward-fill. Bands use sample standard deviation (zero for singleton groups).
Defaults: groupAgg mean, groupArea minmax, smoothingType exponential, weight 0.
EMA weight is 0–1; Gaussian sigma 0–100; average window 0–10000. Overrides use
runId:metricName or groupKey:metricName. limit is 1–10 output runs/groups.

Expressions, regex selection, full-fidelity modes, non-sampling point strategies,
non-line plots, groupAgg last/samples, groupArea samples and other native
data-changing options return unsupported_config with fields listed. Ignored
native cosmetic options are listed in coverage. Non-native keys are invalid_input.

Returns wandb.graph/v1: Graph-shaped series, style/metadata maps, axis/domains,
effectiveConfig, sampling/truncation coverage, per-run errors and W&B links.
Datetime coordinates are milliseconds. MCP Apps hosts display an offline chart
with hover, legend toggles and zoom; other hosts receive compact per-series
point counts and first/last/min/max statistics. samples defaults to 500 per run
and is clamped to the workload history cap. Optional inclusive min_step/max_step
follow history caps; managed ranges require both. Step zoom preserves config
and requeries bounded windows. Reads are sequential and deadline-aware; failed
runs do not discard successful ones. Response-budget downsampling may reduce
returned points. Statistics describe returned samples, not exhaustive history.
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
    ui: dict[str, Any] = {"csp": {"connectDomains": [], "resourceDomains": []}}
    if domain := os.environ.get("WANDB_MCP_APP_DOMAIN"):
        ui["domain"] = domain

    @mcp.resource(RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def line_plot_html() -> str:
        return files("wandb_mcp_server").joinpath("data/line-plot.html").read_text(encoding="utf-8")


def _invalid(message):
    raise ToolInputValidationError(message, field="config")


def _text(name, value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _invalid(f"{name} must be non-empty text of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _invalid(f"{name} must contain valid UTF-8")


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _group_key(value):
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


def _validate(entity, project, run_ids, config, samples, min_step, max_step):
    for name, value in (("entity_name", entity), ("project_name", project)):
        _text(name, value)
    if not isinstance(config, dict) or any(not isinstance(k, str) for k in config):
        _invalid("config must be a native panel config object")
    unknown = set(config) - SUPPORTED - IGNORED - DATA_OPTIONS
    if unknown:
        _invalid("Unknown config fields: " + ", ".join(sorted(unknown)))
    try:
        if len(json.dumps(config, allow_nan=False).encode("utf-8")) > 32768:
            _invalid("config exceeds 32 KiB")
    except (TypeError, ValueError, UnicodeEncodeError):
        _invalid("config must be finite JSON")
    for name, values, cap in (("run_ids", run_ids, 10), ("metrics", config.get("metrics"), 5)):
        if not isinstance(values, list) or not 1 <= len(values) <= cap:
            _invalid(f"{name} must contain 1–{cap} entries")
        for value in values:
            _text(name, value)
        if len(set(values)) != len(values):
            _invalid(f"{name} must contain unique entries")
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
    cfg = {
        "xAxis": "_step",
        "smoothingWeight": 0,
        "smoothingType": "exponential",
        "groupAgg": "mean",
        "groupArea": "minmax",
        "plotType": "line",
        "showLegend": True,
        "legendPosition": "north",
        **deepcopy(config),
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
    unsupported = set(config) & DATA_OPTIONS
    if config.get("pointVisualizationMethod") == "sampling":
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
    ignored = sorted(set(config) & IGNORED)
    for key in ignored:
        del cfg[key]
    return cfg, ignored, sorted(unsupported)


def _error(code, message, **details):
    payload = {"error": code, "message": message, **details}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], structuredContent=payload, isError=True
    )


def _summary(payload):
    lines = [payload["title"], f"Open in W&B: {payload['url']}"]
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
    lines.append(f"sampled={payload['coverage']['sampled']}, truncated={payload['coverage']['truncated']}")
    return "\n".join(lines)


def _result(payload, failed=False):
    for series in payload["series"]:
        points = series["historySeries"]
        series["historySeries"] = [p for p in points if all(_finite(v) for v in p.values())]
        payload["coverage"]["truncated"] |= len(points) != len(series["historySeries"])
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
                series["historySeries"] = (
                    points[::2] + (points[-1:] if len(points) % 2 == 0 else []) if len(points) > 2 else []
                )
                reduced = True
        if not reduced:
            return _error("response_too_large", "Line plot exceeds the response budget; request fewer runs or metrics.")
        payload["coverage"]["truncated"] = True


def render_line_plot(
    entity_name: str,
    project_name: str,
    run_ids: list[str],
    config: dict,
    samples: int = 500,
    min_step: int | None = None,
    max_step: int | None = None,
) -> CallToolResult:
    try:
        cfg, ignored, unsupported = _validate(entity_name, project_name, run_ids, config, samples, min_step, max_step)
    except (ToolInputValidationError, RecursionError, ValueError) as exc:
        return _error(
            "invalid_input", str(exc) if isinstance(exc, ToolInputValidationError) else "Invalid config object"
        )
    if unsupported:
        return _error("unsupported_config", "Unsupported native data configuration", fields=unsupported)
    axis = cfg["xAxis"]
    grouped = bool(cfg.get("groupBy") or cfg.get("aggregate"))
    url = public_wandb_url(entity_name, project_name)
    payload = {
        "schema": "wandb.graph/v1",
        "series": [],
        "seriesStylesById": {},
        "seriesMetadataById": {},
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
        },
        "url": url,
        "range_cap": MCP_MAX_HISTORY_RANGE_STEPS,
        "arguments": dict(
            entity_name=entity_name,
            project_name=project_name,
            run_ids=run_ids,
            config=deepcopy(config),
            samples=min(samples, MCP_MAX_HISTORY_SAMPLES),
            min_step=min_step,
            max_step=max_step,
        ),
    }
    runs = []
    for run_id in run_ids:
        try:
            raise_if_tool_deadline_exceeded()
            group = "All runs"
            if cfg.get("groupBy"):
                key = _group_key(cfg["groupBy"])
                snapshot = fetch_projected_run(
                    WandBApiManager.get_api(),
                    entity=entity_name,
                    project=project_name,
                    run_id=run_id,
                    config_keys=[key] if key else [],
                )
                if snapshot is None:
                    raise ValueError("Run unavailable")
                group = snapshot.get("config", {}).get(key) if key else snapshot.get("group")
                if isinstance(group, (dict, list)) or (isinstance(group, float) and not _finite(group)):
                    raise ValueError("Non-scalar group value")
            raise_if_tool_deadline_exceeded()
            history = json.loads(
                get_run_history(
                    entity_name=entity_name,
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
            series = {}
            for metric in cfg["metrics"]:
                points = {
                    row[axis]: row[metric]
                    for row in history["rows"]
                    if isinstance(row, dict) and _finite(row.get(axis)) and _finite(row.get(metric))
                }
                series[metric] = [
                    {"x": x * (1000 if axis == "_timestamp" else 1), "y": y}
                    for x, y in sorted(points.items())
                    if _finite(x * (1000 if axis == "_timestamp" else 1))
                ]
            runs.append(
                {"id": run_id, "name": str(history.get("run_name") or run_id)[:512], "group": group, "series": series}
            )
            payload["coverage"]["sampled"] |= bool(history.get("sampled", True))
            payload["coverage"]["truncated"] |= bool(
                history.get("truncated") or history.get("source_truncated") or history.get("source_values_truncated")
            )
        except Exception:
            payload["coverage"]["perRunErrors"].append(
                {
                    "runId": run_id,
                    "error": "history_fetch_failed",
                    "url": public_wandb_url(entity_name, project_name, "runs", run_id),
                }
            )
    sources = []
    if grouped:
        groups = defaultdict(list)
        for run in runs:
            groups[json.dumps(run["group"], sort_keys=True)].append(run)
        for members in groups.values():
            label = str(members[0]["group"]) if members[0]["group"] is not None else "(missing)"
            for metric in cfg["metrics"]:
                buckets = defaultdict(list)
                for run in members:
                    for p in run["series"][metric]:
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
                sources.append((label, label, metric, points, band, url))
    else:
        for run in runs:
            for metric, points in run["series"].items():
                sources.append(
                    (
                        run["id"],
                        run["name"],
                        metric,
                        points,
                        [],
                        public_wandb_url(entity_name, project_name, "runs", run["id"]),
                    )
                )
    identities = list(dict.fromkeys(source[0] for source in sources))
    selected = identities[: cfg.get("limit", 10)]
    payload["coverage"]["truncated"] |= len(selected) < len(identities)
    x_max = max((p["x"] for source in sources for p in source[3]), default=100)
    for owner, name, metric, points, band, href in sources:
        if owner not in selected:
            continue
        color_key = owner + ":" + metric
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
        metadata_id = json.dumps([owner, metric], separators=(",", ":"))
        payload["seriesMetadataById"][metadata_id] = {
            "displayName": name,
            "title": cfg.get("overrideSeriesTitles", {}).get(color_key, f"{name} · {metric}"),
            "runName": None if grouped else owner,
            "legendHref": href,
            "entityName": entity_name,
            "projectName": project_name,
        }
        smoothed = smooth(points, cfg["smoothingWeight"], cfg["smoothingType"], x_max)
        variants = [("primary", "line", smoothed)]
        if cfg["smoothingWeight"] and cfg["smoothingType"] != "none" and cfg.get("showOriginalAfterSmoothing"):
            variants.append(("original", "line", points))
        if band:
            variants.append(("band", "area", band))
        for role, kind, data in variants:
            sid = json.dumps([owner, metric, role], separators=(",", ":"))
            entry = {
                "seriesId": sid,
                "metadataId": metadata_id,
                "role": role,
                "kind": kind,
                "metricName": metric,
                "historySeries": data,
            }
            if grouped:
                entry["groupKey"] = owner
            payload["series"].append(entry)
            payload["seriesStylesById"][sid] = {
                "color": transparent if role in {"original", "band"} else color,
                "lineWidth": cfg.get("overrideLineWidths", {}).get(color_key, 1.6),
                "mark": cfg.get("overrideMarks", {}).get(color_key, "solid"),
                "colorKey": color_key,
            }
    return _result(payload, failed=not runs)
