"""Retrieve sampled time-series metric history for a W&B run.

Uses `wandb.Api().run().history()` for sampled data and a tiered
strategy for step-range queries: scan_history (parquet-backed) first,
history() (sampled) as last resort.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Literal, Optional

import wandb

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.config import (
    MCP_HOSTED_MODE,
    MCP_MAX_HISTORY_KEYS,
    MCP_MAX_HISTORY_RANGE_STEPS,
    MCP_MAX_HISTORY_SAMPLES,
    MCP_WORKLOAD_PROFILE,
    MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
    WANDB_API_BASE_URL,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import SelectiveReadUnavailable, fetch_metric_value_steps

logger = get_rich_logger(__name__)

GET_RUN_HISTORY_TOOL_DESCRIPTION = """Retrieve bounded time-series metric data from a W&B run.

Use SDK sampling for overviews, bounded scans for internal-step ranges, and an
exact custom-axis lookup for a logged x-axis value. Every response states the
retrieval method, whether values are sampled or exact, rows scanned, coverage,
and any profile or response-budget truncation.

<when_to_use>
Call this tool when the user asks about training curves, metric trends over time,
loss plots, or any time-series data logged to a W&B run. This is the only tool
that provides step-by-step metric history -- query_wandb_tool returns run-level
summary metrics but not the full training history.

Typical workflow:
1. Use query_wandb_tool to find runs and their summary metrics.
2. Use get_run_history_tool to drill into a specific run's training curves.
3. Use create_wandb_report_tool to visualize the results.
</when_to_use>

Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
run_id : str
    The 8-character W&B run ID (e.g., "gtng2y4l"). This is the short ID,
    not the display name.
keys : list of str, optional
    Specific metric keys to retrieve (e.g., ["loss", "val_loss", "accuracy"]).
    Hosted deployments require 1-20 explicit keys to prevent accidental retrieval
    of extremely high-cardinality histories.
samples : int, optional
    Number of evenly-spaced sample points to return. Defaults to 500.
    Use fewer samples for quick overviews, more for detailed analysis.
min_step : int, optional
    Minimum step to include. Defaults to None (start from beginning).
max_step : int, optional
    Maximum step to include. Defaults to None (include all steps).
x_axis : str, optional
    History x-axis. Defaults to "_step". Set this to a logged monotonic metric
    such as "validation/step" for custom-axis sampling or target lookup.
target_x : float, optional
    Retrieve the row where x_axis logged this exact value. If no exact value was
    logged, returns target_not_logged.
tolerance : float, optional
    When target_x was not logged exactly, permit a bounded nearest-value
    refinement within this absolute tolerance.
stream : "default" or "system", optional
    Select normal run history or system metrics. Defaults to "default".

Returns
-------
JSON with:
  - rows: list of {_step, key1, key2, ...} dicts
  - run_id: the queried run ID
  - run_name: the run's display name
  - total_steps: last logged step number
  - sampled_points: number of rows returned
  - keys_returned: list of metric keys in the response
  - retrieval_method, exact, sampled, rows_scanned, coverage, truncation

Examples
--------
>>> get_run_history_tool("my-team", "my-project", "gtng2y4l", keys=["loss", "val_loss"])
>>> get_run_history_tool(
...     "my-team", "my-project", "h0fm5qp5",
...     keys=["validation/loss"], x_axis="validation/step", target_x=1000,
... )
"""

MAX_HISTORY_ROWS = MCP_MAX_HISTORY_SAMPLES
_EXACT_EPSILON = 1e-9


@dataclass(frozen=True)
class _HistoryFetch:
    rows: list[dict[str, Any]]
    method: str
    sampled: bool
    exact: bool
    rows_scanned: int
    compatibility_caveat: str | None = None


def get_run_history(
    entity_name: str,
    project_name: str,
    run_id: str,
    keys: Optional[List[str]] = None,
    samples: int = 500,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    x_axis: str = "_step",
    target_x: float | None = None,
    tolerance: float | None = None,
    stream: Literal["default", "system"] = "default",
) -> str:
    """Fetch bounded metric history for a W&B run."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not isinstance(x_axis, str) or not x_axis.strip():
        raise ValueError("x_axis must be a non-empty string")
    if stream not in {"default", "system"}:
        raise ValueError("stream must be 'default' or 'system'")
    if target_x is not None and (
        isinstance(target_x, bool) or not isinstance(target_x, (int, float)) or not math.isfinite(float(target_x))
    ):
        raise ValueError("target_x must be a finite number")
    if tolerance is not None and (
        target_x is None
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        raise ValueError("tolerance must be a finite non-negative number and requires target_x")
    if target_x is not None and (min_step is not None or max_step is not None):
        raise ValueError("target_x cannot be combined with min_step or max_step")
    if stream == "system" and target_x is not None:
        raise ValueError("target_x is supported only for the default history stream")
    if stream == "system" and (min_step is not None or max_step is not None):
        raise ValueError("step-range scans are supported only for the default history stream")
    if keys is not None and (
        not isinstance(keys, list)
        or not all(isinstance(key, str) and key.strip() for key in keys)
        or len(keys) > MCP_MAX_HISTORY_KEYS
    ):
        raise ValueError(f"keys must contain at most {MCP_MAX_HISTORY_KEYS} non-empty strings")
    if (MCP_WORKLOAD_PROFILE == "shared" or MCP_HOSTED_MODE) and not keys:
        raise ValueError(
            f"The shared workload profile requires explicit history keys (1-{MCP_MAX_HISTORY_KEYS} metrics)."
        )
    if min_step is not None and max_step is not None:
        if max_step < min_step:
            raise ValueError("max_step must be greater than or equal to min_step")
        if max_step - min_step + 1 > MCP_MAX_HISTORY_RANGE_STEPS:
            raise ValueError(f"history step range cannot exceed {MCP_MAX_HISTORY_RANGE_STEPS} steps")
    elif (MCP_WORKLOAD_PROFILE == "shared" or MCP_HOSTED_MODE) and (min_step is not None or max_step is not None):
        raise ValueError("The shared workload profile requires both min_step and max_step for range scans")

    with track_tool_execution(
        "get_run_history",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "run_id": run_id,
            "keys": keys,
            "samples": samples,
            "stream": stream,
            "x_axis_is_custom": x_axis != "_step",
            "has_target_x": target_x is not None,
            "has_tolerance": tolerance is not None,
        },
    ):
        api_key = WandBApiManager.get_api_key()
        if not api_key:
            raise ValueError("W&B API key is required to fetch run history.")

        try:
            wandb_api = wandb.Api(
                api_key=api_key,
                overrides={"base_url": WANDB_API_BASE_URL},
                timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
            )
            run_path = f"{entity_name}/{project_name}/{run_id}"
            run = wandb_api.run(run_path)
        except wandb.errors.CommError as e:
            raise ValueError(f"Run not found: {run_path}. Error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to access run {entity_name}/{project_name}/{run_id}: {type(e).__name__}")

        profile_limit_applied = samples > MCP_MAX_HISTORY_SAMPLES
        clamped_samples = min(samples, MCP_MAX_HISTORY_SAMPLES)

        try:
            if target_x is not None:
                fetched = _fetch_target_x(
                    wandb_api,
                    run,
                    entity_name=entity_name,
                    project_name=project_name,
                    run_id=run_id,
                    keys=keys,
                    x_axis=x_axis,
                    target_x=float(target_x),
                    tolerance=float(tolerance) if tolerance is not None else None,
                )
                if not fetched.rows:
                    return json.dumps(
                        {
                            "error": "target_not_logged",
                            "message": f"No logged {x_axis!r} value matched target {target_x}.",
                            "run_id": run_id,
                            "x_axis": x_axis,
                            "target_x": target_x,
                            "tolerance": tolerance,
                            "retrieval_method": fetched.method,
                            "exact": False,
                            "sampled": False,
                            "rows_scanned": fetched.rows_scanned,
                            "compatibility_caveat": fetched.compatibility_caveat,
                        }
                    )
            elif min_step is not None or max_step is not None:
                rows, rows_scanned, used_sampled_fallback = _fetch_step_range_with_metadata(
                    run,
                    clamped_samples,
                    keys,
                    min_step,
                    max_step,
                    stream=stream,
                )
                fetched = _HistoryFetch(
                    rows=rows,
                    method="sdk_sample_fallback" if used_sampled_fallback else "sdk_bounded_scan",
                    sampled=used_sampled_fallback or rows_scanned > len(rows),
                    exact=False,
                    rows_scanned=rows_scanned,
                )
            else:
                history_kwargs: Dict[str, Any] = {"samples": clamped_samples, "pandas": False}
                if keys:
                    history_kwargs["keys"] = keys
                history_kwargs["x_axis"] = x_axis
                history_kwargs["stream"] = stream
                rows = list(run.history(**history_kwargs))
                fetched = _HistoryFetch(
                    rows=rows,
                    method="sdk_sample",
                    sampled=True,
                    exact=False,
                    rows_scanned=len(rows),
                )
        except Exception as e:
            raise ValueError(f"Failed to fetch history for run {run_id}: {e}")

        clean_rows = []
        for row in fetched.rows:
            clean_row = {}
            for k, v in row.items():
                if k.startswith("_") and k not in ("_step", "_timestamp", "_runtime"):
                    continue
                if v is None or (isinstance(v, float) and v != v):
                    continue
                clean_row[k] = v
            clean_rows.append(clean_row)

        keys_in_response = set()
        for row in clean_rows:
            keys_in_response.update(row.keys())
        keys_in_response.discard("_step")

        from wandb_mcp_server.config import MAX_RESPONSE_TOKENS

        total_steps = getattr(run, "lastHistoryStep", len(clean_rows))
        original_count = len(clean_rows)

        budget_chars = MAX_RESPONSE_TOKENS * 4
        clean_rows = _enforce_row_budget(clean_rows, budget_chars)
        truncated = len(clean_rows) < original_count

        x_values = [
            float(row[x_axis])
            for row in clean_rows
            if isinstance(row.get(x_axis), (int, float)) and math.isfinite(float(row[x_axis]))
        ]
        result_dict: Dict[str, Any] = {
            "rows": clean_rows,
            "run_id": run_id,
            "run_name": run.name,
            "total_steps": total_steps,
            "sampled_points": len(clean_rows),
            "keys_returned": sorted(keys_in_response),
            "retrieval_method": fetched.method,
            "exact": fetched.exact,
            "sampled": fetched.sampled,
            "requested_samples": samples,
            "effective_samples": clamped_samples,
            "rows_scanned": fetched.rows_scanned,
            "stream": stream,
            "x_axis": x_axis,
            "coverage": {
                "first_x": x_values[0] if x_values else None,
                "last_x": x_values[-1] if x_values else None,
                "requested_min_step": min_step,
                "requested_max_step": max_step,
                "target_x": target_x,
                "tolerance": tolerance,
            },
            "truncated": truncated or profile_limit_applied,
        }
        if truncated:
            result_dict["truncation_note"] = (
                f"Downsampled from {original_count} to {len(clean_rows)} rows to fit token budget. "
                "Use keys= to select fewer metrics or reduce samples."
            )
        if profile_limit_applied:
            result_dict["profile_limit_note"] = (
                f"The {MCP_WORKLOAD_PROFILE} workload profile limits history responses to "
                f"{MCP_MAX_HISTORY_SAMPLES} rows."
            )
            if MCP_HOSTED_MODE:
                result_dict["hosted_limit_note"] = result_dict["profile_limit_note"]
        if fetched.compatibility_caveat:
            result_dict["compatibility_caveat"] = fetched.compatibility_caveat
        return json.dumps(result_dict)


def _fetch_step_range(
    run: Any,
    clamped_samples: int,
    keys: Optional[List[str]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning only rows from a bounded step scan."""
    rows, _, _ = _fetch_step_range_with_metadata(
        run,
        clamped_samples,
        keys,
        min_step,
        max_step,
        stream="default",
    )
    return rows


def _scan_history_rows(
    run: Any,
    *,
    keys: list[str] | None,
    min_step: int | None,
    max_step: int | None,
    scan_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    scan_kwargs: Dict[str, Any] = {}
    if keys:
        scan_kwargs["keys"] = keys
    if min_step is not None:
        scan_kwargs["min_step"] = min_step
    if max_step is not None:
        # Public SDK scan_history treats max_step as exclusive; the MCP
        # interface documents max_step as inclusive.
        scan_kwargs["max_step"] = max_step + 1
    scan_kwargs["page_size"] = min(1000, max(1, scan_limit))
    scanned_rows: list[dict[str, Any]] = []
    rows_scanned = 0
    for index, row in enumerate(run.scan_history(**scan_kwargs)):
        if index >= scan_limit:
            break
        if index % 100 == 0:
            raise_if_tool_deadline_exceeded()
        rows_scanned += 1
        scanned_rows.append(row)
    return scanned_rows, rows_scanned


def _fetch_step_range_with_metadata(
    run: Any,
    clamped_samples: int,
    keys: Optional[List[str]],
    min_step: Optional[int],
    max_step: Optional[int],
    *,
    stream: Literal["default", "system"],
) -> tuple[List[Dict[str, Any]], int, bool]:
    """Fetch history rows for a step range using a tiered strategy.

    Strategy order:
      1. scan_history (parquet-backed when available with SDK-managed fallback;
         fails silently when lastHistoryStep == -1)
      2. history() sampled fallback (always works, ignores step bounds)
    """
    scan_limit = MCP_MAX_HISTORY_RANGE_STEPS
    if min_step is not None and max_step is not None:
        scan_limit = min(scan_limit, max_step - min_step + 1)
    scanned_rows, rows_scanned = _scan_history_rows(
        run,
        keys=keys,
        min_step=min_step,
        max_step=max_step,
        scan_limit=scan_limit,
    )
    rows = _evenly_sample(scanned_rows, clamped_samples)
    if rows:
        return rows, rows_scanned, False

    # Strategy 2: history() sampled fallback (ignores step bounds but always works)
    last_step = getattr(run, "lastHistoryStep", 0) or 0
    if last_step <= 0:
        logger.warning(
            "scan_history returned 0 rows (lastHistoryStep=%s). "
            "Falling back to history(samples=%d) which ignores step bounds.",
            last_step,
            clamped_samples,
        )
        history_kwargs: Dict[str, Any] = {"samples": clamped_samples, "pandas": False, "stream": stream}
        if keys:
            history_kwargs["keys"] = keys
        fallback_rows = list(run.history(**history_kwargs))
        return fallback_rows, len(fallback_rows), True

    return rows, rows_scanned, False


def _numeric_x(row: Dict[str, Any], x_axis: str) -> float | None:
    value = row.get(x_axis)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _select_target_row(
    rows: list[dict[str, Any]],
    *,
    x_axis: str,
    target_x: float,
    tolerance: float | None,
) -> tuple[list[dict[str, Any]], bool]:
    candidates = [(abs(value - target_x), row) for row in rows if (value := _numeric_x(row, x_axis)) is not None]
    exact = [row for distance, row in candidates if distance <= _EXACT_EPSILON]
    if exact:
        return [exact[0]], True
    if tolerance is None or not candidates:
        return [], False
    distance, nearest = min(candidates, key=lambda candidate: candidate[0])
    return ([nearest], False) if distance <= tolerance else ([], False)


def _fetch_target_x(
    api: Any,
    run: Any,
    *,
    entity_name: str,
    project_name: str,
    run_id: str,
    keys: list[str] | None,
    x_axis: str,
    target_x: float,
    tolerance: float | None,
) -> _HistoryFetch:
    requested_keys = list(dict.fromkeys([x_axis, *(keys or [])]))
    if x_axis == "_step":
        if not target_x.is_integer():
            return _HistoryFetch([], "sdk_exact_step_scan", False, False, 0)
        target_step = int(target_x)
        rows, rows_scanned = _scan_history_rows(
            run,
            keys=requested_keys,
            min_step=target_step,
            max_step=target_step,
            scan_limit=1,
        )
        selected, exact = _select_target_row(rows, x_axis=x_axis, target_x=target_x, tolerance=tolerance)
        return _HistoryFetch(selected, "sdk_exact_step_scan", False, exact, rows_scanned)

    try:
        candidate_steps = fetch_metric_value_steps(
            api,
            entity=entity_name,
            project=project_name,
            run_id=run_id,
            metric=x_axis,
            values=[target_x],
        )
    except SelectiveReadUnavailable as exc:
        rows, rows_scanned = _scan_history_rows(
            run,
            keys=requested_keys,
            min_step=None,
            max_step=None,
            scan_limit=MCP_MAX_HISTORY_RANGE_STEPS,
        )
        selected, exact = _select_target_row(rows, x_axis=x_axis, target_x=target_x, tolerance=tolerance)
        return _HistoryFetch(
            selected,
            "sdk_bounded_compatibility_scan",
            False,
            exact,
            rows_scanned,
            f"{exc}; searched at most {MCP_MAX_HISTORY_RANGE_STEPS} rows without public-URL retry",
        )

    candidate_step = candidate_steps[0] if candidate_steps else None
    if candidate_step is None:
        return _HistoryFetch([], "steps_for_metric_values", False, False, 0)

    refinement_radius = 2 if tolerance is not None else 0
    rows, rows_scanned = _scan_history_rows(
        run,
        keys=requested_keys,
        min_step=max(0, candidate_step - refinement_radius),
        max_step=candidate_step + refinement_radius,
        scan_limit=2 * refinement_radius + 1,
    )
    selected, exact = _select_target_row(rows, x_axis=x_axis, target_x=target_x, tolerance=tolerance)
    return _HistoryFetch(
        selected,
        "steps_for_metric_values",
        False,
        exact,
        rows_scanned,
    )


def _evenly_sample(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    """Return an ordered, evenly spaced sample from an already bounded scan."""
    if len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[0]]
    last_index = len(rows) - 1
    indexes = [round(position * last_index / (max_rows - 1)) for position in range(max_rows)]
    return [rows[index] for index in indexes]


def _enforce_row_budget(rows: List[Dict[str, Any]], budget_chars: int) -> List[Dict[str, Any]]:
    """Downsample rows to fit within a character budget.

    Preserves even spacing across _step values by taking every Nth row.
    """
    if not rows:
        return rows

    serialized = json.dumps(rows, default=str)
    if len(serialized) <= budget_chars:
        return rows

    per_row = max(1, len(serialized) // len(rows))
    target_count = max(2, budget_chars // per_row)

    if target_count >= len(rows):
        return rows

    stride = max(1, len(rows) // target_count)
    sampled = rows[::stride]

    while len(json.dumps(sampled, default=str)) > budget_chars and len(sampled) > 2:
        stride *= 2
        sampled = rows[::stride]

    return sampled
