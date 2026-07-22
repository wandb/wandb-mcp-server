"""Retrieve sampled time-series metric history for a W&B run.

Uses `wandb.Api().run().history()` for sampled data and a tiered
strategy for step-range queries: scan_history (parquet-backed) first,
history() (sampled) as last resort.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import wandb

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.config import (
    MCP_HOSTED_MODE,
    MCP_MAX_HISTORY_KEYS,
    MCP_MAX_HISTORY_RANGE_STEPS,
    MCP_MAX_HISTORY_SAMPLES,
    MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
    WANDB_BASE_URL,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

GET_RUN_HISTORY_TOOL_DESCRIPTION = """Retrieve sampled time-series metric data from a W&B run.

Returns step-indexed rows of logged metrics (loss, accuracy, learning_rate, etc.)
for a specific run. The data is sampled to `samples` evenly-spaced points so even
runs with millions of steps return a manageable response.

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

Returns
-------
JSON with:
  - rows: list of {_step, key1, key2, ...} dicts
  - run_id: the queried run ID
  - run_name: the run's display name
  - total_steps: last logged step number
  - sampled_points: number of rows returned
  - keys_returned: list of metric keys in the response

Examples
--------
>>> get_run_history_tool("my-team", "my-project", "gtng2y4l", keys=["loss", "val_loss"])
>>> get_run_history_tool("my-team", "my-project", "h0fm5qp5", samples=100)
"""

MAX_HISTORY_ROWS = 2000


def get_run_history(
    entity_name: str,
    project_name: str,
    run_id: str,
    keys: Optional[List[str]] = None,
    samples: int = 500,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
) -> str:
    """Fetch sampled metric history for a W&B run."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if keys is not None and (
        not isinstance(keys, list)
        or not all(isinstance(key, str) and key.strip() for key in keys)
        or len(keys) > MCP_MAX_HISTORY_KEYS
    ):
        raise ValueError(f"keys must contain at most {MCP_MAX_HISTORY_KEYS} non-empty strings")
    if MCP_HOSTED_MODE and not keys:
        raise ValueError("Hosted MCP requires explicit history keys (1-20 metrics).")
    if min_step is not None and max_step is not None:
        if max_step < min_step:
            raise ValueError("max_step must be greater than or equal to min_step")
        if max_step - min_step + 1 > MCP_MAX_HISTORY_RANGE_STEPS:
            raise ValueError(f"history step range cannot exceed {MCP_MAX_HISTORY_RANGE_STEPS} steps")
    elif MCP_HOSTED_MODE and (min_step is not None or max_step is not None):
        raise ValueError("Hosted MCP step-range queries require both min_step and max_step")

    with track_tool_execution(
        "get_run_history",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "run_id": run_id,
            "keys": keys,
            "samples": samples,
        },
    ):
        api_key = WandBApiManager.get_api_key()
        if not api_key:
            raise ValueError("W&B API key is required to fetch run history.")

        try:
            wandb_api = wandb.Api(
                api_key=api_key,
                overrides={"base_url": WANDB_BASE_URL},
                timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS,
            )
            run_path = f"{entity_name}/{project_name}/{run_id}"
            run = wandb_api.run(run_path)
        except wandb.errors.CommError as e:
            raise ValueError(f"Run not found: {run_path}. Error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to access run {entity_name}/{project_name}/{run_id}: {type(e).__name__}")

        hosted_limit_applied = MCP_HOSTED_MODE and samples > MCP_MAX_HISTORY_SAMPLES
        clamped_samples = min(
            samples, MAX_HISTORY_ROWS, MCP_MAX_HISTORY_SAMPLES if MCP_HOSTED_MODE else MAX_HISTORY_ROWS
        )

        try:
            if min_step is not None or max_step is not None:
                rows = _fetch_step_range(run, clamped_samples, keys, min_step, max_step)
            else:
                history_kwargs: Dict[str, Any] = {"samples": clamped_samples, "pandas": False}
                if keys:
                    history_kwargs["keys"] = keys
                rows = list(run.history(**history_kwargs))
        except Exception as e:
            raise ValueError(f"Failed to fetch history for run {run_id}: {e}")

        clean_rows = []
        for row in rows:
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

        result_dict: Dict[str, Any] = {
            "rows": clean_rows,
            "run_id": run_id,
            "run_name": run.name,
            "total_steps": total_steps,
            "sampled_points": len(clean_rows),
            "keys_returned": sorted(keys_in_response),
        }
        if truncated:
            result_dict["truncation_note"] = (
                f"Downsampled from {original_count} to {len(clean_rows)} rows to fit token budget. "
                "Use keys= to select fewer metrics or reduce samples."
            )
        if hosted_limit_applied:
            result_dict["hosted_limit_note"] = (
                f"Hosted MCP limits history responses to {MCP_MAX_HISTORY_SAMPLES} sampled rows. "
                "Reduce samples or use keys= to select fewer metrics."
            )
        return json.dumps(result_dict)


def _fetch_step_range(
    run: Any,
    clamped_samples: int,
    keys: Optional[List[str]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> List[Dict[str, Any]]:
    """Fetch history rows for a step range using a tiered strategy.

    Strategy order:
      1. scan_history (parquet-backed when available with SDK-managed fallback;
         fails silently when lastHistoryStep == -1)
      2. history() sampled fallback (always works, ignores step bounds)
    """
    # Strategy 1: scan_history
    scan_kwargs: Dict[str, Any] = {}
    if keys:
        scan_kwargs["keys"] = keys
    if min_step is not None:
        scan_kwargs["min_step"] = min_step
    if max_step is not None:
        scan_kwargs["max_step"] = max_step
    scan_limit = MCP_MAX_HISTORY_RANGE_STEPS
    if min_step is not None and max_step is not None:
        scan_limit = min(scan_limit, max_step - min_step + 1)
    scan_kwargs["page_size"] = min(1000, scan_limit)
    scanned_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(run.scan_history(**scan_kwargs)):
        if index >= scan_limit:
            break
        if index % 100 == 0:
            raise_if_tool_deadline_exceeded()
        scanned_rows.append(row)
    rows = _evenly_sample(scanned_rows, clamped_samples)
    if rows:
        return rows

    # Strategy 2: history() sampled fallback (ignores step bounds but always works)
    last_step = getattr(run, "lastHistoryStep", 0) or 0
    if last_step <= 0:
        logger.warning(
            "scan_history returned 0 rows (lastHistoryStep=%s). "
            "Falling back to history(samples=%d) which ignores step bounds.",
            last_step,
            clamped_samples,
        )
        history_kwargs: Dict[str, Any] = {"samples": clamped_samples, "pandas": False}
        if keys:
            history_kwargs["keys"] = keys
        return list(run.history(**history_kwargs))

    return rows


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
