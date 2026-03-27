"""Retrieve sampled time-series metric history for a W&B run.

Uses `wandb.Api().run().history()` for sampled data and a tiered
strategy for step-range queries: beta_scan_history (parquet) first,
scan_history (GraphQL) second, history() (sampled) as last resort.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import wandb

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import WANDB_BASE_URL
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call
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
    If empty, returns all logged keys (can be large).
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

    try:
        api = WandBApiManager.get_api()
        log_tool_call(
            "get_run_history",
            api.viewer,
            {
                "entity_name": entity_name,
                "project_name": project_name,
                "run_id": run_id,
                "keys": keys,
                "samples": samples,
            },
        )
    except Exception:
        logger.debug("analytics emit failed", exc_info=True)

    api_key = WandBApiManager.get_api_key()
    if not api_key:
        raise ValueError("W&B API key is required to fetch run history.")

    try:
        wandb_api = wandb.Api(api_key=api_key, overrides={"base_url": WANDB_BASE_URL})
        run_path = f"{entity_name}/{project_name}/{run_id}"
        run = wandb_api.run(run_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Run not found: {run_path}. Error: {e}")
    except Exception as e:
        raise ValueError(f"Failed to access run {entity_name}/{project_name}/{run_id}: {type(e).__name__}")

    clamped_samples = min(samples, MAX_HISTORY_ROWS)

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
      1. beta_scan_history (parquet via wandb-core, works on all run types)
      2. scan_history (GraphQL, fails silently when lastHistoryStep == -1)
      3. history() sampled fallback (always works, ignores step bounds)
    """
    # Strategy 1: beta_scan_history
    try:
        beta_kwargs: Dict[str, Any] = {"min_step": min_step or 0}
        if keys:
            beta_kwargs["keys"] = keys
        if max_step is not None:
            beta_kwargs["max_step"] = max_step
        beta_kwargs["page_size"] = min(clamped_samples, 1000)
        rows = _reservoir_sample(run.beta_scan_history(**beta_kwargs), clamped_samples)
        if rows:
            return rows
        logger.info("beta_scan_history returned 0 rows, trying scan_history")
    except Exception as e:
        logger.info(f"beta_scan_history unavailable ({type(e).__name__}), trying scan_history")

    # Strategy 2: scan_history
    scan_kwargs: Dict[str, Any] = {}
    if keys:
        scan_kwargs["keys"] = keys
    if min_step is not None:
        scan_kwargs["min_step"] = min_step
    if max_step is not None:
        scan_kwargs["max_step"] = max_step
    rows = _reservoir_sample(run.scan_history(**scan_kwargs), clamped_samples)
    if rows:
        return rows

    # Strategy 3: history() sampled fallback (ignores step bounds but always works)
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


def _reservoir_sample(iterator: Any, max_rows: int) -> List[Dict[str, Any]]:
    """Single-pass reservoir sampling over an iterator of dicts."""
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(iterator):
        if idx < max_rows:
            rows.append(row)
        else:
            j = random.randint(0, idx)
            if j < max_rows:
                rows[j] = row
    rows.sort(key=lambda r: r.get("_step", 0))
    return rows


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
