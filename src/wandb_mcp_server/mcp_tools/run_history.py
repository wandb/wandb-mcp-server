"""Retrieve sampled time-series metric history for a W&B run.

Uses the actor-isolated W&B public API client, independent bounded series for
multi-key reads, and the public scan API for single-key step ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Literal, Optional

import wandb

from wandb_mcp_server.api_client import (
    WandBApiManager,
    raise_for_wandb_server_busy,
    wandb_server_busy_from_exception,
)
from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.config import (
    MCP_HOSTED_MODE,
    MCP_MAX_HISTORY_KEYS,
    MCP_MAX_HISTORY_RANGE_STEPS,
    MCP_MAX_HISTORY_SAMPLES,
    MCP_WORKLOAD_PROFILE,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.trace_utils import count_tokens_conservative
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import (
    SelectiveReadUnavailable,
    fetch_metric_value_steps,
    fetch_sampled_history_series,
)

logger = get_rich_logger(__name__)

GET_RUN_HISTORY_TOOL_DESCRIPTION = """Retrieve bounded time-series metric data from a W&B run.

Use bounded independent-series sampling for explicit multi-key default-history
overviews and ranges, SDK scans for single-key ranges, and an exact point lookup
for a logged x-axis value. Every response states the
retrieval method, whether values are sampled or exact, rows scanned, coverage,
and any profile or response-budget truncation.

<when_to_use>
Call this tool when the user asks about training curves, metric trends over time,
loss plots, or any time-series data logged to a W&B run. This is the only tool
that provides step-by-step metric history -- query_wandb_tool returns run-level
summary metrics but not the full training history.

Typical workflow:
1. For an unfamiliar project, use probe_project_tool to discover indexed keys.
2. Use query_wandb_tool with summary_keys/config_keys to select runs.
3. Use get_run_history_tool with explicit keys and x_axis for targeted curves.
4. Use create_wandb_report_tool to visualize the results.
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
    For explicit multi-key default-history sampled and ranged collection reads,
    keys logged on different cadences are outer-unioned by step; a row may carry
    only the requested metrics logged at that point. target_x is an exact point
    lookup rather than a collection join.
    Hosted deployments require 1-20 explicit keys to prevent accidental retrieval
    of extremely high-cardinality histories.
samples : int, optional
    Total merged-row budget shared across all requested keys. Defaults to 500.
    Use fewer samples for quick overviews, more for detailed analysis.
min_step : int, optional
    Inclusive non-negative minimum step to include. Defaults to None.
max_step : int, optional
    Inclusive non-negative maximum step to include. Defaults to None.
x_axis : str, optional
    History x-axis. Defaults to "_step". Set this to a logged monotonic metric
    such as "validation/step" for custom-axis projection or target lookup. For
    collection reads, the custom axis and metric must occur on the same row.
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
  - requested_keys, optional join, matching_rows, per-key row counts, and
    unobserved/missing/omitted keys. Unobserved means no usable finite/non-null
    value appeared in a bounded result; missing is emitted only for exact counts.
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
_MAX_SAFE_HISTORY_STEP = (1 << 63) - 2


@dataclass(frozen=True)
class _HistoryFetch:
    rows: list[dict[str, Any]]
    method: str
    sampled: bool
    exact: bool
    rows_scanned: int
    compatibility_caveat: str | None = None
    matching_rows: int | None = None
    observed_key_counts: dict[str, int] | None = None
    key_counts_exact: bool = False
    source_truncated: bool = False


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
    for name, value in (("min_step", min_step), ("max_step", max_step)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        if value > _MAX_SAFE_HISTORY_STEP:
            raise ValueError(f"{name} must be at most {_MAX_SAFE_HISTORY_STEP}")
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

    requested_keys = list(dict.fromkeys(keys or []))

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
            # Reuse the actor-isolated public API client. Constructing a fresh
            # ``wandb.Api`` performs credential validation and creates a new
            # transport before every history read, adding a second upstream
            # request and avoidable connection churn to a latency-sensitive
            # path.
            wandb_api = WandBApiManager.get_api(api_key)
            run_path = f"{entity_name}/{project_name}/{run_id}"
            run = wandb_api.run(run_path)
            # This public property performs a narrow, fresh historyKeys read.
            # Do not call Run.load(force=True): Api.run() already loaded the
            # full Run fragment, and forcing it again would repeatedly fetch
            # unbounded config/summary/system metrics on wide active runs.
            snapshot_last_step = run.lastHistoryStep
        except wandb.errors.CommError as e:
            if busy := wandb_server_busy_from_exception(e):
                raise busy from e
            raise ValueError(f"Run not found: {run_path}. Error: {e}")
        except Exception as e:
            if busy := wandb_server_busy_from_exception(e):
                raise busy from e
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
                    keys=requested_keys,
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
                fetched = _fetch_step_range_with_metadata(
                    wandb_api,
                    run,
                    entity_name=entity_name,
                    project_name=project_name,
                    run_id=run_id,
                    clamped_samples=clamped_samples,
                    keys=requested_keys,
                    min_step=min_step,
                    max_step=max_step,
                    snapshot_max_step=snapshot_last_step,
                    x_axis=x_axis,
                    stream=stream,
                )
            elif stream == "default" and len(requested_keys) > 1:
                fetched = _fetch_independent_sampled_history(
                    wandb_api,
                    entity_name=entity_name,
                    project_name=project_name,
                    run_id=run_id,
                    keys=requested_keys,
                    x_axis=x_axis,
                    clamped_samples=clamped_samples,
                    method="batched_sampled_history",
                    max_step=(
                        snapshot_last_step
                        if isinstance(snapshot_last_step, int)
                        and not isinstance(snapshot_last_step, bool)
                        and snapshot_last_step >= 0
                        else None
                    ),
                )
            else:
                history_kwargs: Dict[str, Any] = {"samples": clamped_samples, "pandas": False}
                # W&B 0.28 rejects keys with the system stream. Sample the
                # bounded stream first, then project requested metrics locally.
                if requested_keys and stream == "default":
                    history_kwargs["keys"] = requested_keys
                history_kwargs["x_axis"] = x_axis
                history_kwargs["stream"] = stream
                rows = list(run.history(**history_kwargs))
                if stream == "system" and requested_keys:
                    rows = _project_system_history_rows(
                        rows,
                        requested_keys=requested_keys,
                        x_axis=x_axis,
                    )
                clean_sampled_rows = _clean_history_rows(rows)
                observed_counts = _key_row_counts(clean_sampled_rows, requested_keys)
                matching_rows = len(_filter_rows_with_requested_values(clean_sampled_rows, requested_keys))
                fetched = _HistoryFetch(
                    rows=_key_aware_sample(
                        clean_sampled_rows,
                        clamped_samples,
                        requested_keys,
                    ),
                    method="sdk_sample",
                    sampled=True,
                    exact=False,
                    rows_scanned=len(rows),
                    matching_rows=matching_rows,
                    observed_key_counts=observed_counts,
                    key_counts_exact=False,
                )
        except Exception as e:
            raise_for_wandb_server_busy(e)
            raise ValueError(f"Failed to fetch history for run {run_id}: {e}")

        clean_rows = _clean_history_rows(fetched.rows)
        observed_counts = fetched.observed_key_counts or _key_row_counts(
            clean_rows,
            requested_keys,
        )
        matching_rows = (
            fetched.matching_rows
            if fetched.matching_rows is not None
            else len(_filter_rows_with_requested_values(clean_rows, requested_keys))
        )

        from wandb_mcp_server.config import MAX_RESPONSE_TOKENS

        total_steps = (
            snapshot_last_step
            if isinstance(snapshot_last_step, int) and not isinstance(snapshot_last_step, bool)
            else len(clean_rows)
        )
        original_count = len(clean_rows)

        returned_counts = _key_row_counts(clean_rows, requested_keys)
        keys_in_response = sorted({key for row in clean_rows for key in row if key != "_step"})
        unobserved_keys = [key for key in requested_keys if observed_counts.get(key, 0) == 0]
        keys_omitted_by_limits = [
            key for key in requested_keys if observed_counts.get(key, 0) > 0 and returned_counts.get(key, 0) == 0
        ]

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
            "keys_returned": keys_in_response,
            "requested_keys": requested_keys,
            "matching_rows": matching_rows,
            "matching_rows_exact": fetched.key_counts_exact,
            "key_row_counts": {
                key: {
                    "observed": observed_counts.get(key, 0),
                    "returned": returned_counts.get(key, 0),
                }
                for key in requested_keys
            },
            "unobserved_keys": unobserved_keys,
            "keys_omitted_by_limits": keys_omitted_by_limits,
            "key_counts_exact": fetched.key_counts_exact,
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
            "source_truncated": fetched.source_truncated,
            "truncated": profile_limit_applied or fetched.source_truncated,
        }
        if stream == "default" and len(requested_keys) > 1 and target_x is None:
            result_dict["join"] = "outer"
        if fetched.key_counts_exact:
            result_dict["missing_keys"] = unobserved_keys
        if profile_limit_applied:
            result_dict["profile_limit_note"] = (
                f"The {MCP_WORKLOAD_PROFILE} workload profile limits history responses to "
                f"{MCP_MAX_HISTORY_SAMPLES} rows."
            )
            if MCP_HOSTED_MODE:
                result_dict["hosted_limit_note"] = result_dict["profile_limit_note"]
        if fetched.compatibility_caveat:
            result_dict["compatibility_caveat"] = fetched.compatibility_caveat
        return _serialize_history_response(
            result_dict,
            source_rows=clean_rows,
            requested_keys=requested_keys,
            observed_counts=observed_counts,
            original_count=original_count,
            max_tokens=MAX_RESPONSE_TOKENS,
        )


def _scan_history_rows(
    run: Any,
    *,
    keys: list[str] | None,
    min_step: int | None,
    max_step: int | None,
    scan_limit: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    scan_kwargs: Dict[str, Any] = {}
    if keys:
        scan_kwargs["keys"] = keys
    if min_step is not None:
        scan_kwargs["min_step"] = min_step
    if max_step is not None:
        # Public SDK scan_history treats max_step as exclusive; the MCP
        # interface documents max_step as inclusive.
        scan_kwargs["max_step"] = max_step + 1
    # Active runs can grow while the actor-scoped Api object remains cached.
    # Always rebuild the SDK history reader instead of reusing cached parquet.
    scan_kwargs["use_cache"] = False
    scan_kwargs["page_size"] = min(1000, max(1, scan_limit))
    scanned_rows: list[dict[str, Any]] = []
    rows_scanned = 0
    source_truncated = False
    for index, row in enumerate(islice(run.scan_history(**scan_kwargs), scan_limit + 1)):
        if index % 100 == 0:
            raise_if_tool_deadline_exceeded()
        rows_scanned += 1
        if index >= scan_limit:
            source_truncated = True
            break
        scanned_rows.append(row)
    return scanned_rows, rows_scanned, source_truncated


def _fetch_independent_sampled_history(
    api: Any,
    *,
    entity_name: str,
    project_name: str,
    run_id: str,
    keys: Sequence[str],
    x_axis: str,
    clamped_samples: int,
    method: str,
    min_step: int | None = None,
    max_step: int | None = None,
) -> _HistoryFetch:
    """Fetch, outer-join, and sample independent key series."""
    batch = fetch_sampled_history_series(
        api,
        entity=entity_name,
        project=project_name,
        run_id=run_id,
        keys=keys,
        x_axis=x_axis,
        samples=clamped_samples,
        min_step=min_step,
        max_step=max_step,
    )
    merged_rows = _outer_join_history_series(batch.series, x_axis=x_axis)
    clean_merged_rows = _clean_history_rows(merged_rows)
    value_rows = _filter_rows_with_requested_values(clean_merged_rows, keys)
    observed_counts = _key_row_counts(clean_merged_rows, keys)
    return _HistoryFetch(
        rows=_key_aware_sample(value_rows, clamped_samples, keys),
        method=method,
        sampled=True,
        exact=False,
        rows_scanned=batch.rows_received,
        compatibility_caveat=(
            "Custom-axis collection reads include only metric values logged on "
            "a row containing the requested x-axis; values without that axis "
            "cannot be aligned and are not returned."
            if x_axis != "_step"
            else None
        ),
        matching_rows=len(value_rows),
        observed_key_counts=observed_counts,
        key_counts_exact=False,
    )


def _fetch_step_range_with_metadata(
    api: Any,
    run: Any,
    *,
    entity_name: str,
    project_name: str,
    run_id: str,
    clamped_samples: int,
    keys: Optional[List[str]],
    min_step: Optional[int],
    max_step: Optional[int],
    snapshot_max_step: int | None,
    x_axis: str,
    stream: Literal["default", "system"],
) -> _HistoryFetch:
    """Fetch history rows for a step range using a bounded strategy.

    Strategy order:
      1. independently sampled, bounded series for multi-key default history
      2. scan_history for single-key/default unprojected ranges
      3. history() sampled fallback when the SDK scan is unavailable
    """
    if stream == "default" and keys and len(keys) > 1:
        effective_max_step = max_step
        if isinstance(snapshot_max_step, int) and not isinstance(snapshot_max_step, bool) and snapshot_max_step >= 0:
            effective_max_step = min(max_step, snapshot_max_step) if max_step is not None else snapshot_max_step
            if min_step is not None and min_step > effective_max_step:
                return _HistoryFetch(
                    rows=[],
                    method="batched_sampled_history_range",
                    sampled=True,
                    exact=False,
                    rows_scanned=0,
                    matching_rows=0,
                    observed_key_counts={key: 0 for key in keys},
                    key_counts_exact=False,
                )
        return _fetch_independent_sampled_history(
            api,
            entity_name=entity_name,
            project_name=project_name,
            run_id=run_id,
            keys=keys,
            x_axis=x_axis,
            clamped_samples=clamped_samples,
            method="batched_sampled_history_range",
            min_step=min_step,
            max_step=effective_max_step,
        )

    scan_limit = MCP_MAX_HISTORY_RANGE_STEPS
    if min_step is not None and max_step is not None:
        scan_limit = min(scan_limit, max_step - min_step + 1)
    scan_keys = list(dict.fromkeys([x_axis, "_step", *(keys or [])])) if keys else None
    scanned_rows, rows_scanned, source_truncated = _scan_history_rows(
        run,
        keys=scan_keys,
        min_step=min_step,
        max_step=max_step,
        scan_limit=scan_limit,
    )
    clean_scanned_rows = _clean_history_rows(scanned_rows)
    matching = _filter_rows_with_requested_values(clean_scanned_rows, keys or [])
    observed_counts = _key_row_counts(matching, keys or [])
    if matching:
        sampled_rows = _key_aware_sample(matching, clamped_samples, keys or [])
        return _HistoryFetch(
            rows=sampled_rows,
            method="sdk_bounded_scan",
            sampled=source_truncated or len(matching) > len(sampled_rows),
            exact=False,
            rows_scanned=rows_scanned,
            matching_rows=len(matching),
            observed_key_counts=observed_counts,
            compatibility_caveat=(
                "The bounded range contained more history rows than its internal-step span; "
                "counts cover only the retained prefix."
                if source_truncated
                else None
            ),
            key_counts_exact=(min_step is not None and max_step is not None and not source_truncated),
            source_truncated=source_truncated,
        )

    # Strategy 2: history() sampled fallback (ignores step bounds but always works)
    last_step = getattr(run, "lastHistoryStep", 0) or 0
    if last_step <= 0:
        logger.warning(
            "scan_history returned 0 rows (lastHistoryStep=%s). "
            "Falling back to history(samples=%d) which ignores step bounds.",
            last_step,
            clamped_samples,
        )
        history_kwargs: Dict[str, Any] = {
            "samples": clamped_samples,
            "pandas": False,
            "stream": stream,
            "x_axis": x_axis,
        }
        if keys and stream == "default":
            history_kwargs["keys"] = keys
        fallback_rows = list(run.history(**history_kwargs))
        fallback_scanned = len(fallback_rows)
        if stream == "system" and keys:
            fallback_rows = _project_system_history_rows(
                fallback_rows,
                requested_keys=keys,
                x_axis=x_axis,
            )

        clean_fallback = _clean_history_rows(fallback_rows)
        clean_fallback = _filter_rows_to_step_bounds(
            clean_fallback,
            min_step=min_step,
            max_step=max_step,
        )
        matching_fallback = _filter_rows_with_requested_values(clean_fallback, keys or [])
        fallback_counts = _key_row_counts(matching_fallback, keys or [])
        sampled_fallback = _key_aware_sample(
            matching_fallback,
            clamped_samples,
            keys or [],
        )
        return _HistoryFetch(
            rows=sampled_fallback,
            method="sdk_sample_fallback",
            sampled=True,
            exact=False,
            rows_scanned=fallback_scanned,
            compatibility_caveat=(
                "scan_history was unavailable; sampled fallback was filtered locally "
                "and may not cover every value in the requested step range"
            ),
            matching_rows=len(matching_fallback),
            observed_key_counts=fallback_counts,
            key_counts_exact=False,
        )

    return _HistoryFetch(
        rows=[],
        method="sdk_bounded_scan",
        sampled=False,
        exact=False,
        rows_scanned=rows_scanned,
        matching_rows=0,
        observed_key_counts=observed_counts,
        key_counts_exact=(min_step is not None and max_step is not None and not source_truncated),
        source_truncated=source_truncated,
    )


_DROP_HISTORY_VALUE = object()


def _is_observed_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _clean_history_value(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Return a bounded JSON-safe history value without following cycles."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP_HISTORY_VALUE
    if isinstance(value, str):
        return value if len(value) <= 10_000 else value[:9_984] + "...[truncated]"
    if depth >= 6:
        return "<max-depth>"

    seen = set() if seen is None else seen
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return "<cycle>"
        seen.add(identity)
        try:
            cleaned: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 100:
                    cleaned["_truncated"] = f"{len(value) - 100} mapping entries omitted"
                    break
                normalized = _clean_history_value(item, depth=depth + 1, seen=seen)
                if normalized is not _DROP_HISTORY_VALUE:
                    cleaned[str(key)] = normalized
            return cleaned
        finally:
            seen.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in seen:
            return "<cycle>"
        seen.add(identity)
        try:
            retained = list(islice(iter(value), 101))
            cleaned_items = []
            for item in retained[:100]:
                normalized = _clean_history_value(item, depth=depth + 1, seen=seen)
                if normalized is not _DROP_HISTORY_VALUE:
                    cleaned_items.append(normalized)
            if len(retained) > 100:
                omitted = max(1, len(value) - 100)
                cleaned_items.append(f"<{omitted} list entries omitted>")
            return cleaned_items
        finally:
            seen.remove(identity)

    rendered = str(value)
    return rendered if len(rendered) <= 10_000 else rendered[:9_984] + "...[truncated]"


def _clean_history_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            normalized_key = str(key)
            if normalized_key.startswith("_") and normalized_key not in {
                "_step",
                "_timestamp",
                "_runtime",
            }:
                continue
            if not _is_observed_value(value):
                continue
            normalized = _clean_history_value(value)
            if normalized is not _DROP_HISTORY_VALUE:
                clean_row[normalized_key] = normalized
        clean_rows.append(clean_row)
    return clean_rows


def _filter_rows_with_requested_values(
    rows: Sequence[dict[str, Any]],
    requested_keys: Sequence[str],
) -> list[dict[str, Any]]:
    if not requested_keys:
        return list(rows)
    return [row for row in rows if any(key in row and _is_observed_value(row[key]) for key in requested_keys)]


def _key_row_counts(
    rows: Sequence[Mapping[str, Any]],
    requested_keys: Sequence[str],
) -> dict[str, int]:
    return {key: sum(key in row and _is_observed_value(row[key]) for row in rows) for key in requested_keys}


def _project_system_history_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_keys: Sequence[str],
    x_axis: str,
) -> list[dict[str, Any]]:
    retained_keys = set(requested_keys) | {x_axis, "_step", "_timestamp", "_runtime"}
    projected = [{str(key): value for key, value in row.items() if str(key) in retained_keys} for row in rows]
    return _filter_rows_with_requested_values(projected, requested_keys)


def _filter_rows_to_step_bounds(
    rows: Sequence[dict[str, Any]],
    *,
    min_step: int | None,
    max_step: int | None,
) -> list[dict[str, Any]]:
    if min_step is None and max_step is None:
        return list(rows)
    bounded: list[dict[str, Any]] = []
    for row in rows:
        step = _numeric_x(row, "_step")
        if step is None:
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        bounded.append(row)
    return bounded


def _axis_token(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _outer_join_history_series(
    series: Sequence[Sequence[Mapping[str, Any]]],
    *,
    x_axis: str,
) -> list[dict[str, Any]]:
    """Outer-join independently sampled series without collapsing duplicates."""
    joined: dict[tuple[Any, ...], dict[str, Any]] = {}
    insertion_order: dict[tuple[Any, ...], int] = {}
    next_order = 0

    for series_index, rows in enumerate(series):
        occurrences: dict[tuple[str, str], int] = {}
        for row_index, row in enumerate(rows):
            axis_name: str | None = None
            axis_value: Any = None
            if _is_observed_value(row.get("_step")):
                axis_name, axis_value = "_step", row.get("_step")
            elif _is_observed_value(row.get(x_axis)):
                axis_name, axis_value = x_axis, row.get(x_axis)

            if axis_name is None:
                identity: tuple[Any, ...] = ("unkeyed", series_index, row_index)
            else:
                base = (axis_name, _axis_token(axis_value))
                occurrence = occurrences.get(base, 0)
                occurrences[base] = occurrence + 1
                identity = (axis_name, base[1], occurrence)

            if identity not in joined:
                joined[identity] = {}
                insertion_order[identity] = next_order
                next_order += 1
            target = joined[identity]
            for key, value in row.items():
                normalized_key = str(key)
                if normalized_key not in target or not _is_observed_value(target[normalized_key]):
                    target[normalized_key] = value

    def sort_key(item: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[Any, ...]:
        identity, row = item
        step_value = row.get("_step")
        axis_value = step_value if _is_observed_value(step_value) else row.get(x_axis)
        if isinstance(axis_value, (int, float)) and not isinstance(axis_value, bool):
            numeric = float(axis_value)
            if math.isfinite(numeric):
                return (0, numeric, insertion_order[identity])
        if axis_value is not None:
            return (1, str(axis_value), insertion_order[identity])
        return (2, insertion_order[identity])

    return [dict(row) for _, row in sorted(joined.items(), key=sort_key)]


def _evenly_spaced_indexes(indexes: Sequence[int], count: int) -> list[int]:
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[0]]
    last_index = len(indexes) - 1
    positions = [round(position * last_index / (count - 1)) for position in range(count)]
    return [indexes[position] for position in positions]


def _progressive_even_indexes(indexes: Sequence[int]) -> list[int]:
    """Order indexes so each prefix expands coverage across the full series."""
    indexes = list(indexes)
    if len(indexes) <= 2:
        return indexes

    ordered_positions = [0, len(indexes) - 1]
    intervals = [(0, len(indexes) - 1)]
    while intervals:
        next_intervals: list[tuple[int, int]] = []
        for left, right in intervals:
            midpoint = (left + right) // 2
            if midpoint not in {left, right}:
                ordered_positions.append(midpoint)
            if midpoint - left > 1:
                next_intervals.append((left, midpoint))
            if right - midpoint > 1:
                next_intervals.append((midpoint, right))
        intervals = next_intervals
    return [indexes[position] for position in ordered_positions]


def _key_aware_sample(
    rows: Sequence[dict[str, Any]],
    max_rows: int,
    requested_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Sample a merged timeline while preserving sparse-key representation."""
    rows = list(rows)
    if len(rows) <= max_rows:
        return rows
    if max_rows <= 0:
        return []
    if not requested_keys:
        return _evenly_sample(rows, max_rows)

    key_indexes = {
        key: [index for index, row in enumerate(rows) if key in row and _is_observed_value(row[key])]
        for key in requested_keys
    }
    key_indexes = {key: indexes for key, indexes in key_indexes.items() if indexes}
    if not key_indexes:
        return _evenly_sample(rows, max_rows)

    ordered_keys = [key for key in requested_keys if key in key_indexes]
    key_order = {key: position for position, key in enumerate(ordered_keys)}
    memberships: list[set[str]] = [set() for _ in rows]
    for key, indexes in key_indexes.items():
        for index in indexes:
            memberships[index].add(key)

    selected: set[int] = set()
    returned_counts = {key: 0 for key in ordered_keys}

    def select(index: int) -> None:
        if index in selected:
            return
        selected.add(index)
        for member in memberships[index]:
            returned_counts[member] += 1

    # First cover as many distinct keys as possible. Rows shared by several
    # metrics win, then rows from the sparsest still-unrepresented series.
    uncovered = set(ordered_keys)
    while uncovered and len(selected) < max_rows:
        candidates = [index for index in range(len(rows)) if index not in selected]
        best = max(
            candidates,
            key=lambda index: (
                len(memberships[index] & uncovered),
                sum(1.0 / len(key_indexes[key]) for key in memberships[index] & uncovered),
                -index,
            ),
        )
        represented = memberships[best] & uncovered
        if not represented:
            break
        select(best)
        uncovered -= represented

    if len(selected) >= max_rows:
        return [rows[index] for index in sorted(selected)]

    # Fully retain sparse series, smallest first, whenever all of their
    # additional union rows fit. This is intentionally stronger than assigning
    # equal per-key quotas: a 10-point validation metric should not lose values
    # merely because it shares an 11-row response with a 100-point loss curve.
    for key in sorted(ordered_keys, key=lambda item: (len(key_indexes[item]), key_order[item])):
        additional = [index for index in key_indexes[key] if index not in selected]
        if len(selected) + len(additional) <= max_rows:
            for index in additional:
                select(index)

    # Spend remaining capacity on the least-represented incomplete series. A
    # progressive, precomputed order avoids repeatedly rebuilding an O(n)
    # sample while still expanding coverage across each dense timeline.
    candidate_orders = {key: _progressive_even_indexes(key_indexes[key]) for key in ordered_keys}
    candidate_offsets = {key: 0 for key in ordered_keys}

    def next_candidate(key: str) -> int | None:
        candidates = candidate_orders[key]
        offset = candidate_offsets[key]
        while offset < len(candidates) and candidates[offset] in selected:
            offset += 1
        candidate_offsets[key] = offset
        return candidates[offset] if offset < len(candidates) else None

    while len(selected) < max_rows:
        active = [
            key
            for key in ordered_keys
            if returned_counts[key] < len(key_indexes[key]) and next_candidate(key) is not None
        ]
        if not active:
            break
        key = min(
            active,
            key=lambda item: (
                returned_counts[item] / len(key_indexes[item]),
                returned_counts[item],
                len(key_indexes[item]),
                key_order[item],
            ),
        )
        candidate = next_candidate(key)
        if candidate is None:
            continue
        candidate_offsets[key] += 1
        select(candidate)

    if len(selected) < max_rows:
        remaining_indexes = [index for index in range(len(rows)) if index not in selected]
        selected.update(_evenly_spaced_indexes(remaining_indexes, max_rows - len(selected)))
    return [rows[index] for index in sorted(selected)[:max_rows]]


def _refresh_history_row_metadata(
    payload: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    requested_keys: Sequence[str],
    observed_counts: Mapping[str, int],
    original_count: int,
) -> None:
    """Update every response field whose truth depends on retained rows."""
    payload["rows"] = rows
    payload["sampled_points"] = len(rows)
    payload["keys_returned"] = sorted({key for row in rows for key in row if key != "_step"})
    returned_counts = _key_row_counts(rows, requested_keys)
    payload["key_row_counts"] = {
        key: {
            "observed": observed_counts.get(key, 0),
            "returned": returned_counts.get(key, 0),
        }
        for key in requested_keys
    }
    payload["keys_omitted_by_limits"] = [
        key for key in requested_keys if observed_counts.get(key, 0) > 0 and returned_counts.get(key, 0) == 0
    ]

    x_axis = str(payload.get("x_axis") or "_step")
    x_values = [
        float(row[x_axis])
        for row in rows
        if isinstance(row.get(x_axis), (int, float))
        and not isinstance(row.get(x_axis), bool)
        and math.isfinite(float(row[x_axis]))
    ]
    coverage = payload.get("coverage")
    if isinstance(coverage, dict):
        coverage["first_x"] = x_values[0] if x_values else None
        coverage["last_x"] = x_values[-1] if x_values else None

    response_truncated = len(rows) < original_count
    payload["truncated"] = bool(payload.get("truncated")) or response_truncated
    if response_truncated:
        payload["truncation_note"] = (
            f"Downsampled from {original_count} to {len(rows)} rows to fit the response token budget. "
            "Use keys= to select fewer metrics or reduce samples."
        )


def _serialize_history_response(
    payload: dict[str, Any],
    *,
    source_rows: list[dict[str, Any]],
    requested_keys: Sequence[str],
    observed_counts: Mapping[str, int],
    original_count: int,
    max_tokens: int,
) -> str:
    """Fit the complete history envelope, not only its rows, to the token cap."""

    def serialize(candidate: Mapping[str, Any]) -> str:
        return json.dumps(
            candidate,
            allow_nan=False,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    candidate = dict(payload)
    candidate["coverage"] = dict(payload.get("coverage") or {})
    rows = _enforce_row_budget(
        source_rows,
        max_tokens * 4,
        requested_keys=requested_keys,
    )
    content_compacted = rows != source_rows
    if content_compacted and len(rows) == len(source_rows):
        candidate["truncated"] = True
        candidate["truncation_note"] = (
            "Compacted oversized history values to fit the response token budget. "
            "Use keys= to select fewer metrics or reduce samples."
        )
    while True:
        _refresh_history_row_metadata(
            candidate,
            rows=rows,
            requested_keys=requested_keys,
            observed_counts=observed_counts,
            original_count=original_count,
        )
        serialized = serialize(candidate)
        token_count = count_tokens_conservative(serialized)
        if token_count <= max_tokens:
            return serialized
        if not rows:
            break
        target_count = (
            0
            if len(rows) == 1
            else min(
                len(rows) - 1,
                max(1, int(len(rows) * max_tokens / token_count * 0.85)),
            )
        )
        rows = _key_aware_sample(source_rows, target_count, requested_keys)

    error_payload = {
        "error": "response_too_large",
        "message": "History response metadata exceeded the configured response budget",
    }
    serialized_error = serialize(error_payload)
    if count_tokens_conservative(serialized_error) <= max_tokens:
        return serialized_error
    # MAX_RESPONSE_TOKENS can technically be configured below the size of a
    # useful structured error. Keep valid JSON and the stable error category.
    return serialize({"error": "response_too_large"})


def _numeric_x(row: Dict[str, Any], x_axis: str) -> float | None:
    value = row.get(x_axis)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _scan_history_for_target(
    run: Any,
    *,
    keys: list[str],
    x_axis: str,
    target_x: float,
    tolerance: float | None,
    scan_limit: int,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Stream a bounded compatibility scan and retain at most one row.

    An exact lookup can stop at the first matching row. A tolerance lookup
    must inspect the bounded window to select the nearest value, but still
    avoids materializing thousands of history rows in memory.
    """
    best_row: dict[str, Any] | None = None
    best_distance = math.inf
    rows_scanned = 0
    scan_kwargs = {
        "keys": keys,
        "page_size": min(1000, max(1, scan_limit)),
        "use_cache": False,
    }

    for index, row in enumerate(islice(run.scan_history(**scan_kwargs), scan_limit)):
        if index % 100 == 0:
            raise_if_tool_deadline_exceeded()
        rows_scanned += 1
        value = _numeric_x(row, x_axis)
        if value is None:
            continue
        distance = abs(value - target_x)
        if distance <= _EXACT_EPSILON:
            return [row], True, rows_scanned
        if tolerance is not None and distance < best_distance:
            best_row = row
            best_distance = distance

    if best_row is not None and best_distance <= tolerance:
        return [best_row], False, rows_scanned
    return [], False, rows_scanned


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
        rows, rows_scanned, _ = _scan_history_rows(
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
        raise_for_wandb_server_busy(exc)
        selected, exact, rows_scanned = _scan_history_for_target(
            run,
            keys=requested_keys,
            x_axis=x_axis,
            target_x=target_x,
            tolerance=tolerance,
            scan_limit=MCP_MAX_HISTORY_RANGE_STEPS,
        )
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
    rows, rows_scanned, _ = _scan_history_rows(
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


def _enforce_row_budget(
    rows: List[Dict[str, Any]],
    budget_chars: int,
    *,
    requested_keys: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Downsample rows to fit within a character budget.

    Preserves even spacing across _step values by taking every Nth row.
    """
    if not rows:
        return rows

    serialized = json.dumps(rows, default=str)
    if len(serialized) <= budget_chars:
        return rows

    per_row = max(1, len(serialized) // len(rows))
    target_count = max(1, budget_chars // per_row)

    target_count = min(target_count, len(rows))
    sampled = _key_aware_sample(rows, target_count, requested_keys)
    while len(json.dumps(sampled, default=str)) > budget_chars and target_count > 1:
        target_count = max(1, target_count - max(1, target_count // 4))
        sampled = _key_aware_sample(rows, target_count, requested_keys)

    if len(json.dumps(sampled, default=str)) <= budget_chars:
        return sampled

    # A single unusually large nested value can exceed the entire budget.
    # Preserve the step and requested-key presence while replacing the value,
    # rather than returning invalid or unbounded JSON.
    compact: dict[str, Any] = {}
    for key, value in sampled[0].items():
        if key not in {"_step", "_timestamp", "_runtime", *requested_keys}:
            continue
        encoded = json.dumps(value, default=str)
        compact[key] = value if len(encoded) <= 256 else "<value-truncated>"
    if len(json.dumps([compact], default=str)) > budget_chars:
        compact = {key: value for key, value in compact.items() if key in {"_step", "_timestamp", "_runtime"}}
        compact["_truncated"] = "history row exceeded response budget"
    return [compact]
