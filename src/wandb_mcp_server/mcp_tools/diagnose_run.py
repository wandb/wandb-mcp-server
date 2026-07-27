"""Diagnose a W&B run: convergence, overfitting, NaN detection, tail statistics."""

import json
import math
from typing import Any, Dict, List, Optional

from wandb_mcp_server.admission import ToolDeadlineExceeded
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import MCP_MAX_HISTORY_KEYS, MCP_MAX_HISTORY_SAMPLES
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import fetch_project_fields, fetch_projected_run

logger = get_rich_logger(__name__)

DIAGNOSE_RUN_TOOL_DESCRIPTION = """Diagnose a W&B run's training health.

Automatically detects convergence, overfitting, NaN values, and provides
tail statistics. Returns actionable recommendations.

<when_to_use>
Call when the user asks "is my run okay?", "has it converged?", "is it
overfitting?", or wants a health check on a training run.

This tool selects a bounded metric set from the project field index, samples
only those history keys, and discloses the selection and coverage behind every
conclusion. Use get_run_history_tool for raw metric data instead.
</when_to_use>

Parameters
----------
entity_name : str
    W&B entity (username or team).
project_name : str
    W&B project name.
run_id : str
    The run to diagnose.
loss_key : str, optional
    Name of the loss metric. Auto-detected if not provided.
val_loss_key : str, optional
    Name of the validation loss metric. Auto-detected if not provided.
config_keys : list of str, optional
    Projected config context to include. When omitted, a bounded indexed set is selected.
summary_keys : list of str, optional
    Summary/history metrics to analyze. When omitted, bounded numeric fields are selected.
x_axis : str, optional
    X-axis used for history sampling. Defaults to "_step".
samples : int, optional
    Number of sampled history rows, capped by the active workload profile.

Returns
-------
JSON with diagnosis (converged/diverging/plateaued/insufficient_data),
overfit_signal, nan_warnings, tail_stats, and recommendations.
"""

DIAGNOSIS_SAMPLES = min(500, MCP_MAX_HISTORY_SAMPLES)
_DIAGNOSIS_CONFIG_LIMIT = 10


def _auto_detect_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    """Find a key matching any of the patterns (case-insensitive)."""
    for pattern in patterns:
        for k in keys:
            if pattern in k.lower():
                return k
    return None


def _compute_trend(values: List[float]) -> str:
    """Classify the trend of a metric series."""
    if len(values) < 10:
        return "insufficient_data"

    half = len(values) // 2
    first_half_mean = sum(values[:half]) / half
    second_half_mean = sum(values[half:]) / (len(values) - half)

    tail_10 = values[-max(1, len(values) // 10) :]
    tail_mean = sum(tail_10) / len(tail_10)

    if second_half_mean < first_half_mean * 0.95:
        if abs(tail_mean - second_half_mean) / max(abs(second_half_mean), 1e-8) < 0.05:
            return "plateaued"
        return "decreasing"
    elif second_half_mean > first_half_mean * 1.05:
        return "increasing"
    else:
        return "plateaued"


def _detect_overfit(train_vals: List[float], val_vals: List[float]) -> Dict[str, Any]:
    """Detect overfitting by comparing train and val loss trends."""
    if len(train_vals) < 10 or len(val_vals) < 10:
        return {"detected": False, "reason": "insufficient_data"}

    min_len = min(len(train_vals), len(val_vals))
    train_vals = train_vals[:min_len]
    val_vals = val_vals[:min_len]

    half = min_len // 2
    train_gap_early = sum(abs(v - t) for t, v in zip(train_vals[:half], val_vals[:half])) / half
    train_gap_late = sum(abs(v - t) for t, v in zip(train_vals[half:], val_vals[half:])) / (min_len - half)

    train_trend = _compute_trend(train_vals)
    val_trend = _compute_trend(val_vals)

    overfit = train_trend == "decreasing" and val_trend in ("increasing", "plateaued")
    gap_growing = train_gap_late > train_gap_early * 1.3

    return {
        "detected": overfit or gap_growing,
        "train_loss_trend": train_trend,
        "val_loss_trend": val_trend,
        "gap_early": round(train_gap_early, 6),
        "gap_late": round(train_gap_late, 6),
        "gap_ratio": round(train_gap_late / max(train_gap_early, 1e-8), 3),
    }


def _strip_field_prefix(path: str, prefixes: tuple[str, ...]) -> str | None:
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return None


def _indexed_diagnosis_keys(api: Any, entity_name: str, project_name: str) -> tuple[list[str], list[str], bool]:
    fields = fetch_project_fields(
        api,
        entity=entity_name,
        project=project_name,
        limit=max(200, MCP_MAX_HISTORY_KEYS * 20),
    )
    configs: list[str] = []
    metrics: list[str] = []
    numeric_types = {"number", "integer", "float", "int", "number[]"}
    for field in fields.items:
        path = field["path"]
        config_key = _strip_field_prefix(path, ("config.", "config/"))
        if config_key and not config_key.startswith(("_", "wandb")):
            configs.append(config_key)
            continue
        summary_key = _strip_field_prefix(
            path,
            ("summary.", "summary/", "summary_metrics.", "summary_metrics/", "summaryMetrics."),
        )
        if (
            summary_key
            and not summary_key.startswith(("_", "wandb/"))
            and field.get("type", "").lower() in numeric_types
        ):
            metrics.append(summary_key)
    priority = ("loss", "accuracy", "score", "precision", "recall", "f1", "auc", "grad")
    metrics.sort(key=lambda key: (not any(token in key.lower() for token in priority), key))
    return (
        list(dict.fromkeys(configs))[:_DIAGNOSIS_CONFIG_LIMIT],
        list(dict.fromkeys(metrics))[:MCP_MAX_HISTORY_KEYS],
        not fields.has_more,
    )


def _validate_optional_keys(name: str, value: Optional[List[str]]) -> None:
    if value is not None and (
        not isinstance(value, list)
        or len(value) > MCP_MAX_HISTORY_KEYS
        or not all(isinstance(key, str) and key.strip() for key in value)
    ):
        raise ValueError(f"{name} must contain at most {MCP_MAX_HISTORY_KEYS} non-empty strings")


def diagnose_run(
    entity_name: str,
    project_name: str,
    run_id: str,
    loss_key: Optional[str] = None,
    val_loss_key: Optional[str] = None,
    config_keys: Optional[List[str]] = None,
    summary_keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    samples: int = DIAGNOSIS_SAMPLES,
) -> str:
    """Diagnose a W&B run's training health."""
    for name, value in (("config_keys", config_keys), ("summary_keys", summary_keys)):
        _validate_optional_keys(name, value)
    if not isinstance(x_axis, str) or not x_axis.strip():
        raise ValueError("x_axis must be a non-empty string")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    effective_samples = min(samples, MCP_MAX_HISTORY_SAMPLES)
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "diagnose_run",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "run_id": run_id,
            "config_key_count": len(config_keys or []),
            "summary_key_count": len(summary_keys or []),
            "x_axis_is_custom": x_axis != "_step",
            "samples": effective_samples,
        },
    ) as ctx:
        selected_config = list(config_keys or [])
        selected_summary = list(summary_keys or [])
        selection_source = "explicit"
        field_index_exhaustive: bool | None = None
        compatibility_caveat: str | None = None
        try:
            if config_keys is None or summary_keys is None:
                auto_config, auto_summary, field_index_exhaustive = _indexed_diagnosis_keys(
                    api, entity_name, project_name
                )
                if config_keys is None:
                    selected_config = auto_config
                if summary_keys is None:
                    selected_summary = auto_summary
                selection_source = "project_field_index"
            projected = fetch_projected_run(
                api,
                entity=entity_name,
                project=project_name,
                run_id=run_id,
                config_keys=selected_config,
                summary_keys=selected_summary,
            )
            if projected is None:
                raise ValueError(f"Run {run_id} not found")
            run = api.run(f"{entity_name}/{project_name}/{run_id}")
        except Exception as e:
            if isinstance(e, ToolDeadlineExceeded):
                raise
            try:
                run = api.run(f"{entity_name}/{project_name}/{run_id}")
            except Exception as run_error:
                ctx.mark_error(f"{type(run_error).__name__}: {run_error}")
                return json.dumps({"error": "run_not_found", "message": str(run_error)[:500]})
            summary = dict(getattr(run, "summary", {}) or {})
            config = dict(getattr(run, "config", {}) or {})
            if config_keys is None:
                selected_config = sorted(key for key in config if not key.startswith(("_", "wandb")))[
                    :_DIAGNOSIS_CONFIG_LIMIT
                ]
            if summary_keys is None:
                selected_summary = sorted(
                    key
                    for key, value in summary.items()
                    if not key.startswith(("_", "wandb/")) and isinstance(value, (int, float))
                )[:MCP_MAX_HISTORY_KEYS]
            projected = {
                "id": run_id,
                "display_name": getattr(run, "name", run_id),
                "state": getattr(run, "state", "unknown"),
                "config": {key: config.get(key) for key in selected_config},
                "summary": {key: summary.get(key) for key in selected_summary},
            }
            selection_source = "bounded_sdk_compatibility"
            compatibility_caveat = (
                f"{type(e).__name__}: indexed/projected fields unavailable; "
                "used the one required SDK run and retained only the disclosed bounded keys"
            )

        if loss_key is None:
            loss_key = _auto_detect_key(selected_summary, ["train_loss", "train/loss", "loss"])
        if val_loss_key is None:
            val_loss_key = _auto_detect_key(
                selected_summary,
                ["val_loss", "val/loss", "validation/loss", "eval_loss", "eval/loss", "validation_loss"],
            )

        requested_keys = list(dict.fromkeys([key for key in (loss_key, val_loss_key, *selected_summary) if key]))[
            :MCP_MAX_HISTORY_KEYS
        ]
        if not requested_keys:
            return json.dumps(
                {
                    "diagnosis": "no_loss_key",
                    "message": "No bounded numeric metric set was available; provide loss_key and val_loss_key.",
                    "run_id": run_id,
                    "run_name": getattr(run, "name", run_id),
                    "selection": {
                        "source": selection_source,
                        "config_keys": selected_config,
                        "summary_keys": selected_summary,
                        "field_index_exhaustive": field_index_exhaustive,
                    },
                }
            )

        try:
            history_rows = list(
                run.history(
                    samples=effective_samples,
                    keys=requested_keys,
                    pandas=False,
                    x_axis=x_axis,
                )
            )
        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "history_fetch_failed", "message": str(e)[:500]})

        if not history_rows:
            return json.dumps(
                {
                    "diagnosis": "no_history",
                    "message": "Run has no logged history steps.",
                    "run_id": run_id,
                    "run_name": getattr(run, "name", run_id),
                    "selection": {
                        "source": selection_source,
                        "config_keys": selected_config,
                        "summary_keys": selected_summary,
                        "field_index_exhaustive": field_index_exhaustive,
                    },
                    "coverage": {
                        "sampled": True,
                        "requested_samples": samples,
                        "effective_samples": effective_samples,
                        "returned_rows": 0,
                        "x_axis": x_axis,
                    },
                }
            )

        all_keys = sorted(
            {key for row in history_rows[:10] for key in row if not key.startswith("_") and key in requested_keys}
        )

        # NaN detection
        nan_warnings = {}
        for key in all_keys:
            vals = [row.get(key) for row in history_rows if row.get(key) is not None]
            nan_count = sum(1 for v in vals if isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            if nan_count > 0:
                nan_warnings[key] = {
                    "nan_count": nan_count,
                    "total": len(vals),
                    "fraction": round(nan_count / max(len(vals), 1), 4),
                }

        # Loss analysis
        diagnosis = "unknown"
        loss_stats = None
        if loss_key:
            loss_vals = [
                row.get(loss_key)
                for row in history_rows
                if isinstance(row.get(loss_key), (int, float)) and not math.isnan(row.get(loss_key, float("nan")))
            ]
            if len(loss_vals) >= 10:
                trend = _compute_trend(loss_vals)
                tail_n = max(1, len(loss_vals) // 10)
                loss_stats = {
                    "key": loss_key,
                    "first_value": round(loss_vals[0], 6),
                    "last_value": round(loss_vals[-1], 6),
                    "min_value": round(min(loss_vals), 6),
                    "tail_mean": round(sum(loss_vals[-tail_n:]) / tail_n, 6),
                    "overall_mean": round(sum(loss_vals) / len(loss_vals), 6),
                    "trend": trend,
                    "total_steps": len(loss_vals),
                }

                if trend == "decreasing":
                    diagnosis = "training"
                elif trend == "plateaued":
                    diagnosis = "converged"
                elif trend == "increasing":
                    diagnosis = "diverging"
            else:
                diagnosis = "insufficient_data"
        else:
            diagnosis = "no_loss_key"

        # Overfit detection
        overfit = {"detected": False, "reason": "no_val_loss_key"}
        if loss_key and val_loss_key:
            train_vals = [
                row.get(loss_key)
                for row in history_rows
                if isinstance(row.get(loss_key), (int, float)) and not math.isnan(row.get(loss_key, float("nan")))
            ]
            val_vals = [
                row.get(val_loss_key)
                for row in history_rows
                if isinstance(row.get(val_loss_key), (int, float))
                and not math.isnan(row.get(val_loss_key, float("nan")))
            ]
            overfit = _detect_overfit(train_vals, val_vals)

        # Recommendations
        recommendations = []
        if diagnosis == "diverging":
            recommendations.append("Loss is increasing -- consider reducing learning rate or checking data pipeline.")
        if diagnosis == "converged":
            recommendations.append("Training appears converged -- run can likely be stopped to save compute.")
        if overfit.get("detected"):
            recommendations.append(
                "Overfitting detected -- consider adding regularization, dropout, or early stopping."
            )
        if nan_warnings:
            recommendations.append(
                f"NaN/Inf values found in {len(nan_warnings)} metric(s) -- check for numerical instability."
            )

        return json.dumps(
            {
                "run_id": run_id,
                "run_name": getattr(run, "name", run_id),
                "run_state": getattr(run, "state", "unknown"),
                "diagnosis": diagnosis,
                "loss_stats": loss_stats,
                "overfit_signal": overfit,
                "nan_warnings": nan_warnings if nan_warnings else None,
                "available_keys": all_keys[:30],
                "detected_keys": {"loss": loss_key, "val_loss": val_loss_key},
                "total_history_rows": len(history_rows),
                "recommendations": recommendations,
                "projected_context": {
                    "config": projected.get("config") or {},
                    "summary": projected.get("summary") or {},
                },
                "selection": {
                    "source": selection_source,
                    "config_keys": selected_config,
                    "summary_keys": selected_summary,
                    "field_index_exhaustive": field_index_exhaustive,
                },
                "coverage": {
                    "sampled": True,
                    "requested_samples": samples,
                    "effective_samples": effective_samples,
                    "returned_rows": len(history_rows),
                    "x_axis": x_axis,
                    "project_exhaustive": False,
                    "conclusions_apply_to_sample": True,
                },
                "compatibility_caveat": compatibility_caveat,
            },
            default=str,
        )
