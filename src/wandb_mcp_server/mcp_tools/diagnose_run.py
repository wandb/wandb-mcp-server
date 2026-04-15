"""Diagnose a W&B run: convergence, overfitting, NaN detection, tail statistics."""

import json
import math
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DIAGNOSE_RUN_TOOL_DESCRIPTION = """Diagnose a W&B run's training health.

Automatically detects convergence, overfitting, NaN values, and provides
tail statistics. Returns actionable recommendations.

<when_to_use>
Call when the user asks "is my run okay?", "has it converged?", "is it
overfitting?", or wants a health check on a training run.

This tool samples the run's history and applies standard ML diagnostic
heuristics. Use get_run_history_tool for raw metric data instead.
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

Returns
-------
JSON with diagnosis (converged/diverging/plateaued/insufficient_data),
overfit_signal, nan_warnings, tail_stats, and recommendations.
"""

DIAGNOSIS_SAMPLES = 500


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


def diagnose_run(
    entity_name: str,
    project_name: str,
    run_id: str,
    loss_key: Optional[str] = None,
    val_loss_key: Optional[str] = None,
) -> str:
    """Diagnose a W&B run's training health."""
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "diagnose_run",
        api.viewer,
        {"entity_name": entity_name, "project_name": project_name, "run_id": run_id},
    ) as ctx:
        try:
            run = api.run(f"{entity_name}/{project_name}/{run_id}")
        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "run_not_found", "message": str(e)[:500]})

        try:
            history_rows = list(run.scan_history(page_size=DIAGNOSIS_SAMPLES))[:DIAGNOSIS_SAMPLES]
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
                }
            )

        all_keys = set()
        for row in history_rows[:10]:
            all_keys.update(k for k in row.keys() if not k.startswith("_"))
        all_keys = sorted(all_keys)

        if loss_key is None:
            loss_key = _auto_detect_key(all_keys, ["train_loss", "train/loss", "loss"])
        if val_loss_key is None:
            val_loss_key = _auto_detect_key(
                all_keys, ["val_loss", "val/loss", "eval_loss", "eval/loss", "validation_loss"]
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
            },
            default=str,
        )
