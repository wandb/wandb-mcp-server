"""Compare two W&B runs: config diff, summary metric diff, optional history overlap."""

import json
import math
from typing import Any, Dict, List, Optional

from wandb_mcp_server.admission import ToolDeadlineExceeded
from wandb_mcp_server.api_client import WandBApiManager, raise_for_wandb_server_busy
from wandb_mcp_server.config import MCP_MAX_HISTORY_KEYS, MCP_MAX_HISTORY_SAMPLES
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import fetch_project_fields, fetch_projected_run

logger = get_rich_logger(__name__)

COMPARE_RUNS_TOOL_DESCRIPTION = """Compare two W&B runs side-by-side.

Returns projected config differences, summary metric deltas, metadata, and
optional sampled history overlap. It never loads complete wide summaries simply
to discover comparison fields.

<when_to_use>
Call when the user asks "what changed between run A and B?", "which run is better?",
or wants to understand why two runs have different performance.

Typical workflow:
1. probe_project_tool for indexed fields when the project is unfamiliar
2. query_wandb_tool with projected keys to identify the two runs
3. compare_runs_tool to see what differs
4. get_run_history_tool for targeted custom-axis detail when needed
</when_to_use>

Parameters
----------
entity_name : str
    W&B entity (username or team).
project_name : str
    W&B project name.
run_id_a : str
    First run ID.
run_id_b : str
    Second run ID.
include_history_overlap : bool, optional
    If True, sample both runs' history and return aligned rows. Default: False.
history_keys : list of str, optional
    Specific metric keys to compare in history. If None, uses common keys.
history_samples : int, optional
    Number of history samples per run. Default: 50.
config_keys : list of str, optional
    Exact config fields to compare. When omitted, a bounded set is selected from
    the project field index and disclosed in the response.
summary_keys : list of str, optional
    Exact summary fields to compare. When omitted, a bounded numeric set is
    selected from the project field index and disclosed in the response.
x_axis : str, optional
    X-axis used for optional history sampling. Defaults to "_step".

Returns
-------
JSON with config_diff, summary_diff, metadata_diff, and optional history_comparison.
"""

DEFAULT_HISTORY_SAMPLES = 50
_AUTO_CONFIG_LIMIT = 10
_AUTO_SUMMARY_LIMIT = 20


def _safe_val(v: Any) -> Any:
    """Make values JSON-serializable."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v


def _diff_dicts(a: Dict, b: Dict) -> Dict[str, Any]:
    """Compute structured diff between two dicts."""
    all_keys = sorted(set(list(a.keys()) + list(b.keys())))
    only_a = {}
    only_b = {}
    changed = {}
    same = {}

    for k in all_keys:
        in_a = k in a
        in_b = k in b
        if in_a and not in_b:
            only_a[k] = _safe_val(a[k])
        elif in_b and not in_a:
            only_b[k] = _safe_val(b[k])
        elif a[k] != b[k]:
            entry: Dict[str, Any] = {"run_a": _safe_val(a[k]), "run_b": _safe_val(b[k])}
            if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
                try:
                    entry["delta"] = round(b[k] - a[k], 6)
                except (TypeError, ValueError):
                    pass
            changed[k] = entry
        else:
            same[k] = _safe_val(a[k])

    return {
        "only_in_run_a": only_a,
        "only_in_run_b": only_b,
        "changed": changed,
        "identical_count": len(same),
    }


def _normalized_indexed_key(path: str, category: str) -> str | None:
    prefixes = {
        "config": ("config.", "config/"),
        "summary": ("summary.", "summary/", "summary_metrics.", "summary_metrics/", "summaryMetrics."),
    }
    for prefix in prefixes[category]:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return None


def _indexed_comparison_keys(api: Any, entity_name: str, project_name: str) -> tuple[list[str], list[str], bool]:
    fields = fetch_project_fields(
        api,
        entity=entity_name,
        project=project_name,
        limit=max(MCP_MAX_HISTORY_KEYS * 20, 200),
    )
    config_keys: list[str] = []
    numeric_summary_keys: list[str] = []
    numeric_types = {"number", "integer", "float", "int", "number[]"}
    for field in fields.items:
        path = field["path"]
        if key := _normalized_indexed_key(path, "config"):
            if not key.startswith(("_", "wandb")):
                config_keys.append(key)
            continue
        if key := _normalized_indexed_key(path, "summary"):
            if not key.startswith(("_", "wandb/")) and field.get("type", "").lower() in numeric_types:
                numeric_summary_keys.append(key)
    priority = ("loss", "accuracy", "score", "precision", "recall", "f1", "auc")
    numeric_summary_keys.sort(key=lambda key: (not any(token in key.lower() for token in priority), key))
    return (
        list(dict.fromkeys(config_keys))[:_AUTO_CONFIG_LIMIT],
        list(dict.fromkeys(numeric_summary_keys))[:_AUTO_SUMMARY_LIMIT],
        not fields.has_more,
    )


def _validate_key_list(name: str, value: Optional[List[str]]) -> None:
    if value is not None and (
        not isinstance(value, list)
        or len(value) > MCP_MAX_HISTORY_KEYS
        or not all(isinstance(key, str) and key.strip() for key in value)
    ):
        raise ValueError(f"{name} must contain at most {MCP_MAX_HISTORY_KEYS} non-empty strings")


def compare_runs(
    entity_name: str,
    project_name: str,
    run_id_a: str,
    run_id_b: str,
    include_history_overlap: bool = False,
    history_keys: Optional[List[str]] = None,
    history_samples: int = DEFAULT_HISTORY_SAMPLES,
    config_keys: Optional[List[str]] = None,
    summary_keys: Optional[List[str]] = None,
    x_axis: str = "_step",
) -> str:
    """Compare two W&B runs."""
    if isinstance(history_samples, bool) or not isinstance(history_samples, int) or history_samples < 1:
        raise ValueError("history_samples must be a positive integer")
    history_samples = min(history_samples, MCP_MAX_HISTORY_SAMPLES)
    for name, value in (
        ("history_keys", history_keys),
        ("config_keys", config_keys),
        ("summary_keys", summary_keys),
    ):
        _validate_key_list(name, value)
    if not isinstance(x_axis, str) or not x_axis.strip():
        raise ValueError("x_axis must be a non-empty string")
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "compare_runs",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "config_key_count": len(config_keys or []),
            "summary_key_count": len(summary_keys or []),
            "history_key_count": len(history_keys or []),
            "include_history_overlap": include_history_overlap,
        },
    ) as ctx:
        selected_config = list(config_keys or [])
        selected_summary = list(summary_keys or [])
        selection_source = "explicit"
        field_index_exhaustive: bool | None = None
        try:
            path = f"{entity_name}/{project_name}"
            if config_keys is None or summary_keys is None:
                auto_config, auto_summary, field_index_exhaustive = _indexed_comparison_keys(
                    api, entity_name, project_name
                )
                if config_keys is None:
                    selected_config = auto_config
                if summary_keys is None:
                    selected_summary = auto_summary
                selection_source = "project_field_index"

            projected_a = fetch_projected_run(
                api,
                entity=entity_name,
                project=project_name,
                run_id=run_id_a,
                config_keys=selected_config,
                summary_keys=selected_summary,
            )
            projected_b = fetch_projected_run(
                api,
                entity=entity_name,
                project=project_name,
                run_id=run_id_b,
                config_keys=selected_config,
                summary_keys=selected_summary,
            )
            if projected_a is None or projected_b is None:
                raise ValueError("one or both runs were not found")
            compatibility_caveat = None
        except Exception as selective_error:
            if isinstance(selective_error, ToolDeadlineExceeded):
                raise
            raise_for_wandb_server_busy(selective_error)
            try:
                run_a = api.run(f"{path}/{run_id_a}")
                run_b = api.run(f"{path}/{run_id_b}")
            except Exception as e:
                raise_for_wandb_server_busy(e)
                ctx.mark_error(f"{type(e).__name__}: {e}")
                return json.dumps({"error": "run_not_found", "message": str(e)[:500]})

            run_config_a = dict(getattr(run_a, "config", {}) or {})
            run_config_b = dict(getattr(run_b, "config", {}) or {})
            run_summary_a = dict(getattr(run_a, "summary", {}) or {})
            run_summary_b = dict(getattr(run_b, "summary", {}) or {})
            if config_keys is None:
                selected_config = sorted(
                    key for key in (set(run_config_a) | set(run_config_b)) if not key.startswith(("_", "wandb"))
                )[:_AUTO_CONFIG_LIMIT]
            if summary_keys is None:
                selected_summary = sorted(
                    key
                    for key in (set(run_summary_a) | set(run_summary_b))
                    if not key.startswith(("_", "wandb/"))
                    and isinstance(run_summary_a.get(key, run_summary_b.get(key)), (int, float))
                )[:_AUTO_SUMMARY_LIMIT]
            projected_a = {
                "id": run_id_a,
                "display_name": getattr(run_a, "name", run_id_a),
                "state": getattr(run_a, "state", "unknown"),
                "created_at": getattr(run_a, "created_at", None),
                "heartbeat_at": getattr(run_a, "heartbeat_at", None),
                "tags": getattr(run_a, "tags", []),
                "group": getattr(run_a, "group", None),
                "config": {key: run_config_a[key] for key in selected_config if key in run_config_a},
                "summary": {key: run_summary_a[key] for key in selected_summary if key in run_summary_a},
            }
            projected_b = {
                "id": run_id_b,
                "display_name": getattr(run_b, "name", run_id_b),
                "state": getattr(run_b, "state", "unknown"),
                "created_at": getattr(run_b, "created_at", None),
                "heartbeat_at": getattr(run_b, "heartbeat_at", None),
                "tags": getattr(run_b, "tags", []),
                "group": getattr(run_b, "group", None),
                "config": {key: run_config_b[key] for key in selected_config if key in run_config_b},
                "summary": {key: run_summary_b[key] for key in selected_summary if key in run_summary_b},
            }
            selection_source = "bounded_sdk_compatibility"
            compatibility_caveat = (
                f"{type(selective_error).__name__}: selective fields unavailable; "
                "loaded exactly two SDK runs and retained only the disclosed bounded keys"
            )

        config_a = dict(projected_a.get("config") or {})
        config_b = dict(projected_b.get("config") or {})
        summary_a = dict(projected_a.get("summary") or {})
        summary_b = dict(projected_b.get("summary") or {})

        result: Dict[str, Any] = {
            "run_a": {
                "id": run_id_a,
                "name": projected_a.get("display_name") or run_id_a,
                "state": projected_a.get("state") or "unknown",
            },
            "run_b": {
                "id": run_id_b,
                "name": projected_b.get("display_name") or run_id_b,
                "state": projected_b.get("state") or "unknown",
            },
            "config_diff": _diff_dicts(config_a, config_b),
            "summary_diff": _diff_dicts(summary_a, summary_b),
            "metadata_diff": {
                "run_a": {
                    "created_at": str(projected_a.get("created_at") or ""),
                    "heartbeat_at": str(projected_a.get("heartbeat_at") or ""),
                    "tags": projected_a.get("tags") or [],
                    "group": projected_a.get("group"),
                },
                "run_b": {
                    "created_at": str(projected_b.get("created_at") or ""),
                    "heartbeat_at": str(projected_b.get("heartbeat_at") or ""),
                    "tags": projected_b.get("tags") or [],
                    "group": projected_b.get("group"),
                },
            },
            "selection": {
                "source": selection_source,
                "config_keys": selected_config,
                "summary_keys": selected_summary,
                "field_index_exhaustive": field_index_exhaustive,
            },
            "coverage": {
                "config_fields_compared": len(selected_config),
                "summary_fields_compared": len(selected_summary),
                "full_run_fields_exhaustive": False,
                "history_sampled": include_history_overlap,
            },
        }
        if compatibility_caveat:
            result["compatibility_caveat"] = compatibility_caveat

        if include_history_overlap:
            try:
                run_a = api.run(f"{path}/{run_id_a}")
                run_b = api.run(f"{path}/{run_id_b}")
                common_keys = list(history_keys or selected_summary)[:MCP_MAX_HISTORY_KEYS]
                history_kwargs = {
                    "keys": common_keys,
                    "samples": history_samples,
                    "pandas": False,
                    "x_axis": x_axis,
                }
                hist_a = list(run_a.history(**history_kwargs)) if common_keys else []
                hist_b = list(run_b.history(**history_kwargs)) if common_keys else []
                result["history_comparison"] = {
                    "keys": common_keys,
                    "x_axis": x_axis,
                    "requested_samples_per_run": history_samples,
                    "run_a_rows": len(hist_a),
                    "run_b_rows": len(hist_b),
                    "sampled": True,
                    "project_exhaustive": False,
                    "run_a_sample": [
                        {key: _safe_val(row.get(key)) for key in [x_axis] + common_keys[:5]} for row in hist_a[:10]
                    ],
                    "run_b_sample": [
                        {key: _safe_val(row.get(key)) for key in [x_axis] + common_keys[:5]} for row in hist_b[:10]
                    ],
                }
            except Exception as e:
                raise_for_wandb_server_busy(e)
                result["history_comparison"] = {"error": str(e)[:300], "sampled": True}

        return json.dumps(result, default=str)
