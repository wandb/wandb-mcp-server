"""Compare two W&B runs: config diff, summary metric diff, optional history overlap."""

import json
import math
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

COMPARE_RUNS_TOOL_DESCRIPTION = """Compare two W&B runs side-by-side.

Returns config differences, summary metric deltas, and metadata comparison.

<when_to_use>
Call when the user asks "what changed between run A and B?", "which run is better?",
or wants to understand why two runs have different performance.

Typical workflow:
1. query_wandb_tool or get_run_history_tool to identify the two runs
2. compare_runs_tool to see what differs
3. create_wandb_report_tool to visualize the comparison
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

Returns
-------
JSON with config_diff, summary_diff, metadata_diff, and optional history_comparison.
"""

DEFAULT_HISTORY_SAMPLES = 50


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


def compare_runs(
    entity_name: str,
    project_name: str,
    run_id_a: str,
    run_id_b: str,
    include_history_overlap: bool = False,
    history_keys: Optional[List[str]] = None,
    history_samples: int = DEFAULT_HISTORY_SAMPLES,
) -> str:
    """Compare two W&B runs."""
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "compare_runs",
        api.viewer,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
        },
    ) as ctx:
        try:
            path = f"{entity_name}/{project_name}"
            run_a = api.run(f"{path}/{run_id_a}")
            run_b = api.run(f"{path}/{run_id_b}")
        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "run_not_found", "message": str(e)[:500]})

        config_a = dict(run_a.config) if run_a.config else {}
        config_b = dict(run_b.config) if run_b.config else {}
        summary_a = dict(run_a.summary) if run_a.summary else {}
        summary_b = dict(run_b.summary) if run_b.summary else {}

        # Filter out internal keys from summary
        skip_prefixes = ("_", "wandb/")
        summary_a = {k: v for k, v in summary_a.items() if not any(k.startswith(p) for p in skip_prefixes)}
        summary_b = {k: v for k, v in summary_b.items() if not any(k.startswith(p) for p in skip_prefixes)}

        result: Dict[str, Any] = {
            "run_a": {
                "id": run_id_a,
                "name": getattr(run_a, "name", run_id_a),
                "state": getattr(run_a, "state", "unknown"),
            },
            "run_b": {
                "id": run_id_b,
                "name": getattr(run_b, "name", run_id_b),
                "state": getattr(run_b, "state", "unknown"),
            },
            "config_diff": _diff_dicts(config_a, config_b),
            "summary_diff": _diff_dicts(summary_a, summary_b),
            "metadata_diff": {
                "run_a": {
                    "created_at": str(getattr(run_a, "created_at", "")),
                    "heartbeat_at": str(getattr(run_a, "heartbeat_at", "")),
                    "tags": getattr(run_a, "tags", []),
                    "group": getattr(run_a, "group", None),
                },
                "run_b": {
                    "created_at": str(getattr(run_b, "created_at", "")),
                    "heartbeat_at": str(getattr(run_b, "heartbeat_at", "")),
                    "tags": getattr(run_b, "tags", []),
                    "group": getattr(run_b, "group", None),
                },
            },
        }

        if include_history_overlap:
            try:
                hist_a = list(run_a.scan_history(keys=history_keys, page_size=history_samples))[:history_samples]
                hist_b = list(run_b.scan_history(keys=history_keys, page_size=history_samples))[:history_samples]

                if history_keys is None:
                    keys_a = set()
                    keys_b = set()
                    for row in hist_a[:5]:
                        keys_a.update(k for k in row.keys() if not k.startswith("_"))
                    for row in hist_b[:5]:
                        keys_b.update(k for k in row.keys() if not k.startswith("_"))
                    common_keys = sorted(keys_a & keys_b)
                else:
                    common_keys = history_keys

                result["history_comparison"] = {
                    "keys": common_keys,
                    "run_a_rows": len(hist_a),
                    "run_b_rows": len(hist_b),
                    "run_a_sample": [
                        {k: _safe_val(r.get(k)) for k in ["_step"] + common_keys[:5]} for r in hist_a[:10]
                    ],
                    "run_b_sample": [
                        {k: _safe_val(r.get(k)) for k in ["_step"] + common_keys[:5]} for r in hist_b[:10]
                    ],
                }
            except Exception as e:
                result["history_comparison"] = {"error": str(e)[:300]}

        return json.dumps(result, default=str)
