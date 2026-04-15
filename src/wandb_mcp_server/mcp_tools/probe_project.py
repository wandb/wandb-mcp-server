"""Probe a W&B project: discover run structure, metric keys, config keys, and recommended strategies."""

import json
from typing import Any, Dict, List

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

PROBE_PROJECT_TOOL_DESCRIPTION = """Probe a W&B project to discover its structure before querying.

Samples runs to discover available metric keys, config keys, tags, groups,
and provides recommended query strategies. This is the run-side equivalent
of infer_trace_schema_tool (which is for Weave traces).

<when_to_use>
Call FIRST when working with a new project to understand what data is available
before making specific queries. Essential for:
- Discovering which metrics are logged (loss, accuracy, custom metrics)
- Finding config keys (learning_rate, model, batch_size)
- Understanding project scale (run count, typical step counts)
- Getting query strategy recommendations

Typical workflow:
1. probe_project_tool to discover structure
2. query_wandb_tool or get_run_history_tool with the discovered keys
3. create_wandb_report_tool to visualize
</when_to_use>

Parameters
----------
entity_name : str
    W&B entity (username or team).
project_name : str
    W&B project name.
sample_runs : int, optional
    Number of runs to sample for key discovery. Default: 5.

Returns
-------
JSON with run_count, metric_keys, config_keys, has_history, typical_steps,
tags, groups, and recommendations.
"""

DEFAULT_SAMPLE_RUNS = 5


def probe_project(
    entity_name: str,
    project_name: str,
    sample_runs: int = DEFAULT_SAMPLE_RUNS,
) -> str:
    """Probe a W&B project to discover its structure."""
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "probe_project",
        api.viewer,
        {"entity_name": entity_name, "project_name": project_name, "sample_runs": sample_runs},
    ) as ctx:
        path = f"{entity_name}/{project_name}"

        try:
            runs_iter = api.runs(path, per_page=sample_runs)
        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "project_not_found", "message": str(e)[:500]})

        run_count = 0
        metric_keys: Dict[str, str] = {}
        config_keys: Dict[str, Any] = {}
        all_tags: List[str] = []
        all_groups: List[str] = []
        step_counts: List[int] = []
        states: Dict[str, int] = {}
        sampled = 0

        try:
            for run in runs_iter:
                run_count += 1
                if sampled >= sample_runs:
                    continue

                state = getattr(run, "state", "unknown")
                states[state] = states.get(state, 0) + 1

                summary = dict(run.summary) if run.summary else {}
                for k, v in summary.items():
                    if k.startswith("_") or k.startswith("wandb/"):
                        continue
                    if k not in metric_keys:
                        metric_keys[k] = type(v).__name__

                config = dict(run.config) if run.config else {}
                for k, v in config.items():
                    if k.startswith("_") or k.startswith("wandb"):
                        continue
                    if k not in config_keys:
                        config_keys[k] = _safe_sample_value(v)

                tags = getattr(run, "tags", [])
                if tags:
                    all_tags.extend(tags)

                group = getattr(run, "group", None)
                if group and group not in all_groups:
                    all_groups.append(group)

                last_step = getattr(run, "lastHistoryStep", 0)
                if last_step and last_step > 0:
                    step_counts.append(last_step)

                sampled += 1
        except Exception as e:
            logger.warning(f"Error iterating runs: {e}")

        unique_tags = sorted(set(all_tags))
        has_history = len(step_counts) > 0
        typical_steps = int(sum(step_counts) / len(step_counts)) if step_counts else 0

        recommendations = []
        if run_count > 100:
            recommendations.append(
                f"Large project ({run_count} runs) -- use filters in query_wandb_tool to narrow results."
            )
        if typical_steps > 10000:
            recommendations.append(
                f"Long runs (~{typical_steps} steps) -- use keys=[...] and samples parameter in get_run_history_tool."
            )
        if len(metric_keys) > 20:
            recommendations.append(f"Many metrics ({len(metric_keys)}) -- specify keys to avoid large responses.")
        if unique_tags:
            recommendations.append(
                f"Tags in use: {unique_tags[:10]}. Filter by tag in query_wandb_tool for focused analysis."
            )
        if all_groups:
            recommendations.append(f"Run groups found: {all_groups[:5]}. Group-based comparison may be useful.")
        if not recommendations:
            recommendations.append("Small project -- standard queries should work well.")

        return json.dumps(
            {
                "entity": entity_name,
                "project": project_name,
                "run_count": run_count,
                "sampled_runs": sampled,
                "run_states": states,
                "metric_keys": metric_keys,
                "metric_count": len(metric_keys),
                "config_keys": config_keys,
                "config_count": len(config_keys),
                "has_history": has_history,
                "typical_steps": typical_steps,
                "tags": unique_tags[:20],
                "groups": all_groups[:10],
                "recommendations": recommendations,
            },
            default=str,
        )


def _safe_sample_value(v: Any) -> Any:
    """Return a JSON-safe sample value for display."""
    if isinstance(v, (str, int, float, bool)):
        if isinstance(v, str) and len(v) > 100:
            return v[:100] + "..."
        return v
    if isinstance(v, (list, tuple)):
        return f"[{type(v).__name__}, len={len(v)}]"
    if isinstance(v, dict):
        return f"{{dict, keys={len(v)}}}"
    return str(v)[:50]
