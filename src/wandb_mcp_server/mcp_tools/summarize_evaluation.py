"""Summarize Weave evaluation results with aggregated pass rates and metrics."""

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.service import TraceService
from wandb_mcp_server.config import WF_TRACE_SERVER_URL

logger = get_rich_logger(__name__)

SUMMARIZE_EVALUATION_TOOL_DESCRIPTION = """Summarize Weave evaluation results with aggregated pass rates and metrics.

Finds Evaluation.evaluate traces in a project and returns aggregated results
including per-scorer pass rates, error counts, and token usage.

<when_to_use>
Call when the user asks "how did my eval go?", "what's the pass rate?", "which
tasks fail most?", or wants a summary of evaluation results without manually
navigating trace hierarchies.

This tool aggregates the Evaluation.evaluate -> predict_and_score trace hierarchy
automatically. Use query_weave_traces_tool for raw trace data instead.
</when_to_use>

Parameters
----------
entity_name : str
    W&B entity (username or team).
project_name : str
    W&B project name.
eval_name : str, optional
    Filter to a specific evaluation by op_name. If None, summarizes all evals.
max_evals : int, optional
    Maximum number of evaluation runs to summarize. Default: 5.
include_per_task : bool, optional
    If True, includes per-input-row breakdown. Default: False.

Returns
-------
JSON with evaluations (list of eval summaries) and optional comparison.
"""


def _extract_scores(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scorer results from a trace's summary."""
    scores = {}
    weave_summary = summary.get("weave", {})
    for key, val in weave_summary.items():
        if isinstance(val, dict) and ("mean" in val or "true_count" in val or "true_fraction" in val):
            scores[key] = val
    return scores


def _aggregate_eval(eval_trace: Dict[str, Any], children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a single evaluation's results."""
    summary = eval_trace.get("summary", {})
    scores = _extract_scores(summary)

    total = len(children)
    errors = sum(
        1 for c in children if c.get("exception") or c.get("summary", {}).get("weave", {}).get("status") == "error"
    )
    successes = total - errors

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for child in children:
        child_usage = child.get("summary", {}).get("usage", {})
        for model_usage in child_usage.values() if isinstance(child_usage, dict) else []:
            if isinstance(model_usage, dict):
                for k in usage_total:
                    usage_total[k] += model_usage.get(k, 0)

    return {
        "eval_id": eval_trace.get("id", ""),
        "op_name": eval_trace.get("op_name", ""),
        "started_at": eval_trace.get("started_at", ""),
        "total_predictions": total,
        "successes": successes,
        "errors": errors,
        "error_rate": round(errors / max(total, 1), 4),
        "scores": scores,
        "token_usage": usage_total,
    }


def summarize_evaluation(
    entity_name: str,
    project_name: str,
    eval_name: Optional[str] = None,
    max_evals: int = 5,
    include_per_task: bool = False,
) -> str:
    """Summarize Weave evaluation results."""
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "summarize_evaluation",
        api.viewer,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "eval_name": eval_name,
            "max_evals": max_evals,
        },
    ) as ctx:
        try:
            service = TraceService(
                api_key=WandBApiManager.get_api_key(),
                server_url=WF_TRACE_SERVER_URL,
            )

            filters: Dict[str, Any] = {"op_name_contains": "Evaluation.evaluate"}
            if eval_name:
                filters["op_name_contains"] = eval_name
            filters["trace_roots_only"] = True

            result = service.query_traces(
                entity_name=entity_name,
                project_name=project_name,
                filters=filters,
                limit=max_evals,
                sort_by="started_at",
                sort_direction="desc",
                return_full_data=True,
            )

            eval_traces = result.traces if result.traces else []
            if not eval_traces:
                return json.dumps(
                    {
                        "evaluations": [],
                        "message": "No Evaluation.evaluate traces found in this project.",
                    }
                )

            evaluations = []
            for eval_trace in eval_traces[:max_evals]:
                child_result = service.query_traces(
                    entity_name=entity_name,
                    project_name=project_name,
                    filters={"parent_ids": [eval_trace.get("id", "")]},
                    limit=500,
                    return_full_data=True,
                )
                children = child_result.traces if child_result.traces else []
                summary = _aggregate_eval(eval_trace, children)

                if include_per_task and children:
                    per_task = []
                    for child in children[:50]:
                        task_entry = {
                            "id": child.get("id", ""),
                            "status": child.get("summary", {}).get("weave", {}).get("status", "unknown"),
                            "has_exception": child.get("exception") is not None,
                        }
                        child_scores = _extract_scores(child.get("summary", {}))
                        if child_scores:
                            task_entry["scores"] = child_scores
                        per_task.append(task_entry)
                    summary["per_task"] = per_task

                evaluations.append(summary)

            return json.dumps(
                {
                    "evaluations": evaluations,
                    "count": len(evaluations),
                    "project": f"{entity_name}/{project_name}",
                },
                default=str,
            )

        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "evaluation_query_failed", "message": str(e)[:500]})
