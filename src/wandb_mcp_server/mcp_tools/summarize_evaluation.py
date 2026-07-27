"""Summarize Weave evaluation results with aggregated pass rates and metrics."""

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.config import (
    MCP_MAX_EVALUATION_ROWS,
    MCP_MAX_QUERY_LIMIT,
    MCP_MAX_WANDB_QUERY_ITEMS,
    MCP_WORKLOAD_PROFILE,
)
from wandb_mcp_server.mcp_tools.count_traces import count_traces
from wandb_mcp_server.mcp_tools.query_weave import get_trace_service
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

SUMMARIZE_EVALUATION_TOOL_DESCRIPTION = """Summarize Weave evaluation results with aggregated pass rates and metrics.

Finds Evaluation.evaluate traces in a project and returns aggregated results
including per-scorer pass rates, error counts, and token usage.

<when_to_use>
Call when the user asks "how did my eval go?", "what's the pass rate?", "which
tasks fail most?", or wants a summary of evaluation results without manually
navigating trace hierarchies.

This tool aggregates the Evaluation.evaluate -> predict_and_score trace hierarchy
automatically. Exact matching child counts are reported separately from bounded
detail aggregates, so a capped sample is never presented as exhaustive. Use
query_weave_traces_tool for raw trace data instead.
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
Each evaluation includes exact total_predictions, detail coverage, and whether
its error/token aggregates are exhaustive or sampled.
"""


def _extract_scores(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scorer results from a trace's summary."""
    scores = {}
    weave_summary = summary.get("weave", {})
    for key, val in weave_summary.items():
        if isinstance(val, dict) and ("mean" in val or "true_count" in val or "true_fraction" in val):
            scores[key] = val
    return scores


def _aggregate_eval(
    eval_trace: Dict[str, Any],
    children: List[Dict[str, Any]],
    total_matching_count: int | None = None,
) -> Dict[str, Any]:
    """Aggregate a single evaluation's results."""
    summary = eval_trace.get("summary", {})
    scores = _extract_scores(summary)

    observed = len(children)
    total = observed if total_matching_count is None else max(0, total_matching_count)
    errors = sum(
        1 for c in children if c.get("exception") or c.get("summary", {}).get("weave", {}).get("status") == "error"
    )
    successes = observed - errors

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for child in children:
        child_usage = child.get("summary", {}).get("usage", {})
        for model_usage in child_usage.values() if isinstance(child_usage, dict) else []:
            if isinstance(model_usage, dict):
                for k in usage_total:
                    usage_total[k] += model_usage.get(k, 0)

    exhaustive = observed >= total
    result: dict[str, Any] = {
        "eval_id": eval_trace.get("id", ""),
        "op_name": eval_trace.get("op_name", ""),
        "started_at": eval_trace.get("started_at", ""),
        "total_predictions": total,
        "observed_predictions": observed,
        "returned_count": observed,
        "coverage": round(observed / total, 4) if total else 1.0,
        "details_exhaustive": exhaustive,
        "aggregate_scope": "exhaustive" if exhaustive else "sample",
        "observed_successes": successes,
        "observed_errors": errors,
        "observed_error_rate": round(errors / max(observed, 1), 4),
        "scores": scores,
        "token_usage": usage_total,
        "token_usage_scope": "exhaustive" if exhaustive else "sample",
    }
    if exhaustive:
        # Compatibility aliases are safe only when the aggregate is complete.
        result.update(
            {
                "successes": successes,
                "errors": errors,
                "error_rate": round(errors / max(total, 1), 4),
            }
        )
    else:
        result["coverage_note"] = (
            f"Detail aggregates cover {observed} of {total} matching child traces; "
            "scores stored on the evaluation root may still be backend-computed totals."
        )
    return result


def _query_evaluation_children(
    service: Any,
    *,
    entity_name: str,
    project_name: str,
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch lightweight evaluation children in profile-bounded hosted pages."""
    children: list[dict[str, Any]] = []
    offset = 0
    chunk_size = min(100, MCP_MAX_QUERY_LIMIT)
    while len(children) < limit:
        raise_if_tool_deadline_exceeded()
        requested = min(chunk_size, limit - len(children))
        result = service.query_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=filters,
            sort_by="started_at",
            sort_direction="asc",
            limit=requested,
            offset=offset,
            include_costs=False,
            include_feedback=False,
            columns=["id", "exception", "summary"],
            return_full_data=False,
        )
        page = list(result.traces or [])
        children.extend(page[:requested])
        if len(page) < requested:
            break
        offset += requested
    return children[:limit]


def summarize_evaluation(
    entity_name: str,
    project_name: str,
    eval_name: Optional[str] = None,
    max_evals: int = 5,
    include_per_task: bool = False,
) -> str:
    """Summarize Weave evaluation results."""
    if isinstance(max_evals, bool) or not isinstance(max_evals, int) or max_evals < 1:
        return json.dumps({"error": "invalid_input", "message": "max_evals must be a positive integer"})
    effective_max_evals = min(max_evals, MCP_MAX_WANDB_QUERY_ITEMS)
    with track_tool_execution(
        "summarize_evaluation",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "eval_name": eval_name,
            "max_evals": max_evals,
        },
    ) as ctx:
        try:
            filters: Dict[str, Any] = {"op_name_contains": "Evaluation.evaluate"}
            if eval_name:
                filters["op_name_contains"] = eval_name
            filters["trace_roots_only"] = True
            service = get_trace_service()
            total_evaluations = count_traces(entity_name, project_name, filters=filters)

            result = service.query_traces(
                entity_name=entity_name,
                project_name=project_name,
                filters=filters,
                limit=effective_max_evals,
                sort_by="started_at",
                sort_direction="desc",
                include_costs=False,
                include_feedback=False,
                columns=["id", "op_name", "started_at", "summary"],
                return_full_data=False,
            )

            eval_traces = result.traces if result.traces else []
            if not eval_traces:
                return json.dumps(
                    {
                        "evaluations": [],
                        "items": [],
                        "returned_count": 0,
                        "total_count": total_evaluations,
                        "has_more": total_evaluations > 0,
                        "limit": effective_max_evals,
                        "project_exhaustive": total_evaluations == 0,
                        "message": "No Evaluation.evaluate traces found in this project.",
                    }
                )

            evaluations = []
            for eval_trace in eval_traces[:effective_max_evals]:
                child_filters = {"parent_ids": [eval_trace.get("id", "")]}
                child_total = count_traces(
                    entity_name=entity_name,
                    project_name=project_name,
                    filters=child_filters,
                )
                detail_limit = min(child_total, MCP_MAX_EVALUATION_ROWS)
                if detail_limit:
                    children = _query_evaluation_children(
                        service,
                        entity_name=entity_name,
                        project_name=project_name,
                        filters=child_filters,
                        limit=detail_limit,
                    )
                else:
                    children = []
                summary = _aggregate_eval(eval_trace, children, child_total)
                summary["detail_limit"] = detail_limit
                summary["workload_profile"] = MCP_WORKLOAD_PROFILE

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
                    summary["per_task_scope"] = {
                        "returned_count": len(per_task),
                        "total_count": child_total,
                        "has_more": len(per_task) < child_total,
                        "limit": 50,
                        "project_exhaustive": len(per_task) >= child_total,
                    }

                evaluations.append(summary)

            return json.dumps(
                {
                    "items": evaluations,
                    "evaluations": evaluations,
                    "returned_count": len(evaluations),
                    "total_count": total_evaluations,
                    "has_more": total_evaluations > len(evaluations),
                    "limit": effective_max_evals,
                    "project_exhaustive": total_evaluations <= len(evaluations),
                    "count": len(evaluations),
                    "project": f"{entity_name}/{project_name}",
                },
                default=str,
            )

        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "evaluation_query_failed", "message": str(e)[:500]})
