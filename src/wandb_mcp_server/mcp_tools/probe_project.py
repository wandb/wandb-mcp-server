"""Bounded W&B project structure and scale discovery."""

from __future__ import annotations

from collections import Counter
from itertools import islice
import json
import re
from typing import Any, Mapping

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import MCP_MAX_PROBE_RUNS, MCP_MAX_PROJECT_FIELDS, MCP_WORKLOAD_PROFILE
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import (
    SelectiveReadUnavailable,
    fetch_artifact_inventory,
    fetch_project_counts,
    fetch_project_fields,
    fetch_projected_runs,
)
from wandb_mcp_server.wandb_urls import publicize_wandb_url

logger = get_rich_logger(__name__)

PROBE_PROJECT_TOOL_DESCRIPTION = """Probe a W&B project before issuing detailed queries.

Use this read-only tool first for an unfamiliar or large project. It returns
exact run/state counts, indexed config and summary field names, and bounded
recent/oldest run samples without loading every run summary.

Typical workflow:
1. probe_project_tool to discover field names and project scale
2. query_wandb_tool with summary_keys/config_keys for projected run rows
3. get_run_history_tool with explicit keys and x_axis for time-series values

Parameters
----------
entity_name : str
    W&B entity or team.
project_name : str
    W&B project.
sample_runs : int, optional
    Total recent/oldest metadata rows to sample. The active workload profile
    applies a protective cap.
field_pattern : str, optional
    Server-side field-name pattern for narrowing very wide schemas.
include_artifacts : bool, optional
    Include a compact project artifact type/collection inventory.

Returns
-------
JSON containing exact counts, bounded field inventory, run samples,
sample_scope, exhaustiveness metadata, and compact recommended next calls.
"""

DEFAULT_SAMPLE_RUNS = 6
_FIELD_RESPONSE_LIMIT = 200


def _field_family(path: str) -> str:
    stripped = re.sub(r"^(config|summary_metrics|summaryMetrics|summary)[./]", "", path)
    return re.split(r"[/.]", stripped, maxsplit=1)[0]


def _field_category(path: str) -> tuple[str | None, str]:
    for prefix, category in (
        ("config.", "config"),
        ("config/", "config"),
        ("summary_metrics.", "summary"),
        ("summary_metrics/", "summary"),
        ("summaryMetrics.", "summary"),
        ("summaryMetrics/", "summary"),
        ("summary.", "summary"),
        ("summary/", "summary"),
    ):
        if path.startswith(prefix):
            return category, path[len(prefix) :]
    return None, path


def _sample_item_from_sdk(run: Any) -> dict[str, Any]:
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    return {
        "id": run_id,
        "display_name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "created_at": getattr(run, "created_at", None),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "tags": list(getattr(run, "tags", []) or []),
        "history_line_count": getattr(run, "lastHistoryStep", None),
        "url": publicize_wandb_url(
            getattr(run, "url", None),
            fallback_segments=(entity, project, "runs", run_id),
        ),
    }


def _sdk_count(api: Any, path: str, filters: Mapping[str, Any] | None = None) -> int:
    runs = api.runs(
        path,
        filters=dict(filters or {}),
        per_page=1,
        include_sweeps=False,
        lazy=True,
    )
    return len(runs)


def _sdk_fallback(
    api: Any,
    *,
    entity_name: str,
    project_name: str,
    applied_samples: int,
) -> tuple[dict[str, int], list[dict[str, str]], list[dict[str, Any]]]:
    """Use bounded hydrated SDK pages when the indexed read API is unavailable."""
    path = f"{entity_name}/{project_name}"
    counts = {
        "all": _sdk_count(api, path),
        **{state: _sdk_count(api, path, {"state": state}) for state in ("finished", "failed", "crashed", "running")},
    }
    recent_count = max(1, (applied_samples + 1) // 2)
    oldest_count = max(0, applied_samples - recent_count)
    sampled_runs: list[Any] = list(
        islice(
            api.runs(
                path,
                order="-created_at",
                per_page=recent_count,
                include_sweeps=False,
                lazy=False,
            ),
            recent_count,
        )
    )
    if oldest_count:
        sampled_runs.extend(
            islice(
                api.runs(
                    path,
                    order="+created_at",
                    per_page=oldest_count,
                    include_sweeps=False,
                    lazy=False,
                ),
                oldest_count,
            )
        )

    deduped: dict[str, Any] = {}
    fields: dict[tuple[str, str], str] = {}
    for run in sampled_runs:
        run_id = str(getattr(run, "id", ""))
        if run_id:
            deduped[run_id] = run
        for key, value in dict(getattr(run, "config", {}) or {}).items():
            if not str(key).startswith(("_", "wandb")):
                fields[("config", str(key))] = type(value).__name__
        for key, value in dict(getattr(run, "summary", {}) or {}).items():
            if not str(key).startswith(("_", "wandb/")):
                fields[("summary", str(key))] = type(value).__name__
    return (
        counts,
        [{"path": f"{category}.{name}", "type": field_type} for (category, name), field_type in sorted(fields.items())],
        [_sample_item_from_sdk(run) for run in deduped.values()],
    )


def _safe_sample_value(value: Any) -> Any:
    """Retained compatibility helper for callers that imported it directly."""
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > 100:
            return value[:100] + "..."
        return value
    if isinstance(value, (list, tuple)):
        return f"[{type(value).__name__}, len={len(value)}]"
    if isinstance(value, dict):
        return f"{{dict, keys={len(value)}}}"
    return str(value)[:50]


def probe_project(
    entity_name: str,
    project_name: str,
    sample_runs: int = DEFAULT_SAMPLE_RUNS,
    field_pattern: str | None = None,
    include_artifacts: bool = False,
) -> str:
    """Return a bounded, explicit project-scale and schema snapshot."""
    if not isinstance(entity_name, str) or not entity_name.strip():
        raise ValueError("entity_name must be a non-empty string")
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("project_name must be a non-empty string")
    if isinstance(sample_runs, bool) or not isinstance(sample_runs, int) or sample_runs < 1:
        raise ValueError("sample_runs must be a positive integer")
    if field_pattern is not None and (not isinstance(field_pattern, str) or not field_pattern.strip()):
        raise ValueError("field_pattern must be a non-empty string when supplied")
    if not isinstance(include_artifacts, bool):
        raise ValueError("include_artifacts must be a boolean")

    applied_samples = min(sample_runs, MCP_MAX_PROBE_RUNS)
    with track_tool_execution(
        "probe_project",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "sample_runs": applied_samples,
            "has_field_pattern": field_pattern is not None,
            "include_artifacts": include_artifacts,
        },
        mcp_tool_name="probe_project_tool",
    ) as ctx:
        try:
            api = WandBApiManager.get_api()
            fallback_caveat: str | None = None
            try:
                counts = fetch_project_counts(api, entity=entity_name, project=project_name)
                fields_page = fetch_project_fields(
                    api,
                    entity=entity_name,
                    project=project_name,
                    limit=MCP_MAX_PROJECT_FIELDS,
                    pattern=field_pattern,
                )
                recent_count = max(1, (applied_samples + 1) // 2)
                oldest_count = max(0, applied_samples - recent_count)
                recent = fetch_projected_runs(
                    api,
                    entity=entity_name,
                    project=project_name,
                    filters=None,
                    order="-created_at",
                    limit=recent_count,
                    page_size=recent_count + 1,
                ).items
                oldest = (
                    fetch_projected_runs(
                        api,
                        entity=entity_name,
                        project=project_name,
                        filters=None,
                        order="+created_at",
                        limit=oldest_count,
                        page_size=oldest_count + 1,
                    ).items
                    if oldest_count
                    else []
                )
                fields = fields_page.items
                field_has_more = fields_page.has_more
                run_samples = list(
                    {
                        str(item.get("id")): {
                            **item,
                            "url": publicize_wandb_url(
                                item.get("url"),
                                fallback_segments=(
                                    entity_name,
                                    project_name,
                                    "runs",
                                    item.get("id"),
                                ),
                            ),
                        }
                        for item in [*recent, *oldest]
                        if item.get("id")
                    }.values()
                )
            except SelectiveReadUnavailable as exc:
                logger.warning("Project selective read unavailable; using bounded SDK fallback: %s", exc)
                counts, fields, run_samples = _sdk_fallback(
                    api,
                    entity_name=entity_name,
                    project_name=project_name,
                    applied_samples=applied_samples,
                )
                field_has_more = True
                fallback_caveat = (
                    f"{exc}; field inventory was derived from {len(run_samples)} bounded hydrated SDK runs"
                )

            categorized: dict[str, list[dict[str, str]]] = {"config": [], "summary": []}
            for field in fields:
                category, name = _field_category(str(field.get("path") or ""))
                if category and name and not name.startswith(("_", "wandb/")):
                    categorized[category].append({"path": name, "type": str(field.get("type") or "unknown")})

            config_fields = categorized["config"]
            summary_fields = categorized["summary"]
            metric_families = Counter(_field_family(item["path"]) for item in summary_fields)
            tags = sorted({tag for item in run_samples for tag in item.get("tags") or []})
            groups = sorted({str(item["group"]) for item in run_samples if item.get("group")})
            has_history = any(
                isinstance(item.get("history_line_count"), (int, float)) and item["history_line_count"] > 0
                for item in run_samples
            )

            recommendations = [
                {
                    "tool": "query_wandb_tool",
                    "reason": "Fetch only the run rows and selected fields needed for analysis.",
                    "parameters": {
                        "resource": "runs",
                        "summary_keys": [item["path"] for item in summary_fields[:5]],
                        "config_keys": [item["path"] for item in config_fields[:5]],
                    },
                }
            ]
            if summary_fields:
                recommendations.append(
                    {
                        "tool": "get_run_history_tool",
                        "reason": "Retrieve a bounded time-series after choosing a run.",
                        "parameters": {"keys": [item["path"] for item in summary_fields[:5]]},
                    }
                )

            result: dict[str, Any] = {
                "source": "wandb_selective_read" if fallback_caveat is None else "wandb_sdk_fallback",
                "entity": entity_name,
                "project": project_name,
                "workload_profile": MCP_WORKLOAD_PROFILE,
                "run_count": counts["all"],
                "state_counts": {state: counts[state] for state in ("finished", "failed", "crashed", "running")},
                "sampled_runs": len(run_samples),
                "run_samples": run_samples,
                "sample_scope": {
                    "strategy": "recent_and_oldest",
                    "requested": sample_runs,
                    "effective": applied_samples,
                    "cap_applied": applied_samples < sample_runs,
                    "project_exhaustive": counts["all"] <= len(run_samples),
                },
                "config_field_count_returned": len(config_fields),
                "summary_field_count_returned": len(summary_fields),
                "config_fields": config_fields[:_FIELD_RESPONSE_LIMIT],
                "summary_fields": summary_fields[:_FIELD_RESPONSE_LIMIT],
                "metric_families": dict(metric_families.most_common(50)),
                "field_inventory": {
                    "pattern": field_pattern,
                    "returned_count": len(fields),
                    "has_more": field_has_more,
                    "project_exhaustive": not field_has_more,
                    "response_truncated": (
                        len(config_fields) > _FIELD_RESPONSE_LIMIT or len(summary_fields) > _FIELD_RESPONSE_LIMIT
                    ),
                },
                "has_history_in_sample": has_history,
                "tags_in_sample": tags[:20],
                "groups_in_sample": groups[:20],
                "recommended_next_calls": recommendations,
            }
            if fallback_caveat:
                result["compatibility_caveat"] = fallback_caveat
            if include_artifacts:
                try:
                    result["artifact_inventory"] = fetch_artifact_inventory(
                        api,
                        entity=entity_name,
                        project=project_name,
                    )
                except SelectiveReadUnavailable as exc:
                    result["artifact_inventory"] = {
                        "error": "artifact_inventory_unavailable",
                        "message": str(exc),
                    }
            return json.dumps(result, default=str)
        except Exception as exc:
            ctx.mark_error(f"{type(exc).__name__}: {exc}")
            return json.dumps({"error": "project_probe_failed", "message": str(exc)[:500]})


__all__ = ["DEFAULT_SAMPLE_RUNS", "PROBE_PROJECT_TOOL_DESCRIPTION", "_safe_sample_value", "probe_project"]
