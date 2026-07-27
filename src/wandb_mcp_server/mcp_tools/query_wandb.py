"""Structured, read-only W&B Models queries through the public SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping
from itertools import islice
from typing import Any, Dict, List, Literal, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import (
    MAX_RESPONSE_TOKENS,
    MCP_MAX_WANDB_QUERY_ITEMS,
    MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
    structured_error,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

WandBResource = Literal["project", "run", "runs", "sweep", "sweeps", "reports"]

QUERY_WANDB_TOOL_DESCRIPTION = """Query W&B Models data through the public W&B Python SDK.

Use this read-only tool for project metadata, individual runs, filtered or sorted
run collections, sweeps, and reports. For run history, artifacts, registries,
automations, and integrations, prefer the dedicated MCP tools.

Prefer the existing specialized tools when they match the request:
- entity/project discovery: list_entities_tool and query_wandb_entity_projects
- time-series metrics: get_run_history_tool
- artifact reads: list_artifact_versions_tool and get_artifact_details_tool
- registry reads: list_registries_tool and list_registry_collections_tool
- automations/integrations: list_wandb_automations_tool and list_wandb_integrations_tool

<when_to_use>
Use this tool for run discovery, summary-metric analysis, project metadata,
sweep inspection, or report discovery. It is the normal W&B Models query path.
</when_to_use>

Parameters
----------
entity_name : str
    W&B entity or team name.
project_name : str
    W&B project name.
resource : "project" | "run" | "runs" | "sweep" | "sweeps" | "reports"
    Resource to read through the SDK.
run_id : str, optional
    Required only for resource="run". This is the short W&B run ID, not its display name.
sweep_id : str, optional
    Required only for resource="sweep".
report_name : str, optional
    Optional report-name filter for resource="reports".
filters : dict, optional
    W&B SDK Mongo-style run filters for resource="runs". Supported fields include
    createdAt, displayName, duration, group, host, jobType, name, state, tags,
    username, config.*, and summary_metrics.*. Operators include $and, $or, $eq,
    $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, and $regex.
order : str, optional
    Run ordering for resource="runs", such as -created_at or
    -summary_metrics.accuracy. Default: -created_at.
limit : int, optional
    Maximum collection items to return. Default: 50; deployment limits apply.
include : list[str], optional
    Additional resource details. run/runs accept config, system_metrics, and sweep;
    sweep/sweeps accept config; reports accepts spec. Run summary metrics are always
    returned without needing an include value.

Returns
-------
dict
    A stable JSON-safe envelope with source="wandb_sdk". Collection results use
    items/count/limit/truncated; single-resource results use item.

For schema introspection, unmodeled fields, aliases, cross-resource nesting, or an
exact GraphQL response shape, an administrator may explicitly enable the separate
query_wandb_graphql_tool with WANDB_MCP_ENABLE_RAW_GRAPHQL=true.
"""

_INCLUDE_FIELDS = {
    "project": frozenset(),
    "run": frozenset({"config", "system_metrics", "sweep"}),
    "runs": frozenset({"config", "system_metrics", "sweep"}),
    "sweep": frozenset({"config"}),
    "sweeps": frozenset({"config"}),
    "reports": frozenset({"spec"}),
}
_OPTIONAL_RESPONSE_FIELDS = ("system_metrics", "config", "spec", "sweep")


class WandBQueryValidationError(ValueError):
    """Raised before any W&B client is obtained for an invalid request."""


def _json_safe(value: Any) -> Any:
    """Convert SDK values into deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        return {str(key): _json_safe(item) for key, item in dict(value).items()}
    except (TypeError, ValueError):
        return str(value)


def _serialize_user(user: Any) -> Any:
    if user is None or isinstance(user, (str, int, float, bool, Mapping)):
        return _json_safe(user)
    data = {
        "username": getattr(user, "username", None),
        "name": getattr(user, "name", None),
        "email": getattr(user, "email", None),
    }
    return {key: _json_safe(value) for key, value in data.items() if value is not None}


def _serialize_project(project: Any) -> Dict[str, Any]:
    return {
        "id": _json_safe(getattr(project, "id", None)),
        "name": _json_safe(getattr(project, "name", None)),
        "entity": _json_safe(getattr(project, "entity", None)),
        "description": _json_safe(getattr(project, "description", None)),
        "visibility": _json_safe(getattr(project, "visibility", None)),
        "created_at": _json_safe(getattr(project, "created_at", None)),
        "updated_at": _json_safe(getattr(project, "updated_at", None)),
        "tags": _json_safe(getattr(project, "tags", [])),
        "url": _json_safe(getattr(project, "url", None)),
    }


def _serialize_sweep(sweep: Any, include: frozenset[str]) -> Dict[str, Any]:
    result = {
        "id": _json_safe(getattr(sweep, "id", None)),
        "name": _json_safe(getattr(sweep, "name", None)),
        "state": _json_safe(getattr(sweep, "state", None)),
        "entity": _json_safe(getattr(sweep, "entity", None)),
        "project": _json_safe(getattr(sweep, "project", None)),
        "expected_run_count": _json_safe(getattr(sweep, "expected_run_count", None)),
        "url": _json_safe(getattr(sweep, "url", None)),
    }
    if "config" in include:
        result["config"] = _json_safe(getattr(sweep, "config", {}))
    return result


def _serialize_run(run: Any, include: frozenset[str]) -> Dict[str, Any]:
    result = {
        "id": _json_safe(getattr(run, "id", None)),
        "display_name": _json_safe(getattr(run, "name", None)),
        "state": _json_safe(getattr(run, "state", None)),
        "entity": _json_safe(getattr(run, "entity", None)),
        "project": _json_safe(getattr(run, "project", None)),
        "url": _json_safe(getattr(run, "url", None)),
        "created_at": _json_safe(getattr(run, "created_at", None)),
        "heartbeat_at": _json_safe(getattr(run, "heartbeat_at", None)),
        "duration": _json_safe(getattr(run, "duration", None)),
        "group": _json_safe(getattr(run, "group", None)),
        "job_type": _json_safe(getattr(run, "job_type", None)),
        "tags": _json_safe(getattr(run, "tags", [])),
        "user": _serialize_user(getattr(run, "user", None)),
        "summary": _json_safe(getattr(run, "summary", {})),
    }
    if "config" in include:
        result["config"] = _json_safe(getattr(run, "config", {}))
    if "system_metrics" in include:
        result["system_metrics"] = _json_safe(getattr(run, "system_metrics", {}))
    if "sweep" in include:
        sweep = getattr(run, "sweep", None)
        result["sweep"] = None if sweep is None else _serialize_sweep(sweep, frozenset())
    return result


def _serialize_report(report: Any, include: frozenset[str]) -> Dict[str, Any]:
    result = {
        "id": _json_safe(getattr(report, "id", None)),
        "name": _json_safe(getattr(report, "name", None)),
        "display_name": _json_safe(getattr(report, "display_name", None)),
        "description": _json_safe(getattr(report, "description", None)),
        "user": _serialize_user(getattr(report, "user", None)),
        "created_at": _json_safe(getattr(report, "created_at", None)),
        "updated_at": _json_safe(getattr(report, "updated_at", None)),
        "url": _json_safe(getattr(report, "url", None)),
    }
    if "spec" in include:
        result["spec"] = _json_safe(getattr(report, "spec", {}))
    return result


def _estimate_tokens(payload: Dict[str, Any]) -> int:
    """Conservative approximation suitable for enforcing the configured budget."""
    return max(1, len(json.dumps(payload, default=str, ensure_ascii=False)) // 4)


def _fit_collection_to_budget(payload: Dict[str, Any]) -> Dict[str, Any]:
    omitted_fields: set[str] = set()
    if _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        for field in _OPTIONAL_RESPONSE_FIELDS:
            removed = False
            for item in payload["items"]:
                if field in item:
                    item.pop(field)
                    removed = True
            if removed:
                omitted_fields.add(field)
            if _estimate_tokens(payload) <= MAX_RESPONSE_TOKENS:
                break

    dropped_items = 0
    while payload["items"] and _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        payload["items"].pop()
        dropped_items += 1

    if omitted_fields or dropped_items:
        payload["count"] = len(payload["items"])
        payload["truncated"] = True
        payload["truncation"] = {
            "applied": True,
            "reason": "response_token_budget",
            "omitted_fields": sorted(omitted_fields),
            "dropped_items": dropped_items,
        }
    return payload


def _fit_single_to_budget(payload: Dict[str, Any]) -> Dict[str, Any]:
    omitted_fields: list[str] = []
    if _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        for field in (*_OPTIONAL_RESPONSE_FIELDS, "summary"):
            if field in payload["item"]:
                payload["item"].pop(field)
                omitted_fields.append(field)
            if _estimate_tokens(payload) <= MAX_RESPONSE_TOKENS:
                break
    if omitted_fields:
        payload["truncated"] = True
        payload["truncation"] = {
            "applied": True,
            "reason": "response_token_budget",
            "omitted_fields": omitted_fields,
            "dropped_items": 0,
        }
    return payload


def _validate_request(
    entity_name: str,
    project_name: str,
    resource: str,
    run_id: Optional[str],
    sweep_id: Optional[str],
    report_name: Optional[str],
    filters: Optional[Dict[str, Any]],
    order: str,
    limit: int,
    include: Optional[List[str]],
) -> frozenset[str]:
    if not isinstance(entity_name, str) or not entity_name.strip():
        raise WandBQueryValidationError("entity_name must be a non-empty string")
    if not isinstance(project_name, str) or not project_name.strip():
        raise WandBQueryValidationError("project_name must be a non-empty string")
    if resource not in _INCLUDE_FIELDS:
        raise WandBQueryValidationError(f"unsupported resource: {resource!r}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise WandBQueryValidationError("limit must be a positive integer")
    if filters is not None and not isinstance(filters, dict):
        raise WandBQueryValidationError("filters must be a dictionary")
    if filters is not None and resource != "runs":
        raise WandBQueryValidationError("filters are supported only for resource='runs'")
    if resource != "runs" and order != "-created_at":
        raise WandBQueryValidationError("order is supported only for resource='runs'")
    if resource == "run" and not run_id:
        raise WandBQueryValidationError("run_id is required for resource='run'")
    if resource != "run" and run_id is not None:
        raise WandBQueryValidationError("run_id is supported only for resource='run'")
    if resource == "sweep" and not sweep_id:
        raise WandBQueryValidationError("sweep_id is required for resource='sweep'")
    if resource != "sweep" and sweep_id is not None:
        raise WandBQueryValidationError("sweep_id is supported only for resource='sweep'")
    if resource != "reports" and report_name is not None:
        raise WandBQueryValidationError("report_name is supported only for resource='reports'")
    if include is not None and (not isinstance(include, list) or not all(isinstance(item, str) for item in include)):
        raise WandBQueryValidationError("include must be a list of strings")

    requested = frozenset(include or [])
    unsupported = requested - _INCLUDE_FIELDS[resource]
    if unsupported:
        allowed = sorted(_INCLUDE_FIELDS[resource])
        raise WandBQueryValidationError(
            f"unsupported include value(s) for resource={resource!r}: {sorted(unsupported)}; allowed: {allowed}"
        )
    return requested


def _collection_envelope(
    resource: str,
    entity_name: str,
    project_name: str,
    items: List[Dict[str, Any]],
    limit: int,
    truncated: bool,
) -> Dict[str, Any]:
    return _fit_collection_to_budget(
        {
            "source": "wandb_sdk",
            "resource": resource,
            "entity": entity_name,
            "project": project_name,
            "items": items,
            "count": len(items),
            "limit": limit,
            "truncated": truncated,
            "truncation": {"applied": False},
        }
    )


def _single_envelope(
    resource: str,
    entity_name: str,
    project_name: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    return _fit_single_to_budget(
        {
            "source": "wandb_sdk",
            "resource": resource,
            "entity": entity_name,
            "project": project_name,
            "item": item,
            "truncated": False,
            "truncation": {"applied": False},
        }
    )


def query_wandb(
    entity_name: str,
    project_name: str,
    resource: WandBResource,
    run_id: Optional[str] = None,
    sweep_id: Optional[str] = None,
    report_name: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "-created_at",
    limit: int = 50,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute a structured read using only public W&B SDK operations."""
    try:
        include_fields = _validate_request(
            entity_name,
            project_name,
            resource,
            run_id,
            sweep_id,
            report_name,
            filters,
            order,
            limit,
            include,
        )
    except WandBQueryValidationError as exc:
        return structured_error("invalid_request", str(exc), source="wandb_sdk", resource=resource)

    api = WandBApiManager.get_api()
    requested_limit = limit
    applied_limit = min(limit, MCP_MAX_WANDB_QUERY_ITEMS)
    per_page = min(applied_limit + 1, MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE)
    path = f"{entity_name}/{project_name}"

    with track_tool_execution(
        "query_wandb",
        api.viewer,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "resource": resource,
            "limit": applied_limit,
        },
        mcp_tool_name="query_wandb_tool",
    ) as ctx:
        try:
            if resource == "project":
                project = api.project(project_name, entity=entity_name)
                return _single_envelope(resource, entity_name, project_name, _serialize_project(project))

            if resource == "run":
                run = api.run(f"{path}/{run_id}")
                return _single_envelope(resource, entity_name, project_name, _serialize_run(run, include_fields))

            if resource == "runs":
                runs = api.runs(
                    path,
                    filters=filters,
                    order=order,
                    per_page=per_page,
                    include_sweeps="sweep" in include_fields,
                    lazy=False,
                )
                page = list(islice(runs, applied_limit + 1))
                has_more = len(page) > applied_limit or requested_limit > applied_limit
                items = [_serialize_run(run, include_fields) for run in page[:applied_limit]]
                return _collection_envelope(resource, entity_name, project_name, items, applied_limit, has_more)

            if resource == "sweep":
                sweep = api.sweep(f"{path}/{sweep_id}")
                return _single_envelope(resource, entity_name, project_name, _serialize_sweep(sweep, include_fields))

            if resource == "sweeps":
                sweeps = api.project(project_name, entity=entity_name).sweeps(per_page=per_page)
                page = list(islice(sweeps, applied_limit + 1))
                has_more = len(page) > applied_limit or requested_limit > applied_limit
                items = [_serialize_sweep(sweep, include_fields) for sweep in page[:applied_limit]]
                return _collection_envelope(resource, entity_name, project_name, items, applied_limit, has_more)

            reports = api.reports(path, name=report_name, per_page=per_page)
            page = list(islice(reports, applied_limit + 1))
            has_more = len(page) > applied_limit or requested_limit > applied_limit
            items = [_serialize_report(report, include_fields) for report in page[:applied_limit]]
            return _collection_envelope(resource, entity_name, project_name, items, applied_limit, has_more)
        except (ValueError, KeyError, IndexError) as exc:
            if resource in {"project", "run", "sweep"}:
                ctx.mark_error(f"resource_not_found: {exc}")
                return structured_error(
                    "resource_not_found",
                    str(exc)[:500],
                    source="wandb_sdk",
                    resource=resource,
                    entity=entity_name,
                    project=project_name,
                )
            logger.error("W&B SDK query failed: %s", exc, exc_info=True)
            ctx.mark_error(f"sdk_query_failed: {exc}")
            return structured_error(
                "sdk_query_failed",
                str(exc)[:500],
                source="wandb_sdk",
                resource=resource,
                entity=entity_name,
                project=project_name,
            )
        except Exception as exc:
            logger.error("W&B SDK query failed: %s", exc, exc_info=True)
            ctx.mark_error(f"sdk_query_failed: {exc}")
            return structured_error(
                "sdk_query_failed",
                str(exc)[:500],
                source="wandb_sdk",
                resource=resource,
                entity=entity_name,
                project=project_name,
            )
