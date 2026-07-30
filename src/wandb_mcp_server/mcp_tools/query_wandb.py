"""Structured, read-only W&B Models queries with bounded field projection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from itertools import islice
import threading
from typing import Any, Dict, List, Literal, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import (
    MAX_RESPONSE_TOKENS,
    MCP_HOSTED_MODE,
    MCP_MAX_FULL_DETAIL_ITEMS,
    MCP_MAX_HISTORY_KEYS,
    MCP_MAX_WANDB_QUERY_ITEMS,
    MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
    MCP_WORKLOAD_PROFILE,
    structured_error,
)
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import (
    SelectiveReadUnavailable,
    fetch_projected_reports,
    fetch_projected_run,
    fetch_projected_runs,
    fetch_projected_sweeps,
)
from wandb_mcp_server.wandb_urls import public_wandb_url, publicize_wandb_url

logger = get_rich_logger(__name__)

WandBResource = Literal["project", "run", "runs", "sweep", "sweeps", "reports"]
WandBResponseMode = Literal["items", "count"]

QUERY_WANDB_TOOL_DESCRIPTION = """Query W&B Models data through bounded W&B read APIs.

Use this read-only tool for project metadata, individual runs, filtered or sorted
run collections, sweeps, and reports. For run history, artifacts, registries,
automations, and integrations, prefer the dedicated MCP tools.

For an unfamiliar project, call probe_project_tool first. Then pass only the
returned summary_keys/config_keys needed for the question. This avoids loading
every metric from wide runs and usually answers the question in one request.

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
    Additional resource details. run/runs accept summary, config, system_metrics,
    and sweep; sweep/sweeps accept config; reports accepts spec. Individual runs
    include summary metrics by default; run collections return metadata by default.
summary_keys : list[str], optional
    Specific summary metrics to include for run/runs. Supplying keys implies
    include=["summary"] and uses a server-side field projection.
config_keys : list[str], optional
    Specific config values to include for run/runs. Supplying keys implies
    include=["config"] and uses a server-side field projection.
response_mode : "items" | "count", optional
    "items" returns bounded resources. "count" is supported for resource="runs"
    and returns only the exact server-side matching count.
cursor : str, optional
    Opaque continuation cursor returned by a previous collection response.

Returns
-------
dict
    Collection results include returned_count, total_count, has_more, limit, and
    project_exhaustive. Single-resource results use item.

For schema introspection, unmodeled fields, aliases, cross-resource nesting, or an
exact GraphQL response shape, an administrator may explicitly enable the separate
query_wandb_graphql_tool with WANDB_MCP_ENABLE_RAW_GRAPHQL=true.
"""

_INCLUDE_FIELDS = {
    "project": frozenset(),
    "run": frozenset({"summary", "config", "system_metrics", "sweep"}),
    "runs": frozenset({"summary", "config", "system_metrics", "sweep"}),
    "sweep": frozenset({"config"}),
    "sweeps": frozenset({"config"}),
    "reports": frozenset({"spec"}),
}
_OPTIONAL_RESPONSE_FIELDS = ("system_metrics", "config", "spec", "sweep", "summary")
_JSON_MAX_DEPTH = 12
_JSON_MAX_MAPPING_KEYS = 500
_JSON_MAX_LIST_ITEMS = 500
_JSON_MAX_STRING_CHARS = 16_000
_SDK_RUN_CACHE_LOCK = threading.Lock()


class WandBQueryValidationError(ValueError):
    """Raised before any W&B client is obtained for an invalid request."""


def _json_safe(
    value: Any,
    *,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> Any:
    """Convert SDK values into deterministic JSON-compatible values."""
    if _depth >= _JSON_MAX_DEPTH:
        return {"_truncated": "max_depth"}
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= _JSON_MAX_STRING_CHARS else value[:_JSON_MAX_STRING_CHARS] + "…"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return {"_truncated": "cycle"}
    seen.add(value_id)
    try:
        scalar_item = getattr(value, "item", None)
        if callable(scalar_item):
            return _json_safe(scalar_item(), _depth=_depth + 1, _seen=seen)
    except (TypeError, ValueError, OverflowError):
        pass
    if isinstance(value, Mapping):
        result = {
            str(key): _json_safe(item, _depth=_depth + 1, _seen=seen)
            for key, item in islice(value.items(), _JSON_MAX_MAPPING_KEYS)
        }
        try:
            mapping_length = len(value)
        except TypeError:
            mapping_length = len(result)
        if mapping_length > _JSON_MAX_MAPPING_KEYS:
            result["_truncated_keys"] = mapping_length - _JSON_MAX_MAPPING_KEYS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(islice(iter(value), _JSON_MAX_LIST_ITEMS + 1))
        result = [_json_safe(item, _depth=_depth + 1, _seen=seen) for item in values[:_JSON_MAX_LIST_ITEMS]]
        if len(values) > _JSON_MAX_LIST_ITEMS:
            try:
                omitted = len(value) - _JSON_MAX_LIST_ITEMS
            except TypeError:
                omitted = 1
            result.append({"_truncated_items": omitted})
        return result
    try:
        mapping = dict(value)
    except (TypeError, ValueError):
        rendered = str(value)
        return rendered if len(rendered) <= _JSON_MAX_STRING_CHARS else rendered[:_JSON_MAX_STRING_CHARS] + "…"
    return _json_safe(mapping, _depth=_depth + 1, _seen=seen)


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
    entity = getattr(project, "entity", None)
    name = getattr(project, "name", None)
    return {
        "id": _json_safe(getattr(project, "id", None)),
        "name": _json_safe(name),
        "entity": _json_safe(entity),
        "description": _json_safe(getattr(project, "description", None)),
        "visibility": _json_safe(getattr(project, "visibility", None)),
        "created_at": _json_safe(getattr(project, "created_at", None)),
        "updated_at": _json_safe(getattr(project, "updated_at", None)),
        "tags": _json_safe(getattr(project, "tags", [])),
        "url": publicize_wandb_url(
            getattr(project, "url", None),
            fallback_segments=(entity, name),
        ),
    }


def _serialize_sweep(sweep: Any, include: frozenset[str]) -> Dict[str, Any]:
    entity = getattr(sweep, "entity", None)
    project = getattr(sweep, "project", None)
    sweep_id = getattr(sweep, "id", None) or getattr(sweep, "name", None)
    result = {
        "id": _json_safe(getattr(sweep, "id", None)),
        "name": _json_safe(getattr(sweep, "name", None)),
        "state": _json_safe(getattr(sweep, "state", None)),
        "entity": _json_safe(entity),
        "project": _json_safe(project),
        "expected_run_count": _json_safe(getattr(sweep, "expected_run_count", None)),
        "url": publicize_wandb_url(
            getattr(sweep, "url", None),
            fallback_segments=(entity, project, "sweeps", sweep_id),
        ),
    }
    if "config" in include:
        result["config"] = _json_safe(getattr(sweep, "config", {}))
    return result


def _select_mapping_values(value: Any, keys: Optional[List[str]]) -> Dict[str, Any]:
    if not value:
        return {}
    if keys is None:
        return _json_safe(value)
    try:
        mapping = dict(value)
    except (TypeError, ValueError):
        return {}
    return {key: _json_safe(mapping[key]) for key in keys if key in mapping}


def _serialize_summary(run: Any, summary_keys: Optional[List[str]]) -> Dict[str, Any]:
    summary = getattr(run, "summary", None)
    return _select_mapping_values(summary, summary_keys)


def _select_config_values(value: Any, keys: Optional[List[str]]) -> tuple[Dict[str, Any], list[str]]:
    if keys is None:
        return _json_safe(value or {}), []
    try:
        mapping = dict(value or {})
    except (TypeError, ValueError):
        return {}, list(keys)
    selected: dict[str, Any] = {}
    missing: list[str] = []
    for key in keys:
        if key in mapping:
            selected[key] = _json_safe(mapping[key])
            continue
        current: Any = mapping
        found = True
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            selected[key] = _json_safe(current)
        else:
            missing.append(key)
    return selected, missing


def _serialize_run(
    run: Any,
    include: frozenset[str],
    *,
    summary_keys: Optional[List[str]] = None,
    config_keys: Optional[List[str]] = None,
    include_summary: bool = False,
) -> Dict[str, Any]:
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    run_id = getattr(run, "id", None)
    result = {
        "id": _json_safe(run_id),
        "display_name": _json_safe(getattr(run, "name", None)),
        "state": _json_safe(getattr(run, "state", None)),
        "entity": _json_safe(entity),
        "project": _json_safe(project),
        "url": publicize_wandb_url(
            getattr(run, "url", None),
            fallback_segments=(entity, project, "runs", run_id),
        ),
        "created_at": _json_safe(getattr(run, "created_at", None)),
        "heartbeat_at": _json_safe(getattr(run, "heartbeat_at", None)),
        "duration": _json_safe(getattr(run, "duration", None)),
        "group": _json_safe(getattr(run, "group", None)),
        "job_type": _json_safe(getattr(run, "job_type", None)),
        "tags": _json_safe(getattr(run, "tags", [])),
        "user": _serialize_user(getattr(run, "user", None)),
    }
    if include_summary:
        result["summary"] = _serialize_summary(run, summary_keys)
        if summary_keys:
            summary = getattr(run, "summary", None)
            try:
                summary_mapping = dict(summary or {})
            except (TypeError, ValueError):
                summary_mapping = {}
            missing = [key for key in summary_keys if key not in summary_mapping]
            if missing:
                result["missing_summary_keys"] = missing
    if "config" in include:
        config = getattr(run, "config", {})
        selected_config, missing = _select_config_values(config, config_keys)
        result["config"] = selected_config
        if missing:
            result["missing_config_keys"] = missing
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
        "url": publicize_wandb_url(getattr(report, "url", None)),
    }
    if "spec" in include:
        result["spec"] = _json_safe(getattr(report, "spec", {}))
    return result


def _estimate_tokens(payload: Dict[str, Any]) -> int:
    """Conservative approximation suitable for enforcing the configured budget."""
    return max(1, len(json.dumps(payload, default=str, ensure_ascii=False)) // 4)


def _fit_collection_to_budget(payload: Dict[str, Any]) -> Dict[str, Any]:
    original_has_more = bool(payload["has_more"])
    original_exhaustive = bool(payload["project_exhaustive"])
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
        payload["returned_count"] = len(payload["items"])
        payload["count"] = len(payload["items"])
        payload["has_more"] = original_has_more or dropped_items > 0
        payload["project_exhaustive"] = original_exhaustive and dropped_items == 0
        if dropped_items:
            payload["next_cursor"] = None
        payload["truncated"] = True
        payload["truncation"] = {
            "applied": True,
            "reason": "response_token_budget",
            "omitted_fields": sorted(omitted_fields),
            "dropped_items": dropped_items,
        }
    if _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        return structured_error(
            "response_too_large",
            "The W&B collection metadata exceeded the configured response budget",
            source=payload.get("source", "wandb_sdk"),
            resource=payload.get("resource"),
        )
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
    if _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        identity_fields = ("id", "name", "display_name", "state", "entity", "project", "url")
        payload["item"] = {key: payload["item"][key] for key in identity_fields if key in payload["item"]}
        payload["truncated"] = True
        payload["truncation"] = {
            "applied": True,
            "reason": "response_token_budget",
            "omitted_fields": ["non_identity_fields"],
            "dropped_items": 0,
        }
    if _estimate_tokens(payload) > MAX_RESPONSE_TOKENS:
        return structured_error(
            "response_too_large",
            "The W&B resource metadata exceeded the configured response budget",
            source=payload.get("source", "wandb_sdk"),
            resource=payload.get("resource"),
        )
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
    summary_keys: Optional[List[str]],
    config_keys: Optional[List[str]],
    response_mode: str,
    cursor: Optional[str],
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
    if response_mode not in {"items", "count"}:
        raise WandBQueryValidationError("response_mode must be 'items' or 'count'")
    if response_mode == "count" and resource != "runs":
        raise WandBQueryValidationError("response_mode='count' is supported only for resource='runs'")
    if cursor is not None and (not isinstance(cursor, str) or not cursor.strip()):
        raise WandBQueryValidationError("cursor must be a non-empty string")
    if cursor is not None and resource not in {"runs", "sweeps", "reports"}:
        raise WandBQueryValidationError("cursor is supported only for collection resources")
    if cursor is not None and response_mode != "items":
        raise WandBQueryValidationError("cursor is not supported with response_mode='count'")
    if include is not None and (not isinstance(include, list) or not all(isinstance(item, str) for item in include)):
        raise WandBQueryValidationError("include must be a list of strings")
    if summary_keys is not None and (
        not isinstance(summary_keys, list)
        or not summary_keys
        or len(summary_keys) > MCP_MAX_HISTORY_KEYS
        or not all(isinstance(item, str) and item.strip() for item in summary_keys)
    ):
        raise WandBQueryValidationError(f"summary_keys must contain 1-{MCP_MAX_HISTORY_KEYS} non-empty strings")
    if summary_keys is not None and resource not in {"run", "runs"}:
        raise WandBQueryValidationError("summary_keys are supported only for resource='run' or resource='runs'")
    if config_keys is not None and (
        not isinstance(config_keys, list)
        or not config_keys
        or len(config_keys) > MCP_MAX_HISTORY_KEYS
        or not all(isinstance(item, str) and item.strip() for item in config_keys)
    ):
        raise WandBQueryValidationError(f"config_keys must contain 1-{MCP_MAX_HISTORY_KEYS} non-empty strings")
    if config_keys is not None and resource not in {"run", "runs"}:
        raise WandBQueryValidationError("config_keys are supported only for resource='run' or resource='runs'")
    if response_mode == "count" and (include or summary_keys or config_keys):
        raise WandBQueryValidationError("response_mode='count' does not accept include, summary_keys, or config_keys")

    requested = frozenset(include or [])
    if summary_keys is not None:
        requested = requested | {"summary"}
    if config_keys is not None:
        requested = requested | {"config"}
    unsupported = requested - _INCLUDE_FIELDS[resource]
    if unsupported:
        allowed = sorted(_INCLUDE_FIELDS[resource])
        raise WandBQueryValidationError(
            f"unsupported include value(s) for resource={resource!r}: {sorted(unsupported)}; allowed: {allowed}"
        )
    full_detail_limit = 3 if MCP_HOSTED_MODE and MCP_WORKLOAD_PROFILE == "local" else MCP_MAX_FULL_DETAIL_ITEMS
    if resource in {"runs", "sweeps", "reports"} and limit > full_detail_limit:
        untargeted_summary = "summary" in requested and summary_keys is None
        untargeted_config = "config" in requested and config_keys is None
        unbounded_details = requested & {"system_metrics", "spec"}
        if untargeted_summary or untargeted_config or unbounded_details:
            raise WandBQueryValidationError(
                f"{MCP_WORKLOAD_PROFILE} collection details require "
                f"limit<={full_detail_limit}; use summary_keys/config_keys for projected run fields"
            )
    return requested


def _collection_envelope(
    resource: str,
    entity_name: str,
    project_name: str,
    items: List[Dict[str, Any]],
    requested_limit: int,
    applied_limit: int,
    has_more: bool,
    total_count: Optional[int],
    *,
    cursor: Optional[str] = None,
    next_cursor: Optional[str] = None,
    source: str = "wandb_sdk",
    compatibility_caveat: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": source,
        "resource": resource,
        "entity": entity_name,
        "project": project_name,
        "items": items,
        "returned_count": len(items),
        "count": len(items),
        "total_count": total_count,
        "has_more": has_more,
        "requested_limit": requested_limit,
        "limit": applied_limit,
        "limit_clamped": requested_limit != applied_limit,
        "next_cursor": next_cursor if has_more else None,
        "project_exhaustive": cursor is None and not has_more,
        "truncated": has_more,
        "truncation": (
            {"applied": True, "reason": "collection_limit", "omitted_fields": [], "dropped_items": 0}
            if has_more
            else {"applied": False}
        ),
    }
    if compatibility_caveat:
        payload["compatibility_caveat"] = compatibility_caveat
    return _fit_collection_to_budget(payload)


def _single_envelope(
    resource: str,
    entity_name: str,
    project_name: str,
    item: Dict[str, Any],
    *,
    source: str = "wandb_sdk",
    compatibility_caveat: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": source,
        "resource": resource,
        "entity": entity_name,
        "project": project_name,
        "item": item,
        "truncated": False,
        "truncation": {"applied": False},
    }
    if compatibility_caveat:
        payload["compatibility_caveat"] = compatibility_caveat
    return _fit_single_to_budget(payload)


def _collection_total_count(collection: Any) -> Optional[int]:
    try:
        return len(collection)
    except (TypeError, AttributeError, NotImplementedError):
        return None


def _collection_cursor(collection: Any, has_more: bool) -> Optional[str]:
    if not has_more:
        return None
    try:
        cursor = collection.cursor
    except (AttributeError, TypeError, ValueError):
        return None
    return str(cursor) if cursor else None


def _decorate_projected_run(item: Dict[str, Any]) -> Dict[str, Any]:
    run_id = item.get("id")
    if run_id:
        item["url"] = public_wandb_url(item.get("entity"), item.get("project"), "runs", run_id)
    sweep = item.get("sweep")
    if isinstance(sweep, dict) and sweep.get("id"):
        sweep["url"] = public_wandb_url(
            item.get("entity"),
            item.get("project"),
            "sweeps",
            sweep["id"],
        )
    return _json_safe(item)


def _decorate_projected_sweep(item: Dict[str, Any]) -> Dict[str, Any]:
    sweep_id = item.get("id")
    if sweep_id:
        item["url"] = public_wandb_url(item.get("entity"), item.get("project"), "sweeps", sweep_id)
    return _json_safe(item)


def _decorate_projected_report(
    item: Dict[str, Any],
    entity_name: str,
    project_name: str,
) -> Dict[str, Any]:
    report_path = item.get("name") or item.get("id")
    if report_path:
        item["url"] = public_wandb_url(entity_name, project_name, "reports", report_path)
    return _json_safe(item)


def _sdk_error_result(
    exc: Exception,
    *,
    resource: str,
    entity_name: str,
    project_name: str,
) -> Dict[str, Any]:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
    retry_after = None
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    common = {
        "source": "wandb_sdk",
        "resource": resource,
        "entity": entity_name,
        "project": project_name,
    }
    if status == 401 or "unauth" in name or "invalid api key" in message or "no w&b api key" in message:
        return structured_error("authentication_failed", "W&B authentication failed", **common)
    if status == 403 or "permission" in message or "forbidden" in message:
        return structured_error("permission_denied", "W&B denied access to this resource", **common)
    if status == 404 or "not found" in message or "could not find" in message:
        return structured_error("resource_not_found", "The requested W&B resource was not found", **common)
    if status == 429 or status == 503 or "rate limit" in message or "overload" in message:
        try:
            retry_after_ms = max(1_000, min(60_000, int(float(retry_after or 1) * 1_000)))
        except (TypeError, ValueError):
            retry_after_ms = 1_000
        return structured_error(
            "server_busy",
            "W&B is temporarily busy; retry this bounded request",
            retryable=True,
            retry_after_ms=retry_after_ms,
            **common,
        )
    if "timeout" in name or "timed out" in message:
        return structured_error(
            "upstream_timeout",
            "The bounded W&B request timed out",
            retryable=True,
            **common,
        )
    return structured_error(
        "sdk_query_failed",
        f"W&B query failed ({type(exc).__name__})",
        **common,
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
    summary_keys: Optional[List[str]] = None,
    config_keys: Optional[List[str]] = None,
    response_mode: WandBResponseMode = "items",
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a structured, bounded W&B read."""
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
            summary_keys,
            config_keys,
            response_mode,
            cursor,
        )
    except WandBQueryValidationError as exc:
        return structured_error("invalid_request", str(exc), source="wandb_sdk", resource=resource)

    requested_limit = limit
    applied_limit = min(limit, MCP_MAX_WANDB_QUERY_ITEMS)
    path = f"{entity_name}/{project_name}"

    with track_tool_execution(
        "query_wandb",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "resource": resource,
            "limit": applied_limit,
            "response_mode": response_mode,
            "projected_summary_keys": len(summary_keys or []),
            "projected_config_keys": len(config_keys or []),
            "continued": cursor is not None,
        },
        mcp_tool_name="query_wandb_tool",
    ) as ctx:
        try:
            api = WandBApiManager.get_api()
            if resource == "project":
                project = api.project(project_name, entity=entity_name)
                return _single_envelope(resource, entity_name, project_name, _serialize_project(project))

            if resource == "run":
                use_projection = bool(summary_keys or config_keys) and not (
                    include_fields & {"system_metrics", "sweep"} or ("config" in include_fields and config_keys is None)
                )
                if use_projection:
                    try:
                        item = fetch_projected_run(
                            api,
                            entity=entity_name,
                            project=project_name,
                            run_id=str(run_id),
                            summary_keys=summary_keys if summary_keys is not None else None,
                            config_keys=config_keys or (),
                        )
                    except SelectiveReadUnavailable as exc:
                        run = api.run(f"{path}/{run_id}")
                        return _single_envelope(
                            resource,
                            entity_name,
                            project_name,
                            _serialize_run(
                                run,
                                include_fields,
                                summary_keys=summary_keys,
                                config_keys=config_keys,
                                include_summary=True,
                            ),
                            compatibility_caveat=(
                                f"{exc}; used one bounded full SDK run because projected fields were unavailable"
                            ),
                        )
                    if item is None:
                        raise ValueError(f"run not found: {run_id}")
                    return _single_envelope(
                        resource,
                        entity_name,
                        project_name,
                        _decorate_projected_run(item),
                        source="wandb_selective_read",
                    )
                run = api.run(f"{path}/{run_id}")
                return _single_envelope(
                    resource,
                    entity_name,
                    project_name,
                    _serialize_run(
                        run,
                        include_fields,
                        summary_keys=summary_keys,
                        config_keys=config_keys,
                        include_summary=True,
                    ),
                )

            if resource == "runs":
                if response_mode == "count":
                    runs = api.runs(
                        path,
                        filters=filters,
                        order=order,
                        per_page=1,
                        include_sweeps=False,
                        lazy=True,
                    )
                    total_count = len(runs)
                    return {
                        "source": "wandb_sdk",
                        "resource": resource,
                        "entity": entity_name,
                        "project": project_name,
                        "response_mode": "count",
                        "total_count": total_count,
                        "project_exhaustive": True,
                    }

                use_projection = bool(cursor or include_fields or summary_keys or config_keys)
                if use_projection:
                    try:
                        projected = fetch_projected_runs(
                            api,
                            entity=entity_name,
                            project=project_name,
                            filters=filters,
                            order=order,
                            limit=applied_limit,
                            page_size=MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
                            summary_keys=summary_keys if summary_keys is not None else None,
                            config_keys=config_keys if config_keys is not None else None,
                            include_summary="summary" in include_fields,
                            include_config="config" in include_fields,
                            include_sweep="sweep" in include_fields,
                            include_system_metrics="system_metrics" in include_fields,
                            cursor=cursor,
                        )
                    except SelectiveReadUnavailable as exc:
                        if cursor is not None or applied_limit > MCP_MAX_FULL_DETAIL_ITEMS:
                            return structured_error(
                                "selective_read_unavailable",
                                (
                                    f"{exc}; this backend cannot project selected fields and the requested "
                                    "continuation or detail shape cannot be reproduced safely with the SDK fallback"
                                ),
                                source="wandb_sdk",
                                resource=resource,
                            )
                        compatibility_caveat = (
                            f"{exc}; used a bounded full-page SDK fallback without per-run lazy loads"
                        )
                    else:
                        return _collection_envelope(
                            resource,
                            entity_name,
                            project_name,
                            [_decorate_projected_run(item) for item in projected.items],
                            requested_limit,
                            applied_limit,
                            projected.has_more,
                            projected.total_count,
                            cursor=cursor,
                            next_cursor=projected.next_cursor,
                            source="wandb_selective_read",
                        )
                else:
                    compatibility_caveat = None

                with _SDK_RUN_CACHE_LOCK:
                    flush = getattr(api, "flush", None)
                    if callable(flush):
                        flush()
                    runs = api.runs(
                        path,
                        filters=filters,
                        order=order,
                        per_page=min(
                            applied_limit + (1 if compatibility_caveat else 0),
                            MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
                        ),
                        include_sweeps="sweep" in include_fields,
                        lazy=not bool(include_fields & {"summary", "config", "system_metrics"}),
                    )
                total_count = _collection_total_count(runs)
                page = list(islice(runs, applied_limit))
                has_more = bool(total_count is not None and total_count > len(page))
                if total_count is None:
                    try:
                        has_more = bool(runs.more)
                    except (AttributeError, TypeError, ValueError):
                        has_more = len(page) == applied_limit
                items = [
                    _serialize_run(
                        run,
                        include_fields,
                        summary_keys=summary_keys,
                        config_keys=config_keys,
                        include_summary="summary" in include_fields,
                    )
                    for run in page[:applied_limit]
                ]
                if total_count is None and not has_more:
                    total_count = len(items)
                return _collection_envelope(
                    resource,
                    entity_name,
                    project_name,
                    items,
                    requested_limit,
                    applied_limit,
                    has_more,
                    total_count,
                    cursor=cursor,
                    next_cursor=_collection_cursor(runs, has_more),
                    compatibility_caveat=compatibility_caveat,
                )

            if resource == "sweep":
                sweep = api.sweep(f"{path}/{sweep_id}")
                return _single_envelope(resource, entity_name, project_name, _serialize_sweep(sweep, include_fields))

            if resource == "sweeps":
                try:
                    sweeps = fetch_projected_sweeps(
                        api,
                        entity=entity_name,
                        project=project_name,
                        limit=applied_limit,
                        page_size=MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
                        include_config="config" in include_fields,
                        cursor=cursor,
                    )
                except SelectiveReadUnavailable as exc:
                    if cursor is not None or applied_limit > MCP_MAX_FULL_DETAIL_ITEMS:
                        return structured_error(
                            "selective_read_unavailable",
                            f"{exc}; a bounded continuation cannot be reproduced safely by the public SDK",
                            source="wandb_sdk",
                            resource=resource,
                        )
                    sdk_sweeps = api.project(project_name, entity=entity_name).sweeps(
                        per_page=min(applied_limit + 1, MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE)
                    )
                    sweep_page = list(islice(sdk_sweeps, applied_limit + 1))
                    has_more = len(sweep_page) > applied_limit or bool(getattr(sdk_sweeps, "more", False))
                    items = [_serialize_sweep(sweep, include_fields) for sweep in sweep_page[:applied_limit]]
                    return _collection_envelope(
                        resource,
                        entity_name,
                        project_name,
                        items,
                        requested_limit,
                        applied_limit,
                        has_more,
                        _collection_total_count(sdk_sweeps),
                        next_cursor=_collection_cursor(sdk_sweeps, has_more),
                        compatibility_caveat=f"{exc}; used a small bounded SDK fallback",
                    )
                return _collection_envelope(
                    resource,
                    entity_name,
                    project_name,
                    [_decorate_projected_sweep(item) for item in sweeps.items],
                    requested_limit,
                    applied_limit,
                    sweeps.has_more,
                    sweeps.total_count,
                    cursor=cursor,
                    next_cursor=sweeps.next_cursor,
                    source="wandb_selective_read",
                )

            try:
                reports = fetch_projected_reports(
                    api,
                    entity=entity_name,
                    project=project_name,
                    report_name=report_name,
                    limit=applied_limit,
                    page_size=MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE,
                    include_spec="spec" in include_fields,
                    cursor=cursor,
                )
            except SelectiveReadUnavailable as exc:
                if cursor is not None or applied_limit > MCP_MAX_FULL_DETAIL_ITEMS:
                    return structured_error(
                        "selective_read_unavailable",
                        f"{exc}; a bounded continuation cannot be reproduced safely by the public SDK",
                        source="wandb_sdk",
                        resource=resource,
                    )
                sdk_reports = api.reports(
                    path,
                    name=report_name,
                    per_page=min(applied_limit + 1, MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE),
                )
                report_page = list(islice(sdk_reports, applied_limit + 1))
                items = [_serialize_report(report, include_fields) for report in report_page[:applied_limit]]
                has_more = len(report_page) > applied_limit or bool(getattr(sdk_reports, "more", False))
                return _collection_envelope(
                    resource,
                    entity_name,
                    project_name,
                    items,
                    requested_limit,
                    applied_limit,
                    has_more,
                    None if has_more else len(items),
                    next_cursor=_collection_cursor(sdk_reports, has_more),
                    compatibility_caveat=f"{exc}; used a small bounded SDK fallback",
                )
            return _collection_envelope(
                resource,
                entity_name,
                project_name,
                [_decorate_projected_report(item, entity_name, project_name) for item in reports.items],
                requested_limit,
                applied_limit,
                reports.has_more,
                reports.total_count,
                cursor=cursor,
                next_cursor=reports.next_cursor,
                source="wandb_selective_read",
            )
        except Exception as exc:
            logger.error("W&B SDK query failed: %s", exc, exc_info=True)
            ctx.mark_error(f"sdk_query_failed: {exc}")
            return _sdk_error_result(
                exc,
                resource=resource,
                entity_name=entity_name,
                project_name=project_name,
            )
