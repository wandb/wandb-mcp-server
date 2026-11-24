"""Interactive Agent Lens Insights Explorer MCP App (`ui://wandb/agent-lens-insights.html`)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from importlib.resources import files
import json
import os
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from wandb_mcp_server.config import structured_error
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.mcp_tools.agent_lens import (
    CLUSTERING_STATUS_PATH,
    CLUSTERS_PATH,
    FAILURE_ATTRIBUTIONS_PATH,
    LATEST_WEEK_PATH,
    _breakdowns_path,
    _bounded_list,
    _serialize_insights_filters,
    _validated_window,
    fetch_agent_lens_json,
)
from wandb_mcp_server.tool_results import ToolErrorJSON
from wandb_mcp_server.wandb_urls import public_forge_agent_lens_url

INSIGHTS_RESOURCE_URI = "ui://wandb/agent-lens-insights.html"
INSIGHTS_TOOL_NAME = "render_agent_lens_insights_tool"
INSIGHTS_SCHEMA = "wandb.agent_lens_insights/v1"
VALID_WINDOW_DAYS = frozenset({1, 7, 14, 30})
MAX_OVERVIEW_CLUSTERS = 50
MAX_OVERVIEW_TRACE_IDS = 50

RENDER_AGENT_LENS_INSIGHTS_TOOL_DESCRIPTION = """Render project-wide Agent Lens failure & intent insights as an interactive MCP App (`ui://wandb/agent-lens-insights.html`) plus compact summary JSON.

Gathers classified turn coverage, clustering status, failure & intent category breakdowns, top topic clusters, and optional failure attributions in one parallel call (`window_days`: 1, 7, 14, or 30, or explicit RFC 3339 `start_at`/`end_at` <= 30 days; optional `agent_name` filter). Interactive hosts support window switching, inline example-turn & attribution expansion, conversation inspection, and `[MCP Agent Lens Insights Context]` sync."""


def register_agent_lens_insights_resource(mcp: FastMCP) -> None:
    """Register the offline single-file Insights Explorer MCP App resource."""
    if not hasattr(mcp, "resource"):
        return
    ui: dict[str, Any] = {
        "prefersBorder": True,
        "csp": {"connectDomains": [], "resourceDomains": ["data:"]},
    }
    if domain := os.environ.get("WANDB_MCP_APP_DOMAIN"):
        ui["domain"] = domain

    @mcp.resource(INSIGHTS_RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def agent_lens_insights_html() -> str:
        return files("wandb_mcp_server").joinpath("data/agent-lens-insights.html").read_text(encoding="utf-8")


def _error_result(code: str, message: str, **details: Any) -> CallToolResult:
    payload = structured_error(code, message, **details)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))],
        structuredContent=payload,
        isError=True,
    )


def _error_from_tool_json(raw: ToolErrorJSON) -> CallToolResult:
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        decoded = structured_error("agent_lens_query_failed", "The Agent Lens request failed.")
    if not isinstance(decoded, dict):
        decoded = structured_error("agent_lens_query_failed", "The Agent Lens request failed.")
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(decoded, separators=(",", ":")))],
        structuredContent=decoded,
        isError=True,
    )


def _current_access_mode() -> Literal["read-only", "read-write"]:
    raw = (os.environ.get("WANDB_MCP_ACCESS_MODE") or "read-write").strip().lower()
    return "read-only" if raw == "read-only" else "read-write"


def _humanize_slug(value: str) -> str:
    cleaned = value.strip().replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() if part.islower() else part for part in cleaned.split()) or value


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _safe_optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _normalize_severity(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    lowered = raw.strip().lower()
    if lowered in {"critical", "high", "medium", "low"}:
        return lowered
    return lowered[:32] if lowered else None


def _normalize_coverage(raw_res: Any) -> Dict[str, Any]:
    data = raw_res.get("data") if isinstance(raw_res, dict) else None
    if not isinstance(data, dict):
        return {
            "firstWeek": None,
            "latestWeek": None,
            "classifiedTurnCount": None,
            "totalTurnCount": None,
            "coveragePct": None,
        }
    first_week = str(data["first_week"]).strip() if data.get("first_week") else None
    latest_week = str(data["latest_week"]).strip() if data.get("latest_week") else None
    classified = _safe_optional_int(
        data.get("classified_turn_count") if "classified_turn_count" in data else data.get("classified_turns")
    )
    total = _safe_optional_int(data.get("total_turn_count") if "total_turn_count" in data else data.get("total_turns"))
    pct = _safe_optional_float(data.get("coverage_pct") if "coverage_pct" in data else data.get("coverage_percentage"))
    if pct is None and classified is not None and total is not None and total > 0:
        pct = round((classified / total) * 100.0, 1)
    elif pct is not None and 0.0 <= pct <= 1.0 and (classified is None or total is None):
        pct = round(pct * 100.0, 1)
    return {
        "firstWeek": first_week,
        "latestWeek": latest_week,
        "classifiedTurnCount": classified,
        "totalTurnCount": total,
        "coveragePct": pct,
    }


def _normalize_clustering_status(raw_res: Any) -> List[Dict[str, Any]]:
    data = raw_res.get("data") if isinstance(raw_res, dict) else None
    if not isinstance(data, list):
        return []
    items: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        sig = str(entry.get("signature_type") or entry.get("type") or "").strip().lower()
        if not sig:
            continue
        completed_at = entry.get("completed_at") or entry.get("updated_at") or entry.get("last_run_at")
        status = entry.get("status") or ("ready" if completed_at else "pending")
        items.append(
            {
                "signatureType": sig,
                "status": str(status).strip(),
                "completedAt": str(completed_at).strip() if completed_at else None,
            }
        )
    return items


def _normalize_signature_section(
    breakdown_res: Any,
    clusters_res: Any,
    top_n_clusters: int,
) -> Dict[str, Any]:
    bd_data = breakdown_res.get("data") if isinstance(breakdown_res, dict) else None
    if not isinstance(bd_data, dict):
        bd_data = {}
    turn_count = _safe_int(bd_data.get("turn_count"), 0)
    raw_categories = bd_data.get("categories") if isinstance(bd_data.get("categories"), list) else []

    cluster_meta_from_bd: Dict[str, Dict[str, Any]] = {}
    categories: List[Dict[str, Any]] = []
    for raw_cat in raw_categories:
        if not isinstance(raw_cat, dict):
            continue
        cat_id = str(raw_cat.get("category") or raw_cat.get("id") or "").strip()
        if not cat_id:
            continue
        count = _safe_int(raw_cat.get("count"), 0)
        raw_share = _safe_optional_float(raw_cat.get("share"))
        if raw_share is None:
            share = round(count / turn_count, 4) if turn_count > 0 else 0.0
        else:
            share = round(raw_share / 100.0, 4) if raw_share > 1.0 else round(max(0.0, raw_share), 4)
        severity = _normalize_severity(raw_cat.get("severity") or raw_cat.get("max_severity"))
        label = str(raw_cat.get("title") or raw_cat.get("label") or _humanize_slug(cat_id)).strip()

        cluster_breakdowns: List[Dict[str, Any]] = []
        raw_cb_list = raw_cat.get("cluster_breakdowns")
        if isinstance(raw_cb_list, list):
            for cb in raw_cb_list:
                if not isinstance(cb, dict):
                    continue
                cid = str(cb.get("id") or cb.get("topic_id") or "").strip()
                if not cid:
                    continue
                c_count = _safe_int(cb.get("count"), 0)
                c_share = _safe_optional_float(cb.get("share"))
                if c_share is None and turn_count > 0:
                    c_share = round(c_count / turn_count, 4)
                c_title = str(cb.get("title") or cb.get("name") or "").strip() or None
                c_sev = _normalize_severity(cb.get("severity"))
                cb_item = {
                    "id": cid,
                    "title": c_title,
                    "count": c_count,
                    "share": c_share,
                    "severity": c_sev,
                }
                cluster_breakdowns.append(cb_item)
                cluster_meta_from_bd[cid] = {
                    "categoryId": cat_id,
                    "count": c_count,
                    "share": c_share,
                    "title": c_title,
                    "severity": c_sev,
                }
        cluster_breakdowns.sort(key=lambda item: item["count"], reverse=True)
        categories.append(
            {
                "categoryId": cat_id,
                "label": label,
                "count": count,
                "share": share,
                "severity": severity,
                "clusterBreakdowns": cluster_breakdowns[:top_n_clusters],
            }
        )

    categories.sort(key=lambda item: item["count"], reverse=True)

    raw_clusters = clusters_res.get("data") if isinstance(clusters_res, dict) else None
    seen_cluster_ids: set[str] = set()
    clusters: List[Dict[str, Any]] = []
    if isinstance(raw_clusters, list):
        for raw_cl in raw_clusters:
            if not isinstance(raw_cl, dict):
                continue
            cid = str(raw_cl.get("id") or raw_cl.get("topic_id") or "").strip()
            if not cid or cid in seen_cluster_ids:
                continue
            seen_cluster_ids.add(cid)
            bd_meta = cluster_meta_from_bd.get(cid, {})
            c_count = _safe_optional_int(raw_cl.get("count"))
            if c_count is None:
                c_count = bd_meta.get("count")
            c_share = _safe_optional_float(raw_cl.get("share"))
            if c_share is None:
                c_share = bd_meta.get("share")
            if c_share is None:
                c_share = round(c_count / turn_count, 4) if (c_count is not None and turn_count > 0) else 0.0
            elif c_share > 1.0:
                c_share = round(c_share / 100.0, 4)
            cat_id = (
                str(raw_cl.get("category") or raw_cl.get("category_id") or bd_meta.get("categoryId") or "").strip()
                or None
            )
            title = str(raw_cl.get("title") or raw_cl.get("name") or bd_meta.get("title") or cid).strip()
            summary_raw = raw_cl.get("description") or raw_cl.get("summary")
            summary = sanitize_sensitive_text(summary_raw, max_chars=280) if summary_raw else None
            severity = _normalize_severity(raw_cl.get("severity") or bd_meta.get("severity"))
            attribution_raw = raw_cl.get("attribution") or raw_cl.get("failure_attribution") or raw_cl.get("root_cause")
            attribution = str(attribution_raw).strip() if attribution_raw else None
            clusters.append(
                {
                    "id": cid,
                    "title": title,
                    "summary": summary,
                    "categoryId": cat_id,
                    "count": c_count,
                    "share": round(max(0.0, c_share), 4),
                    "severity": severity,
                    "attribution": attribution,
                }
            )

    # Backfill clusters present in breakdown cluster_breakdowns if /insights/clusters returned fewer
    for cid, bd_meta in cluster_meta_from_bd.items():
        if cid in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cid)
        clusters.append(
            {
                "id": cid,
                "title": bd_meta.get("title") or cid,
                "summary": None,
                "categoryId": bd_meta.get("categoryId"),
                "count": bd_meta.get("count"),
                "share": round(float(bd_meta.get("share") or 0.0), 4),
                "severity": bd_meta.get("severity"),
                "attribution": None,
            }
        )

    clusters.sort(key=lambda item: (item["share"], item["count"] or 0), reverse=True)
    return {
        "turnCount": turn_count,
        "categories": categories,
        "clusters": clusters[:top_n_clusters],
    }


def _normalize_failure_attributions(raw_res: Any) -> List[Dict[str, Any]]:
    data = raw_res.get("data") if isinstance(raw_res, dict) else None
    rows = data if isinstance(data, list) else (data.get("attributions") if isinstance(data, dict) else None)
    if not isinstance(rows, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in rows[:MAX_OVERVIEW_TRACE_IDS]:
        if not isinstance(item, dict):
            continue
        trace_id = str(item.get("trace_id") or "").strip()
        root_cause = (
            item.get("root_cause")
            or item.get("attribution")
            or item.get("failure_type")
            or item.get("category")
            or "unknown"
        )
        summary_raw = item.get("summary") or item.get("reason") or item.get("explanation") or item.get("failure_reason")
        normalized.append(
            {
                "traceId": trace_id or None,
                "failureId": str(item["failure_id"]).strip() if item.get("failure_id") else None,
                "rootCause": str(root_cause).strip(),
                "severity": _normalize_severity(item.get("severity")),
                "summary": sanitize_sensitive_text(summary_raw, max_chars=280) if summary_raw else None,
                "spanName": str(item["span_name"]).strip() if item.get("span_name") else None,
            }
        )
    return normalized


def _decode_warning(label: str, raw: ToolErrorJSON) -> tuple[Optional[str], Optional[str]]:
    """Return (error_code, sanitized_warning) for a partial sub-request failure."""
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return "agent_lens_query_failed", f"{label} unavailable."
    if not isinstance(decoded, dict):
        return "agent_lens_query_failed", f"{label} unavailable."
    err_code = str(decoded.get("error") or "agent_lens_query_failed")
    msg = sanitize_sensitive_text(decoded.get("message") or f"{label} unavailable.", max_chars=180)
    return err_code, f"{label}: {msg}"


async def render_agent_lens_insights(
    entity_name: str,
    project_name: str,
    window_days: int = 7,
    start_at: Optional[str] = None,
    end_at: Optional[str] = None,
    agent_name: Optional[str] = None,
    top_n_clusters: int = 12,
    trace_ids: Optional[List[str]] = None,
    *,
    _access_mode: Optional[Literal["read-only", "read-write"]] = None,
) -> CallToolResult:
    """Gather Agent Lens coverage, clustering status, failure/intent breakdowns, and clusters for the Insights App."""
    if not isinstance(entity_name, str) or not entity_name.strip() or len(entity_name.strip()) > 256:
        return _error_result("agent_lens_invalid_request", "entity_name must be a non-empty string (<= 256 chars)")
    if not isinstance(project_name, str) or not project_name.strip() or len(project_name.strip()) > 256:
        return _error_result("agent_lens_invalid_request", "project_name must be a non-empty string (<= 256 chars)")
    entity = entity_name.strip()
    project = project_name.strip()

    if (
        isinstance(top_n_clusters, bool)
        or not isinstance(top_n_clusters, int)
        or not 1 <= top_n_clusters <= MAX_OVERVIEW_CLUSTERS
    ):
        return _error_result(
            "agent_lens_invalid_request",
            f"top_n_clusters must be an integer between 1 and {MAX_OVERVIEW_CLUSTERS}",
        )

    effective_window_days: Optional[int] = None
    if start_at is not None or end_at is not None:
        if not start_at or not end_at:
            return _error_result(
                "agent_lens_invalid_request",
                "Provide both start_at and end_at (RFC 3339), or omit both to use window_days.",
            )
        try:
            window_params = _validated_window(start_at, end_at)
        except ValueError as error:
            return _error_result("agent_lens_invalid_request", str(error))
    else:
        if isinstance(window_days, bool) or window_days not in VALID_WINDOW_DAYS:
            return _error_result("agent_lens_invalid_request", "window_days must be one of 1, 7, 14, or 30")
        effective_window_days = int(window_days)
        now = datetime.now(UTC).replace(microsecond=0)
        resolved_end = now.isoformat().replace("+00:00", "Z")
        resolved_start = (now - timedelta(days=effective_window_days)).isoformat().replace("+00:00", "Z")
        window_params = {"start_at": resolved_start, "end_at": resolved_end}

    cleaned_agent_name: Optional[str] = None
    breakdown_params: Dict[str, Any] = dict(window_params)
    if agent_name is not None:
        if not isinstance(agent_name, str) or not agent_name.strip() or len(agent_name.strip()) > 256:
            return _error_result("agent_lens_invalid_request", "agent_name must be a non-empty string (<= 256 chars)")
        cleaned_agent_name = agent_name.strip()
        try:
            serialized = _serialize_insights_filters([{"field": "agent_name", "values": [cleaned_agent_name]}])
            if serialized is not None:
                breakdown_params["filters"] = serialized
        except ValueError as error:
            return _error_result("agent_lens_invalid_request", str(error))

    cleaned_trace_ids: List[str] = []
    if trace_ids is not None:
        try:
            cleaned_trace_ids = _bounded_list(trace_ids, MAX_OVERVIEW_TRACE_IDS, "trace_ids")
        except ValueError as error:
            return _error_result("agent_lens_invalid_request", str(error))

    track_base = {
        "entity_name": entity,
        "project_name": project,
        "window_days": effective_window_days,
        "filters": bool(cleaned_agent_name),
    }

    tasks: List[Any] = [
        fetch_agent_lens_json(
            LATEST_WEEK_PATH,
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params=track_base,
        ),
        fetch_agent_lens_json(
            CLUSTERING_STATUS_PATH,
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params=track_base,
        ),
        fetch_agent_lens_json(
            _breakdowns_path("failure"),
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params={**track_base, "signature_type": "failure"},
            params=breakdown_params,
        ),
        fetch_agent_lens_json(
            _breakdowns_path("intent"),
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params={**track_base, "signature_type": "intent"},
            params=breakdown_params,
        ),
        fetch_agent_lens_json(
            CLUSTERS_PATH,
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params={**track_base, "signature_type": "failure"},
            params={**window_params, "type": "failure"},
        ),
        fetch_agent_lens_json(
            CLUSTERS_PATH,
            tool_name=INSIGHTS_TOOL_NAME,
            method="GET",
            entity_name=entity,
            project_name=project,
            track_params={**track_base, "signature_type": "intent"},
            params={**window_params, "type": "intent"},
        ),
    ]
    if cleaned_trace_ids:
        tasks.append(
            fetch_agent_lens_json(
                FAILURE_ATTRIBUTIONS_PATH,
                tool_name=INSIGHTS_TOOL_NAME,
                method="POST",
                entity_name=entity,
                project_name=project,
                track_params={**track_base, "trace_count": len(cleaned_trace_ids)},
                body={"trace_ids": cleaned_trace_ids},
            )
        )

    results = await asyncio.gather(*tasks)
    coverage_res, clustering_res, failure_bd_res, intent_bd_res, failure_cl_res, intent_cl_res = results[:6]
    attributions_res = results[6] if len(results) > 6 else None

    # Hard-fail immediately on auth/configuration/forbidden errors or if both primary breakdowns failed.
    fatal_codes = {"agent_lens_not_configured", "auth_required", "agent_lens_forbidden"}
    for sub_res in results:
        if isinstance(sub_res, ToolErrorJSON):
            err_code, _ = _decode_warning("Agent Lens", sub_res)
            if err_code in fatal_codes:
                return _error_from_tool_json(sub_res)

    if isinstance(failure_bd_res, ToolErrorJSON) and isinstance(intent_bd_res, ToolErrorJSON):
        return _error_from_tool_json(failure_bd_res)

    warnings: List[str] = []
    for label, sub_res in (
        ("Coverage", coverage_res),
        ("Clustering status", clustering_res),
        ("Failure breakdowns", failure_bd_res),
        ("Intent breakdowns", intent_bd_res),
        ("Failure clusters", failure_cl_res),
        ("Intent clusters", intent_cl_res),
    ):
        if isinstance(sub_res, ToolErrorJSON):
            _, warning_text = _decode_warning(label, sub_res)
            if warning_text:
                warnings.append(warning_text)

    if isinstance(attributions_res, ToolErrorJSON):
        _, warning_text = _decode_warning("Failure attributions", attributions_res)
        if warning_text:
            warnings.append(warning_text)

    forge_url = public_forge_agent_lens_url(entity, project)
    coverage = _normalize_coverage(coverage_res if not isinstance(coverage_res, ToolErrorJSON) else None)
    clustering_status = _normalize_clustering_status(
        clustering_res if not isinstance(clustering_res, ToolErrorJSON) else None
    )
    failure_section = _normalize_signature_section(
        failure_bd_res if not isinstance(failure_bd_res, ToolErrorJSON) else None,
        failure_cl_res if not isinstance(failure_cl_res, ToolErrorJSON) else None,
        top_n_clusters=top_n_clusters,
    )
    intent_section = _normalize_signature_section(
        intent_bd_res if not isinstance(intent_bd_res, ToolErrorJSON) else None,
        intent_cl_res if not isinstance(intent_cl_res, ToolErrorJSON) else None,
        top_n_clusters=top_n_clusters,
    )
    failure_attributions = _normalize_failure_attributions(
        attributions_res if (attributions_res is not None and not isinstance(attributions_res, ToolErrorJSON)) else None
    )

    structured_payload: Dict[str, Any] = {
        "schema": INSIGHTS_SCHEMA,
        "entityName": entity,
        "projectName": project,
        "forgeUrl": forge_url,
        "accessMode": _access_mode or _current_access_mode(),
        "window": {
            "startAt": str(window_params["start_at"]),
            "endAt": str(window_params["end_at"]),
            "windowDays": effective_window_days,
            "agentName": cleaned_agent_name,
        },
        "coverage": coverage,
        "clusteringStatus": clustering_status,
        "failure": failure_section,
        "intent": intent_section,
        "failureAttributions": failure_attributions,
        "warnings": warnings,
    }

    compact_text_summary = {
        "schema": INSIGHTS_SCHEMA,
        "entity_name": entity,
        "project_name": project,
        "forge_url": forge_url,
        "window": structured_payload["window"],
        "coverage": coverage,
        "clustering_status": clustering_status,
        "failure": {
            "turn_count": failure_section["turnCount"],
            "categories": [
                {
                    "category_id": c["categoryId"],
                    "label": c["label"],
                    "count": c["count"],
                    "share": c["share"],
                    "severity": c["severity"],
                }
                for c in failure_section["categories"][:10]
            ],
            "top_clusters": [
                {
                    "id": cl["id"],
                    "title": cl["title"],
                    "category_id": cl["categoryId"],
                    "count": cl["count"],
                    "share": cl["share"],
                    "severity": cl["severity"],
                }
                for cl in failure_section["clusters"][:8]
            ],
        },
        "intent": {
            "turn_count": intent_section["turnCount"],
            "categories": [
                {
                    "category_id": c["categoryId"],
                    "label": c["label"],
                    "count": c["count"],
                    "share": c["share"],
                }
                for c in intent_section["categories"][:10]
            ],
            "top_clusters": [
                {
                    "id": cl["id"],
                    "title": cl["title"],
                    "category_id": cl["categoryId"],
                    "count": cl["count"],
                    "share": cl["share"],
                }
                for cl in intent_section["clusters"][:8]
            ],
        },
        "failure_attributions": failure_attributions[:10],
        "warnings": warnings,
    }

    return CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(compact_text_summary, ensure_ascii=False, separators=(",", ":")))
        ],
        structuredContent=structured_payload,
        isError=False,
    )


__all__ = [
    "INSIGHTS_RESOURCE_URI",
    "INSIGHTS_SCHEMA",
    "INSIGHTS_TOOL_NAME",
    "RENDER_AGENT_LENS_INSIGHTS_TOOL_DESCRIPTION",
    "register_agent_lens_insights_resource",
    "render_agent_lens_insights",
]
