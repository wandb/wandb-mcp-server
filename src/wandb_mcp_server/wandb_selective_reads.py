"""Bounded, application-owned W&B reads without caller-supplied GraphQL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.wandb_graphql import execute_graphql


PROJECTED_RUNS_QUERY = """
query MCPProjectedRuns(
  $entity: String!
  $project: String!
  $filters: JSONString
  $order: String
  $first: Int!
  $after: String
  $summaryKeys: [String!]
  $configKeys: [String!]
  $includeSummary: Boolean!
  $includeConfig: Boolean!
) {
  project(name: $project, entityName: $entity) {
    runCount(filters: $filters)
    runs(filters: $filters, order: $order, first: $first, after: $after) {
      edges {
        cursor
        node {
          id
          name
          displayName
          state
          createdAt
          heartbeatAt
          computeSeconds
          historyLineCount
          group
          jobType
          tags
          user {
            username
            name
          }
          summaryMetrics(keys: $summaryKeys) @include(if: $includeSummary)
          config(keys: $configKeys) @include(if: $includeConfig)
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

PROJECTED_RUN_QUERY = """
query MCPProjectedRun(
  $entity: String!
  $project: String!
  $run: String!
  $summaryKeys: [String!]
  $configKeys: [String!]
  $includeSummary: Boolean!
  $includeConfig: Boolean!
) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      id
      name
      displayName
      state
      createdAt
      heartbeatAt
      computeSeconds
      historyLineCount
      group
      jobType
      tags
      user {
        username
        name
      }
      summaryMetrics(keys: $summaryKeys) @include(if: $includeSummary)
      config(keys: $configKeys) @include(if: $includeConfig)
    }
  }
}
"""

PROJECT_FIELDS_QUERY = """
query MCPProjectFields(
  $entity: String!
  $project: String!
  $first: Int!
  $after: String
  $pattern: String
) {
  project(name: $project, entityName: $entity) {
    fields(first: $first, after: $after, pattern: $pattern) {
      edges {
        cursor
        node {
          path
          type
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

PROJECT_COUNTS_QUERY = """
query MCPProjectCounts(
  $entity: String!
  $project: String!
  $all: JSONString
  $finished: JSONString
  $failed: JSONString
  $crashed: JSONString
  $running: JSONString
) {
  project(name: $project, entityName: $entity) {
    total: runCount(filters: $all)
    finished: runCount(filters: $finished)
    failed: runCount(filters: $failed)
    crashed: runCount(filters: $crashed)
    running: runCount(filters: $running)
  }
}
"""

ARTIFACT_INVENTORY_QUERY = """
query MCPArtifactInventory($entity: String!, $project: String!, $first: Int!) {
  project(name: $project, entityName: $entity) {
    artifactTypes(first: $first) {
      edges {
        node {
          id
          name
          artifactCollections(first: 10) {
            totalCount
            edges {
              node {
                id
                name
                description
              }
            }
          }
        }
      }
      pageInfo {
        hasNextPage
      }
    }
  }
}
"""


@dataclass(frozen=True)
class ProjectedRunPage:
    """One bounded projected run collection."""

    items: list[dict[str, Any]]
    total_count: int
    has_more: bool
    requests: int


@dataclass(frozen=True)
class ProjectFieldPage:
    """One bounded project-field inventory."""

    items: list[dict[str, str]]
    has_more: bool
    requests: int


class SelectiveReadUnavailable(RuntimeError):
    """Raised when a backend cannot serve an application-owned read shape."""


def _parse_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {str(key): item for key, item in decoded.items()} if isinstance(decoded, Mapping) else {}


def _unwrap_config_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value and set(value).issubset({"value", "desc"}):
        return value.get("value")
    return value


def _nested_config_value(config: Mapping[str, Any], key: str) -> Any:
    if key in config:
        return _unwrap_config_value(config[key])
    current: Any = config
    for part in key.split("."):
        current = _unwrap_config_value(current)
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return _unwrap_config_value(current)


def _normalize_run_node(
    node: Mapping[str, Any],
    *,
    entity: str,
    project: str,
    summary_keys: Sequence[str],
    config_keys: Sequence[str],
) -> dict[str, Any]:
    summary = _parse_json_mapping(node.get("summaryMetrics"))
    config = _parse_json_mapping(node.get("config"))
    item: dict[str, Any] = {
        "id": node.get("name") or node.get("id"),
        "graphql_id": node.get("id"),
        "display_name": node.get("displayName") or node.get("name"),
        "state": node.get("state"),
        "entity": entity,
        "project": project,
        "created_at": node.get("createdAt"),
        "heartbeat_at": node.get("heartbeatAt"),
        "duration": node.get("computeSeconds"),
        "history_line_count": node.get("historyLineCount"),
        "group": node.get("group"),
        "job_type": node.get("jobType"),
        "tags": node.get("tags") or [],
        "user": node.get("user"),
    }
    if summary_keys:
        item["summary"] = {key: summary.get(key) for key in summary_keys if summary.get(key) is not None}
    if config_keys:
        item["config"] = {key: value for key in config_keys if (value := _nested_config_value(config, key)) is not None}
    return item


def _project_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    project = data.get("project")
    if not isinstance(project, Mapping):
        raise ValueError("W&B project was not found or is not accessible")
    return project


def fetch_projected_runs(
    api: Any,
    *,
    entity: str,
    project: str,
    filters: Mapping[str, Any] | None,
    order: str,
    limit: int,
    page_size: int,
    summary_keys: Sequence[str] = (),
    config_keys: Sequence[str] = (),
) -> ProjectedRunPage:
    """Fetch selected run fields without hydrating full SDK run objects."""
    target = max(1, limit + 1)
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    total_count = 0
    requests = 0
    has_next_page = False

    while len(items) < target:
        raise_if_tool_deadline_exceeded()
        variables = {
            "entity": entity,
            "project": project,
            "filters": json.dumps(dict(filters or {}), separators=(",", ":")),
            "order": order,
            "first": min(max(1, page_size), target - len(items)),
            "after": cursor,
            "summaryKeys": list(summary_keys),
            "configKeys": list(config_keys),
            "includeSummary": bool(summary_keys),
            "includeConfig": bool(config_keys),
        }
        try:
            data = execute_graphql(api, PROJECTED_RUNS_QUERY, variables)
        except Exception as exc:
            raise SelectiveReadUnavailable(f"projected run query unavailable: {type(exc).__name__}") from exc
        requests += 1
        project_payload = _project_payload(data)
        total_count = int(project_payload.get("runCount") or 0)
        connection = project_payload.get("runs")
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("projected run query returned no run connection")
        for edge in connection.get("edges") or []:
            node = edge.get("node") if isinstance(edge, Mapping) else None
            if isinstance(node, Mapping):
                items.append(
                    _normalize_run_node(
                        node,
                        entity=entity,
                        project=project,
                        summary_keys=summary_keys,
                        config_keys=config_keys,
                    )
                )
                if len(items) >= target:
                    break
        page_info = connection.get("pageInfo") or {}
        has_next_page = bool(page_info.get("hasNextPage"))
        cursor = page_info.get("endCursor")
        if not has_next_page or not cursor:
            break

    return ProjectedRunPage(
        items=items[:limit],
        total_count=total_count,
        has_more=len(items) > limit or has_next_page or total_count > limit,
        requests=requests,
    )


def fetch_projected_run(
    api: Any,
    *,
    entity: str,
    project: str,
    run_id: str,
    summary_keys: Sequence[str] = (),
    config_keys: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Fetch one run with only the requested summary and config keys."""
    variables = {
        "entity": entity,
        "project": project,
        "run": run_id,
        "summaryKeys": list(summary_keys),
        "configKeys": list(config_keys),
        "includeSummary": bool(summary_keys),
        "includeConfig": bool(config_keys),
    }
    try:
        data = execute_graphql(api, PROJECTED_RUN_QUERY, variables)
    except Exception as exc:
        raise SelectiveReadUnavailable(f"projected run query unavailable: {type(exc).__name__}") from exc
    node = _project_payload(data).get("run")
    if not isinstance(node, Mapping):
        return None
    return _normalize_run_node(
        node,
        entity=entity,
        project=project,
        summary_keys=summary_keys,
        config_keys=config_keys,
    )


def fetch_project_fields(
    api: Any,
    *,
    entity: str,
    project: str,
    limit: int,
    page_size: int = 200,
    pattern: str | None = None,
) -> ProjectFieldPage:
    """Read the indexed project field vocabulary without hydrating runs."""
    target = max(1, limit + 1)
    items: list[dict[str, str]] = []
    cursor: str | None = None
    requests = 0
    has_next_page = False
    while len(items) < target:
        raise_if_tool_deadline_exceeded()
        try:
            data = execute_graphql(
                api,
                PROJECT_FIELDS_QUERY,
                {
                    "entity": entity,
                    "project": project,
                    "first": min(max(1, page_size), target - len(items)),
                    "after": cursor,
                    "pattern": pattern,
                },
            )
        except Exception as exc:
            raise SelectiveReadUnavailable(f"project field index unavailable: {type(exc).__name__}") from exc
        requests += 1
        connection = _project_payload(data).get("fields")
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("project field query returned no field connection")
        for edge in connection.get("edges") or []:
            node = edge.get("node") if isinstance(edge, Mapping) else None
            if isinstance(node, Mapping) and node.get("path"):
                items.append({"path": str(node["path"]), "type": str(node.get("type") or "unknown")})
                if len(items) >= target:
                    break
        page_info = connection.get("pageInfo") or {}
        has_next_page = bool(page_info.get("hasNextPage"))
        cursor = page_info.get("endCursor")
        if not has_next_page or not cursor:
            break
    return ProjectFieldPage(
        items=items[:limit],
        has_more=len(items) > limit or has_next_page,
        requests=requests,
    )


def fetch_project_counts(api: Any, *, entity: str, project: str) -> dict[str, int]:
    """Fetch exact total and state counts in one server-side query."""
    filters = {
        "all": {},
        "finished": {"state": "finished"},
        "failed": {"state": "failed"},
        "crashed": {"state": "crashed"},
        "running": {"state": "running"},
    }
    variables = {
        "entity": entity,
        "project": project,
        **{name: json.dumps(value, separators=(",", ":")) for name, value in filters.items()},
    }
    try:
        project_payload = _project_payload(execute_graphql(api, PROJECT_COUNTS_QUERY, variables))
    except Exception as exc:
        raise SelectiveReadUnavailable(f"project count query unavailable: {type(exc).__name__}") from exc
    return {name: int(project_payload.get("total" if name == "all" else name) or 0) for name in filters}


def fetch_artifact_inventory(
    api: Any,
    *,
    entity: str,
    project: str,
    type_limit: int = 20,
) -> dict[str, Any]:
    """Return a compact, bounded artifact type and collection inventory."""
    try:
        project_payload = _project_payload(
            execute_graphql(
                api,
                ARTIFACT_INVENTORY_QUERY,
                {"entity": entity, "project": project, "first": max(1, min(type_limit, 50))},
            )
        )
    except Exception as exc:
        raise SelectiveReadUnavailable(f"artifact inventory unavailable: {type(exc).__name__}") from exc
    connection = project_payload.get("artifactTypes")
    if not isinstance(connection, Mapping):
        raise SelectiveReadUnavailable("artifact inventory returned no artifact type connection")
    rows: list[dict[str, Any]] = []
    for edge in connection.get("edges") or []:
        node = edge.get("node") if isinstance(edge, Mapping) else None
        if not isinstance(node, Mapping):
            continue
        collections = node.get("artifactCollections") or {}
        rows.append(
            {
                "type": node.get("name"),
                "collection_count": int(collections.get("totalCount") or 0),
                "sample_collections": [
                    {
                        "name": collection_node.get("name"),
                        "description": collection_node.get("description"),
                    }
                    for collection_edge in collections.get("edges") or []
                    if isinstance(collection_edge, Mapping)
                    and isinstance((collection_node := collection_edge.get("node")), Mapping)
                ],
            }
        )
    return {
        "types": rows,
        "returned_count": len(rows),
        "has_more": bool((connection.get("pageInfo") or {}).get("hasNextPage")),
    }


__all__ = [
    "ARTIFACT_INVENTORY_QUERY",
    "PROJECTED_RUNS_QUERY",
    "PROJECTED_RUN_QUERY",
    "PROJECT_COUNTS_QUERY",
    "PROJECT_FIELDS_QUERY",
    "ProjectFieldPage",
    "ProjectedRunPage",
    "SelectiveReadUnavailable",
    "fetch_artifact_inventory",
    "fetch_project_counts",
    "fetch_project_fields",
    "fetch_projected_run",
    "fetch_projected_runs",
]
