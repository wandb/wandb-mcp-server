"""Bounded, application-owned W&B reads without caller-supplied GraphQL."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, NoReturn, Sequence

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
  $includeSweep: Boolean!
  $includeSystemMetrics: Boolean!
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
          sweepName @include(if: $includeSweep)
          user {
            username
            name
          }
          summaryMetrics(keys: $summaryKeys) @include(if: $includeSummary)
          config(keys: $configKeys) @include(if: $includeConfig)
          systemMetrics @include(if: $includeSystemMetrics)
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

PROJECTED_SWEEPS_QUERY = """
query MCPProjectedSweeps(
  $entity: String!
  $project: String!
  $first: Int!
  $after: String
  $includeConfig: Boolean!
) {
  project(name: $project, entityName: $entity) {
    totalSweeps
    sweeps(first: $first, after: $after) {
      edges {
        cursor
        node {
          id
          name
          displayName
          state
          method
          description
          createdAt
          updatedAt
          runCount
          runCountExpected
          config @include(if: $includeConfig)
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

PROJECTED_REPORTS_QUERY = """
query MCPProjectedReports(
  $entity: String!
  $project: String!
  $name: String
  $first: Int!
  $after: String
  $includeSpec: Boolean!
) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: "runs", viewName: $name, first: $first, after: $after) {
      edges {
        cursor
        node {
          id
          name
          displayName
          description
          user {
            username
            email
          }
          createdAt
          updatedAt
          spec @include(if: $includeSpec)
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

METRIC_VALUE_STEPS_QUERY = """
query MCPMetricValueSteps(
  $entity: String!
  $project: String!
  $run: String!
  $metric: String!
  $values: [Float!]!
) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      stepsForMetricValues(metric: $metric, values: $values)
    }
  }
}
"""

REGISTRY_ARTIFACT_VERSIONS_QUERY = """
query MCPRegistryArtifactVersions(
  $organization: String!
  $registryFilter: JSONString!
  $collectionFilter: JSONString!
  $order: String!
  $first: Int!
  $after: String
) {
  organization(name: $organization) {
    orgEntity {
      artifactMemberships(
        projectFilters: $registryFilter
        collectionFilters: $collectionFilter
        order: $order
        first: $first
        after: $after
      ) {
        edges {
          cursor
          node {
            versionIndex
            aliases {
              alias
            }
            artifactCollection {
              name
            }
            artifact {
              id
              state
              description
              size
              fileCount
              createdAt
              updatedAt
              digest
              tags {
                name
              }
            }
          }
        }
        pageInfo {
          endCursor
          hasNextPage
        }
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
    next_cursor: str | None = None


@dataclass(frozen=True)
class ProjectedResourcePage:
    """One bounded page of an application-owned resource projection."""

    items: list[dict[str, Any]]
    total_count: int | None
    has_more: bool
    next_cursor: str | None
    requests: int


@dataclass(frozen=True)
class ProjectFieldPage:
    """One bounded project-field inventory."""

    items: list[dict[str, str]]
    has_more: bool
    requests: int


@dataclass(frozen=True)
class ArtifactVersionPage:
    """One bounded, ordered registry artifact-version scan."""

    items: list[dict[str, Any]]
    has_more: bool
    requests: int


class SelectiveReadUnavailable(RuntimeError):
    """Raised when a backend cannot serve an application-owned read shape."""


_MAX_PROJECTED_PAGE_REQUESTS = 10


def _projected_page_request_limit(target: int, page_size: int) -> int:
    """Bound requests even when a backend returns short or empty pages."""
    expected_requests = math.ceil(max(1, target) / max(1, page_size))
    return min(_MAX_PROJECTED_PAGE_REQUESTS, expected_requests + 1)


def _next_projected_cursor(
    *,
    context: str,
    current: str | None,
    candidate: Any,
    seen: set[str],
) -> str:
    """Return a usable continuation cursor or reject a broken connection."""
    if not isinstance(candidate, str) or not candidate:
        raise SelectiveReadUnavailable(f"{context} reported another page without a continuation cursor")
    if candidate == current or candidate in seen:
        raise SelectiveReadUnavailable(f"{context} repeated a continuation cursor")
    seen.add(candidate)
    return candidate


def _projected_connection_page(
    connection: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], bool]:
    """Validate one fixed-projection connection page before consuming it."""
    edges = connection.get("edges")
    if not isinstance(edges, list) or not all(isinstance(edge, Mapping) for edge in edges):
        raise SelectiveReadUnavailable(f"{context} returned invalid edges")
    page_info = connection.get("pageInfo")
    if not isinstance(page_info, Mapping) or not isinstance(page_info.get("hasNextPage"), bool):
        raise SelectiveReadUnavailable(f"{context} returned invalid pagination metadata")
    return edges, page_info, page_info["hasNextPage"]


def _retained_projected_cursor(
    edge: Mapping[str, Any],
    *,
    context: str,
    current: str | None,
    seen: set[str],
) -> str:
    """Resume immediately after the last edge retained from an oversized page."""
    return _next_projected_cursor(
        context=context,
        current=current,
        candidate=edge.get("cursor"),
        seen=seen,
    )


def _raise_selective_failure(context: str, exc: Exception) -> NoReturn:
    """Preserve actionable upstream errors; wrap only projection compatibility failures."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if status in {401, 403, 404, 429, 503} or "timeout" in name or "timed out" in message:
        raise exc
    raise SelectiveReadUnavailable(f"{context}: {type(exc).__name__}") from exc


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


_MISSING = object()


def _nested_config_value(config: Mapping[str, Any], key: str) -> Any:
    if key in config:
        return _unwrap_config_value(config[key])
    current: Any = config
    for part in key.split("."):
        current = _unwrap_config_value(current)
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return _unwrap_config_value(current)


def _normalize_run_node(
    node: Mapping[str, Any],
    *,
    entity: str,
    project: str,
    summary_keys: Sequence[str],
    config_keys: Sequence[str],
    include_summary: bool,
    include_config: bool,
    include_sweep: bool = False,
    include_system_metrics: bool = False,
) -> dict[str, Any]:
    summary = _parse_json_mapping(node.get("summaryMetrics"))
    config = _parse_json_mapping(node.get("config"))
    item: dict[str, Any] = {
        "id": node.get("name") or node.get("id"),
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
    if include_summary:
        if summary_keys:
            item["summary"] = {key: summary[key] for key in summary_keys if key in summary}
            missing_summary_keys = [key for key in summary_keys if key not in summary]
            if missing_summary_keys:
                item["missing_summary_keys"] = missing_summary_keys
        else:
            item["summary"] = summary
    if include_config:
        selected_config: dict[str, Any] = {}
        missing_config_keys: list[str] = []
        for key in config_keys:
            value = _nested_config_value(config, key)
            if value is _MISSING:
                missing_config_keys.append(key)
            else:
                selected_config[key] = value
        item["config"] = selected_config if config_keys else config
        if missing_config_keys:
            item["missing_config_keys"] = missing_config_keys
    if include_sweep:
        sweep_name = node.get("sweepName")
        item["sweep"] = (
            None
            if not sweep_name
            else {
                "id": sweep_name,
                "name": sweep_name,
                "entity": entity,
                "project": project,
            }
        )
    if include_system_metrics:
        item["system_metrics"] = _parse_json_mapping(node.get("systemMetrics"))
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
    summary_keys: Sequence[str] | None = (),
    config_keys: Sequence[str] | None = (),
    include_summary: bool = False,
    include_config: bool = False,
    include_sweep: bool = False,
    include_system_metrics: bool = False,
    cursor: str | None = None,
) -> ProjectedRunPage:
    """Fetch selected run fields without hydrating full SDK run objects."""
    include_summary = include_summary or bool(summary_keys)
    include_config = include_config or bool(config_keys)
    target = max(1, limit)
    items: list[dict[str, Any]] = []
    page_cursor = cursor
    total_count = 0
    requests = 0
    has_next_page = False
    seen_cursors = {cursor} if cursor else set()
    max_requests = _projected_page_request_limit(target, page_size)

    while len(items) < target and requests < max_requests:
        raise_if_tool_deadline_exceeded()
        variables = {
            "entity": entity,
            "project": project,
            "filters": json.dumps(dict(filters or {}), separators=(",", ":")),
            "order": order,
            "first": min(max(1, page_size), target - len(items)),
            "after": page_cursor,
            "summaryKeys": None if summary_keys is None else list(summary_keys),
            "configKeys": None if config_keys is None else list(config_keys),
            "includeSummary": include_summary,
            "includeConfig": include_config,
            "includeSweep": include_sweep,
            "includeSystemMetrics": include_system_metrics,
        }
        try:
            data = execute_graphql(api, PROJECTED_RUNS_QUERY, variables)
        except Exception as exc:
            _raise_selective_failure("projected run query unavailable", exc)
        requests += 1
        project_payload = _project_payload(data)
        total_count = int(project_payload.get("runCount") or 0)
        connection = project_payload.get("runs")
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("projected run query returned no run connection")
        edges, page_info, backend_has_next = _projected_connection_page(
            connection,
            context="projected run query",
        )
        retained_edge: Mapping[str, Any] | None = None
        consumed_edges = 0
        for edge in edges:
            if len(items) >= target:
                break
            node = edge.get("node")
            if not isinstance(node, Mapping):
                raise SelectiveReadUnavailable("projected run query returned an invalid run edge")
            items.append(
                _normalize_run_node(
                    node,
                    entity=entity,
                    project=project,
                    summary_keys=summary_keys or (),
                    config_keys=config_keys or (),
                    include_summary=include_summary,
                    include_config=include_config,
                    include_sweep=include_sweep,
                    include_system_metrics=include_system_metrics,
                )
            )
            retained_edge = edge
            consumed_edges += 1
        cut_mid_page = consumed_edges < len(edges)
        has_next_page = backend_has_next or cut_mid_page
        if not has_next_page:
            break
        if cut_mid_page:
            assert retained_edge is not None
            page_cursor = _retained_projected_cursor(
                retained_edge,
                context="projected run query",
                current=page_cursor,
                seen=seen_cursors,
            )
        else:
            page_cursor = _next_projected_cursor(
                context="projected run query",
                current=page_cursor,
                candidate=page_info.get("endCursor"),
                seen=seen_cursors,
            )

    return ProjectedRunPage(
        items=items[:limit],
        total_count=total_count,
        has_more=has_next_page,
        next_cursor=page_cursor if has_next_page else None,
        requests=requests,
    )


def fetch_projected_run(
    api: Any,
    *,
    entity: str,
    project: str,
    run_id: str,
    summary_keys: Sequence[str] | None = (),
    config_keys: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Fetch one run with only the requested summary and config keys."""
    variables = {
        "entity": entity,
        "project": project,
        "run": run_id,
        "summaryKeys": None if summary_keys is None else list(summary_keys),
        "configKeys": list(config_keys),
        "includeSummary": summary_keys is None or bool(summary_keys),
        "includeConfig": bool(config_keys),
    }
    try:
        data = execute_graphql(api, PROJECTED_RUN_QUERY, variables)
    except Exception as exc:
        _raise_selective_failure("projected run query unavailable", exc)
    node = _project_payload(data).get("run")
    if not isinstance(node, Mapping):
        return None
    return _normalize_run_node(
        node,
        entity=entity,
        project=project,
        summary_keys=summary_keys or (),
        config_keys=config_keys,
        include_summary=summary_keys is None or bool(summary_keys),
        include_config=bool(config_keys),
    )


def fetch_projected_sweeps(
    api: Any,
    *,
    entity: str,
    project: str,
    limit: int,
    page_size: int,
    include_config: bool = False,
    cursor: str | None = None,
) -> ProjectedResourcePage:
    """Fetch sweep rows without hydrating one SDK Sweep per result."""
    target = max(1, limit)
    items: list[dict[str, Any]] = []
    page_cursor = cursor
    requests = 0
    total_count: int | None = None
    has_next_page = False
    seen_cursors = {cursor} if cursor else set()
    max_requests = _projected_page_request_limit(target, page_size)
    while len(items) < target and requests < max_requests:
        raise_if_tool_deadline_exceeded()
        try:
            data = execute_graphql(
                api,
                PROJECTED_SWEEPS_QUERY,
                {
                    "entity": entity,
                    "project": project,
                    "first": min(max(1, page_size), target - len(items)),
                    "after": page_cursor,
                    "includeConfig": include_config,
                },
            )
        except Exception as exc:
            _raise_selective_failure("projected sweep query unavailable", exc)
        requests += 1
        project_payload = _project_payload(data)
        total_count = int(project_payload.get("totalSweeps") or 0)
        connection = project_payload.get("sweeps")
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("projected sweep query returned no sweep connection")
        edges, page_info, backend_has_next = _projected_connection_page(
            connection,
            context="projected sweep query",
        )
        retained_edge: Mapping[str, Any] | None = None
        consumed_edges = 0
        for edge in edges:
            if len(items) >= target:
                break
            node = edge.get("node")
            if not isinstance(node, Mapping):
                raise SelectiveReadUnavailable("projected sweep query returned an invalid sweep edge")
            sweep_id = node.get("name") or node.get("id")
            item = {
                "id": sweep_id,
                "name": node.get("displayName") or node.get("name"),
                "state": node.get("state"),
                "method": node.get("method"),
                "description": node.get("description"),
                "entity": entity,
                "project": project,
                "expected_run_count": node.get("runCountExpected"),
                "run_count": node.get("runCount"),
                "created_at": node.get("createdAt"),
                "updated_at": node.get("updatedAt"),
            }
            if include_config:
                item["config"] = _parse_json_mapping(node.get("config"))
            items.append(item)
            retained_edge = edge
            consumed_edges += 1
        cut_mid_page = consumed_edges < len(edges)
        has_next_page = backend_has_next or cut_mid_page
        if not has_next_page:
            break
        if cut_mid_page:
            assert retained_edge is not None
            page_cursor = _retained_projected_cursor(
                retained_edge,
                context="projected sweep query",
                current=page_cursor,
                seen=seen_cursors,
            )
        else:
            page_cursor = _next_projected_cursor(
                context="projected sweep query",
                current=page_cursor,
                candidate=page_info.get("endCursor"),
                seen=seen_cursors,
            )
    has_more = has_next_page
    return ProjectedResourcePage(
        items=items[:limit],
        total_count=total_count,
        has_more=has_more,
        next_cursor=page_cursor if has_more else None,
        requests=requests,
    )


def fetch_projected_reports(
    api: Any,
    *,
    entity: str,
    project: str,
    report_name: str | None,
    limit: int,
    page_size: int,
    include_spec: bool = False,
    cursor: str | None = None,
) -> ProjectedResourcePage:
    """Fetch report metadata without downloading report specs by default."""
    target = max(1, limit)
    items: list[dict[str, Any]] = []
    page_cursor = cursor
    requests = 0
    has_next_page = False
    seen_cursors = {cursor} if cursor else set()
    max_requests = _projected_page_request_limit(target, page_size)
    while len(items) < target and requests < max_requests:
        raise_if_tool_deadline_exceeded()
        try:
            data = execute_graphql(
                api,
                PROJECTED_REPORTS_QUERY,
                {
                    "entity": entity,
                    "project": project,
                    "name": report_name,
                    "first": min(max(1, page_size), target - len(items)),
                    "after": page_cursor,
                    "includeSpec": include_spec,
                },
            )
        except Exception as exc:
            _raise_selective_failure("projected report query unavailable", exc)
        requests += 1
        connection = _project_payload(data).get("allViews")
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("projected report query returned no report connection")
        edges, page_info, backend_has_next = _projected_connection_page(
            connection,
            context="projected report query",
        )
        retained_edge: Mapping[str, Any] | None = None
        consumed_edges = 0
        for edge in edges:
            if len(items) >= target:
                break
            node = edge.get("node")
            if not isinstance(node, Mapping):
                raise SelectiveReadUnavailable("projected report query returned an invalid report edge")
            item = {
                "id": node.get("id"),
                "name": node.get("name"),
                "display_name": node.get("displayName"),
                "description": node.get("description"),
                "user": node.get("user"),
                "created_at": node.get("createdAt"),
                "updated_at": node.get("updatedAt"),
            }
            if include_spec:
                item["spec"] = _parse_json_mapping(node.get("spec"))
            items.append(item)
            retained_edge = edge
            consumed_edges += 1
        cut_mid_page = consumed_edges < len(edges)
        has_next_page = backend_has_next or cut_mid_page
        if not has_next_page:
            break
        if cut_mid_page:
            assert retained_edge is not None
            page_cursor = _retained_projected_cursor(
                retained_edge,
                context="projected report query",
                current=page_cursor,
                seen=seen_cursors,
            )
        else:
            page_cursor = _next_projected_cursor(
                context="projected report query",
                current=page_cursor,
                candidate=page_info.get("endCursor"),
                seen=seen_cursors,
            )
    has_more = has_next_page
    total_count = len(items[:limit]) if cursor is None and not has_more else None
    return ProjectedResourcePage(
        items=items[:limit],
        total_count=total_count,
        has_more=has_more,
        next_cursor=page_cursor if has_more else None,
        requests=requests,
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
            _raise_selective_failure("project field index unavailable", exc)
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
        _raise_selective_failure("project count query unavailable", exc)
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
        _raise_selective_failure("artifact inventory unavailable", exc)
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


def fetch_metric_value_steps(
    api: Any,
    *,
    entity: str,
    project: str,
    run_id: str,
    metric: str,
    values: Sequence[float],
) -> list[int | None]:
    """Resolve monotonic metric values to candidate internal history steps.

    The backend resolver returns candidate steps rather than proof that a target
    value was logged. Callers must read and compare the returned history rows.
    """
    try:
        data = execute_graphql(
            api,
            METRIC_VALUE_STEPS_QUERY,
            {
                "entity": entity,
                "project": project,
                "run": run_id,
                "metric": metric,
                "values": [float(value) for value in values],
            },
        )
    except Exception as exc:
        _raise_selective_failure("metric value step lookup unavailable", exc)
    node = _project_payload(data).get("run")
    if not isinstance(node, Mapping):
        raise ValueError("W&B run was not found or is not accessible")
    steps = node.get("stepsForMetricValues")
    if not isinstance(steps, list):
        raise SelectiveReadUnavailable("metric value step lookup returned no step list")
    return [int(step) if isinstance(step, (int, float)) else None for step in steps]


def fetch_registry_artifact_versions(
    api: Any,
    *,
    organization: str,
    registry_name: str,
    collection_name: str,
    order: str,
    scan_limit: int,
    page_size: int = 100,
) -> ArtifactVersionPage:
    """Fetch an ordered, bounded registry-version page.

    W&B 0.28's public Registry versions iterator does not expose its query's
    ordering parameter. This fixed query-only adapter fills only that parity
    gap; filtering remains bounded and explicit in the caller.
    """
    target = max(1, scan_limit)
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    requests = 0
    has_next_page = False
    registry_filter = json.dumps({"name": f"wandb-registry-{registry_name}"}, separators=(",", ":"))
    collection_filter = json.dumps({"name": collection_name}, separators=(",", ":"))

    while len(items) < target:
        raise_if_tool_deadline_exceeded()
        try:
            data = execute_graphql(
                api,
                REGISTRY_ARTIFACT_VERSIONS_QUERY,
                {
                    "organization": organization,
                    "registryFilter": registry_filter,
                    "collectionFilter": collection_filter,
                    "order": order,
                    "first": min(max(1, page_size), target - len(items)),
                    "after": cursor,
                },
            )
        except Exception as exc:
            _raise_selective_failure("ordered registry artifact query unavailable", exc)
        requests += 1
        organization_payload = data.get("organization")
        org_entity = organization_payload.get("orgEntity") if isinstance(organization_payload, Mapping) else None
        connection = org_entity.get("artifactMemberships") if isinstance(org_entity, Mapping) else None
        if not isinstance(connection, Mapping):
            raise SelectiveReadUnavailable("ordered registry artifact query returned no version connection")

        for edge in connection.get("edges") or []:
            membership = edge.get("node") if isinstance(edge, Mapping) else None
            artifact = membership.get("artifact") if isinstance(membership, Mapping) else None
            if not isinstance(membership, Mapping) or not isinstance(artifact, Mapping):
                continue
            version_index = membership.get("versionIndex")
            collection = membership.get("artifactCollection") or {}
            items.append(
                {
                    "version": f"v{version_index}" if version_index is not None else None,
                    "name": collection.get("name"),
                    "aliases": [
                        alias.get("alias")
                        for alias in membership.get("aliases") or []
                        if isinstance(alias, Mapping) and alias.get("alias")
                    ],
                    "tags": [
                        tag.get("name")
                        for tag in artifact.get("tags") or []
                        if isinstance(tag, Mapping) and tag.get("name")
                    ],
                    "state": artifact.get("state"),
                    "size": artifact.get("size"),
                    "file_count": artifact.get("fileCount"),
                    "description": artifact.get("description"),
                    "created_at": artifact.get("createdAt"),
                    "updated_at": artifact.get("updatedAt"),
                    "digest": artifact.get("digest"),
                }
            )
            if len(items) >= target:
                break
        page_info = connection.get("pageInfo") or {}
        has_next_page = bool(page_info.get("hasNextPage"))
        cursor = page_info.get("endCursor")
        if not has_next_page or not cursor:
            break

    return ArtifactVersionPage(
        items=items,
        has_more=has_next_page,
        requests=requests,
    )


__all__ = [
    "ARTIFACT_INVENTORY_QUERY",
    "REGISTRY_ARTIFACT_VERSIONS_QUERY",
    "METRIC_VALUE_STEPS_QUERY",
    "PROJECTED_RUNS_QUERY",
    "PROJECTED_RUN_QUERY",
    "PROJECTED_SWEEPS_QUERY",
    "PROJECTED_REPORTS_QUERY",
    "PROJECT_COUNTS_QUERY",
    "PROJECT_FIELDS_QUERY",
    "ProjectFieldPage",
    "ProjectedRunPage",
    "ProjectedResourcePage",
    "ArtifactVersionPage",
    "SelectiveReadUnavailable",
    "fetch_artifact_inventory",
    "fetch_metric_value_steps",
    "fetch_registry_artifact_versions",
    "fetch_project_counts",
    "fetch_project_fields",
    "fetch_projected_run",
    "fetch_projected_runs",
    "fetch_projected_sweeps",
    "fetch_projected_reports",
]
