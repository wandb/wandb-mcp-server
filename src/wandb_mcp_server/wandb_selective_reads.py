"""Bounded, application-owned W&B reads without caller-supplied GraphQL."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, NoReturn, Sequence

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge, execute_graphql

MAX_SAFE_HISTORY_STEP = (1 << 63) - 2


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

HISTORY_RUN_SNAPSHOT_QUERY = """
query MCPHistoryRunSnapshot($entity: String!, $project: String!, $run: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      name
      displayName
      state
      historyLineCount
      historyTail
    }
  }
}
"""

PROJECT_METADATA_QUERY = """
query MCPProjectMetadata($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    id
    name
    entityName
    description
    runCount
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

SAMPLED_HISTORY_SERIES_QUERY = """
query MCPSampledHistorySeries(
  $entity: String!
  $project: String!
  $run: String!
  $specs: [JSONString!]!
) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      sampledHistory(specs: $specs)
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


@dataclass(frozen=True)
class SampledHistoryBatch:
    """Bounded sampled history series returned by fixed read requests."""

    series: list[list[dict[str, Any]]]
    keys: list[str]
    # Raw rows returned by Core before fork-segment normalization.
    rows_received: int
    # Rows retained for the pre-merge bounded working set.
    rows_retained: int
    bytes_retained: int
    values_truncated: int
    non_finite_counts: dict[str, int]
    samples_per_series: int
    requests: int


class SelectiveReadUnavailable(RuntimeError):
    """Raised when a backend cannot serve an application-owned read shape."""


class ProjectedReportCursorError(ValueError):
    """Raised before transport when a filtered report cursor is invalid."""


_MAX_PROJECTED_PAGE_REQUESTS = 10
_PROJECTED_REPORT_CURSOR_PREFIX = "mcp-report-v1:"
_MAX_SAMPLED_HISTORY_INPUT_ROWS = 10_000
_MAX_SAMPLED_HISTORY_RAW_ROWS = 100_000
_MAX_SAMPLED_HISTORY_SPECS_PER_REQUEST = 8
_MAX_SAMPLED_HISTORY_ROW_BYTES = 64 * 1024
_MAX_SAMPLED_HISTORY_RETAINED_BYTES = 8 * 1024 * 1024
_MAX_SAMPLED_HISTORY_VALUE_NODES = 500
_MAX_SAMPLED_HISTORY_VALUE_DEPTH = 6
_MAX_SAMPLED_HISTORY_COLLECTION_ITEMS = 100
_TRUNCATED_HISTORY_VALUE = "<history value truncated>"
_CORE_NON_FINITE_TOKENS = frozenset({"NaN", "Infinity", "-Infinity"})


@dataclass
class _SampledValueBudget:
    remaining_bytes: int = 60 * 1024
    remaining_nodes: int = _MAX_SAMPLED_HISTORY_VALUE_NODES
    truncated: bool = False


def _bounded_sampled_history_value(
    value: Any,
    budget: _SampledValueBudget,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Copy one decoded JSON value without exceeding a fixed working budget."""
    budget.remaining_nodes -= 1
    if budget.remaining_nodes < 0 or budget.remaining_bytes <= 0 or depth >= _MAX_SAMPLED_HISTORY_VALUE_DEPTH:
        budget.truncated = True
        return _TRUNCATED_HISTORY_VALUE
    if value is None or isinstance(value, (bool, int)):
        budget.remaining_bytes -= len(str(value))
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            budget.truncated = True
            return None
        budget.remaining_bytes -= len(repr(value))
        return value
    if isinstance(value, str):
        candidate = value[: min(len(value), budget.remaining_bytes)]
        encoded = candidate.encode("utf-8")
        if len(encoded) > budget.remaining_bytes:
            candidate = encoded[: budget.remaining_bytes].decode("utf-8", errors="ignore")
            encoded = candidate.encode("utf-8")
        budget.remaining_bytes -= len(encoded)
        if len(candidate) != len(value):
            budget.truncated = True
        return candidate or (_TRUNCATED_HISTORY_VALUE if value else "")

    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        budget.truncated = True
        return "<cyclic history value>"
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for index, (key, child) in enumerate(value.items()):
                if index >= _MAX_SAMPLED_HISTORY_COLLECTION_ITEMS:
                    result["_truncated"] = "additional mapping entries omitted"
                    budget.truncated = True
                    break
                safe_key = _bounded_sampled_history_value(str(key), budget, depth=depth + 1, seen=active)
                result[str(safe_key)] = _bounded_sampled_history_value(child, budget, depth=depth + 1, seen=active)
                if budget.remaining_bytes <= 0:
                    budget.truncated = True
                    break
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result = []
            for index, child in enumerate(value):
                if index >= _MAX_SAMPLED_HISTORY_COLLECTION_ITEMS:
                    result.append("<additional list entries omitted>")
                    budget.truncated = True
                    break
                result.append(_bounded_sampled_history_value(child, budget, depth=depth + 1, seen=active))
                if budget.remaining_bytes <= 0:
                    budget.truncated = True
                    break
            return result
        budget.truncated = True
        return _bounded_sampled_history_value(str(value), budget, depth=depth + 1, seen=active)
    finally:
        active.discard(identity)


def _bounded_sampled_history_row(
    row: Mapping[str, Any],
    *,
    requested_key: str,
    x_axis: str,
) -> tuple[dict[str, Any], int, bool]:
    """Retain only selected fields and cap the copied row by bytes and nodes."""
    budget = _SampledValueBudget()
    bounded: dict[str, Any] = {}
    for key in dict.fromkeys((x_axis, "_step", requested_key)):
        if key in row:
            bounded[key] = _bounded_sampled_history_value(row[key], budget)
    encoded_size = len(
        json.dumps(
            bounded,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if encoded_size > _MAX_SAMPLED_HISTORY_ROW_BYTES:
        bounded = {key: value for key, value in bounded.items() if key in {x_axis, "_step"}}
        bounded[requested_key] = _TRUNCATED_HISTORY_VALUE
        budget.truncated = True
        encoded_size = len(json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return bounded, encoded_size, budget.truncated


def _is_non_finite_history_value(value: Any) -> bool:
    """Recognize Python and Core 0.82's JSON-safe non-finite encodings."""
    return (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, str) and value in _CORE_NON_FINITE_TOKENS
    )


def is_projected_report_cursor(cursor: str) -> bool:
    """Return whether a cursor belongs to a filtered projected report read."""
    return cursor.startswith(_PROJECTED_REPORT_CURSOR_PREFIX)


def _projected_report_cursor_fingerprint(
    *,
    entity: str,
    project: str,
    report_name: str,
    include_spec: bool,
) -> str:
    """Bind a filtered continuation to the report query that created it."""
    payload = json.dumps(
        {
            "entity": entity,
            "project": project,
            "report_name": report_name,
            "include_spec": include_spec,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _encode_projected_report_cursor(*, mode: str, after: str, fingerprint: str) -> str:
    """Wrap a backend cursor with its filtered report connection mode."""
    payload = json.dumps(
        {"after": after, "mode": mode, "query": fingerprint},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return _PROJECTED_REPORT_CURSOR_PREFIX + encoded


def _decode_projected_report_cursor(cursor: str, *, fingerprint: str) -> tuple[str, str]:
    """Decode and validate one mode-bound filtered report cursor."""
    if not is_projected_report_cursor(cursor):
        raise ProjectedReportCursorError("cursor is not a valid filtered report continuation")
    encoded = cursor.removeprefix(_PROJECTED_REPORT_CURSOR_PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.b64decode((encoded + padding).encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(raw)
    except (UnicodeEncodeError, ValueError, json.JSONDecodeError):
        raise ProjectedReportCursorError("cursor is not a valid filtered report continuation") from None
    if not isinstance(payload, Mapping) or set(payload) != {"after", "mode", "query"}:
        raise ProjectedReportCursorError("cursor is not a valid filtered report continuation")
    mode = payload.get("mode")
    after = payload.get("after")
    query = payload.get("query")
    if mode not in {"internal", "display"} or not isinstance(after, str) or not after:
        raise ProjectedReportCursorError("cursor is not a valid filtered report continuation")
    if query != fingerprint:
        raise ProjectedReportCursorError("cursor does not match this filtered report query")
    return mode, after


def validate_projected_report_cursor(
    cursor: str,
    *,
    entity: str,
    project: str,
    report_name: str,
    include_spec: bool,
) -> None:
    """Validate a filtered report cursor without constructing a W&B client."""
    fingerprint = _projected_report_cursor_fingerprint(
        entity=entity,
        project=project,
        report_name=report_name,
        include_spec=include_spec,
    )
    _decode_projected_report_cursor(cursor, fingerprint=fingerprint)


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
    if isinstance(exc, GraphQLResponseTooLarge):
        raise exc
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


def fetch_project_metadata(api: Any, *, entity: str, project: str) -> dict[str, Any]:
    """Fetch stable project metadata in one fixed, query-only request."""
    raise_if_tool_deadline_exceeded()
    try:
        data = execute_graphql(
            api,
            PROJECT_METADATA_QUERY,
            {"entity": entity, "project": project},
        )
    except Exception as exc:
        _raise_selective_failure("project metadata query unavailable", exc)

    project_payload = _project_payload(data)
    project_id = project_payload.get("id")
    project_name = project_payload.get("name")
    run_count = project_payload.get("runCount")
    if not isinstance(project_id, str) or not project_id:
        raise SelectiveReadUnavailable("project metadata query returned no project id")
    if not isinstance(project_name, str) or not project_name:
        raise SelectiveReadUnavailable("project metadata query returned no project name")
    if isinstance(run_count, bool) or not isinstance(run_count, int) or run_count < 0:
        raise SelectiveReadUnavailable("project metadata query returned an invalid run count")

    return {
        "id": project_id,
        "name": project_name,
        "entity": project_payload.get("entityName") or entity,
        "description": project_payload.get("description"),
        "run_count": run_count,
    }


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


def fetch_history_run_snapshot(
    api: Any,
    *,
    entity: str,
    project: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Read only the identity and upper step needed to construct history."""
    raise_if_tool_deadline_exceeded()
    try:
        data = execute_graphql(
            api,
            HISTORY_RUN_SNAPSHOT_QUERY,
            {"entity": entity, "project": project, "run": run_id},
        )
    except Exception as exc:
        _raise_selective_failure("history run snapshot query unavailable", exc)
    node = _project_payload(data).get("run")
    if not isinstance(node, Mapping):
        return None

    history_line_count = node.get("historyLineCount")
    if isinstance(history_line_count, bool) or not isinstance(history_line_count, int) or history_line_count < 0:
        raise SelectiveReadUnavailable("history run snapshot returned an invalid history line count")
    history_tail = node.get("historyTail")
    if not isinstance(history_tail, str):
        raise SelectiveReadUnavailable("history run snapshot returned no bounded history tail")
    try:
        encoded_rows = json.loads(history_tail)
        if not isinstance(encoded_rows, list) or len(encoded_rows) > 8:
            raise ValueError
        if not encoded_rows:
            if history_line_count == 0:
                return {
                    "id": node.get("name") or run_id,
                    "display_name": node.get("displayName") or node.get("name") or run_id,
                    "state": node.get("state"),
                    "history_line_count": history_line_count,
                    "last_step": -1,
                }
            raise ValueError
        if not isinstance(encoded_rows[-1], str):
            raise ValueError
        # W&B's resume contract treats the final encoded row as the tail. Core
        # 0.82 returns one row; accepting a short legacy list keeps older
        # compatible servers correct without processing an unbounded history.
        tail_row = json.loads(encoded_rows[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        raise SelectiveReadUnavailable("history run snapshot returned an invalid bounded history tail") from None
    raw_step = tail_row.get("_step") if isinstance(tail_row, Mapping) else None
    if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0 or raw_step > MAX_SAFE_HISTORY_STEP:
        raise SelectiveReadUnavailable("history run snapshot returned an invalid upper step")
    # Core returns step zero for a genuinely empty history.  The line count
    # distinguishes that case from a run whose only logged row is step zero.
    last_step = -1 if history_line_count == 0 and raw_step == 0 else raw_step
    return {
        "id": node.get("name") or run_id,
        "display_name": node.get("displayName") or node.get("name") or run_id,
        "state": node.get("state"),
        "history_line_count": history_line_count,
        "last_step": last_step,
    }


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
    """Fetch report metadata without downloading report specs by default.

    W&B's ``viewName`` argument addresses the report's generated internal
    name, while users normally know its display title. A filtered first page
    therefore tries the internal name and, only when that lookup is empty,
    scans one bounded metadata page for an exact display-title match.
    Mode-bound cursors keep each continuation on the connection that produced
    it, including when multiple reports share an internal name.
    """
    target = max(1, limit)
    if report_name is not None:
        return _fetch_projected_reports_by_name(
            api,
            entity=entity,
            project=project,
            report_name=report_name,
            target=target,
            page_size=page_size,
            include_spec=include_spec,
            cursor=cursor,
        )

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
                    "name": None,
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
            items.append(_projected_report_item(edge, include_spec=include_spec))
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


def _fetch_projected_report_page(
    api: Any,
    *,
    entity: str,
    project: str,
    internal_name: str | None,
    first: int,
    cursor: str | None,
    include_spec: bool,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], bool]:
    """Fetch and validate one bounded report connection page."""
    raise_if_tool_deadline_exceeded()
    try:
        data = execute_graphql(
            api,
            PROJECTED_REPORTS_QUERY,
            {
                "entity": entity,
                "project": project,
                "name": internal_name,
                "first": max(1, first),
                "after": cursor,
                "includeSpec": include_spec,
            },
        )
    except Exception as exc:
        _raise_selective_failure("projected report query unavailable", exc)
    connection = _project_payload(data).get("allViews")
    if not isinstance(connection, Mapping):
        raise SelectiveReadUnavailable("projected report query returned no report connection")
    return _projected_connection_page(connection, context="projected report query")


def _projected_report_item(edge: Mapping[str, Any], *, include_spec: bool) -> dict[str, Any]:
    """Convert one validated report edge without hydrating omitted specs."""
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
    return item


def _fetch_projected_reports_by_name(
    api: Any,
    *,
    entity: str,
    project: str,
    report_name: str,
    target: int,
    page_size: int,
    include_spec: bool,
    cursor: str | None,
) -> ProjectedResourcePage:
    """Resolve an exact internal report name or exact display title."""
    requests = 0
    request_size = min(max(1, page_size), target)
    fingerprint = _projected_report_cursor_fingerprint(
        entity=entity,
        project=project,
        report_name=report_name,
        include_spec=include_spec,
    )
    mode: str | None = None
    backend_cursor: str | None = None
    if cursor is not None:
        mode, backend_cursor = _decode_projected_report_cursor(cursor, fingerprint=fingerprint)

    if mode in {None, "internal"}:
        edges, page_info, backend_has_next = _fetch_projected_report_page(
            api,
            entity=entity,
            project=project,
            internal_name=report_name,
            first=request_size,
            cursor=backend_cursor,
            include_spec=include_spec,
        )
        requests += 1
        if edges or mode == "internal":
            retained_edges = edges[:target]
            items = [_projected_report_item(edge, include_spec=include_spec) for edge in retained_edges]
            cut_mid_page = len(retained_edges) < len(edges)
            has_more = backend_has_next or cut_mid_page
            next_backend_cursor = None
            if has_more:
                seen_cursors = {backend_cursor} if backend_cursor else set()
                if cut_mid_page:
                    next_backend_cursor = _retained_projected_cursor(
                        retained_edges[-1],
                        context="projected report query",
                        current=backend_cursor,
                        seen=seen_cursors,
                    )
                else:
                    next_backend_cursor = _next_projected_cursor(
                        context="projected report query",
                        current=backend_cursor,
                        candidate=page_info.get("endCursor"),
                        seen=seen_cursors,
                    )
            return ProjectedResourcePage(
                items=items,
                total_count=len(items) if cursor is None and not has_more else None,
                has_more=has_more,
                next_cursor=(
                    _encode_projected_report_cursor(
                        mode="internal",
                        after=next_backend_cursor,
                        fingerprint=fingerprint,
                    )
                    if next_backend_cursor is not None
                    else None
                ),
                requests=requests,
            )

    edges, page_info, backend_has_next = _fetch_projected_report_page(
        api,
        entity=entity,
        project=project,
        internal_name=None,
        first=request_size,
        cursor=backend_cursor,
        include_spec=include_spec,
    )
    requests += 1

    items: list[dict[str, Any]] = []
    last_scanned_edge: Mapping[str, Any] | None = None
    scanned_edges = 0
    for edge in edges:
        if len(items) >= target:
            break
        item = _projected_report_item(edge, include_spec=include_spec)
        last_scanned_edge = edge
        scanned_edges += 1
        if item.get("display_name") == report_name:
            items.append(item)

    cut_mid_page = scanned_edges < len(edges)
    has_more = backend_has_next or cut_mid_page
    next_backend_cursor = None
    if has_more:
        seen_cursors = {backend_cursor} if backend_cursor else set()
        if cut_mid_page:
            assert last_scanned_edge is not None
            next_backend_cursor = _retained_projected_cursor(
                last_scanned_edge,
                context="projected report display-title scan",
                current=backend_cursor,
                seen=seen_cursors,
            )
        else:
            next_backend_cursor = _next_projected_cursor(
                context="projected report display-title scan",
                current=backend_cursor,
                candidate=page_info.get("endCursor"),
                seen=seen_cursors,
            )
    return ProjectedResourcePage(
        items=items,
        total_count=len(items) if cursor is None and not has_more else None,
        has_more=has_more,
        next_cursor=(
            _encode_projected_report_cursor(
                mode="display",
                after=next_backend_cursor,
                fingerprint=fingerprint,
            )
            if next_backend_cursor is not None
            else None
        ),
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


def fetch_sampled_history_series(
    api: Any,
    *,
    entity: str,
    project: str,
    run_id: str,
    keys: Sequence[str],
    x_axis: str,
    samples: int,
    min_step: int | None = None,
    max_step: int | None = None,
) -> SampledHistoryBatch:
    """Fetch one independently sampled series per key in bounded read requests.

    W&B's public ``Run.history(keys=[...])`` samples only rows where every
    requested key co-occurs. Real training loops commonly log those keys on
    different cadences, so this fixed projection sends one bounded spec per key
    and lets the caller outer-join the returned series. Specs are chunked to
    bound the independent history-store reads executed by W&B Core.
    """
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        raise ValueError("keys must contain at least one history key")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    for name, value in (("min_step", min_step), ("max_step", max_step)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an integer")
    if min_step is not None and max_step is not None and max_step < min_step:
        raise ValueError("max_step must be greater than or equal to min_step")

    samples_per_series = min(
        samples,
        max(1, _MAX_SAMPLED_HISTORY_INPUT_ROWS // len(unique_keys)),
    )
    specs_by_key: list[tuple[str, str]] = []
    for key in unique_keys:
        # Server 0.82 applies AND semantics within a spec, so each metric gets
        # an independent spec. A custom x-axis intentionally remains in that
        # spec: a value cannot be plotted against an axis logged on another row.
        spec_keys = list(dict.fromkeys([x_axis, "_step", key]))
        spec: dict[str, Any] = {"keys": spec_keys, "samples": samples_per_series}
        if min_step is not None:
            spec["minStep"] = min_step
        if max_step is not None:
            # sampledHistory treats maxStep as inclusive, matching the MCP
            # interface (unlike public Run.scan_history).
            spec["maxStep"] = max_step
        specs_by_key.append((key, json.dumps(spec, separators=(",", ":"))))

    series: list[list[dict[str, Any]]] = []
    rows_received = 0
    rows_retained = 0
    bytes_retained = 0
    values_truncated = 0
    non_finite_counts = {key: 0 for key in unique_keys}
    requests = 0
    for offset in range(0, len(specs_by_key), _MAX_SAMPLED_HISTORY_SPECS_PER_REQUEST):
        batch = specs_by_key[offset : offset + _MAX_SAMPLED_HISTORY_SPECS_PER_REQUEST]
        raise_if_tool_deadline_exceeded()
        try:
            data = execute_graphql(
                api,
                SAMPLED_HISTORY_SERIES_QUERY,
                {
                    "entity": entity,
                    "project": project,
                    "run": run_id,
                    "specs": [spec for _, spec in batch],
                },
            )
        except Exception as exc:
            _raise_selective_failure("sampled history series query unavailable", exc)
        requests += 1

        run_node = _project_payload(data).get("run")
        if not isinstance(run_node, Mapping):
            raise ValueError("W&B run was not found or is not accessible")
        payload = run_node.get("sampledHistory")
        if not isinstance(payload, list) or len(payload) != len(batch):
            raise SelectiveReadUnavailable("sampled history series query returned an invalid series count")

        for (requested_key, _), item in zip(batch, payload, strict=True):
            if not isinstance(item, list) or not all(isinstance(row, Mapping) for row in item):
                raise SelectiveReadUnavailable("sampled history series query returned an invalid series")
            rows_received += len(item)
            if rows_received > _MAX_SAMPLED_HISTORY_RAW_ROWS:
                raise SelectiveReadUnavailable("sampled history series query exceeded its raw row safety limit")

            non_finite_counts[requested_key] += sum(
                _is_non_finite_history_value(row.get(requested_key)) for row in item
            )
            valid_count = sum(
                1
                for row in item
                if row.get(requested_key) is not None and not _is_non_finite_history_value(row.get(requested_key))
            )
            retained_count = min(valid_count, samples_per_series)
            if 0 < retained_count < valid_count:
                last = valid_count - 1
                retained_positions = (
                    {round(position * last / (retained_count - 1)) for position in range(retained_count)}
                    if retained_count > 1
                    else {0}
                )
            else:
                retained_positions = None

            filtered_rows: list[dict[str, Any]] = []
            valid_index = -1
            for row in item:
                requested_value = row.get(requested_key)
                if requested_value is None or _is_non_finite_history_value(requested_value):
                    continue
                valid_index += 1
                if retained_positions is not None and valid_index not in retained_positions:
                    continue
                bounded_row, row_bytes, row_was_truncated = _bounded_sampled_history_row(
                    row,
                    requested_key=requested_key,
                    x_axis=x_axis,
                )
                if bytes_retained + row_bytes > _MAX_SAMPLED_HISTORY_RETAINED_BYTES:
                    raise SelectiveReadUnavailable("sampled history series query exceeded its retained byte limit")
                bytes_retained += row_bytes
                values_truncated += int(row_was_truncated)
                filtered_rows.append(bounded_row)
            # Core 0.82 samples fork segments independently and concatenates
            # them, so a valid forked-run response may exceed the requested
            # per-spec sample hint. Selection happens before copying values, so
            # even a valid fork over-return cannot inflate the working set.
            rows_retained += len(filtered_rows)
            if rows_retained > _MAX_SAMPLED_HISTORY_INPUT_ROWS:
                raise SelectiveReadUnavailable("sampled history series query exceeded its retained row limit")
            series.append(filtered_rows)

    return SampledHistoryBatch(
        series=series,
        keys=unique_keys,
        rows_received=rows_received,
        rows_retained=rows_retained,
        bytes_retained=bytes_retained,
        values_truncated=values_truncated,
        non_finite_counts=non_finite_counts,
        samples_per_series=samples_per_series,
        requests=requests,
    )


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
    "SAMPLED_HISTORY_SERIES_QUERY",
    "PROJECT_METADATA_QUERY",
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
    "SampledHistoryBatch",
    "ProjectedReportCursorError",
    "SelectiveReadUnavailable",
    "fetch_artifact_inventory",
    "fetch_metric_value_steps",
    "fetch_sampled_history_series",
    "fetch_registry_artifact_versions",
    "fetch_project_counts",
    "fetch_project_metadata",
    "fetch_project_fields",
    "fetch_projected_run",
    "fetch_projected_runs",
    "fetch_projected_sweeps",
    "fetch_projected_reports",
    "is_projected_report_cursor",
    "validate_projected_report_cursor",
]
