"""Bounded read projection that does not hydrate editable SDK Registry objects."""

from collections.abc import Iterator
import json
from types import SimpleNamespace
from typing import Any

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.registry_support import RegistryMalformedResponse
from wandb_mcp_server.wandb_graphql import execute_graphql

REGISTRIES_QUERY = """
query MCPFetchRegistries($organization: String!, $filters: JSONString, $cursor: String, $perPage: Int!) {
  organization(name: $organization) {
    orgEntity {
      projects(filters: $filters, after: $cursor, first: $perPage) {
        pageInfo { endCursor hasNextPage }
        edges { node {
          name description access createdAt updatedAt
          entity { name organization { name } }
          artifactTypes(first: 101, includeAll: true) { edges { node { name } } }
        } }
      }
    }
  }
}
"""
_PREFIX = "wandb-registry-"


def _prefix_filter_names(value: Any, *, in_name: bool = False) -> Any:
    """Keep registry name selection independent of private SDK utilities.

    Tool-boundary filter validation already bounds depth and JSON size. Only
    literal names are prefixed; regular expressions are passed through.
    """
    if isinstance(value, str):
        return _PREFIX + value if in_name and not value.startswith(_PREFIX) else value
    if isinstance(value, dict):
        return {
            key: child if key == "$regex" else _prefix_filter_names(child, in_name=in_name or key == "name")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_prefix_filter_names(child, in_name=in_name) for child in value]
    return value


def registry_records(api: Any, *, organization: str, filter: dict | None = None, per_page: int = 51) -> Iterator[Any]:
    """Include only registry projects; unknown visibility is explicitly unknown.

    Unlike editable SDK Registry models, a metadata read must not discard the
    entire list when the server adds an access mode. We do not reinterpret an
    unknown access value as organization-wide or restricted permissions.
    """
    prefix = {"name": {"$regex": "^wandb-registry-"}}
    filters = {"$and": [prefix, _prefix_filter_names(filter)]} if filter else prefix
    variables = {"organization": organization, "filters": json.dumps(filters), "cursor": None, "perPage": per_page}
    seen = set()
    for _ in range(8):
        raise_if_tool_deadline_exceeded()
        data = execute_graphql(api, REGISTRIES_QUERY, variables)
        try:
            connection = data["organization"]["orgEntity"]["projects"]
            edges, page = connection["edges"], connection["pageInfo"]
            if not isinstance(edges, list) or len(edges) > per_page or type(page["hasNextPage"]) is not bool:
                raise ValueError
            for edge in edges:
                row = edge["node"]
                name, entity, access = row["name"], row["entity"], row["access"]
                if not isinstance(name, str) or not name.startswith(_PREFIX) or not name.removeprefix(_PREFIX):
                    raise ValueError
                if (access is not None and not isinstance(access, str)) or not isinstance(entity["name"], str):
                    raise ValueError
                types = row["artifactTypes"]["edges"]
                if not isinstance(types, list) or len(types) > 101:
                    raise ValueError
                types = [entry["node"]["name"] for entry in types]
                if not all(isinstance(value, str) for value in types):
                    raise ValueError
                yield SimpleNamespace(
                    name=name.removeprefix(_PREFIX),
                    full_name=name,
                    entity=entity["name"],
                    organization=entity["organization"]["name"],
                    description=row.get("description"),
                    visibility={"PRIVATE": "organization", "RESTRICTED": "restricted"}.get(access, "unknown"),
                    artifact_types=types,
                    created_at=row.get("createdAt"),
                    updated_at=row.get("updatedAt"),
                )
            if not page["hasNextPage"]:
                return
            cursor = page.get("endCursor")
            if not isinstance(cursor, str) or not cursor or len(cursor) > 4096 or cursor in seen:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryMalformedResponse("registry metadata page malformed") from exc
        seen.add(cursor)
        variables["cursor"] = cursor
    raise RegistryMalformedResponse("registry pagination exceeded its request limit")
