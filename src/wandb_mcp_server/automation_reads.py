"""Fixed, bounded Automation projections, including actions not hydrated by SDK."""

from collections.abc import Iterator
import json
from typing import Any

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.wandb_graphql import execute_app_graphql

_FIELDS = """
id name enabled description createdAt updatedAt
scope {
  __typename
  ... on Project { id name }
  ... on Entity { id name }
  ... on ArtifactSequence { id name }
  ... on ArtifactPortfolio { id name }
}
event: triggeringCondition {
  __typename
  ... on FilterEventTriggeringCondition { eventType filter }
}
action: triggeredAction {
  __typename
  ... on NotificationTriggeredAction { title message severity integration { __typename id } }
  ... on GenericWebhookTriggeredAction { requestPayload integration { __typename id } }
  ... on NoOpTriggeredAction { noOp }
  ... on ARIATriggeredAction { prompt }
}
"""
_CONNECTION = (
    """
triggers(first: $first, after: $after, filters: $filters) {
  pageInfo { endCursor hasNextPage }
  edges { node { %s } }
}
"""
    % _FIELDS
)
ENTITY_QUERY = (
    """
query MCPAutomationEntity($entity: String!, $first: Int!, $after: String, $filters: JSONString) {
  entity(name: $entity) { %s }
}
"""
    % _CONNECTION
)
VIEWER_QUERY = (
    """
query MCPAutomationViewer($first: Int!, $after: String, $filters: JSONString) {
  %s
}
"""
    % _CONNECTION
)
MAX_PAGES = 8


def automation_records(*, entity: str | None, name: str | None, per_page: int) -> Iterator[dict[str, Any]]:
    """Traverse the trigger connection, not SDK project pages with empty triggers.

    Backend authorization can leave empty pages. Advance only on a new cursor,
    stop within eight requests, and never turn an incomplete traversal into an
    apparently exhaustive empty list. Callers stop at their item limit + 1.
    """
    for value in (entity, name):
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value.encode()) > 512):
            raise ValueError("Automation selectors must be bounded non-empty strings.")
    variables: dict[str, Any] = {
        "first": min(max(per_page, 1), 100),
        "after": None,
        "filters": json.dumps({"name": name} if name is not None else {}),
    }
    if entity is not None:
        variables["entity"] = entity
    seen = set()
    for _ in range(MAX_PAGES):
        raise_if_tool_deadline_exceeded()
        data = execute_app_graphql(ENTITY_QUERY if entity is not None else VIEWER_QUERY, variables)
        scope = data.get("entity") if entity is not None else data
        connection = scope.get("triggers") if isinstance(scope, dict) else None
        if not isinstance(connection, dict):
            raise ValueError("Automation connection missing.")
        edges, page = connection.get("edges"), connection.get("pageInfo")
        if (
            not isinstance(edges, list)
            or len(edges) > variables["first"]
            or not isinstance(page, dict)
            or type(page.get("hasNextPage")) is not bool
        ):
            raise ValueError("Automation page malformed.")
        for edge in edges:
            if not isinstance(edge, dict) or not isinstance(edge.get("node"), dict):
                raise ValueError("Automation record malformed.")
            if name is not None and edge["node"].get("name") != name:
                raise ValueError("Automation exact-name filter was not honored.")
            yield edge["node"]
        if not page["hasNextPage"]:
            return
        cursor = page.get("endCursor")
        if not isinstance(cursor, str) or not cursor or len(cursor) > 4096 or cursor in seen:
            raise ValueError("Automation pagination cursor missing or repeated.")
        seen.add(cursor)
        variables["after"] = cursor
    raise ValueError("Automation pagination exceeded its request limit.")
