"""Narrow viewer identity lookup for tools that need entity membership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import WANDB_BASE_URL

_VIEWER_IDENTITY_QUERY = """
query MCPViewerIdentity {
  viewer {
    entity
    username
    teams {
      edges {
        node {
          name
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class ViewerIdentity:
    """Minimal identity fields used by entity-discovery tools."""

    entity: str | None
    username: str | None
    teams: tuple[str, ...]

    def as_analytics_viewer(self) -> dict[str, str]:
        """Return a small analytics identity payload."""
        payload: dict[str, str] = {}
        if self.username:
            payload["username"] = self.username
        if self.entity:
            payload["entity"] = self.entity
        return payload

    def entity_names(self) -> list[str]:
        """Return the personal entity followed by unique team entities."""
        names: list[str] = []
        if self.entity:
            names.append(self.entity)
        for team in self.teams:
            if team and team not in names:
                names.append(team)
        return names


def get_viewer_identity() -> ViewerIdentity:
    """Fetch only the viewer fields needed for entity discovery.

    This is intentionally isolated so it can move to a future non-GraphQL
    identity endpoint without changing every tool. Avoid ``wandb.Api.viewer``
    here because its generated query includes ``viewer.apiKeys``.
    """
    api_key = WandBApiManager.get_api_key()
    if not api_key:
        raise ValueError("W&B API key is required to resolve viewer identity.")

    response = requests.post(
        f"{WANDB_BASE_URL.rstrip('/')}/graphql",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": _VIEWER_IDENTITY_QUERY, "variables": {}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Viewer identity response was not an object")

    errors = payload.get("errors")
    if errors:
        message = _first_error_message(errors)
        raise ValueError(f"Viewer identity lookup failed: {message}")

    viewer = payload.get("data", {}).get("viewer")
    if not isinstance(viewer, dict):
        raise ValueError("Viewer identity response did not include viewer data")

    teams = _extract_team_names(viewer.get("teams"))
    return ViewerIdentity(
        entity=_optional_str(viewer.get("entity")),
        username=_optional_str(viewer.get("username")),
        teams=tuple(teams),
    )


def _extract_team_names(teams_payload: Any) -> list[str]:
    """Extract team names from a W&B connection payload."""
    if not isinstance(teams_payload, dict):
        return []
    edges = teams_payload.get("edges")
    if not isinstance(edges, list):
        return []

    names: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node = edge.get("node")
        if not isinstance(node, dict):
            continue
        name = _optional_str(node.get("name"))
        if name:
            names.append(name)
    return names


def _first_error_message(errors: Any) -> str:
    """Extract the first GraphQL error message from a response."""
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict) and first.get("message"):
            return str(first["message"])
    return str(errors)


def _optional_str(value: Any) -> str | None:
    """Return a non-empty string or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text or None
