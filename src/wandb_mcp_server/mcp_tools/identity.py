"""Narrow viewer identity lookup for tools that need entity context."""

from __future__ import annotations

from dataclasses import dataclass

from wandb_mcp_server.api_client import get_wandb_api


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
    """Fetch safe identity fields through the W&B Public API.

    Avoid ``wandb.Api.viewer`` because its generated query includes
    ``viewer.apiKeys``. ``default_entity`` is narrower and goes through the SDK
    transport instead of manually forwarding the API key to GraphQL.
    """
    api = get_wandb_api()
    entity = api.default_entity
    if not entity:
        raise ValueError("Viewer identity response did not include default entity")

    return ViewerIdentity(
        entity=entity,
        username=entity,
        teams=(),
    )
