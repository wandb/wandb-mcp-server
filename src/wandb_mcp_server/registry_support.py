"""Shared organization resolution and error handling for Registry tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wandb_mcp_server.wandb_graphql import fetch_registry_organization_info

_MAX_ORGANIZATION_CANDIDATES = 20


class OrganizationRequired(ValueError):
    """Raised when an entity belongs to more than one organization."""

    def __init__(self, candidates: tuple[str, ...], *, truncated: bool = False) -> None:
        super().__init__("Specify organization because this entity can access more than one organization.")
        self.candidates = candidates
        self.truncated = truncated


class OrganizationResolutionFailed(ValueError):
    """Raised when an organization cannot be resolved for registry access."""


def resolve_registry_organization(api: Any, organization_or_entity: str | None) -> str:
    """Resolve either an organization display name or an entity to one organization."""
    if organization_or_entity is not None:
        if not isinstance(organization_or_entity, str) or not organization_or_entity.strip():
            raise ValueError("organization must be a non-empty string when provided")
        requested = organization_or_entity.strip()
    else:
        settings = getattr(api, "settings", None)
        configured_org = settings.get("organization") if isinstance(settings, Mapping) else None
        if isinstance(configured_org, str) and configured_org.strip():
            return configured_org.strip()

        configured_entity = settings.get("entity") if isinstance(settings, Mapping) else None
        requested = (
            configured_entity.strip() if isinstance(configured_entity, str) and configured_entity.strip() else None
        )

    data = fetch_registry_organization_info(api, requested)
    if not isinstance(data, Mapping):
        raise OrganizationResolutionFailed("W&B returned malformed organization data.")

    direct_org = data.get("organization")
    direct_name = _organization_name(direct_org)
    if direct_name:
        return direct_name

    if requested is not None:
        entity = data.get("entity")
        if not isinstance(entity, Mapping):
            raise OrganizationResolutionFailed(f"No W&B organization or entity named {requested!r} is accessible.")
        team_org = _organization_name(entity.get("organization"))
        if team_org:
            return team_org
        user = entity.get("user")
        organizations = user.get("organizations") if isinstance(user, Mapping) else None
    else:
        viewer = data.get("viewer")
        if not isinstance(viewer, Mapping):
            raise OrganizationResolutionFailed("No W&B organization is available for the authenticated user.")
        organizations = viewer.get("organizations")
        default_entity = viewer.get("entity")
        if isinstance(default_entity, str) and isinstance(organizations, list):
            matching = [
                item
                for item in organizations
                if isinstance(item, Mapping)
                and isinstance(item.get("orgEntity"), Mapping)
                and item["orgEntity"].get("name") == default_entity
            ]
            if len(matching) == 1:
                organizations = matching

    candidates = _organization_candidates(organizations)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        truncated = len(candidates) > _MAX_ORGANIZATION_CANDIDATES
        raise OrganizationRequired(candidates[:_MAX_ORGANIZATION_CANDIDATES], truncated=truncated)
    raise OrganizationResolutionFailed("No W&B organization is available for registry access.")


def registry_error_result(exc: Exception) -> dict[str, Any]:
    """Map registry failures to stable, actionable tool errors."""
    if isinstance(exc, OrganizationRequired):
        return {
            "error": "organization_required",
            "message": str(exc),
            "organization_candidates": list(exc.candidates),
            "candidates_truncated": exc.truncated,
        }
    if isinstance(exc, OrganizationResolutionFailed):
        return {"error": "organization_resolution_failed", "message": str(exc)}

    message = str(exc)
    lowered = message.lower()
    if "authentication" in lowered or "api key" in lowered or "unauth" in lowered:
        return {"error": "authentication_failed", "message": "W&B authentication failed."}
    if "not found" in lowered or "could not find" in lowered:
        return {
            "error": "resource_not_found",
            "message": "The requested W&B registry resource was not found.",
        }
    if isinstance(exc, TypeError) and "nonetype" in lowered:
        return {
            "error": "malformed_response",
            "message": "W&B returned incomplete registry data. Verify the organization or entity and try again.",
        }
    return {"error": "api_error", "message": message[:500]}


def require_registry(registries: Any) -> Any:
    """Require one exact registry match without using the SDK's eager registry lookup."""
    iterator = iter(registries)
    try:
        return next(iterator)
    except StopIteration:
        raise LookupError("Registry not found") from None


def _organization_name(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    org_entity = value.get("orgEntity")
    if not isinstance(name, str) or not name.strip() or not isinstance(org_entity, Mapping):
        return None
    entity_name = org_entity.get("name")
    if not isinstance(entity_name, str) or not entity_name.strip():
        return None
    return name.strip()


def _organization_candidates(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    names = {_organization_name(value) for value in values}
    return tuple(sorted(name for name in names if name))
