"""List W&B registries and registry collections.

Provides two read-only tools for discovering registries and their
collections via the ``wandb.Api`` public interface.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import MCP_MAX_WANDB_QUERY_ITEMS
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.registry_support import (
    RegistryInputError,
    bounded_sdk_page,
    fit_registry_response,
    nullable_string,
    registry_error_result,
    require_registry,
    resolve_registry_organization,
    validate_optional_identifier,
    validate_registry_filter,
)
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_ITEMS = 50


# ---------------------------------------------------------------------------
# Tool 1 – list_registries
# ---------------------------------------------------------------------------

LIST_REGISTRIES_TOOL_DESCRIPTION = """List W&B registries for an organization.

Returns registry names, descriptions, visibility, and allowed artifact types.

<when_to_use>
Call this tool FIRST when the user asks about model registries, registered
models, or registered datasets. Use the output to identify which registry to
drill into with list_registry_collections_tool.

Typical workflow:
1. list_registries_tool → discover available registries
2. list_registry_collections_tool → browse collections in a registry
3. list_artifact_versions_tool → see versions of a specific collection
4. get_artifact_details_tool → inspect a single version
</when_to_use>

<critical_info>
Requires the user's API key to have access to the organization. If no
organization is specified, one accessible organization is selected. Accounts
with more than one accessible organization receive `organization_required`
with a bounded candidate list.
Supports MongoDB-style filters on name, description, etc.
(e.g., {"name": {"$regex": "model.*"}}).
</critical_info>

Parameters
----------
organization : str, optional
    W&B organization name. Omit to resolve one accessible organization.
filter : dict, optional
    MongoDB-style filter dict (e.g., {"name": {"$regex": "model.*"}}).
max_items : int, optional
    Maximum registries to return. Default: 50; workload-profile limits apply.

Returns
-------
JSON with:
  - registries: list of registry objects with name, description, visibility, etc.
  - returned_count / total_count: returned and exact totals when known
  - has_more / project_exhaustive: explicit pagination scope
"""


def list_registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List W&B registries for an organization."""

    with track_tool_execution(
        "list_registries",
        None,
        {
            "has_organization": organization is not None,
            "has_filter": filter is not None,
            "max_items": max_items,
        },
    ) as ctx:
        try:
            _validate_request(
                organization=organization,
                registry_name=None,
                filter=filter,
                max_items=max_items,
            )
            max_items = min(max_items, MCP_MAX_WANDB_QUERY_ITEMS)
            api = WandBApiManager.get_api()
            resolved_organization = resolve_registry_organization(api, organization)
            kwargs: Dict[str, Any] = {
                "organization": resolved_organization,
                "per_page": min(max_items + 1, 100),
            }
            if filter is not None:
                kwargs["filter"] = filter

            page, has_more = bounded_sdk_page(api.registries(**kwargs), max_items)
            registries: List[Dict[str, Any]] = []
            for reg in page:
                artifact_types, artifact_types_truncated = _bounded_string_values(
                    getattr(reg, "artifact_types", []),
                )
                registries.append(
                    {
                        "name": _bounded_text(getattr(reg, "name", None)),
                        "full_name": _bounded_text(getattr(reg, "full_name", None)),
                        "organization": _bounded_text(getattr(reg, "organization", None)),
                        "entity": _bounded_text(getattr(reg, "entity", None)),
                        "description": _bounded_text(getattr(reg, "description", None)),
                        "visibility": _bounded_text(getattr(reg, "visibility", None)),
                        "artifact_types": artifact_types,
                        "artifact_types_truncated": artifact_types_truncated,
                        "created_at": nullable_string(getattr(reg, "created_at", None)),
                        "updated_at": nullable_string(getattr(reg, "updated_at", None)),
                    }
                )

            total_count = None if has_more else len(registries)
            result = fit_registry_response(
                {
                    "items": registries,
                    "registries": registries,
                    "returned_count": len(registries),
                    "total_count": total_count,
                    "has_more": has_more,
                    "limit": max_items,
                    "project_exhaustive": not has_more,
                    "count": len(registries),
                    "truncated": has_more,
                },
                aliases=("registries",),
            )
            return json.dumps(result, allow_nan=False)

        except Exception as e:
            logger.error("Registry listing failed (%s)", type(e).__name__)
            ctx.mark_error(type(e).__name__)
            return json.dumps(registry_error_result(e))


# ---------------------------------------------------------------------------
# Tool 2 – list_registry_collections
# ---------------------------------------------------------------------------

LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION = """List collections within a W&B registry.

Collections are named groups of artifact versions (e.g., a model with v0..v12).

<when_to_use>
Call this when the user wants to see what models or datasets exist within a
specific registry. Use the collection name to then call
list_artifact_versions_tool or get_artifact_details_tool.
</when_to_use>

<critical_info>
The registry_name is the short name (e.g., "model"), NOT the full name with
the "wandb-registry-" prefix.
Supports MongoDB-style filters on name, description, tags, etc.
</critical_info>

Parameters
----------
registry_name : str
    The registry short name (e.g., "model", "dataset", "my-registry").
organization : str, optional
    W&B organization name. Omit to resolve one accessible organization.
filter : dict, optional
    MongoDB-style filter (e.g., {"tag": "production"}).
max_items : int, optional
    Maximum collections to return. Default: 50; workload-profile limits apply.

Returns
-------
JSON with:
  - registry: the queried registry name
  - collections: list of collection objects with name, type, tags, and metadata.
    Collection-wide aliases are intentionally not loaded; use
    list_artifact_versions_tool for version aliases.
  - returned_count / total_count: returned and exact totals when known
  - has_more / project_exhaustive: explicit pagination scope
"""


def list_registry_collections(
    registry_name: str,
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List collections within a W&B registry."""

    with track_tool_execution(
        "list_registry_collections",
        None,
        {
            "has_organization": organization is not None,
            "has_filter": filter is not None,
            "max_items": max_items,
        },
    ) as ctx:
        try:
            _validate_request(
                organization=organization,
                registry_name=registry_name,
                filter=filter,
                max_items=max_items,
            )
            max_items = min(max_items, MCP_MAX_WANDB_QUERY_ITEMS)
            api = WandBApiManager.get_api()
            resolved_organization = resolve_registry_organization(api, organization)
            registry_search = api.registries(
                organization=resolved_organization,
                filter={"name": registry_name},
                per_page=1,
            )
            require_registry(registry_search)

            coll_kwargs: Dict[str, Any] = {"per_page": min(max_items + 1, 100)}
            if filter is not None:
                coll_kwargs["filter"] = filter

            collection_iter = registry_search.collections(**coll_kwargs)
            page, has_more = bounded_sdk_page(collection_iter, max_items)
            last_response = getattr(collection_iter, "last_response", None)
            exact_total = getattr(last_response, "total_count", None)
            if not isinstance(exact_total, int) or isinstance(exact_total, bool):
                exact_total = None
            collections: List[Dict[str, Any]] = []
            for coll in page:
                tags, tags_truncated = _bounded_string_values(getattr(coll, "tags", []))
                collections.append(
                    {
                        "name": _bounded_text(getattr(coll, "name", None)),
                        "type": _bounded_text(getattr(coll, "type", None)),
                        "description": _bounded_text(getattr(coll, "description", None)),
                        "tags": tags,
                        "tags_truncated": tags_truncated,
                        "aliases": None,
                        "aliases_loaded": False,
                        "created_at": nullable_string(getattr(coll, "created_at", None)),
                        "updated_at": nullable_string(getattr(coll, "updated_at", None)),
                        "is_sequence": coll.is_sequence() if hasattr(coll, "is_sequence") else None,
                    }
                )

            result = fit_registry_response(
                {
                    "registry": registry_name,
                    "items": collections,
                    "collections": collections,
                    "returned_count": len(collections),
                    "total_count": exact_total if exact_total is not None else (None if has_more else len(collections)),
                    "has_more": has_more,
                    "limit": max_items,
                    "project_exhaustive": not has_more,
                    "count": len(collections),
                    "truncated": has_more,
                },
                aliases=("collections",),
            )
            return json.dumps(result, allow_nan=False)

        except Exception as e:
            logger.error("Registry collection listing failed (%s)", type(e).__name__)
            ctx.mark_error(type(e).__name__)
            return json.dumps(registry_error_result(e))


def _validate_request(
    *,
    organization: str | None,
    registry_name: str | None,
    filter: Mapping[str, Any] | None,
    max_items: int,
) -> None:
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        raise RegistryInputError("max_items must be a positive integer")
    validate_optional_identifier("organization", organization)
    validate_optional_identifier("registry_name", registry_name)
    validate_registry_filter(filter)


def _bounded_text(value: Any, *, max_chars: int = 16_000) -> str | None:
    rendered = nullable_string(value)
    if rendered is None or len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}…"


def _bounded_string_values(values: Any, *, limit: int = 100) -> tuple[list[str], bool]:
    if values is None:
        return [], False
    result: list[str] = []
    iterator = iter(values)
    for _ in range(limit + 1):
        raise_if_tool_deadline_exceeded()
        try:
            value = next(iterator)
        except StopIteration:
            return result, False
        if len(result) < limit:
            name = value if isinstance(value, str) else getattr(value, "name", value)
            rendered = _bounded_text(name, max_chars=512)
            if rendered is not None:
                result.append(rendered)
    return result, True
