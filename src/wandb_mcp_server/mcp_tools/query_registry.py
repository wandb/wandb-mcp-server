"""List W&B registries and registry collections.

Provides two read-only tools for discovering registries and their
collections via the ``wandb.Api`` public interface.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_ITEMS = 50
MAX_ITEMS_CEILING = 200


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
organization is specified, uses the default organization for the
authenticated user.
Supports MongoDB-style filters on name, description, etc.
(e.g., {"name": {"$regex": "model.*"}}).
</critical_info>

Parameters
----------
organization : str, optional
    W&B organization name. Omit to use the authenticated user's default org.
filter : dict, optional
    MongoDB-style filter dict (e.g., {"name": {"$regex": "model.*"}}).
max_items : int, optional
    Maximum registries to return. Default: 50, max: 200.

Returns
-------
JSON with:
  - registries: list of registry objects with name, description, visibility, etc.
  - count: number of registries returned
  - truncated: whether more registries exist beyond max_items
"""


def list_registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List W&B registries for an organization."""

    try:
        api = WandBApiManager.get_api()
        log_tool_call(
            "list_registries",
            api.viewer,
            {"organization": organization, "filter": filter, "max_items": max_items},
        )
    except Exception:
        logger.debug("analytics emit failed", exc_info=True)

    max_items = min(max_items, MAX_ITEMS_CEILING)

    try:
        api = WandBApiManager.get_api()
        kwargs: Dict[str, Any] = {"per_page": min(max_items, 100)}
        if organization is not None:
            kwargs["organization"] = organization
        if filter is not None:
            kwargs["filter"] = filter

        registries: List[Dict[str, Any]] = []
        truncated = False
        for reg in api.registries(**kwargs):
            if len(registries) >= max_items:
                truncated = True
                break
            registries.append(
                {
                    "name": getattr(reg, "name", None),
                    "full_name": getattr(reg, "full_name", None),
                    "organization": getattr(reg, "organization", None),
                    "entity": getattr(reg, "entity", None),
                    "description": getattr(reg, "description", None),
                    "visibility": getattr(reg, "visibility", None),
                    "artifact_types": list(getattr(reg, "artifact_types", [])),
                    "created_at": str(getattr(reg, "created_at", "")),
                    "updated_at": str(getattr(reg, "updated_at", "")),
                }
            )

        return json.dumps({"registries": registries, "count": len(registries), "truncated": truncated})

    except Exception as e:
        logger.error(f"Error in list_registries: {e}", exc_info=True)
        return json.dumps({"error": "api_error", "message": str(e)[:500]})


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
    W&B organization name. Omit to use default.
filter : dict, optional
    MongoDB-style filter (e.g., {"tag": "production"}).
max_items : int, optional
    Maximum collections to return. Default: 50, max: 200.

Returns
-------
JSON with:
  - registry: the queried registry name
  - collections: list of collection objects with name, type, tags, aliases, etc.
  - count: number of collections returned
  - truncated: whether more collections exist beyond max_items
"""


def list_registry_collections(
    registry_name: str,
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List collections within a W&B registry."""

    try:
        api = WandBApiManager.get_api()
        log_tool_call(
            "list_registry_collections",
            api.viewer,
            {
                "registry_name": registry_name,
                "organization": organization,
                "filter": filter,
                "max_items": max_items,
            },
        )
    except Exception:
        logger.debug("analytics emit failed", exc_info=True)

    max_items = min(max_items, MAX_ITEMS_CEILING)

    try:
        api = WandBApiManager.get_api()
        reg_kwargs: Dict[str, Any] = {}
        if organization is not None:
            reg_kwargs["organization"] = organization
        registry = api.registry(registry_name, **reg_kwargs)

        coll_kwargs: Dict[str, Any] = {"per_page": min(max_items, 100)}
        if filter is not None:
            coll_kwargs["filter"] = filter

        collections: List[Dict[str, Any]] = []
        truncated = False
        for coll in registry.collections(**coll_kwargs):
            if len(collections) >= max_items:
                truncated = True
                break
            collections.append(
                {
                    "name": getattr(coll, "name", None),
                    "type": getattr(coll, "type", None),
                    "description": getattr(coll, "description", None),
                    "tags": getattr(coll, "tags", []),
                    "aliases": getattr(coll, "aliases", []),
                    "created_at": str(getattr(coll, "created_at", "")),
                    "updated_at": str(getattr(coll, "updated_at", "")),
                    "is_sequence": coll.is_sequence() if hasattr(coll, "is_sequence") else None,
                }
            )

        return json.dumps(
            {
                "registry": registry_name,
                "collections": collections,
                "count": len(collections),
                "truncated": truncated,
            }
        )

    except Exception as e:
        logger.error(f"Error in list_registry_collections: {e}", exc_info=True)
        return json.dumps({"error": "api_error", "message": str(e)[:500]})
