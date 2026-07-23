"""List W&B registries and registry collections.

Provides two read-only tools for discovering registries and their
collections via the ``wandb.Api`` public interface.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.config import MCP_MAX_WANDB_QUERY_ITEMS
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
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
        {"organization": organization, "filter": filter, "max_items": max_items},
    ) as ctx:
        try:
            if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
                return json.dumps({"error": "invalid_input", "message": "max_items must be a positive integer"})
            max_items = min(max_items, MCP_MAX_WANDB_QUERY_ITEMS)
            api = WandBApiManager.get_api()
            kwargs: Dict[str, Any] = {"per_page": min(max_items + 1, 100)}
            if organization is not None:
                kwargs["organization"] = organization
            if filter is not None:
                kwargs["filter"] = filter

            page, has_more = _bounded_page(api.registries(**kwargs), max_items)
            registries: List[Dict[str, Any]] = []
            for reg in page:
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

            total_count = None if has_more else len(registries)
            return json.dumps(
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
                }
            )

        except Exception as e:
            logger.error(f"Error in list_registries: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
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
    Maximum collections to return. Default: 50; workload-profile limits apply.

Returns
-------
JSON with:
  - registry: the queried registry name
  - collections: list of collection objects with name, type, tags, aliases, etc.
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
            "registry_name": registry_name,
            "organization": organization,
            "filter": filter,
            "max_items": max_items,
        },
    ) as ctx:
        try:
            if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
                return json.dumps({"error": "invalid_input", "message": "max_items must be a positive integer"})
            max_items = min(max_items, MCP_MAX_WANDB_QUERY_ITEMS)
            api = WandBApiManager.get_api()
            reg_kwargs: Dict[str, Any] = {}
            if organization is not None:
                reg_kwargs["organization"] = organization
            registry = api.registry(registry_name, **reg_kwargs)

            coll_kwargs: Dict[str, Any] = {"per_page": min(max_items + 1, 100)}
            if filter is not None:
                coll_kwargs["filter"] = filter

            collection_iter = registry.collections(**coll_kwargs)
            exact_total: int | None = None
            try:
                exact_total = len(collection_iter)
            except (TypeError, NotImplementedError):
                pass
            page, has_more = _bounded_page(collection_iter, max_items)
            collections: List[Dict[str, Any]] = []
            for coll in page:
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
                    "items": collections,
                    "collections": collections,
                    "returned_count": len(collections),
                    "total_count": exact_total if exact_total is not None else (None if has_more else len(collections)),
                    "has_more": has_more,
                    "limit": max_items,
                    "project_exhaustive": not has_more,
                    "count": len(collections),
                    "truncated": has_more,
                }
            )

        except Exception as e:
            logger.error(f"Error in list_registry_collections: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


def _bounded_page(values: Any, limit: int) -> tuple[list[Any], bool]:
    """Consume at most limit-plus-one values from a lazy SDK collection."""
    rows: list[Any] = []
    iterator = iter(values)
    for _ in range(limit + 1):
        raise_if_tool_deadline_exceeded()
        try:
            rows.append(next(iterator))
        except StopIteration:
            break
    return rows[:limit], len(rows) > limit
