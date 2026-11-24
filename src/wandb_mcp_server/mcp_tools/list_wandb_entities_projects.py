"""List projects for a specific W&B entity."""

import json
from typing import Optional

from wandb_mcp_server.api_client import raise_for_wandb_server_busy
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.tool_results import tool_error_json
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_selective_reads import SelectiveReadUnavailable, fetch_entity_projects

logger = get_rich_logger(__name__)

DEFAULT_MAX_PROJECTS = 50
MAX_PROJECTS_CEILING = 200
MAX_ENTITIES_WHEN_NONE = 5
_ORDER_MAP = {
    "-updated_at": "-updatedAt",
    "+updated_at": "+updatedAt",
    "updated_at": "-updatedAt",
    "-updatedAt": "-updatedAt",
    "+updatedAt": "+updatedAt",
    "-last_active": "-lastActive",
    "last_active": "-lastActive",
    "-lastActive": "-lastActive",
    "-created_at": "-createdAt",
    "+created_at": "+createdAt",
    "created_at": "-createdAt",
    "-createdAt": "-createdAt",
    "+createdAt": "+createdAt",
}

LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """List projects for a W&B entity (username or team), ordered by recency.

<when_to_use>
Call this when the user has specified an entity but needs to find a project name,
or when queries fail due to "project not found" errors.

Preferred workflow:
1. Call list_entities first to discover available entity names
2. Call this tool with entity="<specific name>" to list that entity's most recently active projects
</when_to_use>

<critical_info>
Always pass a specific entity name when possible. Uses a bounded server-ordered
GraphQL projection (`order="-updated_at"` by default, with `run_count`) so large
teams with thousands of projects return in one round-trip instead of paging the
entire entity. If entity is omitted, only the first 5 entities are listed.
</critical_info>

Parameters
----------
entity : str, optional
    W&B entity (username or team name). If omitted, lists projects for
    the current user + up to 5 teams (bounded).
max_projects : int, optional
    Maximum projects to return per entity. Default: 50, max: 200.
order : str, optional
    Sort order: "-updated_at" (default), "-last_active", "-created_at", "+created_at", or "+updated_at".

Returns
-------
JSON with:
  - projects: dict mapping entity name -> list of project objects (including run_count when available)
  - truncated: whether any entity's project list was capped
"""


def list_entity_projects(
    entity: Optional[str] = None,
    max_projects: int = DEFAULT_MAX_PROJECTS,
    order: str = "-updated_at",
) -> str:
    """List projects for a W&B entity with bounded, server-ordered results."""
    from wandb_mcp_server.api_client import get_wandb_api

    gql_order = _ORDER_MAP.get(order)
    if gql_order is None:
        return tool_error_json(
            {
                "error": "invalid_input",
                "message": (
                    "order must be one of '-updated_at', '-last_active', '-created_at', '+created_at', or '+updated_at'"
                ),
            }
        )

    api = get_wandb_api()
    max_projects = min(max(1, max_projects), MAX_PROJECTS_CEILING)

    viewer = None
    if entity is None:
        viewer = api.viewer
        all_entities = [viewer.entity] + getattr(viewer, "teams", [])
        entities = all_entities[:MAX_ENTITIES_WHEN_NONE]
        entities_truncated = len(all_entities) > MAX_ENTITIES_WHEN_NONE
    else:
        entities = [entity]
        entities_truncated = False

    with track_tool_execution(
        "list_entity_projects",
        viewer or entity,
        {"entity": entity, "max_projects": max_projects, "order": gql_order},
    ) as ctx:
        entities_projects = {}
        any_projects_truncated = False

        for ent in entities:
            try:
                try:
                    projects_data, has_more = fetch_entity_projects(
                        api,
                        entity=ent,
                        limit=max_projects,
                        order=gql_order,
                    )
                    if has_more:
                        any_projects_truncated = True
                except SelectiveReadUnavailable as exc:
                    raise_for_wandb_server_busy(exc)
                    projects = api.projects(ent)
                    projects_data = []
                    for project in projects:
                        if len(projects_data) >= max_projects:
                            any_projects_truncated = True
                            break
                        projects_data.append(
                            {
                                "name": project.name,
                                "entity": project.entity,
                                "description": getattr(project, "description", None),
                                "visibility": getattr(project, "visibility", None),
                                "created_at": str(getattr(project, "created_at", "")),
                                "updated_at": str(getattr(project, "updated_at", "")),
                                "tags": getattr(project, "tags", []),
                            }
                        )
                entities_projects[ent] = projects_data
            except Exception as e:
                raise_for_wandb_server_busy(e)
                logger.warning("Failed to list projects for one entity (%s)", type(e).__name__)
                ctx.mark_error(type(e).__name__)
                entities_projects[ent] = [
                    {
                        "error": "project_query_failed",
                        "message": "W&B project listing failed for this entity.",
                    }
                ]

        return json.dumps(
            {
                "projects": entities_projects,
                "truncated": any_projects_truncated or entities_truncated,
                "max_projects_per_entity": max_projects,
            }
        )
