"""List projects for a specific W&B entity."""

import json
from typing import Optional

from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_PROJECTS = 50
MAX_PROJECTS_CEILING = 200
MAX_ENTITIES_WHEN_NONE = 5

LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """List projects for a W&B entity (username or team).

<when_to_use>
Call this when the user has specified an entity but needs to find a project name,
or when queries fail due to "project not found" errors.

Preferred workflow:
1. Call list_entities first to discover available entity names
2. Call this tool with entity="<specific name>" to list that entity's projects
</when_to_use>

<critical_info>
Always pass a specific entity name when possible. If entity is omitted, only
the first 5 entities are listed with up to max_projects each to avoid
overwhelming the response. Use list_entities for full entity discovery.
</critical_info>

Parameters
----------
entity : str, optional
    W&B entity (username or team name). If omitted, lists projects for
    the current user + up to 5 teams (bounded).
max_projects : int, optional
    Maximum projects to return per entity. Default: 50, max: 200.

Returns
-------
JSON with:
  - projects: dict mapping entity name -> list of project objects
  - truncated: whether any entity's project list was capped
"""


def list_entity_projects(
    entity: Optional[str] = None,
    max_projects: int = DEFAULT_MAX_PROJECTS,
) -> str:
    """List projects for a W&B entity with bounded results."""
    from wandb_mcp_server.api_client import get_wandb_api

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
        {"entity": entity, "max_projects": max_projects},
    ) as ctx:
        entities_projects = {}
        any_projects_truncated = False

        for ent in entities:
            try:
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
                logger.warning(f"Failed to list projects for entity {ent}: {e}")
                ctx.mark_error(f"{type(e).__name__}: {e}")
                entities_projects[ent] = [{"error": str(e)}]

        return json.dumps(
            {
                "projects": entities_projects,
                "truncated": any_projects_truncated or entities_truncated,
                "max_projects_per_entity": max_projects,
            }
        )
