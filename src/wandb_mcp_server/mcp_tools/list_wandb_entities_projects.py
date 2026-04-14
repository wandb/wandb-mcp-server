from typing import Any

from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution


LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """
Fetch all projects for a specific wandb or weave entity. Useful to use when
the user hasn't specified a project name or queries are failing due to a
missing or incorrect Weights & Biases project name.

If no entity is provided, the tool will fetch all projects for the current user
as well as all the project in the teams they are part of.

<when_to_use>
Call this tool FIRST when the user mentions a W&B entity but has not specified a
project name, or when queries fail due to "project not found" errors. This
resolves entity/project discovery before any data query.
</when_to_use>

<critical_info>

**Important:**

Do not use this tool if the user has not specified a W&BB entity name. Instead ask
the user to provide either their W&B username or W&B team name.
</critical_info>

<debugging_tips>

**Error Handling:**

If this function throws an error, it's likely because the W&B entity name is incorrect.
If this is the case, ask the user to double check the W&B entity name given by the user,
either their personal user or their W&B Team name.

**Expected Project Name Not Found:**

If the user doesn't see the project they're looking for in the list of projects,
ask them to double check the W&B entity name, either their personal W&B username or their
W&B Team name.
</debugging_tips>

Args:
    entity (str): The wandb entity (username or team name)

Returns:
    List[Dict[str, Any]]: List of project dictionaries containing:
        - name: Project name
        - entity: Entity name
        - description: Project description
        - visibility: Project visibility (public/private)
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - tags: List of project tags
"""


logger = get_rich_logger(__name__)


def list_entity_projects(entity: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch all projects for a specific wandb entity. If no entity is provided,
    fetches projects for the current user and their teams.

    Args:
        entity (str, optional): The wandb entity (username or team name). If None,
                               fetches projects for the current user and their teams.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping entity names to lists of project dictionaries.
            Each project dictionary contains:
            - name: Project name
            - entity: Entity name
            - description: Project description
            - visibility: Project visibility (public/private)
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - tags: List of project tags
    """
    from wandb_mcp_server.api_client import get_wandb_api

    api = get_wandb_api()

    viewer = None
    if entity is None:
        viewer = api.viewer
        entities = [viewer.entity] + viewer.teams
    else:
        entities = [entity]

    with track_tool_execution(
        "list_entity_projects",
        viewer,
        {"entity": entity},
    ):
        entities_projects = {}
        for entity in entities:
            projects = api.projects(entity)

            projects_data = []
            for project in projects:
                project_dict = {
                    "name": project.name,
                    "entity": project.entity,
                    "description": getattr(project, "description", None),
                    "visibility": getattr(project, "visibility", None),
                    "created_at": getattr(project, "created_at", None),
                    "updated_at": getattr(project, "updated_at", None),
                    "tags": getattr(project, "tags", []),
                }
                projects_data.append(project_dict)

            entities_projects[entity] = projects_data

        return entities_projects
