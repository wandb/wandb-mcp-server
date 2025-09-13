from typing import Any

import wandb


LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """
Fetch all projects for a specific wandb or weave entity. Useful to use when 
the user hasn't specified a project name or queries are failing due to a 
missing or incorrect Weights & Biases project name.

If no entity is provided, the tool will fetch all projects for the current user 
as well as all the project in the teams they are part of.

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
    # Initialize wandb API
    api = wandb.Api()

    # Merge entity and teams into a single list
    if entity is None:
        viewer = api.viewer
        entities = [viewer.entity] + viewer.teams
    else:
        entities = [entity]

    # Get all projects for the entity

    entities_projects = {}
    for entity in entities:
        projects = api.projects(entity)

        # Convert projects to a list of dictionaries
        projects_data = []
        for project in projects:
            project_dict = {
                "name": project.name,
                "entity": project.entity,
                "description": getattr(project, "description", None),
                "visibility": getattr(project, "visibility", None),
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "tags": project.tags,
            }
            projects_data.append(project_dict)

        entities_projects[entity] = projects_data

    return entities_projects
