"""Lightweight entity discovery tool.

Returns the authenticated user's personal entity and team memberships
without enumerating projects. Agents use this to discover which entity
name to pass to other tools (query_traces, list_entity_projects, etc.).
"""

import json

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

LIST_ENTITIES_TOOL_DESCRIPTION = """List W&B entities (username + teams) the current API key has access to.

Returns entity names and types WITHOUT listing projects (fast, lightweight).

<when_to_use>
Call this FIRST when:
- The user hasn't specified which entity (username or team) to use
- You need to discover available entities before querying projects or data
- The user asks "what teams do I have?" or "what's my entity?"

After getting entity names, call list_entity_projects with a SPECIFIC entity
to get that entity's projects.
</when_to_use>

<critical_info>
This is a discovery-only tool. It does NOT list projects.
Use list_entity_projects(entity="<name>") to get projects for a specific entity.
</critical_info>

Returns
-------
JSON with:
  - entities: list of {name, type} where type is "user" or "team"
  - count: total number of entities
"""


def list_entities() -> str:
    """List W&B entities (user + teams) accessible with the current API key."""
    api = WandBApiManager.get_api()
    viewer = api.viewer

    with track_tool_execution("list_entities", viewer, {}):
        entities = [{"name": viewer.entity, "type": "user"}]
        for team in getattr(viewer, "teams", []):
            entities.append({"name": team, "type": "team"})

        return json.dumps({"entities": entities, "count": len(entities)})
