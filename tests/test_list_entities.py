"""Tests for list_entities tool and list_entity_projects max_projects bounds."""

import json
from unittest.mock import MagicMock, patch

from wandb_mcp_server.mcp_tools.identity import ViewerIdentity


def _make_project(name, entity):
    p = MagicMock()
    p.name = name
    p.entity = entity
    p.description = f"Desc for {name}"
    p.visibility = "private"
    p.created_at = "2026-01-01"
    p.updated_at = "2026-04-01"
    p.tags = []
    return p


class TestListEntities:
    """list_entities returns viewer entity + teams without project enumeration."""

    @patch("wandb_mcp_server.mcp_tools.list_entities.get_viewer_identity")
    def test_returns_user_and_teams(self, mock_get_identity):
        mock_get_identity.return_value = ViewerIdentity(
            entity="alice",
            username="alice",
            teams=("team-a", "team-b", "team-c"),
        )
        from wandb_mcp_server.mcp_tools.list_entities import list_entities

        result = json.loads(list_entities())
        assert result["count"] == 4
        assert result["entities"][0] == {"name": "alice", "type": "user"}
        assert result["entities"][1] == {"name": "team-a", "type": "team"}
        assert result["entities"][3] == {"name": "team-c", "type": "team"}

    @patch("wandb_mcp_server.mcp_tools.list_entities.get_viewer_identity")
    def test_no_teams(self, mock_get_identity):
        mock_get_identity.return_value = ViewerIdentity(entity="solo-user", username="solo-user", teams=())
        from wandb_mcp_server.mcp_tools.list_entities import list_entities

        result = json.loads(list_entities())
        assert result["count"] == 1
        assert result["entities"][0]["name"] == "solo-user"
        assert result["entities"][0]["type"] == "user"

    @patch("wandb_mcp_server.mcp_tools.list_entities.get_viewer_identity")
    def test_viewer_without_teams_attr(self, mock_get_identity):
        mock_get_identity.return_value = ViewerIdentity(entity="no-attr-user", username=None, teams=())
        from wandb_mcp_server.mcp_tools.list_entities import list_entities

        result = json.loads(list_entities())
        assert result["count"] == 1


class TestListEntityProjectsBounds:
    """list_entity_projects respects max_projects and max_entities."""

    def _mock_api_with_projects(self, entity, count):
        api = MagicMock()
        api.projects.return_value = iter([_make_project(f"proj-{i}", entity) for i in range(count)])
        return api

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    def test_max_projects_caps_results(self, mock_get_api):
        mock_get_api.return_value = self._mock_api_with_projects("alice", 100)
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = json.loads(list_entity_projects(entity="alice", max_projects=5))
        assert len(result["projects"]["alice"]) == 5
        assert result["truncated"] is True

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    def test_max_projects_ceiling(self, mock_get_api):
        mock_get_api.return_value = self._mock_api_with_projects("alice", 300)
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = json.loads(list_entity_projects(entity="alice", max_projects=999))
        assert len(result["projects"]["alice"]) == 200

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    @patch("wandb_mcp_server.mcp_tools.list_wandb_entities_projects.get_viewer_identity")
    def test_entity_none_caps_entities(self, mock_get_identity, mock_get_api):
        mock_get_identity.return_value = ViewerIdentity(
            entity="user",
            username="user",
            teams=tuple(f"team-{i}" for i in range(20)),
        )
        api = MagicMock()
        api.projects.return_value = iter([_make_project("p1", "user")])
        mock_get_api.return_value = api

        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = json.loads(list_entity_projects(entity=None, max_projects=10))
        assert len(result["projects"]) <= 5
        assert result["truncated"] is True

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    def test_specific_entity_no_truncation(self, mock_get_api):
        mock_get_api.return_value = self._mock_api_with_projects("alice", 3)
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = json.loads(list_entity_projects(entity="alice", max_projects=50))
        assert len(result["projects"]["alice"]) == 3
        assert result["truncated"] is False

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    def test_returns_json_string(self, mock_get_api):
        mock_get_api.return_value = self._mock_api_with_projects("alice", 0)
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = list_entity_projects(entity="alice")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "projects" in parsed
        assert "truncated" in parsed

    @patch("wandb_mcp_server.api_client.get_wandb_api")
    def test_relogin_required_returns_structured_sdk_error(self, mock_get_api):
        api = MagicMock()
        api.projects.side_effect = Exception("[relogin required; relogin required; relogin required]")
        mock_get_api.return_value = api

        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = json.loads(list_entity_projects(entity="alice"))
        error = result["projects"]["alice"][0]

        assert error["error"] == "sdk_relogin_required"
        assert "API-key metadata" in error["message"]
        assert "relogin required; relogin required" not in json.dumps(result)
