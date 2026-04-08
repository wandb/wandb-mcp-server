"""Tests for registry discovery tools (list_registries, list_registry_collections)."""

import json
from unittest.mock import MagicMock, patch

from wandb_mcp_server.mcp_tools.query_registry import (
    LIST_REGISTRIES_TOOL_DESCRIPTION,
    LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION,
    list_registries,
    list_registry_collections,
)


def _make_registry(name="my-registry", **overrides):
    reg = MagicMock()
    reg.name = name
    reg.full_name = f"wandb-registry-{name}"
    reg.organization = overrides.get("organization", "my-org")
    reg.entity = overrides.get("entity", "my-org")
    reg.description = overrides.get("description", "A test registry")
    reg.visibility = overrides.get("visibility", "organization")
    reg.artifact_types = overrides.get("artifact_types", ["model"])
    reg.created_at = overrides.get("created_at", "2025-01-01T00:00:00")
    reg.updated_at = overrides.get("updated_at", "2025-06-01T00:00:00")
    return reg


def _make_collection(name="my-model", **overrides):
    coll = MagicMock()
    coll.name = name
    coll.type = overrides.get("type", "model")
    coll.description = overrides.get("description", "A test collection")
    coll.tags = overrides.get("tags", ["production"])
    coll.aliases = overrides.get("aliases", ["latest"])
    coll.created_at = overrides.get("created_at", "2025-01-01T00:00:00")
    coll.updated_at = overrides.get("updated_at", "2025-06-01T00:00:00")
    coll.is_sequence.return_value = overrides.get("is_sequence", True)
    return coll


class TestListRegistries:
    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_basic(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.registries.return_value = iter(
            [
                _make_registry("models"),
                _make_registry("datasets"),
            ]
        )
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registries())

        assert result["count"] == 2
        assert result["truncated"] is False
        assert result["registries"][0]["name"] == "models"
        assert result["registries"][1]["name"] == "datasets"

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_filter_passed_through(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.registries.return_value = iter([_make_registry("models")])
        mock_api_mgr.get_api.return_value = mock_api

        filt = {"name": {"$regex": "model.*"}}
        list_registries(filter=filt)

        call_kwargs = mock_api.registries.call_args[1]
        assert call_kwargs["filter"] == filt

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_max_items_ceiling(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        regs = [_make_registry(f"reg-{i}") for i in range(250)]
        mock_api.registries.return_value = iter(regs)
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registries(max_items=999))

        assert result["count"] == 200
        assert result["truncated"] is True

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_api_error_returns_json(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.registries.side_effect = Exception("Connection refused")
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registries())

        assert "error" in result
        assert "Connection refused" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_empty_result(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.registries.return_value = iter([])
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registries())

        assert result["count"] == 0
        assert result["registries"] == []
        assert result["truncated"] is False


class TestListRegistryCollections:
    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_basic(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_registry = MagicMock()
        mock_registry.collections.return_value = iter(
            [
                _make_collection("model-a"),
                _make_collection("model-b"),
            ]
        )
        mock_api.registry.return_value = mock_registry
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registry_collections("my-registry"))

        assert result["registry"] == "my-registry"
        assert result["count"] == 2
        assert result["collections"][0]["name"] == "model-a"
        assert result["collections"][1]["name"] == "model-b"

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_empty(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_registry = MagicMock()
        mock_registry.collections.return_value = iter([])
        mock_api.registry.return_value = mock_registry
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registry_collections("my-registry"))

        assert result["count"] == 0
        assert result["collections"] == []

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_collection_properties(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_registry = MagicMock()
        coll = _make_collection("prod-model", tags=["production", "v2"], is_sequence=True)
        mock_registry.collections.return_value = iter([coll])
        mock_api.registry.return_value = mock_registry
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registry_collections("my-registry"))
        c = result["collections"][0]

        assert c["name"] == "prod-model"
        assert c["tags"] == ["production", "v2"]
        assert c["is_sequence"] is True

    @patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager")
    def test_api_error_returns_json(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.registry.side_effect = Exception("Registry not found")
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_registry_collections("nonexistent"))

        assert "error" in result
        assert "Registry not found" in result["message"]


class TestToolDescriptions:
    def test_list_registries_has_when_to_use(self):
        assert "<when_to_use>" in LIST_REGISTRIES_TOOL_DESCRIPTION
        assert "</when_to_use>" in LIST_REGISTRIES_TOOL_DESCRIPTION

    def test_list_registry_collections_has_when_to_use(self):
        assert "<when_to_use>" in LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION
        assert "</when_to_use>" in LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION
