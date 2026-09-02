"""Tests for registry organization and entity resolution."""

from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.registry_support import (
    OrganizationRequired,
    resolve_registry_organization,
)


@patch("wandb_mcp_server.registry_support.fetch_registry_organization_info")
def test_explicit_organization_is_preserved(mock_fetch):
    mock_fetch.return_value = {
        "organization": {"name": "acme", "orgEntity": {"name": "acme-org"}},
        "entity": None,
    }

    assert resolve_registry_organization(MagicMock(), "acme") == "acme"


@patch("wandb_mcp_server.registry_support.fetch_registry_organization_info")
def test_team_entity_resolves_to_organization(mock_fetch):
    mock_fetch.return_value = {
        "organization": None,
        "entity": {
            "organization": {"name": "Acme Corp", "orgEntity": {"name": "acme-org"}},
            "user": None,
        },
    }

    assert resolve_registry_organization(MagicMock(), "acme-team") == "Acme Corp"


@patch("wandb_mcp_server.registry_support.fetch_registry_organization_info")
def test_personal_entity_with_multiple_orgs_returns_candidates(mock_fetch):
    mock_fetch.return_value = {
        "organization": None,
        "entity": {
            "organization": None,
            "user": {
                "organizations": [
                    {"name": "Beta", "orgEntity": {"name": "beta-org"}},
                    {"name": "Alpha", "orgEntity": {"name": "alpha-org"}},
                ]
            },
        },
    }

    with pytest.raises(OrganizationRequired) as error:
        resolve_registry_organization(MagicMock(), "personal-entity")

    assert error.value.candidates == ("Alpha", "Beta")


@patch("wandb_mcp_server.registry_support.fetch_registry_organization_info")
def test_configured_organization_avoids_resolution_query(mock_fetch):
    api = MagicMock()
    api.settings = {"organization": "configured-org"}

    assert resolve_registry_organization(api, None) == "configured-org"
    mock_fetch.assert_not_called()
