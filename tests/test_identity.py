"""Tests for narrow MCP viewer identity lookup."""

from wandb_mcp_server.mcp_tools.identity import get_viewer_identity


def test_get_viewer_identity_uses_public_api_default_entity(monkeypatch):
    class Api:
        default_entity = "alice"

    monkeypatch.setattr(
        "wandb_mcp_server.mcp_tools.identity.get_wandb_api",
        lambda: Api(),
    )

    identity = get_viewer_identity()

    assert identity.entity == "alice"
    assert identity.username == "alice"
    assert identity.teams == ()
    assert identity.entity_names() == ["alice"]
