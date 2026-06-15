"""Tests for narrow MCP viewer identity lookup."""

from wandb_mcp_server.mcp_tools.identity import get_viewer_identity


def test_get_viewer_identity_uses_bearer_key_and_minimal_query(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            calls.append(("raise_for_status",))

        def json(self):
            return {
                "data": {
                    "viewer": {
                        "entity": "alice",
                        "username": "alice",
                        "teams": {
                            "edges": [
                                {"node": {"name": "team-a"}},
                                {"node": {"name": "team-b"}},
                            ]
                        },
                    }
                }
            }

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return Response()

    monkeypatch.setattr(
        "wandb_mcp_server.mcp_tools.identity.WandBApiManager.get_api_key",
        staticmethod(lambda: "local-wandb_v1_test"),
    )
    monkeypatch.setattr(
        "wandb_mcp_server.mcp_tools.identity.requests.post",
        fake_post,
    )

    identity = get_viewer_identity()

    assert identity.entity == "alice"
    assert identity.teams == ("team-a", "team-b")
    url, headers, payload, timeout = calls[0]
    assert url == "https://api.wandb.ai/graphql"
    assert headers["Authorization"] == "Bearer local-wandb_v1_test"
    assert "apiKeys" not in payload["query"]
    assert timeout == 30
    assert calls[1] == ("raise_for_status",)
