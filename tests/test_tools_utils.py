"""Tests for MCP tool execution helpers."""


def test_safe_viewer_info_returns_viewer_when_available():
    from wandb_mcp_server.mcp_tools.tools_utils import safe_viewer_info

    class Api:
        viewer = {"username": "alice"}

    assert safe_viewer_info(Api()) == {"username": "alice"}


def test_safe_viewer_info_swallows_viewer_errors():
    from wandb_mcp_server.mcp_tools.tools_utils import safe_viewer_info

    class Api:
        @property
        def viewer(self):
            raise RuntimeError("relogin required")

    assert safe_viewer_info(Api()) is None


def test_track_tool_execution_swallows_analytics_failure(monkeypatch):
    from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

    class BrokenTracker:
        def track_tool_call(self, **kwargs):
            raise RuntimeError("analytics unavailable")

    monkeypatch.setattr(
        "wandb_mcp_server.analytics.get_analytics_tracker",
        lambda: BrokenTracker(),
    )

    with track_tool_execution("safe_tool", None, {"entity_name": "ent"}):
        result = "tool completed"

    assert result == "tool completed"
