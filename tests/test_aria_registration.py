import asyncio

from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.server import register_tools


def test_aria_tools_are_registered_with_self_guiding_descriptions() -> None:
    mcp = FastMCP("test")
    register_tools(mcp)

    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

    assert {"aria_send_message", "aria_get_turn"} <= tools.keys()
    assert "hand off W&B-native work" in tools["aria_send_message"].description
    assert "parent_turn_id" in tools["aria_send_message"].description
    assert "bounded interval" in tools["aria_get_turn"].description
    assert tools["aria_send_message"].inputSchema["required"] == ["message"]
    assert tools["aria_get_turn"].inputSchema["required"] == ["turn_id"]
