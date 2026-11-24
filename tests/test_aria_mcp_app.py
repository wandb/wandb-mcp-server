from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

import wandb_mcp_server.server as server
from wandb_mcp_server.mcp_tools import aria
from wandb_mcp_server.mcp_tools.aria_app import ARIA_RESOURCE_URI


def _enable_aria(monkeypatch: pytest.MonkeyPatch, access_mode: str = "read-write") -> None:
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-agents-aria")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", access_mode)
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    monkeypatch.setenv("MCP_CAPACITY_CLASS", "small")
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://wb-agent.example", transport=handler)


def test_aria_mcp_app_resource_and_tool_metadata_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_aria(monkeypatch, "read-write")
    mcp = FastMCP("test-aria-app")
    server.register_tools(mcp)

    async def inspect() -> tuple[dict[str, Any], list[Any]]:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        resources = await mcp.list_resources()
        return tools, resources

    tools, resources = asyncio.run(inspect())
    for tool_name in ("aria_send_message", "aria_get_turn", "aria_list_threads", "aria_get_thread"):
        assert tools[tool_name].meta is not None
        assert tools[tool_name].meta["ui"] == {"resourceUri": ARIA_RESOURCE_URI}
        assert tools[tool_name].meta["ui/resourceUri"] == ARIA_RESOURCE_URI

    assert "parallel_turn_ids" in tools["aria_get_turn"].inputSchema["properties"]
    assert tools["aria_get_turn"].inputSchema["required"] == ["turn_id"]

    aria_resources = [res for res in resources if str(res.uri) == ARIA_RESOURCE_URI]
    assert len(aria_resources) == 1
    assert aria_resources[0].mimeType == "text/html;profile=mcp-app"


def test_turn_result_extracts_prompt_parts_tool_steps_and_forge_url_without_leaking_outputs() -> None:
    secret_stdout = "SUPER_SECRET_STDOUT_VALUE"
    turn = {
        "id": "turn-42",
        "thread_id": "thread-alpha",
        "parent_turn_id": None,
        "title": "Looped Block Ablation",
        "user_prompt": [
            {"type": "text", "text": "Compare Arm A vs Arm C across 3 seeds."},
            {"type": "wandb_client", "platform": "agent"},
        ],
        "wandb_entity": "wandb",
        "wandb_project": "looped-transformer-lab",
        "state": "in_progress",
        "created_at": "2026-11-24T10:00:00Z",
        "updated_at": "2026-11-24T10:00:15Z",
        "messages": [],
        "tool_calls": [
            {
                "type": "invocation",
                "call_id": "call-1",
                "name": "query_wandb",
                "timestamp": "2026-11-24T10:00:02Z",
                "arguments": {"secret": secret_stdout},
            },
            {
                "type": "response",
                "call_id": "call-1",
                "timestamp": "2026-11-24T10:00:05Z",
                "is_error": False,
                "output": {"stdout": secret_stdout},
            },
            {
                "type": "invocation",
                "call_id": "call-2",
                "name": "shell",
                "timestamp": "2026-11-24T10:00:06Z",
                "arguments": {"cmd": "python eval.py"},
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-42", api_key="token", client=client)

    result = asyncio.run(run())
    assert result["ok"] is True
    assert result["user_prompt"] == "Compare Arm A vs Arm C across 3 seeds."
    assert result["title"] == "Looped Block Ablation"
    assert result["forge_url"] == "https://forge.coreweave.com/home/aria/threads/thread-alpha"
    assert result["tool_steps"] == [
        {
            "call_id": "call-1",
            "name": "query_wandb",
            "status": "ok",
            "started_at": "2026-11-24T10:00:02Z",
            "ended_at": "2026-11-24T10:00:05Z",
        },
        {
            "call_id": "call-2",
            "name": "shell",
            "status": "running",
            "started_at": "2026-11-24T10:00:06Z",
        },
    ]
    assert secret_stdout not in json.dumps(result)


def test_get_aria_turn_polls_parallel_sub_agents_concurrently() -> None:
    turns_by_id = {
        "arm-a": {
            "id": "arm-a",
            "thread_id": "th-a",
            "title": "Arm A",
            "wandb_entity": "wandb",
            "wandb_project": "looped-lab",
            "state": "completed",
            "updated_at": "2026-11-24T10:05:00Z",
            "messages": [{"role": "assistant", "content": "Arm A done"}],
            "tool_calls": [],
        },
        "arm-b": {
            "id": "arm-b",
            "thread_id": "th-b",
            "title": "Arm B",
            "wandb_entity": "wandb",
            "wandb_project": "looped-lab",
            "state": "in_progress",
            "updated_at": "2026-11-24T10:05:00Z",
            "messages": [],
            "tool_calls": [{"type": "invocation", "call_id": "c1", "name": "shell"}],
        },
        "arm-c": {
            "id": "arm-c",
            "thread_id": "th-c",
            "title": "Arm C",
            "wandb_entity": "wandb",
            "wandb_project": "looped-lab",
            "state": "completed",
            "updated_at": "2026-11-24T10:05:00Z",
            "messages": [],
            "tool_calls": [],
            "permission_requests": [{"type": "network_access", "reason": "Read arXiv paper", "domains": ["arxiv.org"]}],
        },
    }

    class SlowConcurrentTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.08)
            tid = request.url.path.rsplit("/", 1)[-1]
            payload = turns_by_id.get(tid)
            if payload is None:
                return httpx.Response(404, json={"error": {"code": "not_found"}}, request=request)
            return httpx.Response(200, json=payload, request=request)

    async def run() -> tuple[Dict[str, Any], float]:
        t0 = time.monotonic()
        async with _client(SlowConcurrentTransport()) as client:
            res = await aria.get_aria_turn(
                "arm-a",
                parallel_turn_ids=["arm-b", "arm-c", "arm-a", "missing-turn"],
                wait_seconds=0,
                api_key="token",
                client=client,
            )
        return res, time.monotonic() - t0

    result, elapsed = asyncio.run(run())
    # 4 concurrent 80ms requests should finish well below sequential 320ms
    assert elapsed < 0.28
    assert result["ok"] is True
    assert [m["turn_id"] for m in result["parallel_turns"]] == [
        "arm-a",
        "arm-b",
        "arm-c",
        "missing-turn",
    ]
    assert result["parallel_summary"] == {
        "total_count": 4,
        "completed_count": 1,
        "active_count": 1,
        "awaiting_permission_count": 1,
        "awaiting_user_input_count": 0,
        "errored_count": 1,
        "cancelled_count": 0,
        "all_terminal": False,
    }
    assert result["poll_after_seconds"] == 2
    assert "arm-b" in result["next_action"]


def test_mcp_boundary_attaches_access_mode_on_read_only_aria_get_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_aria(monkeypatch, "read-only")

    async def fake_get_turn(**kwargs: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "turn_id": kwargs["turn_id"],
            "thread_id": "th-ro",
            "scope": {"entity": "wandb", "project": "demo"},
            "state": "completed",
            "is_terminal": True,
            "latest_response": "Completed answer.",
        }

    monkeypatch.setattr(server, "get_aria_turn", fake_get_turn)
    mcp = FastMCP("test-ro")
    server.register_tools(mcp)

    async def call() -> Any:
        async with create_connected_server_and_client_session(mcp) as client_session:
            return await client_session.call_tool("aria_get_turn", {"turn_id": "turn-ro"})

    res = asyncio.run(call())
    assert res.isError is False
    assert res.structuredContent["result"]["access_mode"] == "read-only"
    assert res.structuredContent["result"]["forge_url"] == "https://forge.coreweave.com/home/aria/threads/th-ro"
