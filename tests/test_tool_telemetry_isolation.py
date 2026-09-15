"""Identity follows the authenticated actor through concurrent worker calls."""

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from wandb_mcp_server import analytics
from wandb_mcp_server.analytics_datadog import map_to_datadog_log
from wandb_mcp_server.analytics_segment import map_to_segment_track
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["off", "standard", "strict"])
async def test_concurrent_worker_calls_emit_only_their_own_identity(monkeypatch, level):
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "false")
    monkeypatch.setenv("MCP_LOG_PRIVACY_LEVEL", level)
    WandBApiManager._clear_api_cache()
    actors = {"synthetic-key-a": "synthetic-alice", "synthetic-key-b": "synthetic-bob"}
    constructed = []

    def fake_api(*, api_key, **kwargs):
        constructed.append(api_key)
        return SimpleNamespace(_viewer=SimpleNamespace(_attrs={"username": actors[api_key]}))

    monkeypatch.setattr("wandb_mcp_server.api_client.wandb.Api", fake_api)
    tracker = analytics.AnalyticsTracker()
    events = []
    monkeypatch.setattr(tracker, "_emit", lambda event, labels: events.append(analytics._prepare_event(event)))
    monkeypatch.setattr(analytics, "get_analytics_tracker", lambda: tracker)
    barrier = threading.Barrier(2, timeout=5)
    server = InstrumentedFastMCP("identity-isolation")

    @server.tool(name="query_wandb_tool")
    def query_wandb_tool() -> str:
        WandBApiManager.get_api()
        barrier.wait()
        return "ok"

    async def invoke(key):
        token = WandBApiManager.set_context_api_key(key)
        try:
            await server.call_tool("query_wandb_tool", {})
        finally:
            WandBApiManager.reset_context_api_key(token)

    try:
        await asyncio.gather(*(invoke(key) for key in actors))
        assert len(constructed) == len(actors), "analytics must not construct additional SDK clients"
        assert len(events) == len(actors)
        assert all(event["success"] for event in events)
        expected = (
            {"wandb_key:" + hashlib.sha256(key.encode()).hexdigest()[:24] for key in actors}
            if level == "strict"
            else set(actors.values())
        )
        assert {event["actor_id"] for event in events} == expected
        assert {event["user_id"] for event in events} == expected
        sinks = [
            (
                map_to_datadog_log(event, dd_env="test", dd_version="test", dd_service="test"),
                map_to_segment_track(event),
            )
            for event in events
        ]
        assert {dd["attributes"]["usr"]["id"] for dd, _ in sinks} == expected
        pseudonyms = {"wandb_key:" + hashlib.sha256(key.encode()).hexdigest()[:24] for key in actors}
        assert {segment["userId"] for _, segment in sinks} == pseudonyms
        assert all(username not in json.dumps([segment for _, segment in sinks]) for username in actors.values())
        serialized = json.dumps([events, sinks])
        assert all(key not in serialized for key in actors)
        if level == "strict":
            assert all(username not in serialized for username in actors.values())
        assert WandBApiManager.get_api_key() is None
    finally:
        server._sync_executor.shutdown(wait=True)
        WandBApiManager._clear_api_cache()
