import json
import time

import pytest

import wandb_mcp_server.config as cfg
import wandb_mcp_server.server as server
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata
from wandb_mcp_server.weave_api.service import TraceService


class FakeMCP:
    """Capture FastMCP-decorated tool functions."""

    def __init__(self):
        self.tools = {}

    def tool(self, name=None, description=None):
        # Mirror FastMCP.tool(name=..., description=...): the agent tools are now
        # registered directly with an explicit name (server.register_tools), so the
        # double must honor `name`, falling back to __name__ for the decorator form.
        def decorator(func):
            self.tools[name or func.__name__] = func
            return func

        return decorator


def _registered_tools(monkeypatch):
    monkeypatch.setattr(WandBApiManager, "get_api_key", staticmethod(lambda: "test-key"))
    fake = FakeMCP()
    server.register_tools(fake)
    return fake.tools


def _empty_query_result():
    return QueryResult(metadata=TraceMetadata(total_traces=0), traces=[])


@pytest.mark.asyncio
async def test_metadata_only_over_hosted_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(cfg, "MCP_MAX_QUERY_LIMIT", 100)
    called = False

    async def fake_query(*args, **kwargs):
        nonlocal called
        called = True
        return _empty_query_result()

    monkeypatch.setattr(server, "query_paginated_weave_traces", fake_query)

    tool = _registered_tools(monkeypatch)["query_weave_traces_tool"]
    result = json.loads(await tool("entity", "project", limit=101, metadata_only=True))

    assert result["error"] == "quota_exceeded"
    assert result["max_limit"] == 100
    assert called is False


@pytest.mark.asyncio
async def test_hosted_omitted_limit_defaults_to_query_cap(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(cfg, "MCP_MAX_QUERY_LIMIT", 100)
    seen_kwargs = {}

    async def fake_query(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return _empty_query_result()

    monkeypatch.setattr(server, "query_paginated_weave_traces", fake_query)

    async def no_count(*args, **kwargs):
        return None

    monkeypatch.setattr(server, "_count_traces_or_none", no_count)

    tool = _registered_tools(monkeypatch)["query_weave_traces_tool"]
    result = json.loads(await tool("entity", "project", metadata_only=True))

    assert "error" not in result
    assert seen_kwargs["target_limit"] == 100


@pytest.mark.asyncio
async def test_full_trace_hosted_limit_uses_full_cap(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(cfg, "MCP_MAX_FULL_TRACE_LIMIT", 25)

    tool = _registered_tools(monkeypatch)["query_weave_traces_tool"]
    result = json.loads(await tool("entity", "project", detail_level="full", limit=26))

    assert result["error"] == "quota_exceeded"
    assert result["max_limit"] == 25


@pytest.mark.parametrize("sort_by", ["total_cost", "completion_cost", "prompt_cost"])
@pytest.mark.asyncio
async def test_cost_sort_rejected_in_hosted_mode(monkeypatch, sort_by):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "shared")
    called = False

    async def fake_query(*args, **kwargs):
        nonlocal called
        called = True
        return _empty_query_result()

    monkeypatch.setattr(server, "query_paginated_weave_traces", fake_query)

    tool = _registered_tools(monkeypatch)["query_weave_traces_tool"]
    result = json.loads(await tool("entity", "project", sort_by=sort_by, limit=10))

    assert result["error"] == "quota_exceeded"
    assert result["sort_by"] == sort_by
    assert called is False


def test_service_rejects_direct_hosted_over_limit(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "dedicated")
    monkeypatch.setattr(cfg, "MCP_MAX_QUERY_LIMIT", 100)

    service = TraceService(api_key="test-key")

    with pytest.raises(cfg.HostedLimitExceeded):
        service.query_paginated_traces("entity", "project", target_limit=101)


def test_service_defaults_direct_hosted_limit(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "dedicated")
    monkeypatch.setattr(cfg, "MCP_MAX_QUERY_LIMIT", 3)

    service = TraceService(api_key="test-key")
    calls = []

    def fake_query(request_body):
        calls.append(request_body)
        for i in range(5):
            yield {
                "id": f"call-{i}",
                "project_id": "entity/project",
                "op_name": "op",
                "trace_id": f"trace-{i}",
                "started_at": "2026-01-01T00:00:00Z",
            }

    monkeypatch.setattr(service.client, "query_traces", fake_query)

    result = service.query_paginated_traces("entity", "project", target_limit=None, chunk_size=10)

    assert len(result.traces) == 3
    assert calls[0]["limit"] == 3


def test_service_rejects_direct_hosted_cost_sort(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "dedicated")

    service = TraceService(api_key="test-key")

    with pytest.raises(cfg.HostedLimitExceeded):
        service._query_for_cost_sorting("entity", "project", sort_by="total_cost")


@pytest.mark.asyncio
async def test_count_tool_timeout_returns_within_deadline(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_TOOL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(WandBApiManager, "get_api_key", staticmethod(lambda: "test-key"))

    def slow_count(*args, **kwargs):
        time.sleep(3)
        return 1

    monkeypatch.setattr(server, "count_traces", slow_count)

    tool = _registered_tools(monkeypatch)["count_weave_traces_tool"]
    start = time.monotonic()
    result = json.loads(await tool("entity", "project"))
    elapsed = time.monotonic() - start

    assert elapsed < 2.5
    assert result["error"] == "timeout"
    assert result["timeout_seconds"] == 1


@pytest.mark.asyncio
async def test_count_tool_passes_bounded_request_timeout(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_TOOL_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(WandBApiManager, "get_api_key", staticmethod(lambda: "test-key"))
    request_timeouts = []

    def capture_count(*args, **kwargs):
        request_timeouts.append(kwargs["request_timeout"])
        return 2

    monkeypatch.setattr(server, "count_traces", capture_count)

    tool = _registered_tools(monkeypatch)["count_weave_traces_tool"]
    result = json.loads(await tool("entity", "project"))

    assert result == {"total_count": 2, "root_traces_count": 2}
    assert request_timeouts == [1, 1]


@pytest.mark.asyncio
async def test_count_tool_preserves_session_context_in_executor(monkeypatch):
    from wandb_mcp_server.session_manager import current_session_id

    monkeypatch.setattr(cfg, "MCP_TOOL_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(WandBApiManager, "get_api_key", staticmethod(lambda: "test-key"))
    observed_session_ids = []

    def capture_count(*args, **kwargs):
        observed_session_ids.append(current_session_id.get())
        return 2

    monkeypatch.setattr(server, "count_traces", capture_count)

    token = current_session_id.set("sess_test")
    try:
        tool = _registered_tools(monkeypatch)["count_weave_traces_tool"]
        result = json.loads(await tool("entity", "project"))
    finally:
        current_session_id.reset(token)

    assert result == {"total_count": 2, "root_traces_count": 2}
    assert observed_session_ids == ["sess_test", "sess_test"]


@pytest.mark.asyncio
async def test_trace_query_preflight_timeout_does_not_block_main_query(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(cfg, "MCP_MAX_QUERY_LIMIT", 100)
    monkeypatch.setattr(cfg, "MCP_TOOL_TIMEOUT_SECONDS", 1)

    def slow_count(*args, **kwargs):
        time.sleep(3)
        return 10

    async def fake_query(*args, **kwargs):
        return QueryResult(metadata=TraceMetadata(total_traces=1), traces=[])

    monkeypatch.setattr(server, "count_traces", slow_count)
    monkeypatch.setattr(server, "query_paginated_weave_traces", fake_query)

    tool = _registered_tools(monkeypatch)["query_weave_traces_tool"]
    start = time.monotonic()
    result = json.loads(await tool("entity", "project", limit=100))
    elapsed = time.monotonic() - start

    assert elapsed < 3
    assert "error" not in result
