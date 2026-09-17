"""Exercise the docs proxy with real HTTPX streaming and the MCP boundary."""

import asyncio
import json
import time

import httpx
import pytest

from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.mcp_tools import docs_search as docs
from wandb_mcp_server.trace_utils import count_tokens_conservative


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def upstream(monkeypatch):
    real_client = httpx.AsyncClient
    requests = []

    def install(body=None, *, chunks=None, status=200, headers=None, failure=None):
        if body is None:
            body = {"result": {"content": [{"type": "text", "text": "Short documentation."}]}}
        raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        stream = Chunks(chunks if chunks is not None else [raw])

        def handler(request):
            requests.append(request)
            if failure:
                raise failure
            return httpx.Response(status, headers=headers or {}, stream=stream)

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(docs.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
        return requests, stream

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query", ["", "  ", None, [], "界" * 21846], ids=["empty", "blank", "null", "array", "oversized-utf8"]
)
async def test_input_rejected_before_network(upstream, query):
    requests, _ = upstream()
    result = json.loads(await docs.search_wandb_docs(query))
    assert result["error"] == "invalid_input"
    assert not requests


@pytest.mark.asyncio
async def test_json_and_single_request(upstream):
    requests, stream = upstream()
    assert await docs.search_wandb_docs("private-query-canary") == "Short documentation."
    assert len(requests) == 1
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert str(requests[0].url) == docs.DOCS_MCP_URL
    assert stream.closed


@pytest.mark.asyncio
async def test_empty_results_remain_a_sanitized_error(upstream):
    upstream({"result": {"content": []}})
    result = await docs.search_wandb_docs("private-query-canary")
    assert json.loads(result)["error"] == "upstream_error"
    assert "canary" not in result


@pytest.mark.asyncio
async def test_exact_budget_preserves_text(upstream, monkeypatch):
    text = "Useful documentation without truncation."
    monkeypatch.setattr(docs, "MAX_RESPONSE_TOKENS", count_tokens_conservative(text))
    upstream({"result": {"content": [{"text": text}]}})
    assert await docs.search_wandb_docs("query") == text


@pytest.mark.asyncio
async def test_sse_response(upstream):
    upstream(
        'event: message\ndata: {"result":{"content":[{"type":"text","text":"SSE docs"}]}}\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    assert await docs.search_wandb_docs("query") == "SSE docs"


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Content-Length": "1"}, {"Content-Length": "1025"}])
async def test_streaming_byte_ceiling(upstream, monkeypatch, headers):
    monkeypatch.setattr(docs, "MAX_ACCUMULATED_BYTES", 1024)
    requests, stream = upstream(chunks=[b"x" * 600, b"y" * 600, b"z"], headers=headers)
    assert json.loads(await docs.search_wandb_docs("private-query-canary"))["error"] == "response_too_large"
    assert stream.reads <= 2
    assert len(requests) == 1
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"Content-Encoding": "gzip"}, {"Content-Length": "invalid"}, {"Content-Length": "-1"}]
)
async def test_malformed_transport_does_not_read(upstream, headers):
    _, stream = upstream(headers=headers)
    assert json.loads(await docs.search_wandb_docs("query"))["error"] == "malformed_response"
    assert stream.reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", ["not json", "[]", '{"result":{"content":[1]}}', '{"result":{"content":[{"text":{}}]}}']
)
async def test_malformed_body_is_sanitized(upstream, body):
    upstream(body)
    result = await docs.search_wandb_docs("private-query-canary")
    assert json.loads(result)["error"] == "malformed_response"
    assert "private-query-canary" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code",
    [
        (302, "upstream_error"),
        (401, "upstream_error"),
        (500, "upstream_error"),
        (429, "server_busy"),
        (503, "server_busy"),
    ],
)
async def test_http_failures_never_echo_or_retry(upstream, status, code):
    requests, stream = upstream(
        "private-upstream-canary",
        status=status,
        headers={"Location": "https://other.invalid", "Retry-After": "99999999"},
    )
    result = await docs.search_wandb_docs("private-query-canary")
    payload = json.loads(result)
    assert payload["error"] == code
    assert "canary" not in result
    assert len(requests) == 1
    assert stream.reads == 0
    if status in (429, 503):
        assert payload["retryable"] is True
        assert 1000 <= payload["retry_after_ms"] <= 60000


@pytest.mark.asyncio
async def test_upstream_jsonrpc_error_is_sanitized(upstream):
    upstream({"error": {"message": "private-upstream-canary"}})
    result = await docs.search_wandb_docs("private-query-canary")
    assert json.loads(result)["error"] == "upstream_error"
    assert "canary" not in result


@pytest.mark.asyncio
async def test_deadline_expired_never_requests(upstream):
    requests, _ = upstream()
    token = current_tool_deadline.set(time.monotonic() - 1)
    try:
        assert json.loads(await docs.search_wandb_docs("query"))["error"] == "tool_timeout"
    finally:
        current_tool_deadline.reset(token)
    assert not requests


@pytest.mark.asyncio
async def test_timeout_and_cancellation(upstream):
    upstream(failure=httpx.ReadTimeout("private-upstream-canary"))
    result = await docs.search_wandb_docs("private-query-canary")
    assert json.loads(result)["error"] == "tool_timeout"
    assert "canary" not in result
    upstream(failure=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await docs.search_wandb_docs("query")


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 3, 30, 100, 30000])
async def test_success_always_fits_exact_budget(upstream, monkeypatch, budget):
    monkeypatch.setattr(docs, "MAX_RESPONSE_TOKENS", budget)
    text = "界🙂 documentation" * 10000
    upstream({"result": {"content": [{"type": "text", "text": text}]}})
    result = await docs.search_wandb_docs("query")
    assert result != text
    assert count_tokens_conservative(result) <= budget
    assert "\ufffd" not in result


@pytest.mark.asyncio
async def test_mcp_boundary_does_not_return_40002_tokens(upstream, monkeypatch):
    from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
    from wandb_mcp_server.server import register_tools
    from wandb_mcp_server.runtime_contract import resolve_runtime_selection

    monkeypatch.setattr(docs, "MAX_RESPONSE_TOKENS", 30000)
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-only")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "dedicated")
    upstream({"result": {"content": [{"type": "text", "text": "documentation " * 40001}]}})
    server = InstrumentedFastMCP("docs-audit")
    register_tools(server, resolve_runtime_selection())
    result = await server.call_tool("search_wandb_docs_tool", {"query": "query"})
    content = result[0] if isinstance(result, tuple) else result
    assert sum(count_tokens_conservative(item.text) for item in content) <= 30000


@pytest.mark.asyncio
@pytest.mark.parametrize("internal_url", ["http://api", "http://internal", "http://limit"])
@pytest.mark.parametrize("budget", [11, 31, 30000])
async def test_mcp_response_budget_includes_redaction(upstream, monkeypatch, internal_url, budget):
    from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
    from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
    from wandb_mcp_server.server import register_tools
    from wandb_mcp_server.runtime_contract import resolve_runtime_selection

    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", internal_url)
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-only")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "dedicated")
    monkeypatch.setattr(docs, "MAX_RESPONSE_TOKENS", budget)
    upstream({"result": {"content": [{"type": "text", "text": (internal_url + " ") * 10000}]}})
    server = InstrumentedFastMCP("docs-redaction-audit")
    register_tools(server, resolve_runtime_selection())
    try:
        result = await server.call_tool("search_wandb_docs_tool", {"query": "query"})
    finally:
        server.shutdown_sync_executor()
    content = result[0] if isinstance(result, tuple) else result
    assert sum(count_tokens_conservative(item.text) for item in content) <= budget
    assert all(internal_url not in item.text for item in content)
    assert all(sanitize_sensitive_text(item.text) == item.text for item in content)
