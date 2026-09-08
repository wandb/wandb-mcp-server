"""Regression tests for non-amplifying functional Weave requests."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Iterator

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from wandb_mcp_server.api_client import WandBApiManager, wandb_server_busy_from_exception
from wandb_mcp_server.mcp_tools import count_traces as count_module
from wandb_mcp_server.server import create_mcp_server
from wandb_mcp_server.weave_api.client import WeaveApiClient


@contextmanager
def _overloaded_weave_server(
    *,
    status_code: int,
    retry_after: str,
    body: bytes,
) -> Iterator[tuple[str, list[str]]]:
    attempts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            attempts.append(self.path)
            self.send_response(status_code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Retry-After", retry_after)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", attempts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_trace_query_attempts_once_and_preserves_retry_after() -> None:
    with _overloaded_weave_server(
        status_code=429,
        retry_after="7",
        body=b"Too many requests",
    ) as (server_url, attempts):
        client = WeaveApiClient(
            api_key="test-key",
            server_url=server_url,
            retries=99,
        )

        with pytest.raises(Exception) as caught:
            list(client.query_traces({"project_id": "entity/project"}))

    busy = wandb_server_busy_from_exception(caught.value)
    assert attempts == ["/calls/stream_query"]
    assert busy is not None
    assert busy.status_code == 429
    assert busy.retry_after_ms == 7_000
    assert busy.as_dict() == {
        "error": "server_busy",
        "message": "The W&B service is busy; retry this tool call.",
        "retryable": True,
        "retry_after_ms": 7_000,
    }


def test_trace_count_attempts_once_and_preserves_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    with _overloaded_weave_server(
        status_code=503,
        retry_after="4",
        body=b"Service unavailable: capacity exhausted",
    ) as (server_url, attempts):
        monkeypatch.setattr(count_module.WandBApiManager, "get_api_key", lambda: "test-key")
        monkeypatch.setattr(count_module.WandBApiManager, "get_api", lambda: object())
        monkeypatch.setattr("wandb_mcp_server.config.WF_TRACE_SERVER_URL", server_url)

        with pytest.raises(Exception) as caught:
            count_module.count_traces("entity", "project")

    busy = wandb_server_busy_from_exception(caught.value)
    assert attempts == ["/calls/query_stats"]
    assert busy is not None
    assert busy.status_code == 503
    assert busy.retry_after_ms == 4_000
    assert busy.as_dict() == {
        "error": "server_busy",
        "message": "The W&B service is busy; retry this tool call.",
        "retryable": True,
        "retry_after_ms": 4_000,
    }


def test_trace_count_never_logs_customer_request_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_canary = "customer-filter-canary"
    with _overloaded_weave_server(
        status_code=400,
        retry_after="0",
        body=b"upstream-response-canary",
    ) as (server_url, _attempts):
        monkeypatch.setattr(count_module.WandBApiManager, "get_api_key", lambda: "test-key")
        monkeypatch.setattr(count_module.WandBApiManager, "get_api", lambda: object())
        monkeypatch.setattr("wandb_mcp_server.config.WF_TRACE_SERVER_URL", server_url)

        with caplog.at_level("DEBUG"):
            with pytest.raises(Exception):
                count_module.count_traces(
                    "customer-entity",
                    "customer-project",
                    filters={"trace_id": request_canary},
                )

    for protected in (
        request_canary,
        "customer-entity",
        "customer-project",
        "upstream-response-canary",
    ):
        assert protected not in caplog.text


def test_trace_count_default_uses_validated_wandb_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, int]:
            return {"count": 3}

    class Session:
        @staticmethod
        def post(url: str, **kwargs: object) -> Response:
            captured["url"] = url
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(count_module.WandBApiManager, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(count_module.WandBApiManager, "get_api", lambda: object())
    monkeypatch.setattr(count_module, "get_no_retry_session", lambda: Session())

    assert count_module.count_traces("entity", "project") == 3
    assert captured["timeout"] == count_module.MCP_WANDB_REQUEST_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_trace_query_preflight_stops_after_one_overloaded_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    with _overloaded_weave_server(
        status_code=429,
        retry_after="7",
        body=b"Too many requests",
    ) as (server_url, attempts):
        monkeypatch.setattr(count_module.WandBApiManager, "get_api", lambda: object())
        monkeypatch.setattr("wandb_mcp_server.config.WF_TRACE_SERVER_URL", server_url)
        token = WandBApiManager.set_context_api_key("test-key")
        mcp = create_mcp_server("stdio")
        try:
            with pytest.raises(ToolError) as caught:
                await mcp.call_tool(
                    "query_weave_traces_tool",
                    {
                        "entity_name": "entity",
                        "project_name": "project",
                        "limit": 101,
                    },
                )
        finally:
            WandBApiManager.reset_context_api_key(token)
            mcp.shutdown_sync_executor()

    assert attempts == ["/calls/query_stats"]
    assert '"error": "server_busy"' in str(caught.value)
    assert '"retry_after_ms": 7000' in str(caught.value)
