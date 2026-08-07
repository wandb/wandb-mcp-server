import asyncio
import json
from typing import Any, Dict

import httpx

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools import aria


def _turn(state: str = "queued", **overrides: Any) -> Dict[str, Any]:
    turn: Dict[str, Any] = {
        "id": "turn-1",
        "thread_id": "thread-1",
        "parent_turn_id": None,
        "state": state,
        "updated_at": "2026-08-07T12:00:00Z",
        "messages": [],
        "tool_calls": [],
        "error_info": None,
    }
    turn.update(overrides)
    return turn


def _client(handler: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://wb-agent.example",
        transport=handler,
    )


def test_send_root_turn_uses_bearer_auth_and_openapi_payload(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json=_turn(), request=request)

    def new_client(api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="https://wb-agent.example",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(aria, "_new_http_client", new_client)

    result = asyncio.run(
        aria.send_aria_message(
            "Compare the latest eval runs",
            entity="team",
            project="evals",
            api_key="user-wandb-token",
        )
    )

    assert captured == {
        "method": "POST",
        "path": "/api/v1/turns",
        "authorization": "Bearer user-wandb-token",
        "body": {
            "user_prompt": "Compare the latest eval runs",
            "entity": "team",
            "project": "evals",
        },
    }
    assert result["ok"] is True
    assert result["turn_id"] == "turn-1"
    assert result["state"] == "queued"
    assert result["is_terminal"] is False
    assert result["poll_after_seconds"] == 2


def test_send_reuses_request_scoped_wandb_token(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        return httpx.Response(202, json=_turn(), request=request)

    def new_client(api_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="https://wb-agent.example",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(aria, "_new_http_client", new_client)
    context_token = WandBApiManager.set_context_api_key("context-wandb-token-123")
    try:
        result = asyncio.run(aria.send_aria_message("Inspect my runs"))
    finally:
        WandBApiManager.reset_context_api_key(context_token)

    assert result["ok"] is True
    assert captured["authorization"] == "Bearer context-wandb-token-123"


def test_send_continuation_uses_parent_and_omits_scope() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            202,
            json=_turn(parent_turn_id="turn-0"),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Now limit that to failed runs",
                parent_turn_id="turn-0",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert captured == {
        "user_prompt": "Now limit that to failed runs",
        "parent_turn_id": "turn-0",
    }
    assert result["parent_turn_id"] == "turn-0"


def test_send_rejects_scope_on_continuation_without_calling_service() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Continue",
                entity="team",
                parent_turn_id="turn-0",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert called is False
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"


def test_bounded_send_wait_polls_until_completed(monkeypatch) -> None:
    responses = [
        _turn("queued"),
        _turn(
            "in_progress",
            tool_calls=[
                {
                    "type": "invocation",
                    "name": "query_runs",
                    "call_id": "call-1",
                }
            ],
        ),
        _turn(
            "completed",
            messages=[
                {
                    "role": "assistant",
                    "content": "The regression is isolated to nightly runs.",
                    "message": {},
                }
            ],
        ),
    ]
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=responses.pop(0), request=request)

    monkeypatch.setattr(aria, "POLL_INTERVAL_SECONDS", 0)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Diagnose the regression",
                wait_seconds=2,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert paths == [
        "/api/v1/turns",
        "/api/v1/turns/turn-1",
        "/api/v1/turns/turn-1",
    ]
    assert result["state"] == "completed"
    assert result["is_terminal"] is True
    assert result["latest_response"] == ("The regression is isolated to nightly runs.")
    assert result["poll_after_seconds"] is None


def test_get_turn_returns_partial_progress() -> None:
    partial = _turn(
        "in_progress",
        messages=[
            {
                "role": "assistant",
                "content": "I am checking the eval traces now.",
                "message": {},
            }
        ],
        tool_calls=[
            {
                "type": "invocation",
                "name": "query_weave",
                "call_id": "call-1",
                "timestamp": "2026-08-07T12:00:01Z",
                "arguments": {"project": "evals"},
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=partial, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert result["ok"] is True
    assert result["state"] == "in_progress"
    assert result["latest_response"] == "I am checking the eval traces now."
    assert result["progress"] == {
        "message_count": 1,
        "tool_call_record_count": 1,
        "latest_tool_activity": {
            "type": "invocation",
            "name": "query_weave",
            "call_id": "call-1",
            "timestamp": "2026-08-07T12:00:01Z",
        },
    }
    assert result["turn"] == partial


def test_get_turn_service_error_is_retryable_and_keeps_handle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": "worker pool unavailable"},
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert result["ok"] is False
    assert result["turn_id"] == "turn-1"
    assert result["error"] == {
        "type": "service_unavailable",
        "message": "ARIA is currently unavailable or failed to process the request.",
        "retryable": True,
        "status_code": 503,
        "details": "worker pool unavailable",
    }
    assert "same turn_id" in result["next_action"]


def test_send_service_error_is_not_automatically_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message("Analyze runs", api_key="token", client=client)

    result = asyncio.run(run())

    assert result["ok"] is False
    assert result["error"]["type"] == "service_unavailable"
    assert result["error"]["retryable"] is False
    assert "outcome may be unknown" in result["next_action"]


def test_auth_rejection_is_legible_and_does_not_expose_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"detail": "entity is not enabled"},
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Analyze runs",
                api_key="super-secret-token",
                client=client,
            )

    result = asyncio.run(run())
    serialized = json.dumps(result)

    assert result["error"]["type"] == "authentication_error"
    assert result["error"]["status_code"] == 403
    assert result["error"]["retryable"] is False
    assert "super-secret-token" not in serialized


def test_wait_is_capped() -> None:
    result = asyncio.run(aria.get_aria_turn("turn-1", wait_seconds=31, api_key="token"))

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"
    assert "between 0 and 30" in result["error"]["message"]
