import asyncio
from importlib.metadata import version as distribution_version
import json
import logging
import time
from typing import Any, Dict

import httpx
import pytest

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools import aria


def _turn(state: str = "queued", **overrides: Any) -> Dict[str, Any]:
    turn: Dict[str, Any] = {
        "id": "turn-1",
        "thread_id": "thread-1",
        "parent_turn_id": None,
        "wandb_entity": "team",
        "wandb_project": "project",
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
    requests: list[tuple[str, bytes, Dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.method, request.url.raw_path, body))
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": "thread/1"}, request=request)
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["Authorization"]
        captured["wandb_client"] = request.headers["X-Wandb-Client"]
        captured["body"] = body
        return httpx.Response(202, json=_turn(thread_id="thread/1"), request=request)

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
            title="Eval regressions",
            api_key="user-wandb-token",
        )
    )

    assert captured == {
        "method": "POST",
        "path": "/api/v1/turns",
        "authorization": "Bearer user-wandb-token",
        "wandb_client": aria.ARIA_CLIENT_IDENTITY,
        "body": {
            "user_prompt": [
                {"type": "text", "text": "Compare the latest eval runs"},
                {"type": "wandb_client", "platform": "agent"},
            ],
            "entity": "team",
            "project": "evals",
        },
    }
    assert requests == [
        (
            "POST",
            b"/api/v1/turns",
            {
                "user_prompt": [
                    {"type": "text", "text": "Compare the latest eval runs"},
                    {"type": "wandb_client", "platform": "agent"},
                ],
                "entity": "team",
                "project": "evals",
            },
        ),
        ("PATCH", b"/api/v1/threads/thread%2F1", {"title": "Eval regressions"}),
    ]
    assert result["ok"] is True
    assert result["turn_id"] == "turn-1"
    assert result["title"] == "Eval regressions"
    assert result["state"] == "queued"
    assert result["is_terminal"] is False
    assert result["poll_after_seconds"] == 2
    assert result["scope"] == {"entity": "team", "project": "project"}
    assert "turn" not in result


@pytest.mark.parametrize("title", [None, "   "])
def test_send_root_turn_requires_title_without_contacting_service(title) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            kwargs = {} if title is None else {"title": title}
            return await aria.send_aria_message("Start a conversation", api_key="token", client=client, **kwargs)

    result = asyncio.run(run())

    assert calls == 0
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"
    assert result["error"]["retryable"] is False


def test_send_rejects_title_on_continuation_without_contacting_service() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Continue the conversation",
                title="Should not be supplied",
                parent_turn_id="turn-0",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == 0
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"


def test_send_rejects_title_longer_than_255_characters_without_contacting_service() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Start a conversation",
                title="t" * 256,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == 0
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"
    assert "255" in result["error"]["message"]


def test_send_title_patch_failure_keeps_durable_turn_handle() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json=_turn(), request=request)
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Start a conversation",
                title="A useful title",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == ["POST", "PATCH"]
    assert result["ok"] is True
    assert result["turn_id"] == "turn-1"
    assert result["title_error"] == "service_unavailable"
    assert "error" not in result


def test_send_poll_failure_preserves_title_patch_error() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(202, json=_turn(), request=request)
        if request.method == "PATCH":
            return httpx.Response(500, request=request)
        return httpx.Response(429, json={"detail": "capacity"}, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Start a conversation",
                title="A useful title",
                wait_seconds=2,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == ["POST", "PATCH", "GET"]
    assert result["ok"] is False
    assert result["turn_id"] == "turn-1"
    assert result["title_error"] == "service_unavailable"
    assert result["error"]["type"] == "rate_limited"


def test_send_title_patch_is_skipped_when_post_omits_thread_id() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        turn = _turn()
        turn.pop("thread_id")
        return httpx.Response(202, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Start a conversation",
                title="A useful title",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == ["POST"]
    assert result["ok"] is True
    assert result["turn_id"] == "turn-1"
    assert result["title_error"] == "missing_thread_id"


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
        result = asyncio.run(aria.send_aria_message("Inspect my runs", title="Inspect my runs"))
    finally:
        WandBApiManager.reset_context_api_key(context_token)

    assert result["ok"] is True
    assert captured["authorization"] == "Bearer context-wandb-token-123"


def test_every_aria_request_overrides_client_attribution_header() -> None:
    captured: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.method, request.headers["X-Wandb-Client"]))
        return httpx.Response(200, json=_turn("completed"), request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="https://wb-agent.example",
            headers={"X-Wandb-Client": "caller-controlled"},
            transport=httpx.MockTransport(handler),
        ) as client:
            await aria._request_turn(client, "POST", "/api/v1/turns", payload={"user_prompt": "safe"})
            await aria._request_turn(client, "GET", "/api/v1/turns/turn-1")

    asyncio.run(run())

    assert captured == [
        ("POST", aria.ARIA_CLIENT_IDENTITY),
        ("GET", aria.ARIA_CLIENT_IDENTITY),
    ]
    assert aria.ARIA_CLIENT_IDENTITY == f"wandb-mcp-server/{distribution_version('wandb-mcp-server')}"


def test_concurrent_callers_keep_request_scoped_tokens_isolated(monkeypatch) -> None:
    captured: Dict[str, str] = {}

    def new_client(api_key: str) -> httpx.AsyncClient:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "PATCH":
                return httpx.Response(200, json={"id": "thread-1"}, request=request)
            prompt = json.loads(request.content)["user_prompt"][0]["text"]
            await asyncio.sleep(0)
            captured[prompt] = request.headers["Authorization"]
            return httpx.Response(202, json=_turn(id=f"turn-{prompt}"), request=request)

        return httpx.AsyncClient(
            base_url="https://wb-agent.example",
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(aria, "_new_http_client", new_client)

    async def call(prompt: str, token: str) -> Dict[str, Any]:
        context_token = WandBApiManager.set_context_api_key(token)
        try:
            return await aria.send_aria_message(prompt, title=f"Conversation for {prompt}")
        finally:
            WandBApiManager.reset_context_api_key(context_token)

    async def run() -> list[Dict[str, Any]]:
        return list(
            await asyncio.gather(
                call("caller-a", "caller-a-secret-token"),
                call("caller-b", "caller-b-secret-token"),
            )
        )

    results = asyncio.run(run())

    assert all(result["ok"] is True for result in results)
    assert captured == {
        "caller-a": "Bearer caller-a-secret-token",
        "caller-b": "Bearer caller-b-secret-token",
    }


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
        "user_prompt": [
            {"type": "text", "text": "Now limit that to failed runs"},
            {"type": "wandb_client", "platform": "agent"},
        ],
        "parent_turn_id": "turn-0",
    }
    assert result["parent_turn_id"] == "turn-0"


def test_send_project_only_resolves_default_entity(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    async def resolve_default_entity(api_key: str) -> str:
        assert api_key == "token"
        return "default-team"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": "thread-1"}, request=request)
        captured.update(json.loads(request.content))
        return httpx.Response(
            202,
            json=_turn(wandb_entity="default-team", wandb_project="evals"),
            request=request,
        )

    monkeypatch.setattr(aria, "_resolve_default_entity", resolve_default_entity)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Inspect this project",
                project="evals",
                title="Inspect this project",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert captured == {
        "user_prompt": [
            {"type": "text", "text": "Inspect this project"},
            {"type": "wandb_client", "platform": "agent"},
        ],
        "entity": "default-team",
        "project": "evals",
    }
    assert result["scope"] == {"entity": "default-team", "project": "evals"}


def test_default_entity_resolution_uses_bounded_sync_dispatch(monkeypatch) -> None:
    class FakeApi:
        default_entity = "default-team"

    calls: list[Any] = []

    async def run_sync(call):
        calls.append(call)
        return call()

    monkeypatch.setattr(WandBApiManager, "get_api", lambda api_key: FakeApi())
    monkeypatch.setattr(
        "wandb_mcp_server.instrumented_server.run_sync_in_current_tool",
        run_sync,
    )

    assert asyncio.run(aria._resolve_default_entity("token")) == "default-team"
    assert len(calls) == 1


def test_send_preserves_confirmed_turn_handle_when_followup_poll_fails() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.method == "POST":
            return httpx.Response(202, json=_turn(), request=request)
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": "thread-1"}, request=request)
        return httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"detail": "capacity"},
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Analyze runs",
                title="Analyze runs",
                wait_seconds=2,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == 3
    assert result == {
        "ok": False,
        "error": {
            "type": "rate_limited",
            "message": "ARIA is rate limiting requests.",
            "retryable": True,
            "status_code": 429,
            "retry_after_ms": 7_000,
            "details": "capacity",
        },
        "next_action": "Retry aria_get_turn with the same turn_id; the ARIA turn remains server-side.",
        "turn_id": "turn-1",
        "title": "Analyze runs",
    }


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
        if request.method == "PATCH":
            return httpx.Response(200, json={"id": "thread-1"}, request=request)
        return httpx.Response(200, json=responses.pop(0), request=request)

    monkeypatch.setattr(aria, "POLL_INTERVAL_SECONDS", 0)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Diagnose the regression",
                title="Diagnose the regression",
                wait_seconds=2,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert paths == [
        "/api/v1/turns",
        "/api/v1/threads/thread-1",
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
        "tool_call_records_inspected": 1,
        "tool_error_count_exact": True,
        "latest_tool_activity": {
            "type": "invocation",
            "name": "query_weave",
            "call_id": "call-1",
            "timestamp": "2026-08-07T12:00:01Z",
        },
        "phase": "responding",
        "status_text": "ARIA has produced a partial response but is still working; keep polling.",
    }
    assert "turn" not in result


def test_get_turn_can_include_full_snapshot() -> None:
    turn = _turn("completed", messages=[{"role": "assistant", "content": "Done"}])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn(
                "turn-1",
                include_turn=True,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert result["turn"] == turn


def test_compact_poll_omits_large_internal_snapshot_fields() -> None:
    turn = _turn(
        "in_progress",
        messages=[{"role": "assistant", "content": "Partial answer"}],
        tool_calls=[
            {
                "type": "invocation",
                "name": "shell",
                "call_id": "call-1",
                "arguments": {"encrypted_reasoning": "x" * 100_000},
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())
    serialized = json.dumps(result)

    assert "turn" not in result
    assert "encrypted_reasoning" not in serialized
    assert len(serialized) < 2_000


def test_latest_response_survives_more_than_two_hundred_trailing_records() -> None:
    turn = _turn(
        "completed",
        messages=[{"role": "assistant", "content": "final answer"}]
        + [{"role": "tool", "content": "done"} for _ in range(250)],
    )

    assert aria._latest_assistant_response(turn) == "final answer"


def test_compact_progress_surfaces_recovered_internal_tool_errors() -> None:
    turn = _turn(
        "in_progress",
        tool_calls=[
            {"type": "invocation", "name": "shell", "call_id": "call-1"},
            {
                "type": "response",
                "name": "shell",
                "call_id": "call-1",
                "is_error": True,
                "response": {"stderr": "x" * 100_000},
            },
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert result["progress"]["tool_error_count"] == 1
    assert "continued working" in result["progress"]["internal_error_note"]
    assert result["progress"]["status_text"].startswith("ARIA's latest shell attempt failed")
    assert "stderr" not in json.dumps(result)


def test_compact_progress_detects_nonzero_executor_exit_code() -> None:
    turn = _turn(
        "in_progress",
        tool_calls=[
            {"type": "invocation", "name": "shell", "call_id": "call-1"},
            {
                "type": "response",
                "name": "shell",
                "call_id": "call-1",
                "is_error": False,
                "response": {
                    "result": {
                        "exit_code": 7,
                        "stdout": "recovery details " * 10_000,
                    }
                },
            },
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert result["progress"]["tool_error_count"] == 1
    assert "executor error" in result["progress"]["internal_error_note"]
    assert result["progress"]["latest_tool_activity"] == {
        "type": "response",
        "name": "shell",
        "call_id": "call-1",
        "is_error": False,
        "recovered_error": True,
    }
    assert result["progress"]["status_text"].startswith("ARIA's latest shell attempt failed")
    assert "stdout" not in json.dumps(result)


def test_compact_progress_does_not_treat_zero_exit_code_as_error() -> None:
    turn = _turn(
        "in_progress",
        tool_calls=[
            {
                "type": "response",
                "name": "shell",
                "call_id": "call-1",
                "is_error": False,
                "response": {"exit_code": 0},
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert "tool_error_count" not in result["progress"]
    assert "internal_error_note" not in result["progress"]
    assert "recovered_error" not in result["progress"]["latest_tool_activity"]
    assert result["progress"]["status_text"].startswith("ARIA received a result")


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
            return await aria.send_aria_message(
                "Analyze runs",
                title="Analyze runs",
                api_key="token",
                client=client,
            )

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
                title="Analyze runs",
                api_key="super-secret-token",
                client=client,
            )

    result = asyncio.run(run())
    serialized = json.dumps(result)

    assert result["error"]["type"] == "authentication_error"
    assert result["error"]["status_code"] == 403
    assert result["error"]["retryable"] is False
    assert "details" not in result["error"]
    assert "entity is not enabled" not in serialized
    assert "super-secret-token" not in serialized


def test_validation_details_remain_structured() -> None:
    validation_details = [
        {
            "type": "value_error",
            "loc": ["body", "project"],
            "msg": "entity must be provided when project is provided",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": validation_details}, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Analyze runs",
                entity="team",
                project="project",
                title="Analyze runs",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert result["error"]["details"] == validation_details
    assert isinstance(result["error"]["details"], list)


def test_wait_is_capped() -> None:
    result = asyncio.run(aria.get_aria_turn("turn-1", wait_seconds=31, api_key="token"))

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_request"
    assert "between 0 and 30" in result["error"]["message"]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://wb-agent.example",
        "wb-agent.example",
        "https://user:password@wb-agent.example",
        "https://wb-agent.example?token=secret",
        "https://wb-agent.example#fragment",
        "https://wb-agent.example/api",
    ],
)
def test_new_client_rejects_unsafe_base_urls(monkeypatch, base_url: str) -> None:
    monkeypatch.delenv("WB_AGENT_BASE_URL", raising=False)
    monkeypatch.setattr(aria, "WB_AGENT_BASE_URL", base_url)

    with pytest.raises(ValueError, match="WB_AGENT_BASE_URL"):
        aria._new_http_client("token")


def test_new_client_disables_redirects(monkeypatch) -> None:
    monkeypatch.delenv("WB_AGENT_BASE_URL", raising=False)
    monkeypatch.setattr(aria, "WB_AGENT_BASE_URL", "https://wb-agent.example/")
    client = aria._new_http_client("token")
    try:
        assert client.follow_redirects is False
        assert str(client.base_url) == "https://wb-agent.example"
    finally:
        asyncio.run(client.aclose())


def test_new_client_honors_safe_base_url_configured_after_import(monkeypatch) -> None:
    monkeypatch.setattr(aria, "WB_AGENT_BASE_URL", "https://stale.example")
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://late-config.example/")

    client = aria._new_http_client("token")
    try:
        assert str(client.base_url) == "https://late-config.example"
    finally:
        asyncio.run(client.aclose())


def test_send_does_not_follow_redirect_or_retry_post() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            307,
            headers={"Location": "https://attacker.example/collect"},
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Analyze runs",
                title="Analyze runs",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == 1
    assert result["ok"] is False
    assert result["error"]["type"] == "upstream_error"


@pytest.mark.parametrize("exception_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_ambiguous_send_transport_failure_returns_outcome_unknown(exception_type) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception_type("ambiguous transport failure", request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "Analyze runs",
                title="Analyze runs",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == 1
    assert result["error"] == {
        "type": "outcome_unknown",
        "message": "ARIA did not confirm whether the message submission created a turn.",
        "retryable": False,
    }
    assert "do not retry blindly" in result["next_action"]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"message": "x" * (32 * 1024 + 1), "title": "A conversation"}, "message"),
        ({"message": "😀" * 8193, "title": "A conversation"}, "message"),
        ({"message": "ok", "entity": "e" * 513, "title": "A conversation"}, "entity"),
        ({"message": "ok", "project": "p" * 513, "title": "A conversation"}, "project"),
        ({"message": "ok", "parent_turn_id": "t" * 513}, "parent_turn_id"),
    ],
)
def test_send_rejects_oversized_inputs_without_contacting_service(kwargs, field: str) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(api_key="token", client=client, **kwargs)

    result = asyncio.run(run())

    assert called is False
    assert result["error"]["type"] == "invalid_request"
    assert field in result["error"]["message"]


def test_get_rejects_oversized_identifiers_without_contacting_service() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    async def run() -> tuple[Dict[str, Any], Dict[str, Any]]:
        async with _client(httpx.MockTransport(handler)) as client:
            single = await aria.get_aria_turn("t" * 513, api_key="token", client=client)
            thread = await aria.get_aria_thread("t" * 513, api_key="token", client=client)
            return single, thread

    single, thread = asyncio.run(run())

    assert called is False
    assert single["error"]["type"] == "invalid_request"
    assert thread["error"]["type"] == "invalid_request"


@pytest.mark.parametrize(
    "turn",
    [
        _turn(id="t" * 513),
        _turn(id=123),
        _turn(state="s" * 65),
        _turn(state="waiting_on_unknown_backend_phase"),
        _turn(state=123),
    ],
)
def test_rejects_invalid_upstream_turn_identifiers_before_polling(turn: Dict[str, Any]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=turn, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", wait_seconds=2, api_key="token", client=client)

    result = asyncio.run(run())
    assert calls == 1
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_response"


def test_single_get_rejects_mismatched_upstream_turn_without_exposing_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_turn(
                "completed",
                id="different-turn",
                messages=[{"role": "assistant", "content": "wrong turn secret"}],
            ),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn(
                "requested-turn",
                include_turn=True,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())
    serialized = json.dumps(result)
    assert result["ok"] is False
    assert result["turn_id"] == "requested-turn"
    assert result["error"]["type"] == "invalid_response"
    assert "different-turn" not in serialized
    assert "wrong turn secret" not in serialized


def test_concurrent_callers_share_one_eight_request_outbound_limit() -> None:
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            turn_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=_turn("completed", id=turn_id), request=request)
        finally:
            active -= 1

    async def run() -> list[Dict[str, Any]]:
        async with _client(httpx.MockTransport(handler)) as client_a:
            async with _client(httpx.MockTransport(handler)) as client_b:
                clients = (client_a, client_b)
                return list(
                    await asyncio.gather(
                        *(
                            aria.get_aria_turn(f"turn-{index}", api_key="token", client=clients[index % 2])
                            for index in range(20)
                        )
                    )
                )

    results = asyncio.run(run())
    assert all(result["ok"] is True for result in results)
    assert max_active == 8


@pytest.mark.parametrize("value", ["nan", "inf", "1e10000"])
def test_nonfinite_retry_after_never_escapes_stable_error_mapping(value: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": value},
            json={"detail": "capacity"},
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())
    assert result["ok"] is False
    assert result["error"]["type"] == "rate_limited"
    assert "retry_after_ms" not in result["error"]


def test_poll_returns_pending_before_outer_tool_deadline(monkeypatch) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls > 1:
            await asyncio.sleep(1)
        return httpx.Response(200, json=_turn("queued"), request=request)

    monkeypatch.setattr(aria, "POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(aria, "POLL_DEADLINE_SAFETY_SECONDS", 0.05)

    async def run() -> Dict[str, Any]:
        deadline_token = aria.current_tool_deadline.set(time.monotonic() + 0.1)
        try:
            async with _client(httpx.MockTransport(handler)) as client:
                return await aria.get_aria_turn(
                    "turn-1",
                    wait_seconds=30,
                    api_key="token",
                    client=client,
                )
        finally:
            aria.current_tool_deadline.reset(deadline_token)

    started = time.monotonic()
    result = asyncio.run(run())

    assert time.monotonic() - started < 0.5
    assert calls == 2
    assert result["ok"] is True
    assert result["state"] == "queued"
    assert result["is_terminal"] is False


def test_include_turn_is_sanitized_and_truncated_to_response_budget(monkeypatch) -> None:
    api_key = "explicit-super-secret-token"
    internal_url = "http://wandb-api.default.svc.cluster.local:8081"
    turn = _turn(
        "completed",
        messages=[
            {
                "role": "assistant",
                "content": f"{api_key} {internal_url} " + "large-result " * 20_000,
            }
        ],
        debug={"authorization": f"Bearer {api_key}"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=turn, request=request)

    monkeypatch.setattr(aria, "MAX_RESPONSE_TOKENS", 500)
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", internal_url)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn(
                "turn-1",
                include_turn=True,
                api_key=api_key,
                client=client,
            )

    result = asyncio.run(run())
    serialized = json.dumps(result)

    assert aria._estimate_response_tokens(result) <= 500
    assert result["_truncation"]["applied"] is True
    assert result["_truncation"]["reason"] == "response_token_budget"
    assert result["_truncation"].get("omitted_fields") or result["_truncation"].get("compaction")
    assert result["turn"]["_truncated"] is True
    assert api_key not in serialized
    assert "wandb-api.default.svc" not in serialized
    assert "cluster.local" not in serialized


def test_minimum_enabled_response_budget_is_still_a_hard_limit(monkeypatch) -> None:
    monkeypatch.setattr(aria, "MAX_RESPONSE_TOKENS", 64)
    result = aria._finalize_result(
        {
            "ok": True,
            "turn_id": "turn-1",
            "state": "completed",
            "latest_response": "x" * 50_000,
        }
    )

    assert result["_truncation"]["applied"] is True
    assert aria._estimate_response_tokens(result) <= 64


def test_finalize_normalizes_all_nonfinite_values_to_json_safe_sentinels() -> None:
    result = aria._finalize_result(
        {
            "ok": True,
            "turn_id": "turn-1",
            "turn": {
                "metrics": [float("nan"), float("inf"), float("-inf"), 1.5],
                "nested": {"error_info": {"score": float("nan")}},
            },
        }
    )

    # `allow_nan=False` is the strict JSON guarantee used at the MCP boundary.
    json.dumps(result, allow_nan=False)
    assert result["turn"]["metrics"] == [
        "<non-finite float: NaN>",
        "<non-finite float: Infinity>",
        "<non-finite float: -Infinity>",
        1.5,
    ]
    assert result["turn"]["nested"]["error_info"]["score"] == "<non-finite float: NaN>"


def test_upstream_nonfinite_values_are_consistent_in_compact_and_included_turn() -> None:
    turn = _turn(
        "completed",
        score=float("nan"),
        messages=[
            {
                "role": "assistant",
                "content": "done",
                "metrics": {"positive": float("inf"), "negative": float("-inf")},
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # Python's decoder accepts these legacy service values; the MCP result
        # still has to normalize them before strict JSON serialization.
        body = json.dumps(turn, allow_nan=True).encode()
        return httpx.Response(200, content=body, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn(
                "turn-1",
                include_turn=True,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())
    json.dumps(result, allow_nan=False)
    assert result["ok"] is True
    assert result["turn"]["score"] == "<non-finite float: NaN>"
    assert result["turn"]["messages"][0]["metrics"] == {
        "positive": "<non-finite float: Infinity>",
        "negative": "<non-finite float: -Infinity>",
    }


def test_upstream_response_body_is_bounded_before_json_decode(monkeypatch) -> None:
    monkeypatch.setattr(aria, "MAX_UPSTREAM_RESPONSE_BYTES", 128)

    class ChunkedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{" + b'"payload":"'
            yield b"x" * 256
            yield b'"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=ChunkedBody(),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())
    assert result["ok"] is False
    assert result["error"]["type"] == "response_too_large"
    assert result["error"]["retryable"] is False


def test_compressed_get_response_is_rejected_before_decoding() -> None:
    captured_accept_encoding = None

    class CompressedWireBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"compressed-body-must-not-be-decoded"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_accept_encoding
        captured_accept_encoding = request.headers.get("Accept-Encoding")
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=CompressedWireBody(),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key="token", client=client)

    result = asyncio.run(run())
    assert captured_accept_encoding == "identity"
    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_response"


def test_compressed_post_response_has_unknown_outcome() -> None:
    class CompressedWireBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"compressed-body-must-not-be-decoded"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            202,
            headers={"Content-Encoding": "br"},
            stream=CompressedWireBody(),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.send_aria_message(
                "do work",
                title="Do work",
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())
    assert result["ok"] is False
    assert result["error"]["type"] == "outcome_unknown"
    assert result["error"]["retryable"] is False


def test_http_logs_hide_internal_aria_origin_for_post_and_get(caplog) -> None:
    internal_host = "aria.private.svc.cluster.local"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_turn("completed"), request=request)

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url=f"https://{internal_host}",
            transport=httpx.MockTransport(handler),
        ) as client:
            await aria.send_aria_message(
                "safe synthetic prompt",
                title="Safe synthetic prompt",
                api_key="token",
                client=client,
            )
            await aria.get_aria_turn("turn-1", api_key="token", client=client)

    caplog.set_level(logging.INFO, logger="httpx")
    asyncio.run(run())
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "httpx")
    assert "POST" in logs
    assert "GET" in logs
    assert "<aria-service>/api/v1/turns" in logs
    assert internal_host not in logs
    assert "turn-1" not in logs


def test_upstream_error_sanitizes_request_secret_and_internal_host(monkeypatch) -> None:
    api_key = "explicit-super-secret-token"
    internal_url = "http://wandb-api.default.svc.cluster.local:8081"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            headers={"Retry-After": "2"},
            json={"detail": (f"Bearer {api_key} failed through {internal_url}; token={api_key}")},
            request=request,
        )

    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", internal_url)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key=api_key, client=client)

    result = asyncio.run(run())
    serialized = json.dumps(result)

    assert result["error"]["retry_after_ms"] == 2_000
    assert api_key not in serialized
    assert "wandb-api.default.svc" not in serialized
    assert "cluster.local" not in serialized


def test_upstream_error_sanitizes_canaries_before_boundary_truncation(monkeypatch) -> None:
    api_key = "boundary-super-secret-token"
    internal_url = "http://wandb-api.default.svc.cluster.local:8081"
    prefix = "x" * 995

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": f"{prefix}{api_key} {prefix}{internal_url}"},
            request=request,
        )

    monkeypatch.setenv("WANDB_API_KEY", api_key)
    monkeypatch.setenv("WANDB_INTERNAL_BASE_URL", internal_url)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turn("turn-1", api_key=api_key, client=client)

    serialized = json.dumps(asyncio.run(run()))
    assert api_key not in serialized
    assert "wandb-api.default.svc" not in serialized
    assert "cluster.local" not in serialized


def test_compact_progress_inspection_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(aria, "MAX_PROGRESS_TOOL_RECORDS", 3)
    turn = _turn(
        "in_progress",
        tool_calls=[{"type": "response", "result": {"exit_code": 1}} for _ in range(10)],
    )

    progress = aria._progress_summary(turn)
    assert progress["tool_call_record_count"] == 10
    assert progress["tool_call_records_inspected"] == 3
    assert progress["tool_error_count"] == 3
    assert progress["tool_error_count_exact"] is False
    assert "at least 3" in progress["internal_error_note"]


def test_logs_omit_prompt_scope_and_turn_identifiers(caplog) -> None:
    prompt = "private prompt canary"
    entity = "private-entity-canary"
    project = "private-project-canary"
    turn_id = "private-turn-id-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_turn("completed", id=turn_id, wandb_entity=entity, wandb_project=project),
            request=request,
        )

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            await aria.send_aria_message(
                prompt,
                entity=entity,
                project=project,
                title="Private conversation",
                api_key="token",
                client=client,
            )
            await aria.get_aria_turn(turn_id, api_key="token", client=client)

    caplog.set_level("INFO")
    asyncio.run(run())

    assert prompt not in caplog.text
    assert entity not in caplog.text
    assert project not in caplog.text
    assert turn_id not in caplog.text
    assert "/api/v1/turns/<redacted>" in caplog.text


def _thread(**overrides: Any) -> Dict[str, Any]:
    thread: Dict[str, Any] = {
        "id": "thread-1",
        "title": "Eval regressions",
        "prompt_preview": "Compare the latest eval runs",
        "wandb_entity": "team",
        "wandb_project": "evals",
        "is_pinned": False,
        "visibility": "private",
        "created_at": "2026-08-01T12:00:00Z",
        "last_turn_created_at": "2026-08-07T12:00:00Z",
        "internal_debug": "not exposed",
    }
    thread.update(overrides)
    return thread


def test_list_threads_sends_scoped_page_and_returns_compact_summaries() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["wandb_client"] = request.headers["X-Wandb-Client"]
        return httpx.Response(
            200,
            json=[_thread(), _thread(id="thread-2", title="", prompt_preview="p" * 600)],
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.list_aria_threads(
                entity="team", project="evals", limit=2, offset=4, api_key="token", client=client
            )

    result = asyncio.run(run())

    assert captured == {
        "method": "GET",
        "path": "/api/v1/threads",
        "params": {"entity": "team", "project": "evals", "limit": "2", "offset": "4", "sort_desc": "true"},
        "wandb_client": aria.ARIA_CLIENT_IDENTITY,
    }
    assert result["ok"] is True
    assert [thread["thread_id"] for thread in result["threads"]] == ["thread-1", "thread-2"]
    assert result["threads"][0]["scope"] == {"entity": "team", "project": "evals"}
    assert result["threads"][0]["title"] == "Eval regressions"
    assert result["threads"][0]["prompt_preview"] == "Compare the latest eval runs"
    assert result["threads"][1]["title"] is None
    assert result["threads"][1]["prompt_preview"] == "p" * 500 + "..."
    assert "internal_debug" not in json.dumps(result)
    assert result["has_more"] is True
    assert result["next_offset"] == 6
    assert "aria_get_thread" in result["next_action"]


def test_list_threads_last_page_has_no_next_offset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_thread()], request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.list_aria_threads(api_key="token", client=client)

    result = asyncio.run(run())

    assert result["count"] == 1
    assert result["has_more"] is False
    assert result["next_offset"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": 51}, "limit"),
        ({"limit": True}, "limit"),
        ({"offset": -1}, "offset"),
        ({"offset": 10_001}, "offset"),
        ({"entity": "e" * 513}, "entity"),
    ],
)
def test_list_threads_rejects_invalid_inputs_without_contacting_service(kwargs: Dict[str, Any], message: str) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.list_aria_threads(api_key="token", client=client, **kwargs)

    result = asyncio.run(run())

    assert called is False
    assert result["error"]["type"] == "invalid_request"
    assert message in result["error"]["message"]


@pytest.mark.parametrize(
    "body",
    [{"threads": []}, [{"title": "missing id"}], [_thread(id="t" * 513)], [_thread()] * 21],
)
def test_list_threads_rejects_invalid_upstream_lists(body: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.list_aria_threads(api_key="token", client=client)

    result = asyncio.run(run())

    assert result["ok"] is False
    assert result["error"]["type"] == "invalid_response"


def test_get_thread_returns_summary_and_newest_first_turn_metadata() -> None:
    requests: list[tuple[str, Dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.raw_path.decode().split("?")[0], dict(request.url.params)))
        if request.url.path.endswith("/turns"):
            return httpx.Response(
                200,
                json=[
                    _turn("in_progress", id="turn-3", parent_turn_id="turn-2", user_prompt=""),
                    _turn("completed", id="turn-2", parent_turn_id="turn-1", messages=[{"secret": "answer"}]),
                    _turn("completed", id="turn-1"),
                ],
                request=request,
            )
        return httpx.Response(200, json=_thread(id="thread/1"), request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_thread("thread/1", limit=3, api_key="token", client=client)

    result = asyncio.run(run())

    assert requests == [
        ("/api/v1/threads/thread%2F1", {}),
        (
            "/api/v1/threads/thread%2F1/turns",
            {"field_set": "metadata", "sort_by": "created_at", "sort_desc": "true", "limit": "3", "offset": "0"},
        ),
    ]
    assert result["ok"] is True
    assert result["thread"]["thread_id"] == "thread/1"
    assert [turn["turn_id"] for turn in result["turns"]] == ["turn-3", "turn-2", "turn-1"]
    assert result["turns"][0]["is_terminal"] is False
    assert result["turns"][1]["parent_turn_id"] == "turn-1"
    assert "answer" not in json.dumps(result["turns"])
    assert result["newest_completed_turn_id"] == "turn-2"
    assert "parent_turn_id=turn-2" in result["next_action"]
    assert result["has_more"] is True
    assert result["next_offset"] == 3


def test_get_thread_rejects_mismatched_thread_without_fetching_turns() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=_thread(id="other-thread", title="secret title"), request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_thread("thread-1", api_key="token", client=client)

    result = asyncio.run(run())

    assert paths == ["/api/v1/threads/thread-1"]
    assert result["error"]["type"] == "invalid_response"
    assert "other-thread" not in json.dumps(result)
    assert "secret title" not in json.dumps(result)


def test_get_thread_maps_missing_thread_and_invalid_turns() -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "nope"}, request=request)

    def bad_turns(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/turns"):
            return httpx.Response(200, json=[_turn("mystery_state")], request=request)
        return httpx.Response(200, json=_thread(), request=request)

    async def run(handler: Any) -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_thread("thread-1", api_key="token", client=client)

    assert asyncio.run(run(missing))["error"]["type"] == "thread_not_found"
    assert asyncio.run(run(bad_turns))["error"]["type"] == "invalid_response"


def test_thread_http_logs_hide_scope_and_identifiers(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/threads":
            return httpx.Response(200, json=[], request=request)
        if request.url.path.endswith("/turns"):
            return httpx.Response(200, json=[], request=request)
        return httpx.Response(200, json=_thread(id="private-thread-canary"), request=request)

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            await aria.list_aria_threads(entity="private-entity-canary", api_key="token", client=client)
            await aria.get_aria_thread("private-thread-canary", api_key="token", client=client)

    caplog.set_level(logging.INFO, logger="httpx")
    asyncio.run(run())
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "httpx")
    assert "<aria-service>/api/v1/threads" in logs
    assert "private-entity-canary" not in logs
    assert "private-thread-canary" not in logs
