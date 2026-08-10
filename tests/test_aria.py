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
    assert result["scope"] == {"entity": "team", "project": "project"}
    assert "turn" not in result


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


def test_send_project_only_resolves_default_entity(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    async def resolve_default_entity(api_key: str) -> str:
        assert api_key == "token"
        return "default-team"

    def handler(request: httpx.Request) -> httpx.Response:
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
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert captured == {
        "user_prompt": "Inspect this project",
        "entity": "default-team",
        "project": "evals",
    }
    assert result["scope"] == {"entity": "default-team", "project": "evals"}


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


def test_get_turns_fetches_concurrently_and_preserves_order() -> None:
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        turn_id = request.url.path.rsplit("/", 1)[-1]
        active -= 1
        return httpx.Response(
            200,
            json=_turn("completed", id=turn_id, thread_id=f"thread-{turn_id}"),
            request=request,
        )

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turns(
                ["turn-3", "turn-1", "turn-2"],
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert result["ok"] is True
    assert max_active == 3
    assert [item["turn_id"] for item in result["results"]] == ["turn-3", "turn-1", "turn-2"]
    assert result["terminal_count"] == 3
    assert result["pending_count"] == 0
    assert result["error_count"] == 0


def test_get_turns_uses_one_shared_polling_window(monkeypatch) -> None:
    calls: Dict[str, int] = {"turn-1": 0, "turn-2": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        turn_id = request.url.path.rsplit("/", 1)[-1]
        calls[turn_id] += 1
        state = "queued" if calls[turn_id] == 1 else "completed"
        return httpx.Response(200, json=_turn(state, id=turn_id), request=request)

    monkeypatch.setattr(aria, "POLL_INTERVAL_SECONDS", 0)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turns(
                ["turn-1", "turn-2"],
                wait_seconds=1,
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert calls == {"turn-1": 2, "turn-2": 2}
    assert result["terminal_count"] == 2


def test_get_turns_returns_successes_with_per_turn_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        turn_id = request.url.path.rsplit("/", 1)[-1]
        if turn_id == "missing":
            return httpx.Response(404, json={"detail": "not found"}, request=request)
        return httpx.Response(200, json=_turn("completed", id=turn_id), request=request)

    async def run() -> Dict[str, Any]:
        async with _client(httpx.MockTransport(handler)) as client:
            return await aria.get_aria_turns(
                ["turn-1", "missing"],
                api_key="token",
                client=client,
            )

    result = asyncio.run(run())

    assert result["ok"] is True
    assert result["terminal_count"] == 1
    assert result["error_count"] == 1
    assert result["failed_turn_ids"] == ["missing"]
    assert result["results"][1]["error"]["type"] == "turn_not_found"
