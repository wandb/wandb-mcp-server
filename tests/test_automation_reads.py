"""Exercise real fixed Automation projections without SDK action filtering."""

import copy
import json
import time

import httpx
import pytest

from wandb_mcp_server import automation_reads as reads, wandb_graphql as transport
from wandb_mcp_server.admission import current_tool_deadline
from wandb_mcp_server.api_client import api_key_context
from wandb_mcp_server.mcp_tools.automations import list_automations


def node(kind="ARIATriggeredAction", scope="Project"):
    return {
        "id": "fixture",
        "name": "fixture",
        "enabled": True,
        "description": None,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": None,
        "scope": {"__typename": scope, "id": "scope", "name": "fixture"},
        "event": {
            "__typename": "FilterEventTriggeringCondition",
            "eventType": "RUN_STATE",
            "filter": json.dumps({"run_filter": {}, "run_state_filter": {"states": ["FAILED"]}}),
        },
        "action": {"__typename": kind, "prompt": "Synthetic prompt"},
    }


def page(rows, cursor=None, more=False):
    return {"edges": [{"node": r} for r in rows], "pageInfo": {"endCursor": cursor, "hasNextPage": more}}


def install(monkeypatch, pages):
    calls = []

    def execute(query, variables):
        transport.validate_read_only_graphql(query)
        calls.append((query, copy.deepcopy(variables)))
        conn = pages[len(calls) - 1]
        return {"entity": {"triggers": conn}} if "entity" in variables else {"triggers": conn}

    monkeypatch.setattr(reads, "execute_app_graphql", execute)
    return calls


@pytest.mark.parametrize("scope", ["Project", "Entity", "ArtifactSequence", "ArtifactPortfolio"])
@pytest.mark.parametrize(
    "kind,expected", [("ARIATriggeredAction", "ARIA"), ("PushNotificationTriggeredAction", "PUSH_NOTIFICATION")]
)
def test_new_actions_and_entity_scope_are_not_lost(monkeypatch, scope, kind, expected):
    calls = install(monkeypatch, [page([node(kind, scope)])])
    result = json.loads(list_automations(entity="team", name="fixture"))
    assert result["count"] == 1
    assert result["automations"][0]["action"]["type"] == expected
    assert json.loads(calls[0][1]["filters"]) == {"name": "fixture"}
    assert result["truncated"] is False


def test_viewer_lists_accessible_triggers_not_only_created_by_viewer(monkeypatch):
    calls = install(monkeypatch, [page([node()])])
    assert json.loads(list_automations())["count"] == 1
    assert "entity" not in calls[0][1]
    assert "viewer {" not in calls[0][0]


def test_new_backend_event_is_readable_without_sdk_enum_hydration(monkeypatch):
    row = node()
    row["event"]["eventType"] = "WEAVE_METRIC_THRESHOLD"
    row["event"]["filter"] = '{"threshold":2}'
    install(monkeypatch, [page([row])])
    result = json.loads(list_automations())
    assert result["automations"][0]["event"] == {
        "type": "WEAVE_METRIC_THRESHOLD",
        "filter": {"threshold": 2},
        "filter_format": "backend",
    }
    row["event"]["filter"] = "invalid-canary"
    install(monkeypatch, [page([row])])
    result = list_automations()
    assert json.loads(result)["error"] == "api_error" and "canary" not in result


def test_empty_authorization_page_then_real_rows_and_limit(monkeypatch):
    calls = install(monkeypatch, [page([], "next", True), page([node()], "last", True), page([node()])])
    result = json.loads(list_automations(entity="team", max_items=1))
    assert result["count"] == 1 and result["truncated"] is True
    assert [variables["after"] for _, variables in calls] == [None, "next", "last"]


@pytest.mark.parametrize(
    "pages",
    [
        [page([], None, True)],
        [page([], "x", True), page([], "x", True)],
        [page([], str(i), True) for i in range(8)],
        [{"edges": [None], "pageInfo": {"hasNextPage": False}}],
    ],
)
def test_incomplete_and_malformed_pages_fail_not_empty(monkeypatch, pages):
    calls = install(monkeypatch, pages)
    assert json.loads(list_automations(entity="team"))["error"] == "api_error"
    assert len(calls) <= 8


def test_name_mismatch_and_malformed_action_are_errors(monkeypatch):
    install(monkeypatch, [page([node()])])
    assert json.loads(list_automations(name="different"))["error"] == "api_error"
    row = node()
    row["action"]["prompt"] = {"secret-canary": "not-a-string"}
    install(monkeypatch, [page([row])])
    result = list_automations()
    assert json.loads(result)["error"] == "api_error" and "canary" not in result


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk


@pytest.fixture
def http(monkeypatch):
    real = httpx.Client
    calls = []
    token = api_key_context.set("credential-canary")

    def setup(*, status=200, headers=None, chunks=None):
        stream = Chunks(chunks or [b'{"data":{"triggers":{}}}'])

        def handler(request):
            calls.append(request)
            return httpx.Response(status, headers=headers, stream=stream)

        monkeypatch.setattr(
            transport.httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)
        )
        return calls, stream

    yield setup
    api_key_context.reset(token)


def test_transport_preserves_actor_and_internal_routing_without_sdk_user_agent(http, monkeypatch):
    monkeypatch.setattr(transport, "WANDB_API_BASE_URL", "https://internal.invalid")
    calls, _ = http()
    assert transport.execute_app_graphql("query { viewer { id } }", {}) == {"triggers": {}}
    assert len(calls) == 1 and str(calls[0].url) == "https://internal.invalid/graphql"
    assert calls[0].headers["User-Agent"] == "wandb-mcp-server"
    assert calls[0].headers["Accept-Encoding"] == "identity"
    assert calls[0].headers["Authorization"].startswith("Basic ")


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500, 503])
def test_transport_never_follows_redirects_or_retries(http, status):
    calls, stream = http(status=status, headers={"Location": "https://secret-canary.invalid"})
    with pytest.raises(Exception):
        transport.execute_app_graphql("query { viewer { id } }", {})
    assert len(calls) == 1 and stream.reads == 0


@pytest.mark.parametrize("headers", [{"Content-Length": "9000"}, {"Content-Encoding": "gzip"}, {"Content-Length": "x"}])
def test_transport_rejects_malformed_and_oversize_before_read(http, monkeypatch, headers):
    monkeypatch.setattr(transport, "MAX_ACCUMULATED_BYTES", 100)
    _, stream = http(headers=headers)
    with pytest.raises(ValueError):
        transport.execute_app_graphql("query { viewer { id } }", {})
    assert stream.reads == 0


def test_stream_limit_missing_result_and_deadline(http, monkeypatch):
    monkeypatch.setattr(transport, "MAX_ACCUMULATED_BYTES", 100)
    _, stream = http(chunks=[b"x" * 60, b"y" * 60, b"z"])
    with pytest.raises(transport.GraphQLResponseTooLarge):
        transport.execute_app_graphql("query { viewer { id } }", {})
    assert stream.reads == 2
    calls, _ = http(chunks=[b'{"errors":[{"message":"secret-canary"}]}'])
    with pytest.raises(ValueError, match="unsuccessful GraphQL") as error:
        transport.execute_app_graphql("query { viewer { id } }", {})
    assert "canary" not in str(error.value)
    token = current_tool_deadline.set(time.monotonic() - 1)
    before = len(calls)
    try:
        with pytest.raises(TimeoutError):
            transport.execute_app_graphql("query { viewer { id } }", {})
    finally:
        current_tool_deadline.reset(token)
    assert len(calls) == before
