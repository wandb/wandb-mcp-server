"""Unit tests for the list_automations and list_integrations tools.

These tests use real wandb pydantic instances (via ``Automation.model_validate``
and ``SlackIntegration.model_validate`` / ``WebhookIntegration.model_validate``)
rather than mocks. The production serializer (``_jsonify_*``) runs against the
actual model contracts the wandb SDK guarantees, so if wandb renames a field
or changes a discriminator, these tests catch it. The one ``MagicMock`` left
in the file is in ``test_unknown_typename_falls_back``. It simulates an
integration kind that is not ``SlackIntegration`` or ``WebhookIntegration``,
which no real type satisfies by construction.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from unittest.mock import MagicMock

from pytest import fixture, mark
from wandb.automations import (
    Automation,
    EventType,
    SlackIntegration,
    WebhookIntegration,
)

PATCH_TARGET = "wandb_mcp_server.mcp_tools.automations.WandBApiManager"


# ---------------------------------------------------------------------------
# Fixtures: factories that build real wandb pydantic instances via
# ``model_validate``, the canonical "construct from API response" path.
# Tests get factories rather than fixed instances so individual cases can
# override only the fields they care about.
# ---------------------------------------------------------------------------


@fixture
def mock_api(mocker) -> MagicMock:
    """Patch ``WandBApiManager`` and return the mocked ``api`` instance.

    Production calls ``WandBApiManager.get_api()``. We replace the manager
    with a MagicMock and return the api so tests can configure
    ``api.automations.return_value`` etc.
    """
    mgr = mocker.patch(PATCH_TARGET)
    api = MagicMock()
    mgr.get_api.return_value = api
    return api


# ---------------------------------------------------------------------------
# Scope payload factories. These return GraphQL-shaped dicts that
# make_automation assembles and validates through Automation.model_validate.
# ---------------------------------------------------------------------------


@fixture
def make_project_scope() -> Callable[..., dict[str, Any]]:
    def _make(*, id: str = "scope_proj_1", name: str = "my-project") -> dict[str, Any]:
        return {"__typename": "Project", "id": id, "name": name}

    return _make


@fixture
def make_collection_scope() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        id: str = "scope_coll_1",
        name: str = "my-models",
        kind: str = "ArtifactSequence",
    ) -> dict[str, Any]:
        # wandb has two artifact-collection variants: ArtifactSequence
        # (versioned artifact) and ArtifactPortfolio (registry collection).
        # Both serialize to ARTIFACT_COLLECTION scope_type so either works.
        return {"__typename": kind, "id": id, "name": name}

    return _make


# ---------------------------------------------------------------------------
# Event payload factories.
# ---------------------------------------------------------------------------


def _filter_event_payload(event_type: str, filter_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "__typename": "FilterEventTriggeringCondition",
        "eventType": event_type,
        # The wire format stores the filter as a JSON-encoded string.
        "filter": json.dumps(filter_obj),
    }


@fixture
def make_threshold_event() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        metric_name: str = "acc",
        agg: str | None = "MAX",
        window: int = 5,
        cmp: str = "$gt",
        threshold: float = 0.9,
    ) -> dict[str, Any]:
        threshold_filter: dict[str, Any] = {
            "name": metric_name,
            "window_size": window,
            "cmp_op": cmp,
            "threshold": threshold,
        }
        if agg is not None:
            threshold_filter["agg_op"] = agg
        return _filter_event_payload(
            "RUN_METRIC",
            {
                "run_filter": {"$and": []},
                "run_metric_filter": {"threshold_filter": threshold_filter},
            },
        )

    return _make


@fixture
def make_change_event() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        metric_name: str = "loss",
        agg: str | None = "AVERAGE",
        window: int = 3,
        prior_window: int = 3,
        change_type: str = "RELATIVE",
        change_dir: str = "DECREASE",
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        change_filter: dict[str, Any] = {
            "name": metric_name,
            "current_window_size": window,
            "prior_window_size": prior_window,
            "change_type": change_type,
            "change_dir": change_dir,
            "change_amount": threshold,
        }
        if agg is not None:
            change_filter["agg_op"] = agg
        return _filter_event_payload(
            "RUN_METRIC_CHANGE",
            {
                "run_filter": {"$and": []},
                "run_metric_filter": {"change_filter": change_filter},
            },
        )

    return _make


@fixture
def make_zscore_event() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        metric_name: str = "loss",
        window: int = 30,
        change_dir: str = "ANY",
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        return _filter_event_payload(
            "RUN_METRIC_ZSCORE",
            {
                "run_filter": {"$and": []},
                "run_metric_filter": {
                    "zscore_filter": {
                        "name": metric_name,
                        "window_size": window,
                        "change_dir": change_dir,
                        "threshold": threshold,
                    },
                },
            },
        )

    return _make


@fixture
def make_run_state_event() -> Callable[..., dict[str, Any]]:
    def _make(*, states: tuple[str, ...] = ("FINISHED", "FAILED")) -> dict[str, Any]:
        # wandb's StateFilter validator dedupes + sorts these on the way in,
        # so the order tests assert on is the canonical sorted order.
        return _filter_event_payload(
            "RUN_STATE",
            {
                "run_filter": {"$and": []},
                "run_state_filter": {"states": list(states)},
            },
        )

    return _make


@fixture
def make_mutation_event() -> Callable[..., dict[str, Any]]:
    """An event for CREATE_ARTIFACT / ADD_ARTIFACT_ALIAS / LINK_ARTIFACT.

    Mutation-event filters are open-ended MongoLikeFilter objects. The
    production code falls back to ``repr()`` for these, so we only need
    a sentinel filter payload.
    """

    def _make(*, event_type: EventType = EventType.CREATE_ARTIFACT) -> dict[str, Any]:
        return _filter_event_payload(event_type.value, {"filter": {}})

    return _make


# ---------------------------------------------------------------------------
# Action payload factories.
# ---------------------------------------------------------------------------


@fixture
def make_notification_action() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        integration_id: str = "int_slack_1",
        title: str = "t",
        message: str = "m",
        severity: str | None = "INFO",
    ) -> dict[str, Any]:
        return {
            "__typename": "NotificationTriggeredAction",
            "integration": {"__typename": "SlackIntegration", "id": integration_id},
            "title": title,
            "message": message,
            "severity": severity,
        }

    return _make


@fixture
def make_webhook_action() -> Callable[..., dict[str, Any]]:
    def _make(
        *,
        integration_id: str = "int_webhook_1",
        request_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "__typename": "GenericWebhookTriggeredAction",
            "integration": {"__typename": "GenericWebhookIntegration", "id": integration_id},
            # wire format is JSON-encoded. wandb parses it back to a dict
            # in-memory but model_dump round-trips it to a string. Tests
            # assert the string form.
            "requestPayload": json.dumps(request_payload) if request_payload is not None else None,
        }

    return _make


@fixture
def make_no_op_action() -> Callable[..., dict[str, Any]]:
    def _make() -> dict[str, Any]:
        return {"__typename": "NoOpTriggeredAction", "noOp": True}

    return _make


# ---------------------------------------------------------------------------
# Top-level Automation factory.
# ---------------------------------------------------------------------------


@fixture
def make_automation(make_project_scope, make_threshold_event, make_notification_action) -> Callable[..., Automation]:
    """Build a real ``wandb.automations.Automation`` via ``model_validate``.

    Override any of ``scope`` / ``event`` / ``action`` with a payload dict
    from the matching factory (e.g. ``make_threshold_event(...)``). Defaults
    produce a Project-scoped RUN_METRIC threshold automation with a Slack
    notification action.
    """

    def _make(
        *,
        id: str = "auto_1",
        name: str = "my-automation",
        enabled: bool = True,
        description: str | None = "desc",
        created_at: str = "2026-01-01T12:00:00",
        updated_at: str | None = "2026-05-01T12:00:00",
        scope: dict[str, Any] | None = None,
        event: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
    ) -> Automation:
        payload = {
            "__typename": "Trigger",
            "id": id,
            "name": name,
            "enabled": enabled,
            "description": description,
            "createdAt": created_at,
            "updatedAt": updated_at,
            "scope": scope if scope is not None else make_project_scope(),
            "event": event if event is not None else make_threshold_event(),
            "action": action if action is not None else make_notification_action(),
        }
        return Automation.model_validate(payload)

    return _make


# ---------------------------------------------------------------------------
# Integration factories.
# ---------------------------------------------------------------------------


@fixture
def make_slack_integration() -> Callable[..., SlackIntegration]:
    def _make(
        *,
        id: str = "int_slack_1",
        team_name: str = "acme",
        channel_name: str = "alerts",
    ) -> SlackIntegration:
        return SlackIntegration.model_validate(
            {
                "__typename": "SlackIntegration",
                "id": id,
                "teamName": team_name,
                "channelName": channel_name,
            }
        )

    return _make


@fixture
def make_webhook_integration() -> Callable[..., WebhookIntegration]:
    def _make(
        *,
        id: str = "int_webhook_1",
        name: str = "prod-webhook",
        url_endpoint: str = "https://example.com/hook",
    ) -> WebhookIntegration:
        return WebhookIntegration.model_validate(
            {
                "__typename": "GenericWebhookIntegration",
                "id": id,
                "name": name,
                "urlEndpoint": url_endpoint,
            }
        )

    return _make


# ---------------------------------------------------------------------------
# list_automations
# ---------------------------------------------------------------------------


class TestListAutomations:
    def test_no_entity_arg_calls_api_without_entity(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation()])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations())

        assert result["count"] == 1
        assert result["entity"] is None
        assert result["truncated"] is False
        kwargs = mock_api.automations.call_args.kwargs
        assert kwargs.get("entity") is None
        assert kwargs.get("name") is None

    def test_entity_and_name_passed_through(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation(name="exact-match")])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(entity="my-team", name="exact-match"))

        assert result["entity"] == "my-team"
        assert result["count"] == 1
        assert result["automations"][0]["name"] == "exact-match"
        kwargs = mock_api.automations.call_args.kwargs
        assert kwargs["entity"] == "my-team"
        assert kwargs["name"] == "exact-match"

    def test_max_items_truncation(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation(id=f"a{i}") for i in range(10)])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(max_items=3))
        assert result["count"] == 3
        assert result["truncated"] is True

    def test_max_items_ceiling(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation(id=f"a{i}") for i in range(250)])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(max_items=999))
        assert result["count"] == 200
        assert result["truncated"] is True

    def test_empty_result(self, mock_api):
        mock_api.automations.return_value = iter([])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(entity="empty-team"))
        assert result == {
            "automations": [],
            "count": 0,
            "entity": "empty-team",
            "truncated": False,
        }

    def test_project_scope_serialized(self, mock_api, make_automation, make_project_scope):
        mock_api.automations.return_value = iter(
            [make_automation(scope=make_project_scope(id="scope_p1", name="proj-1"))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        scope = json.loads(list_automations())["automations"][0]["scope"]
        assert scope == {"type": "PROJECT", "id": "scope_p1", "name": "proj-1"}

    def test_collection_scope_serialized(self, mock_api, make_automation, make_collection_scope):
        mock_api.automations.return_value = iter(
            [make_automation(scope=make_collection_scope(id="scope_c1", name="my-models"))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        scope = json.loads(list_automations())["automations"][0]["scope"]
        assert scope == {"type": "ARTIFACT_COLLECTION", "id": "scope_c1", "name": "my-models"}

    def test_threshold_event_structured(self, mock_api, make_automation, make_threshold_event):
        mock_api.automations.return_value = iter(
            [
                make_automation(
                    event=make_threshold_event(metric_name="accuracy", agg="MAX", window=5, cmp="$gt", threshold=0.95)
                )
            ]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event == {
            "type": "RUN_METRIC",
            "filter": {
                "kind": "threshold",
                "name": "accuracy",
                "agg": "MAX",
                "window": 5,
                "cmp": "$gt",
                "threshold": 0.95,
            },
        }

    def test_change_event_structured(self, mock_api, make_automation, make_change_event):
        mock_api.automations.return_value = iter(
            [
                make_automation(
                    event=make_change_event(
                        metric_name="loss",
                        agg="AVERAGE",
                        window=3,
                        prior_window=3,
                        change_type="RELATIVE",
                        change_dir="DECREASE",
                        threshold=0.1,
                    )
                )
            ]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event == {
            "type": "RUN_METRIC_CHANGE",
            "filter": {
                "kind": "change",
                "name": "loss",
                "agg": "AVERAGE",
                "window": 3,
                "prior_window": 3,
                "change_type": "RELATIVE",
                "change_dir": "DECREASE",
                "threshold": 0.1,
            },
        }

    def test_zscore_event_structured(self, mock_api, make_automation, make_zscore_event):
        mock_api.automations.return_value = iter(
            [make_automation(event=make_zscore_event(metric_name="loss", window=30, change_dir="ANY", threshold=3.0))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event == {
            "type": "RUN_METRIC_ZSCORE",
            "filter": {
                "kind": "zscore",
                "name": "loss",
                "window": 30,
                "change_dir": "ANY",
                "threshold": 3.0,
            },
        }

    def test_run_state_event_serialized(self, mock_api, make_automation, make_run_state_event):
        mock_api.automations.return_value = iter(
            [make_automation(event=make_run_state_event(states=("FINISHED", "FAILED")))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_STATE"
        # StateFilter dedupes and sorts. ``FAILED`` sorts before ``FINISHED``.
        assert event["filter"] == {"states": ["FAILED", "FINISHED"]}

    @mark.parametrize(
        "event_type",
        [EventType.CREATE_ARTIFACT, EventType.ADD_ARTIFACT_ALIAS, EventType.LINK_ARTIFACT],
    )
    def test_mutation_event_serialized(self, mock_api, make_automation, make_mutation_event, event_type):
        mock_api.automations.return_value = iter([make_automation(event=make_mutation_event(event_type=event_type))])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == event_type.value
        # Mutation-event filter has no structured shape, only a summary string.
        assert set(event["filter"].keys()) == {"summary"}

    def test_notification_action_serialized(self, mock_api, make_automation, make_notification_action):
        mock_api.automations.return_value = iter(
            [
                make_automation(
                    action=make_notification_action(integration_id="int_x", title="T", message="M", severity="WARN")
                )
            ]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        assert action == {
            "type": "NOTIFICATION",
            "integration_id": "int_x",
            "title": "T",
            "message": "M",
            "severity": "WARN",
        }

    def test_webhook_action_serialized(self, mock_api, make_automation, make_webhook_action):
        mock_api.automations.return_value = iter(
            [make_automation(action=make_webhook_action(integration_id="int_w", request_payload={"k": "v"}))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        # wandb's ``request_payload`` round-trips through model_dump as the
        # JSON-encoded wire form, not the in-memory dict.
        assert action == {
            "type": "GENERIC_WEBHOOK",
            "integration_id": "int_w",
            "request_payload": '{"k":"v"}',
        }

    def test_no_op_action_serialized(self, mock_api, make_automation, make_no_op_action):
        mock_api.automations.return_value = iter([make_automation(action=make_no_op_action())])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        assert action == {"type": "NO_OP"}

    def test_iso_timestamps(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation()])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        auto = json.loads(list_automations())["automations"][0]
        assert auto["created_at"] == "2026-01-01T12:00:00"
        assert auto["updated_at"] == "2026-05-01T12:00:00"

    def test_null_updated_at(self, mock_api, make_automation):
        mock_api.automations.return_value = iter([make_automation(updated_at=None)])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        auto = json.loads(list_automations())["automations"][0]
        assert auto["updated_at"] is None

    def test_api_error_returns_error_dict(self, mock_api):
        mock_api.automations.side_effect = RuntimeError("boom")

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations())
        assert result["error"] == "api_error"
        assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# list_integrations
# ---------------------------------------------------------------------------


class TestListIntegrations:
    def test_no_type_returns_both(self, mock_api, make_slack_integration, make_webhook_integration):
        mock_api.integrations.return_value = iter([make_slack_integration(), make_webhook_integration()])

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(entity="my-team"))
        assert result["count"] == 2
        assert {item["type"] for item in result["integrations"]} == {"slack", "webhook"}
        mock_api.slack_integrations.assert_not_called()
        mock_api.webhook_integrations.assert_not_called()

    def test_filter_slack(self, mock_api, make_slack_integration):
        mock_api.slack_integrations.return_value = iter(
            [make_slack_integration(id="s1", team_name="acme", channel_name="alerts")]
        )

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(entity="my-team", kind="slack"))
        assert result["count"] == 1
        assert result["integrations"][0] == {
            "id": "s1",
            "type": "slack",
            "team_name": "acme",
            "channel_name": "alerts",
        }
        mock_api.integrations.assert_not_called()
        mock_api.webhook_integrations.assert_not_called()

    def test_filter_webhook(self, mock_api, make_webhook_integration):
        mock_api.webhook_integrations.return_value = iter(
            [make_webhook_integration(id="w1", name="prod", url_endpoint="https://x.test/hook")]
        )

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(kind="webhook"))
        assert result["count"] == 1
        assert result["integrations"][0] == {
            "id": "w1",
            "type": "webhook",
            "name": "prod",
            "url_endpoint": "https://x.test/hook",
        }
        mock_api.integrations.assert_not_called()
        mock_api.slack_integrations.assert_not_called()

    def test_invalid_type_returns_error(self, mock_api):
        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(kind="email"))
        assert result["error"] == "invalid_input"
        mock_api.integrations.assert_not_called()
        mock_api.slack_integrations.assert_not_called()
        mock_api.webhook_integrations.assert_not_called()

    def test_empty_string_kind_is_invalid(self, mock_api):
        """Regression: empty string is falsy in Python so a naive truthy check
        would silently treat ``kind=""`` as ``None`` and return both kinds.
        It must be rejected like any other non-allowed value.
        """
        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(kind=""))
        assert result["error"] == "invalid_input"
        mock_api.integrations.assert_not_called()
        mock_api.slack_integrations.assert_not_called()
        mock_api.webhook_integrations.assert_not_called()

    def test_empty(self, mock_api):
        mock_api.integrations.return_value = iter([])

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["count"] == 0
        assert result["integrations"] == []
        assert result["truncated"] is False

    def test_max_items_truncation(self, mock_api, make_slack_integration):
        mock_api.integrations.return_value = iter([make_slack_integration(id=f"s{i}") for i in range(20)])

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(max_items=5))
        assert result["count"] == 5
        assert result["truncated"] is True

    def test_max_items_ceiling(self, mock_api, make_webhook_integration):
        mock_api.integrations.return_value = iter([make_webhook_integration(id=f"w{i}") for i in range(250)])

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(max_items=999))
        assert result["count"] == 200
        assert result["truncated"] is True

    def test_api_error_returns_error_dict(self, mock_api):
        mock_api.integrations.side_effect = RuntimeError("kapow")

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["error"] == "api_error"
        assert "kapow" in result["message"]

    def test_unknown_typename_falls_back(self, mock_api):
        """Forward-compat: an Integration kind that's neither Slack nor Webhook
        should still serialize through the wildcard match arm.

        This is the one case where a real wandb instance won't do. The
        whole point is "we don't know about this type yet", so we drop
        down to a MagicMock for this test specifically. The production
        wildcard arm calls ``.model_dump(include={"id"}, ...)``, so we
        wire that up to return the expected dict.
        """
        other = MagicMock()
        other.id = "d1"
        other.typename__ = "DiscordIntegration"
        other.model_dump.return_value = {"id": "d1"}
        mock_api.integrations.return_value = iter([other])

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["count"] == 1
        assert result["integrations"][0] == {"id": "d1", "type": "DiscordIntegration"}


# ---------------------------------------------------------------------------
# Tool descriptions are present and non-trivial (smoke test)
# ---------------------------------------------------------------------------


@mark.parametrize("const_name", ["LIST_AUTOMATIONS_TOOL_DESCRIPTION", "LIST_INTEGRATIONS_TOOL_DESCRIPTION"])
def test_tool_descriptions_present(const_name):
    from wandb_mcp_server.mcp_tools import automations as mod

    val = getattr(mod, const_name)
    assert isinstance(val, str)
    assert "<when_to_use>" in val
    assert "<critical_info>" in val
