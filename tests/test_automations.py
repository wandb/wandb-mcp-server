"""Mock-based unit tests for the list_automations and list_integrations tools.

These tests exercise the flattening logic in ``mcp_tools/automations.py``
against ``MagicMock(spec=...)`` stand-ins for the wandb SDK's pydantic
models -- this gives us correct ``isinstance``/``match-case`` behavior
without paying the cost of constructing real Automation/Integration
objects through pydantic validation. No live wandb API calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable
from unittest.mock import MagicMock

from pytest import fixture, mark
from wandb.automations import (
    ActionType,
    Automation,
    EventType,
    SlackIntegration,
    WebhookIntegration,
)

PATCH_TARGET = "wandb_mcp_server.mcp_tools.automations.WandBApiManager"


# ---------------------------------------------------------------------------
# Fixtures: factories for fake wandb-SDK-shaped objects
#
# We use MagicMock(spec=<real wandb class>) so that isinstance() and the
# match-case class patterns in the production code work correctly. Each
# fixture returns a *factory* so individual tests can override only the
# fields they care about.
# ---------------------------------------------------------------------------


@fixture
def mock_api(mocker) -> MagicMock:
    """Patch WandBApiManager and yield the mocked ``api`` instance.

    The production code under test calls ``WandBApiManager.get_api()``;
    we replace the whole manager with a MagicMock and return the api
    object so tests can configure ``api.automations.return_value`` etc.
    """
    mgr = mocker.patch(PATCH_TARGET)
    api = MagicMock()
    mgr.get_api.return_value = api
    return api


@fixture
def make_project_scope() -> Callable[..., MagicMock]:
    def _make(*, id: str = "scope_proj_1", name: str = "my-project") -> MagicMock:
        s = MagicMock()
        s.scope_type = MagicMock(value="PROJECT")
        s.id = id
        s.name = name
        return s

    return _make


@fixture
def make_collection_scope() -> Callable[..., MagicMock]:
    def _make(*, id: str = "scope_coll_1", name: str = "my-models") -> MagicMock:
        s = MagicMock()
        s.scope_type = MagicMock(value="ARTIFACT_COLLECTION")
        s.id = id
        s.name = name
        return s

    return _make


# NOTE: ``MagicMock(name=x, ...)`` treats ``name`` as the mock's repr name,
# NOT as setting the ``.name`` attribute. For any wandb field literally called
# "name" we always assign it via attribute after construction.


@fixture
def make_threshold_event() -> Callable[..., MagicMock]:
    def _make(
        *,
        metric_name: str = "acc",
        agg: str | None = "MAX",
        window: int = 5,
        cmp: str = "$gt",
        threshold: float = 0.9,
    ) -> MagicMock:
        inner = MagicMock(window=window, cmp=cmp, threshold=threshold)
        inner.name = metric_name
        inner.agg = MagicMock(value=agg) if agg else None
        wrapper = MagicMock(event_type=EventType.RUN_METRIC_THRESHOLD, threshold_filter=inner)
        return MagicMock(event_type=EventType.RUN_METRIC_THRESHOLD, filter=MagicMock(metric=wrapper))

    return _make


@fixture
def make_change_event() -> Callable[..., MagicMock]:
    def _make(
        *,
        metric_name: str = "loss",
        agg: str | None = "AVERAGE",
        window: int = 3,
        prior_window: int = 3,
        change_type: str = "RELATIVE",
        change_dir: str = "DECREASE",
        threshold: float = 0.1,
    ) -> MagicMock:
        inner = MagicMock(
            window=window,
            prior_window=prior_window,
            change_type=MagicMock(value=change_type),
            change_dir=MagicMock(value=change_dir),
            threshold=threshold,
        )
        inner.name = metric_name
        inner.agg = MagicMock(value=agg) if agg else None
        wrapper = MagicMock(event_type=EventType.RUN_METRIC_CHANGE, change_filter=inner)
        return MagicMock(event_type=EventType.RUN_METRIC_CHANGE, filter=MagicMock(metric=wrapper))

    return _make


@fixture
def make_zscore_event() -> Callable[..., MagicMock]:
    def _make(
        *,
        metric_name: str = "loss",
        window: int = 30,
        change_dir: str = "ANY",
        threshold: float = 3.0,
    ) -> MagicMock:
        inner = MagicMock(
            window=window,
            change_dir=MagicMock(value=change_dir),
            threshold=threshold,
        )
        inner.name = metric_name
        wrapper = MagicMock(event_type=EventType.RUN_METRIC_ZSCORE, zscore_filter=inner)
        return MagicMock(event_type=EventType.RUN_METRIC_ZSCORE, filter=MagicMock(metric=wrapper))

    return _make


@fixture
def make_run_state_event() -> Callable[..., MagicMock]:
    def _make(*, states: tuple[str, ...] = ("finished", "failed")) -> MagicMock:
        state = MagicMock(states=[MagicMock(value=s) for s in states])
        return MagicMock(event_type=EventType.RUN_STATE, filter=MagicMock(state=state))

    return _make


@fixture
def make_mutation_event() -> Callable[..., MagicMock]:
    """An event for CREATE_ARTIFACT / ADD_ARTIFACT_ALIAS / LINK_ARTIFACT.

    Mutation-event filters are open-ended MongoLikeFilter objects; the
    production code falls back to ``repr()`` for these, so we only need
    a sentinel filter object whose ``repr`` is recognizable.
    """

    def _make(*, event_type: EventType = EventType.CREATE_ARTIFACT) -> MagicMock:
        return MagicMock(event_type=event_type, filter=MagicMock())

    return _make


@fixture
def make_notification_action() -> Callable[..., MagicMock]:
    def _make(
        *,
        integration_id: str = "int_slack_1",
        title: str = "t",
        message: str = "m",
        severity: str | None = "INFO",
    ) -> MagicMock:
        a = MagicMock(
            action_type=ActionType.NOTIFICATION,
            integration=MagicMock(id=integration_id),
            title=title,
            message=message,
            severity=MagicMock(value=severity) if severity else None,
        )
        return a

    return _make


@fixture
def make_webhook_action() -> Callable[..., MagicMock]:
    def _make(
        *,
        integration_id: str = "int_webhook_1",
        request_payload: dict | None = None,
    ) -> MagicMock:
        return MagicMock(
            action_type=ActionType.GENERIC_WEBHOOK,
            integration=MagicMock(id=integration_id),
            request_payload=request_payload,
        )

    return _make


@fixture
def make_no_op_action() -> Callable[..., MagicMock]:
    def _make() -> MagicMock:
        return MagicMock(action_type=ActionType.NO_OP)

    return _make


@fixture
def make_automation(make_project_scope, make_threshold_event, make_notification_action) -> Callable[..., MagicMock]:
    """Build a MagicMock with the same surface as ``wandb.automations.Automation``.

    Using ``spec=Automation`` would make ``isinstance(m, Automation)`` true,
    but the production code doesn't isinstance-check Automations -- it
    only attribute-accesses them -- so a plain MagicMock keeps fixture
    setup simpler. Override any field via kwargs.
    """

    def _make(
        *,
        id: str = "auto_1",
        name: str = "my-automation",
        enabled: bool = True,
        description: str | None = "desc",
        created_at: datetime = datetime(2026, 1, 1, 12, 0, 0),
        updated_at: datetime | None = datetime(2026, 5, 1, 12, 0, 0),
        scope: MagicMock | None = None,
        event: MagicMock | None = None,
        action: MagicMock | None = None,
    ) -> MagicMock:
        m = MagicMock(
            id=id,
            enabled=enabled,
            description=description,
            created_at=created_at,
            updated_at=updated_at,
            scope=scope or make_project_scope(),
            event=event or make_threshold_event(),
            action=action or make_notification_action(),
        )
        m.name = name  # see note above: ``name`` is reserved in MagicMock()
        return m

    return _make


@fixture
def make_slack_integration() -> Callable[..., MagicMock]:
    """Build a MagicMock that passes ``isinstance(m, SlackIntegration)``.

    Using ``spec=SlackIntegration`` is the canonical way to get a mock
    that the production code's ``match SlackIntegration():`` pattern
    will recognize.
    """

    def _make(
        *,
        id: str = "int_slack_1",
        team_name: str = "acme",
        channel_name: str = "alerts",
    ) -> MagicMock:
        m = MagicMock(spec=SlackIntegration)
        m.id = id
        m.team_name = team_name
        m.channel_name = channel_name
        return m

    return _make


@fixture
def make_webhook_integration() -> Callable[..., MagicMock]:
    def _make(
        *,
        id: str = "int_webhook_1",
        name: str = "prod-webhook",
        url_endpoint: str = "https://example.com/hook",
    ) -> MagicMock:
        m = MagicMock(spec=WebhookIntegration)
        m.id = id
        m.name = name  # safe here -- assigned via attribute, not kwarg
        m.url_endpoint = url_endpoint
        return m

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
        assert "entity" not in kwargs
        assert "name" not in kwargs

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
                "metric": "accuracy",
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
                "metric": "loss",
                "agg": "AVERAGE",
                "current_window": 3,
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
                "metric": "loss",
                "window": 30,
                "change_dir": "ANY",
                "threshold": 3.0,
            },
        }

    def test_run_state_event_serialized(self, mock_api, make_automation, make_run_state_event):
        mock_api.automations.return_value = iter(
            [make_automation(event=make_run_state_event(states=("finished", "failed")))]
        )

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_STATE"
        assert event["filter"] == {"states": ["finished", "failed"]}

    @mark.parametrize(
        "event_type",
        [EventType.CREATE_ARTIFACT, EventType.ADD_ARTIFACT_ALIAS, EventType.LINK_ARTIFACT],
    )
    def test_mutation_event_serialized(self, mock_api, make_automation, make_mutation_event, event_type):
        mock_api.automations.return_value = iter([make_automation(event=make_mutation_event(event_type=event_type))])

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == event_type.value
        # Mutation-event filter has no structured shape -- only a summary string.
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
        assert action == {
            "type": "GENERIC_WEBHOOK",
            "integration_id": "int_w",
            "request_payload": {"k": "v"},
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
        should still serialize (this is the wildcard match arm)."""
        other = MagicMock()
        other.id = "d1"
        other.typename__ = "DiscordIntegration"
        # IMPORTANT: do NOT spec=SlackIntegration/WebhookIntegration so the
        # match-case wildcard fires.
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


# ---------------------------------------------------------------------------
# Sanity check: ``Automation`` import works so spec= would work in future tests
# ---------------------------------------------------------------------------


def test_wandb_automation_class_importable():
    """If wandb ever moves/renames Automation, this test will fail loudly
    so the mock fixtures above can be updated in lockstep."""
    assert isinstance(Automation, type)
