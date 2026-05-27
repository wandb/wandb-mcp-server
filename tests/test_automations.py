"""Mock-based unit tests for the list_automations and list_integrations tools.

These tests stand in for the wandb SDK -- they exercise the flattening logic
in mcp_tools/automations.py against fake Automation / Integration objects
shaped like the pydantic models in wandb/wandb/automations/. No live wandb
API calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake wandb-SDK-shaped objects
#
# We mirror the public pydantic shapes from wandb/wandb/automations/:
#   - Scope:        scope_type (enum), id, name
#   - Event:        event_type (enum), filter (typed per event)
#   - Action:       action_type (enum), integration (with .id), + variant fields
#   - Integration:  typename__ (literal), id, + variant fields
# ---------------------------------------------------------------------------


def _enum(value):
    """Build a fake enum-like object whose .value matches the SDK's LenientStrEnum."""
    return SimpleNamespace(value=value)


def _project_scope(id="scope_proj_1", name="my-project"):
    return SimpleNamespace(scope_type=_enum("PROJECT"), id=id, name=name)


def _collection_scope(id="scope_coll_1", name="my-models"):
    return SimpleNamespace(scope_type=_enum("ARTIFACT_COLLECTION"), id=id, name=name)


# Inner metric filters: each defines exactly one of threshold_filter /
# change_filter / zscore_filter, matching wandb's _Wrapped* pydantic shapes.


def _threshold_metric_filter(name="acc", agg="MAX", window=5, cmp="$gt", threshold=0.9):
    inner = SimpleNamespace(
        name=name,
        agg=_enum(agg) if agg else None,
        window=window,
        cmp=cmp,
        threshold=threshold,
    )
    return SimpleNamespace(threshold_filter=inner)


def _change_metric_filter(
    name="loss",
    agg="AVERAGE",
    window=3,
    prior_window=3,
    change_type="RELATIVE",
    change_dir="DECREASE",
    threshold=0.1,
):
    inner = SimpleNamespace(
        name=name,
        agg=_enum(agg) if agg else None,
        window=window,
        prior_window=prior_window,
        change_type=_enum(change_type),
        change_dir=_enum(change_dir),
        threshold=threshold,
    )
    return SimpleNamespace(change_filter=inner)


def _zscore_metric_filter(name="loss", window=30, change_dir="ANY", threshold=3.0):
    inner = SimpleNamespace(
        name=name,
        window=window,
        change_dir=_enum(change_dir),
        threshold=threshold,
    )
    return SimpleNamespace(zscore_filter=inner)


def _run_metric_event(event_type="RUN_METRIC", metric=None):
    filt = SimpleNamespace(metric=metric or _threshold_metric_filter())
    return SimpleNamespace(event_type=_enum(event_type), filter=filt)


def _run_state_event(states=("finished", "failed")):
    state = SimpleNamespace(states=[_enum(s) for s in states])
    filt = SimpleNamespace(state=state)
    # `filter` for RUN_STATE only carries .state, not .metric -- match that.
    return SimpleNamespace(event_type=_enum("RUN_STATE"), filter=filt)


def _mutation_event(event_type="CREATE_ARTIFACT"):
    # Mutation-event filters are MongoLikeFilter (And() etc.) -- no .metric or .state.
    filt = SimpleNamespace()
    return SimpleNamespace(event_type=_enum(event_type), filter=filt)


def _notification_action(integration_id="int_slack_1", title="t", message="m", severity="INFO"):
    return SimpleNamespace(
        action_type=_enum("NOTIFICATION"),
        integration=SimpleNamespace(id=integration_id),
        title=title,
        message=message,
        severity=_enum(severity) if severity else None,
    )


def _webhook_action(integration_id="int_webhook_1", request_payload=None):
    return SimpleNamespace(
        action_type=_enum("GENERIC_WEBHOOK"),
        integration=SimpleNamespace(id=integration_id),
        request_payload=request_payload,
    )


def _no_op_action():
    return SimpleNamespace(action_type=_enum("NO_OP"), integration=None)


def _automation(
    *,
    id="auto_1",
    name="my-automation",
    enabled=True,
    description="desc",
    scope=None,
    event=None,
    action=None,
    updated_at=datetime(2026, 5, 1, 12, 0, 0),
):
    return SimpleNamespace(
        id=id,
        name=name,
        enabled=enabled,
        description=description,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=updated_at,
        scope=scope or _project_scope(),
        event=event or _run_metric_event(),
        action=action or _notification_action(),
    )


def _slack_integration(id="int_slack_1", team="acme", channel="alerts"):
    return SimpleNamespace(
        typename__="SlackIntegration",
        id=id,
        team_name=team,
        channel_name=channel,
    )


def _webhook_integration(id="int_webhook_1", name="prod-webhook", url="https://example.com/hook"):
    return SimpleNamespace(
        typename__="GenericWebhookIntegration",
        id=id,
        name=name,
        url_endpoint=url,
    )


# ---------------------------------------------------------------------------
# list_automations
# ---------------------------------------------------------------------------


class TestListAutomations:
    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_no_entity_arg_calls_api_without_entity(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation()])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations())

        assert result["count"] == 1
        assert result["entity"] is None
        assert result["truncated"] is False
        kwargs = api.automations.call_args.kwargs
        assert "entity" not in kwargs
        assert "name" not in kwargs

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_entity_and_name_passed_through(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(name="exact-match")])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(entity="my-team", name="exact-match"))

        assert result["entity"] == "my-team"
        assert result["count"] == 1
        assert result["automations"][0]["name"] == "exact-match"
        kwargs = api.automations.call_args.kwargs
        assert kwargs["entity"] == "my-team"
        assert kwargs["name"] == "exact-match"

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_max_items_truncation(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(id=f"a{i}") for i in range(10)])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(max_items=3))
        assert result["count"] == 3
        assert result["truncated"] is True

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_max_items_ceiling(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(id=f"a{i}") for i in range(250)])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(max_items=999))
        assert result["count"] == 200
        assert result["truncated"] is True

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_empty_result(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations(entity="empty-team"))
        assert result == {
            "automations": [],
            "count": 0,
            "entity": "empty-team",
            "truncated": False,
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_serializes_project_scope(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(scope=_project_scope(id="scope_p1", name="proj-1"))])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        scope = json.loads(list_automations())["automations"][0]["scope"]
        assert scope == {"type": "PROJECT", "id": "scope_p1", "name": "proj-1"}

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_serializes_collection_scope(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(scope=_collection_scope(id="scope_c1", name="my-models"))])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        scope = json.loads(list_automations())["automations"][0]["scope"]
        assert scope == {"type": "ARTIFACT_COLLECTION", "id": "scope_c1", "name": "my-models"}

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_threshold_event_serialized_structured(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter(
            [
                _automation(
                    event=_run_metric_event(
                        event_type="RUN_METRIC",
                        metric=_threshold_metric_filter(
                            name="accuracy", agg="MAX", window=5, cmp="$gt", threshold=0.95
                        ),
                    )
                )
            ]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_METRIC"
        assert event["filter"] == {
            "kind": "threshold",
            "metric": "accuracy",
            "agg": "MAX",
            "window": 5,
            "cmp": "$gt",
            "threshold": 0.95,
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_change_event_serialized_structured(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter(
            [
                _automation(
                    event=_run_metric_event(
                        event_type="RUN_METRIC_CHANGE",
                        metric=_change_metric_filter(
                            name="loss",
                            agg="AVERAGE",
                            window=3,
                            prior_window=3,
                            change_type="RELATIVE",
                            change_dir="DECREASE",
                            threshold=0.1,
                        ),
                    )
                )
            ]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_METRIC_CHANGE"
        assert event["filter"] == {
            "kind": "change",
            "metric": "loss",
            "agg": "AVERAGE",
            "current_window": 3,
            "prior_window": 3,
            "change_type": "RELATIVE",
            "change_dir": "DECREASE",
            "threshold": 0.1,
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_zscore_event_serialized_structured(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter(
            [
                _automation(
                    event=_run_metric_event(
                        event_type="RUN_METRIC_ZSCORE",
                        metric=_zscore_metric_filter(name="loss", window=30, change_dir="ANY", threshold=3.0),
                    )
                )
            ]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_METRIC_ZSCORE"
        assert event["filter"] == {
            "kind": "zscore",
            "metric": "loss",
            "window": 30,
            "change_dir": "ANY",
            "threshold": 3.0,
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_run_state_event_serialized(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(event=_run_state_event(states=("finished", "failed")))])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "RUN_STATE"
        assert event["filter"] == {"states": ["finished", "failed"]}

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_mutation_event_serialized(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(event=_mutation_event("CREATE_ARTIFACT"))])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        event = json.loads(list_automations())["automations"][0]["event"]
        assert event["type"] == "CREATE_ARTIFACT"
        # Mutation event filter is open-ended; we just expose a summary.
        assert "summary" in event["filter"]

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_serializes_notification_action(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter(
            [_automation(action=_notification_action(integration_id="int_x", title="T", message="M", severity="WARN"))]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        assert action == {
            "type": "NOTIFICATION",
            "integration_id": "int_x",
            "title": "T",
            "message": "M",
            "severity": "WARN",
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_serializes_webhook_action(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter(
            [_automation(action=_webhook_action(integration_id="int_w", request_payload={"k": "v"}))]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        assert action == {
            "type": "GENERIC_WEBHOOK",
            "integration_id": "int_w",
            "request_payload": {"k": "v"},
        }

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_serializes_no_op_action(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(action=_no_op_action())])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        action = json.loads(list_automations())["automations"][0]["action"]
        assert action == {"type": "NO_OP"}

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_iso_timestamps(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation()])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        auto = json.loads(list_automations())["automations"][0]
        assert auto["created_at"] == "2026-01-01T12:00:00"
        assert auto["updated_at"] == "2026-05-01T12:00:00"

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_null_updated_at(self, mock_mgr):
        api = MagicMock()
        api.automations.return_value = iter([_automation(updated_at=None)])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        auto = json.loads(list_automations())["automations"][0]
        assert auto["updated_at"] is None

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_api_error_returns_error_dict(self, mock_mgr):
        api = MagicMock()
        api.automations.side_effect = RuntimeError("boom")
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_automations

        result = json.loads(list_automations())
        assert result["error"] == "api_error"
        assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# list_integrations
# ---------------------------------------------------------------------------


class TestListIntegrations:
    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_no_type_returns_both(self, mock_mgr):
        api = MagicMock()
        api.integrations.return_value = iter([_slack_integration(), _webhook_integration()])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(entity="my-team"))
        assert result["count"] == 2
        types = {item["type"] for item in result["integrations"]}
        assert types == {"slack", "webhook"}
        api.slack_integrations.assert_not_called()
        api.webhook_integrations.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_filter_slack(self, mock_mgr):
        api = MagicMock()
        api.slack_integrations.return_value = iter([_slack_integration(id="s1", team="acme", channel="alerts")])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(entity="my-team", integration_type="slack"))
        assert result["count"] == 1
        assert result["integrations"][0] == {
            "id": "s1",
            "type": "slack",
            "team_name": "acme",
            "channel_name": "alerts",
        }
        api.integrations.assert_not_called()
        api.webhook_integrations.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_filter_webhook(self, mock_mgr):
        api = MagicMock()
        api.webhook_integrations.return_value = iter(
            [_webhook_integration(id="w1", name="prod", url="https://x.test/hook")]
        )
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(integration_type="webhook"))
        assert result["count"] == 1
        assert result["integrations"][0] == {
            "id": "w1",
            "type": "webhook",
            "name": "prod",
            "url_endpoint": "https://x.test/hook",
        }
        api.integrations.assert_not_called()
        api.slack_integrations.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_invalid_type_returns_error(self, mock_mgr):
        api = MagicMock()
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(integration_type="email"))
        assert result["error"] == "invalid_input"
        api.integrations.assert_not_called()
        api.slack_integrations.assert_not_called()
        api.webhook_integrations.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_empty(self, mock_mgr):
        api = MagicMock()
        api.integrations.return_value = iter([])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["count"] == 0
        assert result["integrations"] == []
        assert result["truncated"] is False

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_max_items_truncation(self, mock_mgr):
        api = MagicMock()
        api.integrations.return_value = iter([_slack_integration(id=f"s{i}") for i in range(20)])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(max_items=5))
        assert result["count"] == 5
        assert result["truncated"] is True

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_max_items_ceiling(self, mock_mgr):
        api = MagicMock()
        api.integrations.return_value = iter([_webhook_integration(id=f"w{i}") for i in range(250)])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations(max_items=999))
        assert result["count"] == 200
        assert result["truncated"] is True

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_api_error_returns_error_dict(self, mock_mgr):
        api = MagicMock()
        api.integrations.side_effect = RuntimeError("kapow")
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["error"] == "api_error"
        assert "kapow" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.automations.WandBApiManager")
    def test_unknown_typename_falls_back(self, mock_mgr):
        # Future integration kinds we don't recognize should still serialize.
        api = MagicMock()
        api.integrations.return_value = iter([SimpleNamespace(typename__="DiscordIntegration", id="d1")])
        mock_mgr.get_api.return_value = api

        from wandb_mcp_server.mcp_tools.automations import list_integrations

        result = json.loads(list_integrations())
        assert result["count"] == 1
        assert result["integrations"][0]["id"] == "d1"
        assert result["integrations"][0]["type"] == "DiscordIntegration"


# ---------------------------------------------------------------------------
# Tool descriptions are present and non-trivial (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "const_name",
    ["LIST_AUTOMATIONS_TOOL_DESCRIPTION", "LIST_INTEGRATIONS_TOOL_DESCRIPTION"],
)
def test_tool_descriptions_present(const_name):
    from wandb_mcp_server.mcp_tools import automations as mod

    val = getattr(mod, const_name)
    assert isinstance(val, str)
    assert "<when_to_use>" in val
    assert "<critical_info>" in val
