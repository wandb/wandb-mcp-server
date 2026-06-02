"""List W&B Automations and the integrations they can target.

Provides two read-only tools for discovering Automations and the
Slack/webhook integrations that Automation actions reference, via the
public ``wandb.Api`` interface introduced in wandb 0.19.11.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator
from itertools import islice
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from pydantic import PositiveInt
from wandb.automations import EventType, SlackIntegration, WebhookIntegration

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

if TYPE_CHECKING:
    from wandb.automations import ArtifactCollectionScope, Automation, Integration, ProjectScope
    from wandb.automations.actions import SavedAction
    from wandb.automations.events import SavedEvent

logger = get_rich_logger(__name__)

MAX_ITEMS_DEFAULT: Final[int] = 50
MAX_ITEMS_CEIL: Final[int] = 200

# Public literal type so MCP clients see the allowed values for the ``kind`` parameter.
IntegrationKind = Literal["slack", "webhook"]
_VALID_INTEGRATION_KINDS: Final[Collection[IntegrationKind]] = frozenset(get_args(IntegrationKind))

_DUMP_KWARGS: Final[dict[str, Any]] = dict(mode="json", by_alias=False, round_trip=True)


# ---------------------------------------------------------------------------
# Tool 1 -- list_automations
# ---------------------------------------------------------------------------

LIST_AUTOMATIONS_TOOL_DESCRIPTION = dedent(
    """\
    List W&B Automations the current API key can access.

    Automations are user-configured rules that fire actions (Slack notification or
    generic webhook) when events happen in W&B: a new artifact is created, an
    alias is added, a model artifact is linked to a registry collection, a run
    metric crosses a threshold or changes significantly, or a run changes state.

    <when_to_use>
    Call this tool when the user asks about:
    - "What automations / alerts / triggers / webhooks do we have?"
    - "Is there an automation that fires when X happens?"
    - "Audit automations on my project / team"
    - "Which automations send to Slack channel #foo?"

    If the user wants to know the available Slack channels or webhook URLs that
    existing or new automations can target, call list_wandb_integrations_tool
    instead (or in addition).
    </when_to_use>

    <critical_info>
    - entity=None lists every automation the authenticated viewer can see across
    all of their teams. Pass entity="<team-or-user>" to scope to one entity.
    - The optional name filter is an exact match (not regex or substring).
    - Results are capped by max_items (default 50, ceiling 200). When more exist,
    the response sets truncated=true.
    - Each returned automation has these fields: id, name, enabled, description,
    created_at, updated_at, scope, event, action.
    - scope is {type, id, name} where type is PROJECT or ARTIFACT_COLLECTION.
    - event is {type, filter} where filter shape depends on the event type.
    - action is {type, ...} with extra fields per action type (NOTIFICATION for
    Slack, GENERIC_WEBHOOK for webhooks, NO_OP for placeholders).
    - The scope only includes id and name. The parent project and entity are
    not on the scope. If you need them, use the `entity` you passed to this
    tool and look up the project separately.
    - This tool is read-only. It cannot create, modify, or delete automations.
    </critical_info>

    Parameters
    ----------
    entity : str, optional
        W&B entity (team or user) to filter by. If omitted, returns every
        automation the authenticated viewer has access to.
    name : str, optional
        Exact-match filter on the automation's name.
    max_items : int, optional
        Maximum automations to return. Default 50, ceiling 200.

    Returns
    -------
    JSON with:
    - automations: list of automation objects (see fields above)
    - count: number of automations returned
    - entity: the entity filter applied (or null)
    - truncated: whether more automations exist beyond max_items
    """
)


def _clamp(value: int, floor: int, ceil: int, /) -> int:
    return max(floor, min(value, ceil))


def _jsonify_scope(scope: ProjectScope | ArtifactCollectionScope) -> dict[str, Any]:
    """Flatten an AutomationScope to ``{type, id, name}``.

    The wandb GraphQL fragments only carry ``id`` and ``name`` on scopes
    (see ``wandb/automations/_generated/fragments.py``: ProjectScopeFields,
    ArtifactSequenceScopeFields, ArtifactPortfolioScopeFields). The public
    ``scope_type`` enum (PROJECT | ARTIFACT_COLLECTION) lets agents branch
    on it without parsing GraphQL typename strings.
    """
    return {"type": scope.scope_type.value, "id": scope.id, "name": scope.name}


def _jsonify_metric_filter(metric_filter: Any) -> dict[str, Any]:
    """Flatten the inner metric filter from a RunMetricFilter wrapper.

    A ``RunMetricFilter.metric`` is one of three pydantic wrapper variants.
    Each carries its own ``event_type`` discriminator and exposes exactly
    one of the threshold, change, or zscore sub-filter attributes. We
    discriminate via the upstream-defined enum value rather than poking at
    attribute presence.
    """
    match metric_filter:
        case object(event_type=EventType.RUN_METRIC_THRESHOLD, threshold_filter=threshold_filter):
            return {"kind": "threshold"} | threshold_filter.model_dump(**_DUMP_KWARGS)
        case object(event_type=EventType.RUN_METRIC_CHANGE, change_filter=change_filter):
            return {"kind": "change"} | change_filter.model_dump(**_DUMP_KWARGS)
        case object(event_type=EventType.RUN_METRIC_ZSCORE, zscore_filter=zscore_filter):
            return {"kind": "zscore"} | zscore_filter.model_dump(**_DUMP_KWARGS)
        case _:
            return {"kind": "unknown"}


def _jsonify_event(event: SavedEvent) -> dict[str, Any]:
    """Flatten a SavedEvent to ``{type, filter}``.

    The filter shape is structured for run-metric and run-state events.
    Mutation events (CREATE_ARTIFACT etc.) carry an open-ended MongoLikeFilter
    that we surface as a brief string ``summary`` since the SDK doesn't
    expose a stable structured shape for it.
    """
    from wandb.automations.events import RunMetricFilter, RunStateFilter, SavedEvent

    match event:
        case SavedEvent(event_type=type_, filter=RunMetricFilter(metric=metric_filter)):
            return {"type": type_.value, "filter": _jsonify_metric_filter(metric_filter)}
        case SavedEvent(event_type=type_, filter=RunStateFilter(state=state_filter)):
            return {"type": type_.value, "filter": state_filter.model_dump(**_DUMP_KWARGS)}
        case _:
            return {"type": event.event_type.value, "filter": {"summary": repr(event.filter)}}


def _jsonify_action(action: SavedAction) -> dict[str, Any]:
    """Flatten a SavedAction to a JSON-safe dict.

    Match-cases on the public ``Saved*Action`` pydantic types
    (SavedNotificationAction, SavedWebhookAction, SavedNoOpAction). Each
    arm emits only the fields its variant guarantees, so we never reach
    for an attribute the variant does not define.
    """
    from wandb.automations.actions import SavedNoOpAction, SavedNotificationAction, SavedWebhookAction

    match action:
        case SavedNotificationAction(action_type=type_, integration=integration):
            jsonable = action.model_dump(include={"title", "message", "severity"}, **_DUMP_KWARGS)
            return {"type": type_.value, "integration_id": integration.id} | jsonable
        case SavedWebhookAction(action_type=type_, integration=integration):
            jsonable = action.model_dump(include={"request_payload"}, **_DUMP_KWARGS)
            return {"type": type_.value, "integration_id": integration.id} | jsonable
        case SavedNoOpAction(action_type=type_):
            return {"type": type_.value}
        case _:
            # QUEUE_JOB / PUSH_NOTIFICATION are currently excluded from the public create API, but they may appear on saved automations. Only expose the type.
            return {"type": action.action_type.value}


def _jsonify_automation(automation: Automation) -> dict[str, Any]:
    """Flatten an Automation pydantic object to a JSON-safe dict."""
    jsonable = automation.model_dump(exclude={"scope", "event", "action"}, **_DUMP_KWARGS)
    jsonable_scope = _jsonify_scope(automation.scope)
    jsonable_event = _jsonify_event(automation.event)
    jsonable_action = _jsonify_action(automation.action)
    return jsonable | {"scope": jsonable_scope, "event": jsonable_event, "action": jsonable_action}


def list_automations(
    entity: str | None = None,
    name: str | None = None,
    max_items: PositiveInt = MAX_ITEMS_DEFAULT,
) -> str:
    """List W&B Automations accessible with the current API key."""
    params = locals()  # Must be first so it only picks up the function args

    api = WandBApiManager.get_api()
    with track_tool_execution("list_automations", api.viewer, params) as ctx:
        max_items = _clamp(max_items, 1, MAX_ITEMS_CEIL)

        try:
            iterator = api.automations(entity=entity, name=name, per_page=_clamp(max_items, 1, 100))
            automations = list(map(_jsonify_automation, islice(iterator, max_items)))
            truncated = next(iterator, None) is not None

            result = {
                "automations": automations,
                "count": len(automations),
                "entity": entity,
                "truncated": truncated,
            }
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in list_automations: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


# ---------------------------------------------------------------------------
# Tool 2 -- list_integrations
# ---------------------------------------------------------------------------

LIST_INTEGRATIONS_TOOL_DESCRIPTION = dedent("""\
    List W&B integrations (Slack channels and webhooks) for an entity.

    Integrations are the destinations that W&B Automations send notifications to.
    A SlackIntegration represents a connected Slack channel. A WebhookIntegration
    represents a configured outbound webhook URL. An Automation action references
    exactly one integration by id.

    <when_to_use>
    Call this tool when the user asks about:
    - "What Slack channels / webhooks can I send alerts to?"
    - "Which integrations are set up for my team?"
    - "List the webhooks configured on entity X"

    You should also call this BEFORE creating an automation (a future tool) when
    you need an integration_id to pass into the action.
    </when_to_use>

    <critical_info>
    - Integrations are configured at the entity (team) level via the W&B UI.
    This tool only lists existing integrations. It does not create them.
    - entity=None defaults to the authenticated viewer's default entity.
    - kind is "slack" (Slack only), "webhook" (webhook only), or null (both).
    - Each returned record always has {id, type}. Slack adds {team_name,
    channel_name}. Webhook adds {name, url_endpoint}.
    </critical_info>

    Parameters
    ----------
    entity : str, optional
        W&B entity (team or user). Omit to use the viewer's default entity.
    kind : "slack" | "webhook" | None
        Omit to return both kinds.
    max_items : int, optional
        Maximum integrations to return. Default 50, ceiling 200.

    Returns
    -------
    JSON with:
    - integrations: list of integration objects
    - count: number of integrations returned
    - entity: the entity filter applied (or null)
    - kind: the type filter applied (or null)
    - truncated: whether more integrations exist beyond max_items
    """)


def _jsonify_integration(integration: Integration) -> dict[str, Any]:
    """Flatten a Slack or Webhook integration to a JSON-safe dict.

    Discriminated via ``isinstance`` against the public wandb pydantic
    classes. pydantic v2 returns concrete subclass instances for
    discriminated unions, so this is the canonical way to branch.
    The wildcard arm keeps the tool forward-compatible with future
    integration kinds the server may add.
    """
    match integration:
        case SlackIntegration():
            jsonable = integration.model_dump(include={"id", "team_name", "channel_name"}, **_DUMP_KWARGS)
            return {"type": "slack"} | jsonable
        case WebhookIntegration():
            jsonable = integration.model_dump(include={"id", "name", "url_endpoint"}, **_DUMP_KWARGS)
            return {"type": "webhook"} | jsonable
        case _:
            jsonable = integration.model_dump(include={"id"}, **_DUMP_KWARGS)
            return {"type": integration.typename__} | jsonable


def list_integrations(
    entity: str | None = None,
    kind: IntegrationKind | str | None = None,
    max_items: int = MAX_ITEMS_DEFAULT,
) -> str:
    """List Slack and/or webhook integrations for an entity."""
    params = locals()  # Must be first so it only picks up the function args

    api = WandBApiManager.get_api()
    with track_tool_execution("list_integrations", api.viewer, params) as ctx:
        max_items = _clamp(max_items, 1, MAX_ITEMS_CEIL)

        if kind is not None and kind not in _VALID_INTEGRATION_KINDS:
            ctx.mark_error(f"invalid kind: {kind!r}")

            msg = f"kind must be one of {sorted(_VALID_INTEGRATION_KINDS)} or null, got {kind!r}"
            return json.dumps({"error": "invalid_input", "message": msg})

        try:
            iterator: Iterator[Integration]
            match kind:
                case "slack":
                    iterator = api.slack_integrations(entity=entity, per_page=_clamp(max_items, 1, 100))
                case "webhook":
                    iterator = api.webhook_integrations(entity=entity, per_page=_clamp(max_items, 1, 100))
                case _:  # None (already validated above)
                    iterator = api.integrations(entity=entity, per_page=_clamp(max_items, 1, 100))

            integrations = list(map(_jsonify_integration, islice(iterator, max_items)))
            truncated = next(iterator, None) is not None

            result = {
                "integrations": integrations,
                "count": len(integrations),
                "entity": entity,
                "kind": kind,
                "truncated": truncated,
            }
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in list_integrations: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})
