"""List W&B Automations and the integrations they can target.

Provides two read-only tools for discovering Automations and the
Slack/webhook integrations that Automation actions reference, via the
public ``wandb.Api`` interface introduced in wandb 0.19.11.
"""

from __future__ import annotations

import json
from typing import Any

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_ITEMS = 50
MAX_ITEMS_CEILING = 200

# Valid integration_type values for list_integrations.
_SLACK = "slack"
_WEBHOOK = "webhook"
_VALID_INTEGRATION_TYPES = frozenset({_SLACK, _WEBHOOK})

# wandb backend `__typename` strings used to discriminate the Integration union.
# See wandb/automations/_generated/fragments.py: SlackIntegrationFields,
# WebhookIntegrationFields.
_TYPENAME_SLACK_INTEGRATION = "SlackIntegration"
_TYPENAME_WEBHOOK_INTEGRATION = "GenericWebhookIntegration"


# ---------------------------------------------------------------------------
# Tool 1 -- list_automations
# ---------------------------------------------------------------------------

LIST_AUTOMATIONS_TOOL_DESCRIPTION = """List W&B Automations the current API key can access.

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
- The optional name filter is an exact match (not regex/substring).
- Results are capped by max_items (default 50, ceiling 200). When more exist,
  the response sets truncated=true.
- Each returned automation has:
    id, name, enabled, description, created_at, updated_at,
    scope:  {type, id, name}      -- PROJECT or ARTIFACT_COLLECTION
    event:  {type, filter}        -- e.g. RUN_METRIC threshold, ADD_ARTIFACT_ALIAS
    action: {type, ...}           -- NOTIFICATION (Slack), GENERIC_WEBHOOK, NO_OP
- The wandb GraphQL schema only stores the scope's id + name; it does NOT
  return the parent project/entity on the scope itself. If you need that
  context, use the `entity` you passed in (for the scope) and look up the
  project separately.
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
    Maximum automations to return. Default: 50, max: 200.

Returns
-------
JSON with:
  - automations: list of automation objects (see fields above)
  - count: number of automations returned
  - entity: the entity filter applied (or null)
  - truncated: whether more automations exist beyond max_items
"""


def _serialize_scope(scope: Any) -> dict[str, Any]:
    """Flatten an AutomationScope to ``{type, id, name}``.

    The wandb GraphQL fragments only carry ``id`` and ``name`` on scopes
    (see ``wandb/automations/_generated/fragments.py:ProjectScopeFields``,
    ``ArtifactSequenceScopeFields``, ``ArtifactPortfolioScopeFields``).
    ``scope_type`` is the public enum (``PROJECT`` or ``ARTIFACT_COLLECTION``)
    added by ``wandb.automations.scopes``.
    """
    return {
        "type": scope.scope_type.value,
        "id": scope.id,
        "name": scope.name,
    }


def _serialize_metric_filter(metric: Any) -> dict[str, Any]:
    """Flatten the inner metric filter from a RunMetricFilter wrapper.

    A ``RunMetricFilter.metric`` is one of three pydantic wrapper variants
    (``_WrappedMetricThresholdFilter`` / ``_WrappedMetricChangeFilter`` /
    ``_WrappedMetricZScoreFilter``); each defines exactly one of the
    ``threshold_filter`` / ``change_filter`` / ``zscore_filter`` attributes.
    We discriminate by attribute presence rather than reaching into wandb's
    private union types.
    """
    if (inner := getattr(metric, "threshold_filter", None)) is not None:
        return {
            "kind": "threshold",
            "metric": inner.name,
            "agg": inner.agg.value if inner.agg else None,
            "window": inner.window,
            "cmp": inner.cmp,
            "threshold": inner.threshold,
        }
    if (inner := getattr(metric, "change_filter", None)) is not None:
        return {
            "kind": "change",
            "metric": inner.name,
            "agg": inner.agg.value if inner.agg else None,
            "current_window": inner.window,
            "prior_window": inner.prior_window,
            "change_type": inner.change_type.value,
            "change_dir": inner.change_dir.value,
            "threshold": inner.threshold,
        }
    if (inner := getattr(metric, "zscore_filter", None)) is not None:
        return {
            "kind": "zscore",
            "metric": inner.name,
            "window": inner.window,
            "change_dir": inner.change_dir.value,
            "threshold": inner.threshold,
        }
    return {"kind": "unknown"}


def _serialize_state_filter(state: Any) -> dict[str, Any]:
    """Flatten a run-state filter to ``{states: [...]}``.

    ``StateFilter`` (from wandb.automations._filters.run_states) carries an
    in-list of states. We expose them as plain strings for easy LLM use.
    """
    states = getattr(state, "states", None)
    if states is None:
        # Older saved automations may store the state differently; fall back to repr.
        return {"summary": repr(state)}
    return {"states": [s.value if hasattr(s, "value") else str(s) for s in states]}


def _serialize_event(event: Any) -> dict[str, Any]:
    """Flatten a SavedEvent to ``{type, filter}``.

    ``event.filter`` is one of:
    - ``_WrappedSavedEventFilter`` for mutation events (CREATE_ARTIFACT etc.),
      whose ``.filter`` is a MongoLikeFilter -- we just expose it as a brief
      string summary, since the structure is open-ended.
    - ``RunMetricFilter`` for RUN_METRIC* events, whose ``.metric`` holds the
      typed threshold/change/zscore filter we flatten via _serialize_metric_filter.
    - ``RunStateFilter`` for RUN_STATE events, whose ``.state`` holds the
      state filter we flatten via _serialize_state_filter.
    """
    type_str = event.event_type.value
    raw_filter = event.filter

    metric = getattr(raw_filter, "metric", None)
    if metric is not None:
        return {"type": type_str, "filter": _serialize_metric_filter(metric)}

    state = getattr(raw_filter, "state", None)
    if state is not None:
        return {"type": type_str, "filter": _serialize_state_filter(state)}

    # Mutation-event filter (And()/MongoLikeFilter) -- no public structured shape.
    return {"type": type_str, "filter": {"summary": repr(raw_filter)}}


def _serialize_action(action: Any) -> dict[str, Any]:
    """Flatten a SavedAction to ``{type, ...}``.

    Saved actions are a discriminated union on ``action_type``
    (``NOTIFICATION`` / ``GENERIC_WEBHOOK`` / ``NO_OP`` / ``QUEUE_JOB``).
    NOTIFICATION carries title/message/severity; GENERIC_WEBHOOK carries
    request_payload; both reference an integration by id.
    """
    type_str = action.action_type.value
    out: dict[str, Any] = {"type": type_str}

    integration = getattr(action, "integration", None)
    if integration is not None:
        out["integration_id"] = integration.id

    if type_str == "NOTIFICATION":
        out["title"] = action.title
        out["message"] = action.message
        out["severity"] = action.severity.value if action.severity else None
    elif type_str == "GENERIC_WEBHOOK":
        # request_payload is JSON-encoded on the wire; the SDK already
        # parses it back into a dict via JsonEncoded[dict[str, Any]].
        out["request_payload"] = getattr(action, "request_payload", None)

    return out


def _serialize_automation(automation: Any) -> dict[str, Any]:
    """Flatten an Automation pydantic object to a JSON-safe dict."""
    created_at = automation.created_at
    updated_at = automation.updated_at
    return {
        "id": automation.id,
        "name": automation.name,
        "enabled": automation.enabled,
        "description": automation.description,
        "created_at": created_at.isoformat(),
        "updated_at": updated_at.isoformat() if updated_at is not None else None,
        "scope": _serialize_scope(automation.scope),
        "event": _serialize_event(automation.event),
        "action": _serialize_action(automation.action),
    }


def list_automations(
    entity: str | None = None,
    name: str | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List W&B Automations accessible with the current API key."""

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "list_automations",
        api.viewer,
        {"entity": entity, "name": name, "max_items": max_items},
    ) as ctx:
        max_items = min(max_items, MAX_ITEMS_CEILING)

        try:
            kwargs: dict[str, Any] = {"per_page": min(max_items, 100)}
            if entity is not None:
                kwargs["entity"] = entity
            if name is not None:
                kwargs["name"] = name

            automations: list[dict[str, Any]] = []
            truncated = False
            for auto in api.automations(**kwargs):
                if len(automations) >= max_items:
                    truncated = True
                    break
                automations.append(_serialize_automation(auto))

            return json.dumps(
                {
                    "automations": automations,
                    "count": len(automations),
                    "entity": entity,
                    "truncated": truncated,
                }
            )

        except Exception as e:
            logger.error(f"Error in list_automations: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


# ---------------------------------------------------------------------------
# Tool 2 -- list_integrations
# ---------------------------------------------------------------------------

LIST_INTEGRATIONS_TOOL_DESCRIPTION = """List W&B integrations (Slack channels and webhooks) for an entity.

Integrations are the destinations that W&B Automations send notifications to.
A SlackIntegration represents a connected Slack channel; a WebhookIntegration
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
- Integrations are configured at the entity (team) level via the W&B UI; this
  tool only lists existing integrations, it does not create them.
- entity=None defaults to the authenticated viewer's default entity.
- integration_type filters: "slack" returns only Slack integrations,
  "webhook" returns only generic webhook integrations, omit/null returns both.
- Each returned record always has {id, type}. Slack adds {team_name,
  channel_name}; webhook adds {name, url_endpoint}.
</critical_info>

Parameters
----------
entity : str, optional
    W&B entity (team or user). Omit to use the viewer's default entity.
integration_type : str, optional
    "slack" or "webhook". Omit to return both kinds.
max_items : int, optional
    Maximum integrations to return. Default: 50, max: 200.

Returns
-------
JSON with:
  - integrations: list of integration objects
  - count: number of integrations returned
  - entity: the entity filter applied (or null)
  - integration_type: the type filter applied (or null)
  - truncated: whether more integrations exist beyond max_items
"""


def _serialize_integration(integration: Any) -> dict[str, Any]:
    """Flatten a SlackIntegration or WebhookIntegration to a JSON-safe dict.

    The Integration union is discriminated by the GraphQL ``__typename``
    field, exposed on the SDK pydantic models as ``typename__``. Both
    variants share ``id``; Slack adds ``team_name`` / ``channel_name``,
    webhook adds ``name`` / ``url_endpoint``.
    """
    typename = integration.typename__
    if typename == _TYPENAME_SLACK_INTEGRATION:
        return {
            "id": integration.id,
            "type": _SLACK,
            "team_name": integration.team_name,
            "channel_name": integration.channel_name,
        }
    if typename == _TYPENAME_WEBHOOK_INTEGRATION:
        return {
            "id": integration.id,
            "type": _WEBHOOK,
            "name": integration.name,
            "url_endpoint": integration.url_endpoint,
        }
    # Forward-compat: surface unknown kinds with at least an id + raw typename
    # rather than dropping or erroring, so future integration kinds added on
    # the server side don't break this tool. ``id`` is on every Integration
    # subclass in the generated fragments.
    return {"id": integration.id, "type": typename}


def list_integrations(
    entity: str | None = None,
    integration_type: str | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List Slack and/or webhook integrations for an entity."""

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "list_integrations",
        api.viewer,
        {"entity": entity, "integration_type": integration_type, "max_items": max_items},
    ) as ctx:
        max_items = min(max_items, MAX_ITEMS_CEILING)

        if integration_type is not None and integration_type not in _VALID_INTEGRATION_TYPES:
            ctx.mark_error(f"invalid integration_type: {integration_type!r}")
            return json.dumps(
                {
                    "error": "invalid_input",
                    "message": (
                        f"integration_type must be one of {sorted(_VALID_INTEGRATION_TYPES)} or null, "
                        f"got {integration_type!r}"
                    ),
                }
            )

        try:
            kwargs: dict[str, Any] = {"per_page": min(max_items, 100)}
            if entity is not None:
                kwargs["entity"] = entity

            if integration_type == _SLACK:
                source = api.slack_integrations(**kwargs)
            elif integration_type == _WEBHOOK:
                source = api.webhook_integrations(**kwargs)
            else:
                source = api.integrations(**kwargs)

            integrations: list[dict[str, Any]] = []
            truncated = False
            for item in source:
                if len(integrations) >= max_items:
                    truncated = True
                    break
                integrations.append(_serialize_integration(item))

            return json.dumps(
                {
                    "integrations": integrations,
                    "count": len(integrations),
                    "entity": entity,
                    "integration_type": integration_type,
                    "truncated": truncated,
                }
            )

        except Exception as e:
            logger.error(f"Error in list_integrations: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})
