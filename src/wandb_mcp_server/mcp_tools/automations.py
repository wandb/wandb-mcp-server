"""List W&B Automations and the integrations they can target.

Provides two read-only tools for discovering Automations and the
Slack/webhook integrations that Automation actions reference, via the
public ``wandb.Api`` interface introduced in wandb 0.19.11.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_ITEMS = 50
MAX_ITEMS_CEILING = 200

# Valid integration_type values for list_integrations.
_SLACK = "slack"
_WEBHOOK = "webhook"
_VALID_INTEGRATION_TYPES = {_SLACK, _WEBHOOK}


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
    scope:  {type, name, ...}     -- PROJECT or ARTIFACT_COLLECTION
    event:  {type, summary}       -- e.g. RUN_METRIC threshold, ADD_ARTIFACT_ALIAS
    action: {type, ...}           -- NOTIFICATION (Slack), GENERIC_WEBHOOK, NO_OP
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


def _serialize_scope(scope: Any) -> Dict[str, Any]:
    """Flatten an AutomationScope (ProjectScope or ArtifactCollectionScope) to a dict.

    Uses getattr with defaults so any future scope-type addition still serializes
    without raising. The ``scope_type`` field is set by the SDK on every scope
    subclass (see wandb/wandb/automations/scopes.py).
    """
    scope_type = getattr(scope, "scope_type", None)
    out: Dict[str, Any] = {
        "type": getattr(scope_type, "value", str(scope_type)) if scope_type is not None else None,
        "name": getattr(scope, "name", None),
    }
    # ProjectScope carries entity/project info directly; ArtifactCollection scopes
    # carry it on a nested project field. Try both flat and nested locations.
    project = getattr(scope, "project", None)
    if project is not None and not isinstance(project, str):
        out["project"] = getattr(project, "name", None)
        out["entity"] = getattr(project, "entity_name", None) or getattr(project, "entity", None)
    else:
        out["project"] = project
        out["entity"] = getattr(scope, "entity_name", None) or getattr(scope, "entity", None)
    return out


def _serialize_event(event: Any) -> Dict[str, Any]:
    """Flatten a SavedEvent into a small dict with type + human-readable summary."""
    event_type = getattr(event, "event_type", None)
    type_str = getattr(event_type, "value", str(event_type)) if event_type is not None else None

    # The filter on a SavedEvent is either a _WrappedSavedEventFilter (mutation
    # events) or a RunMetricFilter / RunStateFilter. For each, prefer the
    # built-in __repr__ on the innermost metric/state filter, which produces a
    # compact human-readable string (see _filters/run_metrics.py:147-150).
    summary: Optional[str] = None
    filt = getattr(event, "filter", None)
    try:
        inner_metric = getattr(filt, "metric", None)
        inner_state = getattr(filt, "state", None)
        if inner_metric is not None:
            inner = (
                getattr(inner_metric, "threshold_filter", None)
                or getattr(inner_metric, "change_filter", None)
                or getattr(inner_metric, "zscore_filter", None)
                or inner_metric
            )
            summary = repr(inner).strip("'")
        elif inner_state is not None:
            summary = repr(inner_state).strip("'")
        elif filt is not None:
            summary = repr(filt).strip("'")
    except Exception:
        summary = None

    return {"type": type_str, "summary": summary}


def _serialize_action(action: Any) -> Dict[str, Any]:
    """Flatten a SavedAction into a dict with type + the fields agents need."""
    action_type = getattr(action, "action_type", None)
    type_str = getattr(action_type, "value", str(action_type)) if action_type is not None else None
    out: Dict[str, Any] = {"type": type_str}

    integration = getattr(action, "integration", None)
    if integration is not None:
        out["integration_id"] = getattr(integration, "id", None)

    # NOTIFICATION (Slack) carries title/message/severity; webhook does not.
    if type_str == "NOTIFICATION":
        out["title"] = getattr(action, "title", None)
        out["message"] = getattr(action, "message", None)
        severity = getattr(action, "severity", None)
        out["severity"] = getattr(severity, "value", str(severity)) if severity is not None else None
    return out


def _serialize_automation(automation: Any) -> Dict[str, Any]:
    """Flatten an Automation pydantic object into a JSON-safe dict."""
    created_at = getattr(automation, "created_at", None)
    updated_at = getattr(automation, "updated_at", None)
    return {
        "id": getattr(automation, "id", None),
        "name": getattr(automation, "name", None),
        "enabled": getattr(automation, "enabled", None),
        "description": getattr(automation, "description", None),
        "created_at": created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else (str(created_at) if created_at is not None else None),
        "updated_at": updated_at.isoformat()
        if hasattr(updated_at, "isoformat")
        else (str(updated_at) if updated_at is not None else None),
        "scope": _serialize_scope(getattr(automation, "scope", None)),
        "event": _serialize_event(getattr(automation, "event", None)),
        "action": _serialize_action(getattr(automation, "action", None)),
    }


def list_automations(
    entity: Optional[str] = None,
    name: Optional[str] = None,
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
            kwargs: Dict[str, Any] = {"per_page": min(max_items, 100)}
            if entity is not None:
                kwargs["entity"] = entity
            if name is not None:
                kwargs["name"] = name

            automations: List[Dict[str, Any]] = []
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


def _serialize_integration(integration: Any) -> Dict[str, Any]:
    """Flatten a SlackIntegration or WebhookIntegration into a JSON-safe dict.

    The two integration kinds are discriminated by the GraphQL ``__typename``
    field, exposed in the SDK as ``typename__``. Slack integrations carry
    team_name + channel_name; webhook integrations carry name + url_endpoint.
    """
    typename = getattr(integration, "typename__", None)
    if typename == "SlackIntegration":
        return {
            "id": getattr(integration, "id", None),
            "type": _SLACK,
            "team_name": getattr(integration, "team_name", None),
            "channel_name": getattr(integration, "channel_name", None),
        }
    if typename == "GenericWebhookIntegration":
        return {
            "id": getattr(integration, "id", None),
            "type": _WEBHOOK,
            "name": getattr(integration, "name", None),
            "url_endpoint": getattr(integration, "url_endpoint", None),
        }
    # Fallback for unknown future integration kinds: surface what we can.
    return {
        "id": getattr(integration, "id", None),
        "type": typename or "unknown",
    }


def list_integrations(
    entity: Optional[str] = None,
    integration_type: Optional[str] = None,
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
            kwargs: Dict[str, Any] = {"per_page": min(max_items, 100)}
            if entity is not None:
                kwargs["entity"] = entity

            if integration_type == _SLACK:
                source = api.slack_integrations(**kwargs)
            elif integration_type == _WEBHOOK:
                source = api.webhook_integrations(**kwargs)
            else:
                source = api.integrations(**kwargs)

            integrations: List[Dict[str, Any]] = []
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
