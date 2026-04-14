"""Analytics tracking for W&B MCP Server.

Structured logging pipeline: Cloud Run -> Cloud Logging -> BigQuery -> Hex.

Event types:
  user_session  -- user login / session start
  tool_call     -- MCP tool invocation (params sanitised)
  request       -- individual HTTP request

Disable with ``MCP_ANALYTICS_DISABLED=true`` env var.

Based on prior art by @NiWaRe (PR #2), rewritten for improved
datetime handling, cleaner auth integration, and structured event schema.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

SCHEMA_VERSION = "1.0"

_SENSITIVE_PARAM_PATTERNS: List[str] = [
    "api_key",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
]

_MAX_PARAM_VALUE_LENGTH = 200


class _StructuredJsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for Cloud Logging ingestion.

    Cloud Run's logging agent parses stdout lines as jsonPayload when
    they are valid JSON with a ``severity`` field. This lets BigQuery
    and Hex query analytics fields (event_type, tool_name, session_id,
    etc.) directly without regex.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {}
        if hasattr(record, "json_fields"):
            payload.update(record.json_fields)
        if hasattr(record, "labels"):
            payload["labels"] = record.labels
        payload["severity"] = record.levelname
        payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


analytics_logger = logging.getLogger("wandb_mcp_server.analytics")
analytics_logger.setLevel(logging.INFO)
analytics_logger.propagate = False

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_StructuredJsonFormatter())
analytics_logger.addHandler(_handler)

_REQUIRED_BASE_FIELDS = frozenset({"schema_version", "event_type", "timestamp"})


def _utcnow_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


class AnalyticsTracker:
    """Emit structured analytics events for the MCP server.

    Events are written as structured JSON via the ``wandb_mcp_server.analytics``
    logger so that Cloud Logging can route them to BigQuery.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and os.environ.get("MCP_ANALYTICS_DISABLED", "false").lower() != "true"
        if not self.enabled:
            logger.info("Analytics tracking is disabled")

    # ------------------------------------------------------------------
    # Viewer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_email_domain(viewer_info: Any) -> Optional[str]:
        """Return the email domain (e.g. ``anthropic.com``) or ``None``."""
        try:
            email = None
            if isinstance(viewer_info, str):
                email = viewer_info
            elif hasattr(viewer_info, "email"):
                email = viewer_info.email
            elif isinstance(viewer_info, dict) and "email" in viewer_info:
                email = viewer_info["email"]
            if email and "@" in email:
                return email.split("@")[1].lower()
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_user_id(viewer_info: Any) -> Optional[str]:
        """Return the best available non-PII user identifier.

        Prefers ``username`` > ``entity`` (the W&B team/org slug).
        Email is deliberately **not** returned to avoid logging PII;
        when only an email is available the domain portion is returned
        instead.  Raw string inputs are returned only when they do not
        look like email addresses.  Returns ``None`` for unrecognised
        types rather than stringifying arbitrary objects.
        """
        try:
            for attr in ("username", "entity"):
                if hasattr(viewer_info, attr):
                    val = getattr(viewer_info, attr)
                    if val:
                        return str(val)
            if hasattr(viewer_info, "email"):
                email = getattr(viewer_info, "email")
                if email and "@" in str(email):
                    return str(email).split("@")[1].lower()
            if isinstance(viewer_info, str):
                if "@" in viewer_info:
                    return viewer_info.split("@")[1].lower()
                return viewer_info
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Param sanitisation
    # ------------------------------------------------------------------

    @classmethod
    def _sanitise_params(cls, params: Optional[Dict[str, Any]], *, _depth: int = 0) -> Dict[str, Any]:
        """Strip sensitive keys and truncate large values.

        Recursively sanitises nested dicts and lists up to 3 levels deep.
        """
        if not params:
            return {}
        safe: Dict[str, Any] = {}
        for key, value in params.items():
            if any(p in key.lower() for p in _SENSITIVE_PARAM_PATTERNS):
                safe[key] = "<redacted>"
            elif isinstance(value, dict) and _depth < 3:
                safe[key] = cls._sanitise_params(value, _depth=_depth + 1)
            elif isinstance(value, list) and _depth < 3:
                safe[key] = cls._sanitise_list(value, _depth=_depth + 1)
            elif isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LENGTH:
                safe[key] = f"<truncated:{len(value)} chars>"
            else:
                safe[key] = value
        return safe

    @classmethod
    def _sanitise_list(cls, items: list, *, _depth: int = 0) -> list:
        """Sanitise each element in a list, recursing into dicts and nested lists."""
        result = []
        for item in items:
            if isinstance(item, dict) and _depth < 3:
                result.append(cls._sanitise_params(item, _depth=_depth))
            elif isinstance(item, list) and _depth < 3:
                result.append(cls._sanitise_list(item, _depth=_depth + 1))
            else:
                result.append(item)
        return result

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_event(event_type: str) -> Dict[str, Any]:
        """Build the required base fields present in every event."""
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "timestamp": _utcnow_iso(),
        }
        deployment_id = os.environ.get("MCP_DEPLOYMENT_ID")
        if deployment_id:
            event["deployment_id"] = deployment_id
        return event

    # ------------------------------------------------------------------
    # Event emitters
    # ------------------------------------------------------------------

    def _emit(self, event: Dict[str, Any], labels: Dict[str, str]) -> None:
        """Write a structured event to the analytics logger and forward to Segment + Datadog.

        Validates required base fields are present and catches
        serialisation errors so analytics never disrupts the server.
        The Segment and Datadog forwarders are called after Cloud Logging
        emission; each is gated by its own env vars and fails silently.
        """
        missing = _REQUIRED_BASE_FIELDS - event.keys()
        if missing:
            logger.warning(f"Analytics event missing required fields: {missing}")
            return
        try:
            analytics_logger.info(
                "ANALYTICS_EVENT",
                extra={"json_fields": event, "labels": labels},
            )
        except Exception as exc:
            logger.debug(f"Analytics emit failed (non-fatal): {exc}")

        try:
            from wandb_mcp_server.analytics_segment import get_segment_forwarder

            forwarder = get_segment_forwarder()
            if forwarder.enabled:
                forwarder.forward(event)
        except Exception as exc:
            logger.debug(f"Segment forwarding failed (non-fatal): {exc}")

        try:
            from wandb_mcp_server.analytics_datadog import get_datadog_forwarder

            dd_forwarder = get_datadog_forwarder()
            if dd_forwarder.enabled:
                dd_forwarder.forward(event)
        except Exception as exc:
            logger.debug(f"Datadog forwarding failed (non-fatal): {exc}")

    def track_user_session(
        self,
        session_id: str,
        viewer_info: Any,
        api_key_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a session start / heartbeat."""
        if not self.enabled:
            return
        try:
            email_domain = self._extract_email_domain(viewer_info)
            event = {
                **self._base_event("user_session"),
                "session_id": session_id,
                "user_id": self._extract_user_id(viewer_info),
                "email_domain": email_domain,
                "api_key_hash": api_key_hash[:16] if api_key_hash else None,
                "metadata": metadata or {},
            }
            self._emit(event, {"event_type": "user_session", "email_domain": email_domain or "unknown"})
        except Exception as exc:
            logger.warning(f"Failed to track user session: {exc}")

    def track_tool_call(
        self,
        tool_name: str,
        session_id: Optional[str],
        viewer_info: Any,
        params: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record an MCP tool invocation."""
        if not self.enabled:
            return
        try:
            email_domain = self._extract_email_domain(viewer_info)
            event = {
                **self._base_event("tool_call"),
                "session_id": session_id,
                "user_id": self._extract_user_id(viewer_info),
                "email_domain": email_domain,
                "tool_name": tool_name,
                "params": self._sanitise_params(params),
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
            }
            self._emit(
                event,
                {
                    "event_type": "tool_call",
                    "tool_name": tool_name,
                    "email_domain": email_domain or "unknown",
                    "success": str(success),
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to track tool call: {exc}")

    def track_request(
        self,
        request_id: str,
        session_id: Optional[str],
        method: str,
        path: str,
        status_code: int,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        email_domain: Optional[str] = None,
    ) -> None:
        """Record an HTTP request."""
        if not self.enabled:
            return
        try:
            event = {
                **self._base_event("request"),
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
            self._emit(
                event,
                {
                    "event_type": "request",
                    "email_domain": email_domain or "unknown",
                    "status_code": str(status_code),
                },
            )
        except Exception as exc:
            logger.warning(f"Failed to track request: {exc}")


# -- Singleton access -------------------------------------------------------

_analytics_tracker: Optional[AnalyticsTracker] = None


def get_analytics_tracker() -> AnalyticsTracker:
    """Get or create the global analytics tracker.

    Respects ``MCP_ANALYTICS_DISABLED=true`` to turn off tracking.
    """
    global _analytics_tracker
    if _analytics_tracker is None:
        _analytics_tracker = AnalyticsTracker(enabled=True)
    return _analytics_tracker


def reset_analytics_tracker() -> None:
    """Reset the global analytics tracker (for testing)."""
    global _analytics_tracker
    _analytics_tracker = None
