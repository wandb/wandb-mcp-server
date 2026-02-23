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

import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

_SENSITIVE_PARAM_PATTERNS = ("api_key", "token", "secret", "password", "credential", "auth")

analytics_logger = logging.getLogger("wandb_mcp_server.analytics")
analytics_logger.setLevel(logging.INFO)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class AnalyticsTracker:
    """Emit structured analytics events for the MCP server.

    Events are written as structured JSON via the ``wandb_mcp_server.analytics``
    logger so that Cloud Logging can route them to BigQuery.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = (
            enabled
            and os.environ.get("MCP_ANALYTICS_DISABLED", "false").lower() != "true"
        )
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
        """Return the best available user identifier."""
        try:
            for attr in ("username", "entity", "email"):
                if hasattr(viewer_info, attr):
                    return getattr(viewer_info, attr)
            if isinstance(viewer_info, str):
                return viewer_info
            return str(viewer_info)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Param sanitisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strip sensitive keys and truncate large values."""
        if not params:
            return {}
        safe: Dict[str, Any] = {}
        for key, value in params.items():
            if any(p in key.lower() for p in _SENSITIVE_PARAM_PATTERNS):
                safe[key] = "<redacted>"
            elif isinstance(value, str) and len(value) > 200:
                safe[key] = f"<truncated:{len(value)} chars>"
            else:
                safe[key] = value
        return safe

    # ------------------------------------------------------------------
    # Event emitters
    # ------------------------------------------------------------------

    def _emit(self, event: Dict[str, Any], labels: Dict[str, str]) -> None:
        analytics_logger.info(
            "ANALYTICS_EVENT",
            extra={"json_fields": event, "labels": labels},
        )

    def track_user_session(
        self,
        session_id: str,
        viewer_info: Any,
        api_key_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a session start."""
        if not self.enabled:
            return
        try:
            email_domain = self._extract_email_domain(viewer_info)
            event = {
                "event_type": "user_session",
                "timestamp": _utcnow_iso(),
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
                "event_type": "tool_call",
                "timestamp": _utcnow_iso(),
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
                "event_type": "request",
                "timestamp": _utcnow_iso(),
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
    """Get or create the global analytics tracker."""
    global _analytics_tracker
    if _analytics_tracker is None:
        enabled = os.environ.get("MCP_ANALYTICS_ENABLED", "true").lower() == "true"
        _analytics_tracker = AnalyticsTracker(enabled=enabled)
    return _analytics_tracker


def reset_analytics_tracker() -> None:
    """Reset the global analytics tracker (for testing)."""
    global _analytics_tracker
    _analytics_tracker = None
