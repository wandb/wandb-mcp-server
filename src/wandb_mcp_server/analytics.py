"""
Analytics tracking for W&B MCP Server.

Implements structured logging for Cloud Logging -> BigQuery pipeline.
Tracks user activity, tool calls, and session metrics.

Events are logged in JSON format with the following structure:
- analytics.user_session: User login/session start
- analytics.tool_call: MCP tool invocation
- analytics.request: Individual API request

These logs can be exported to BigQuery for analysis of:
1. Unique users by email domain
2. Tool call distribution
3. Weekly active users and retention
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

# Analytics logger writes structured JSON for Cloud Logging
analytics_logger = logging.getLogger("wandb_mcp_server.analytics")
analytics_logger.setLevel(logging.INFO)


class AnalyticsTracker:
    """
    Tracks analytics events for the MCP server.

    Events are logged as structured JSON to Cloud Logging, which can be
    exported to BigQuery for analysis.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the analytics tracker.

        Args:
            enabled: Whether analytics tracking is enabled
        """
        self.enabled = enabled and os.environ.get("MCP_ANALYTICS_DISABLED", "false").lower() != "true"

        if not self.enabled:
            logger.info("Analytics tracking is disabled")

    def _extract_email_domain(self, viewer_info: Any) -> Optional[str]:
        """
        Extract email domain from W&B viewer info.

        Args:
            viewer_info: The wandb.Api().viewer object or email string

        Returns:
            Email domain (e.g., "anthropic.com") or None
        """
        try:
            # Handle different viewer formats
            email = None
            if isinstance(viewer_info, str):
                email = viewer_info
            elif hasattr(viewer_info, 'email'):
                email = viewer_info.email
            elif hasattr(viewer_info, '__dict__') and 'email' in viewer_info.__dict__:
                email = viewer_info.__dict__['email']
            elif isinstance(viewer_info, dict) and 'email' in viewer_info:
                email = viewer_info['email']

            if email and '@' in email:
                return email.split('@')[1].lower()

            return None
        except Exception as e:
            logger.debug(f"Could not extract email domain: {e}")
            return None

    def _extract_user_id(self, viewer_info: Any) -> Optional[str]:
        """
        Extract user ID from W&B viewer info.

        Args:
            viewer_info: The wandb.Api().viewer object

        Returns:
            User ID or email
        """
        try:
            # Try to get username first
            if hasattr(viewer_info, 'username'):
                return viewer_info.username
            elif hasattr(viewer_info, 'entity'):
                return viewer_info.entity
            elif hasattr(viewer_info, 'email'):
                return viewer_info.email
            elif isinstance(viewer_info, str):
                return viewer_info

            return str(viewer_info)
        except Exception as e:
            logger.debug(f"Could not extract user ID: {e}")
            return None

    def track_user_session(
        self,
        session_id: str,
        viewer_info: Any,
        api_key_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a user session start.

        Args:
            session_id: Unique session identifier
            viewer_info: W&B viewer information
            api_key_hash: Hashed API key (for debugging, never log raw key)
            metadata: Additional session metadata
        """
        if not self.enabled:
            return

        try:
            user_id = self._extract_user_id(viewer_info)
            email_domain = self._extract_email_domain(viewer_info)

            event = {
                "event_type": "user_session",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "api_key_hash": api_key_hash[:16] if api_key_hash else None,  # Only log prefix
                "metadata": metadata or {}
            }

            # Log as JSON for Cloud Logging structured logging
            analytics_logger.info(
                "ANALYTICS_EVENT",
                extra={
                    "json_fields": event,
                    "labels": {
                        "event_type": "user_session",
                        "email_domain": email_domain or "unknown"
                    }
                }
            )

        except Exception as e:
            logger.warning(f"Failed to track user session: {e}")

    def track_tool_call(
        self,
        tool_name: str,
        session_id: Optional[str],
        viewer_info: Any,
        params: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Track an MCP tool call.

        Args:
            tool_name: Name of the tool called
            session_id: Session identifier
            viewer_info: W&B viewer information
            params: Tool parameters (will be sanitized)
            success: Whether the call succeeded
            error: Error message if failed
            duration_ms: Call duration in milliseconds
        """
        if not self.enabled:
            return

        try:
            user_id = self._extract_user_id(viewer_info)
            email_domain = self._extract_email_domain(viewer_info)

            # Sanitize params - remove sensitive data
            safe_params = {}
            if params:
                for key, value in params.items():
                    # Skip API keys and large data
                    if 'api_key' in key.lower() or 'token' in key.lower():
                        safe_params[key] = "<redacted>"
                    elif isinstance(value, str) and len(value) > 200:
                        safe_params[key] = f"<truncated:{len(value)} chars>"
                    else:
                        safe_params[key] = value

            event = {
                "event_type": "tool_call",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "tool_name": tool_name,
                "params": safe_params,
                "success": success,
                "error": error,
                "duration_ms": duration_ms
            }

            # Log as JSON for Cloud Logging structured logging
            analytics_logger.info(
                "ANALYTICS_EVENT",
                extra={
                    "json_fields": event,
                    "labels": {
                        "event_type": "tool_call",
                        "tool_name": tool_name,
                        "email_domain": email_domain or "unknown",
                        "success": str(success)
                    }
                }
            )

        except Exception as e:
            logger.warning(f"Failed to track tool call: {e}")

    def track_request(
        self,
        request_id: str,
        session_id: Optional[str],
        method: str,
        path: str,
        status_code: int,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        email_domain: Optional[str] = None
    ) -> None:
        """
        Track an HTTP request.

        Args:
            request_id: Unique request identifier
            session_id: Session identifier
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: User identifier
            email_domain: User email domain
        """
        if not self.enabled:
            return

        try:
            event = {
                "event_type": "request",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms
            }

            # Log as JSON for Cloud Logging structured logging
            analytics_logger.info(
                "ANALYTICS_EVENT",
                extra={
                    "json_fields": event,
                    "labels": {
                        "event_type": "request",
                        "email_domain": email_domain or "unknown",
                        "status_code": str(status_code)
                    }
                }
            )

        except Exception as e:
            logger.warning(f"Failed to track request: {e}")


# Global analytics tracker instance
_analytics_tracker: Optional[AnalyticsTracker] = None


def get_analytics_tracker() -> AnalyticsTracker:
    """Get or create the global analytics tracker."""
    global _analytics_tracker
    if _analytics_tracker is None:
        enabled = os.environ.get("MCP_ANALYTICS_ENABLED", "true").lower() == "true"
        _analytics_tracker = AnalyticsTracker(enabled=enabled)
    return _analytics_tracker


def reset_analytics_tracker():
    """Reset the global analytics tracker (for testing)."""
    global _analytics_tracker
    _analytics_tracker = None
