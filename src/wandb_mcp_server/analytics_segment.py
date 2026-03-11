"""Gorilla /analytics/t (Segment Track) compatibility layer.

Maps internal MCP analytics events to the payload shape expected by
Gorilla's ``POST /analytics/t`` endpoint, which forwards to Segment.

The Gorilla handler decodes a ``segmentio/analytics-go/v3.Track`` struct::

    {
        "userId":     "<required, skipped if empty>",
        "event":      "<event name>",
        "properties": { ... },
        "timestamp":  "<optional ISO-8601>"
    }

This module provides:
- Pure mapper functions (no network calls).
- A gated ``SegmentForwarder`` that can dry-run or POST mapped payloads.
- Automatic integration with ``AnalyticsTracker._emit()`` via singleton.

Enable dry-run logging with ``MCP_SEGMENT_DRY_RUN=true``.
Enable live forwarding with ``MCP_SEGMENT_FORWARD=true`` + ``WANDB_BASE_URL``.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

SEGMENT_EVENT_PREFIX = "mcp_server"

_EVENT_NAME_MAP: Dict[str, str] = {
    "user_session": f"{SEGMENT_EVENT_PREFIX}.session_start",
    "tool_call": f"{SEGMENT_EVENT_PREFIX}.tool_call",
    "request": f"{SEGMENT_EVENT_PREFIX}.http_request",
}

_SESSION_PROPERTY_KEYS: List[str] = [
    "session_id",
    "email_domain",
    "api_key_hash",
    "metadata",
]

_TOOL_CALL_PROPERTY_KEYS: List[str] = [
    "session_id",
    "tool_name",
    "params",
    "success",
    "error",
    "duration_ms",
]

_REQUEST_PROPERTY_KEYS: List[str] = [
    "session_id",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
]


def map_to_segment_track(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an internal analytics event dict to a Segment Track payload.

    Returns None if the event cannot be mapped (missing userId or unknown type).

    Args:
        event: Internal analytics event dict (as emitted by AnalyticsTracker).

    Returns:
        Segment Track-compatible dict, or None if unmappable.
    """
    event_type = event.get("event_type")
    segment_event_name = _EVENT_NAME_MAP.get(event_type)
    if not segment_event_name:
        return None

    user_id = event.get("user_id") or "anonymous"

    property_keys = {
        "user_session": _SESSION_PROPERTY_KEYS,
        "tool_call": _TOOL_CALL_PROPERTY_KEYS,
        "request": _REQUEST_PROPERTY_KEYS,
    }.get(event_type, [])

    properties: Dict[str, Any] = {
        "schema_version": event.get("schema_version", "1.0"),
        "source": "wandb-mcp-server",
    }
    for key in property_keys:
        if key in event:
            properties[key] = event[key]

    track_payload: Dict[str, Any] = {
        "userId": user_id,
        "event": segment_event_name,
        "properties": properties,
    }

    ts_raw = event.get("timestamp")
    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw)
            track_payload["timestamp"] = dt.isoformat()
        except (ValueError, TypeError):
            pass

    return track_payload


def _build_retry_session() -> requests.Session:
    """Build a requests session with retry logic for Gorilla POSTs."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class SegmentForwarder:
    """Gated forwarder that maps and optionally sends events to Gorilla /analytics/t.

    Modes (controlled by env vars):
    - Off (default): does nothing.
    - Dry-run (``MCP_SEGMENT_DRY_RUN=true``): logs mapped payloads without sending.
    - Live (``MCP_SEGMENT_FORWARD=true``): POSTs to ``{WANDB_BASE_URL}/analytics/t``.

    Live POSTs run in a daemon thread so they never block the MCP request path.

    Args:
        base_url: Override for WANDB_BASE_URL. If None, reads from env.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.dry_run = os.environ.get("MCP_SEGMENT_DRY_RUN", "false").lower() == "true"
        self.live = os.environ.get("MCP_SEGMENT_FORWARD", "false").lower() == "true"
        self.base_url = (base_url or os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).rstrip("/")
        self._segment_logger = logging.getLogger("wandb_mcp_server.segment_dryrun")
        self._segment_logger.setLevel(logging.INFO)
        self._forwarded_payloads: List[Dict[str, Any]] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._thread_local = threading.local()

    @property
    def enabled(self) -> bool:
        """True if either dry-run or live mode is on."""
        return self.dry_run or self.live

    def forward(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map an internal event and forward it (or dry-run log it).

        Returns the mapped payload if forwarding was attempted, None otherwise.
        Live POSTs are dispatched to a background thread.
        """
        if not self.enabled:
            return None

        payload = map_to_segment_track(event)
        if payload is None:
            return None

        if self.dry_run:
            self._forwarded_payloads.append(payload)
            self._segment_logger.info(
                "SEGMENT_DRY_RUN",
                extra={"json_fields": payload},
            )
            return payload

        if self.live:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=4)
            self._executor.submit(self._post, payload)
            return payload

        return None

    def _post(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST the payload to Gorilla /analytics/t (called from thread pool)."""
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = _build_retry_session()
            self._thread_local.session = session

        url = f"{self.base_url}/analytics/t"
        try:
            resp = session.post(
                url,
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.warning(f"Segment forward failed: {resp.status_code} {resp.text[:200]}")
            return payload
        except Exception as exc:
            logger.warning(f"Segment forward error (non-fatal): {exc}")
            return payload

    def get_forwarded_payloads(self) -> List[Dict[str, Any]]:
        """Return all payloads that were forwarded (for testing/inspection)."""
        return list(self._forwarded_payloads)

    def clear_forwarded_payloads(self) -> None:
        """Clear the forwarded payloads buffer."""
        self._forwarded_payloads.clear()


# -- Singleton access -------------------------------------------------------

_segment_forwarder: Optional[SegmentForwarder] = None


def get_segment_forwarder() -> SegmentForwarder:
    """Get or create the global SegmentForwarder singleton."""
    global _segment_forwarder
    if _segment_forwarder is None:
        _segment_forwarder = SegmentForwarder()
    return _segment_forwarder


def reset_segment_forwarder() -> None:
    """Reset the global SegmentForwarder (for testing)."""
    global _segment_forwarder
    _segment_forwarder = None
