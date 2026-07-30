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
Enable live forwarding with ``MCP_SEGMENT_FORWARD=true``. Server-side delivery
uses ``WANDB_INTERNAL_BASE_URL`` when configured, otherwise ``WANDB_BASE_URL``.
"""

import logging
import os
import threading
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import requests

from wandb_mcp_server.bounded_worker import BoundedWorkerQueue
from wandb_mcp_server.config import (
    MCP_ANALYTICS_QUEUE_CAPACITY,
    MCP_ANALYTICS_TEST_BUFFER_CAPACITY,
)
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

SEGMENT_EVENT_PREFIX = "mcp_server"

_EVENT_NAME_MAP: Dict[str, str] = {
    "user_session": f"{SEGMENT_EVENT_PREFIX}.session_start",
    "tool_call": f"{SEGMENT_EVENT_PREFIX}.tool_call",
}

_SESSION_PROPERTY_KEYS: List[str] = [
    "session_id",
    "metadata",
    "mcp_client_version",
    "mcp_client_confidence",
]

_BASE_PROPERTY_KEYS: List[str] = [
    "release_version",
    "deployment_id",
    "runtime_surface",
    "transport",
    "deployment_type",
    "environment",
    "hosted_mode",
    "wandb_base_host",
    "agent_harness",
    "client_vendor",
    "call_type",
    "mcp_client_family",
    "mcp_client_app",
    "mcp_client_source",
    "mcp_protocol_version",
    "mcp_jsonrpc_method",
]

_TOOL_CALL_PROPERTY_KEYS: List[str] = [
    "session_id",
    "tool_name",
    "mcp_tool_name",
    "usage_dimensions",
    "success",
    "error",
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

    user_id = event.get("actor_id") or event.get("user_id") or "anonymous"

    property_keys = {
        "user_session": _SESSION_PROPERTY_KEYS,
        "tool_call": _TOOL_CALL_PROPERTY_KEYS,
    }.get(event_type, [])

    properties: Dict[str, Any] = {
        "schema_version": event.get("schema_version", "1.1"),
        "source": "wandb-mcp-server",
    }
    for key in [*_BASE_PROPERTY_KEYS, *property_keys]:
        if key in event and event[key] not in (None, "", {}, []):
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
    """Build a single-attempt session for best-effort Gorilla analytics.

    Analytics must not amplify W&B overloads. A failed event remains in the
    canonical local log and is dropped after this one forwarding attempt.
    """
    return requests.Session()


class SegmentForwarder:
    """Gated forwarder that maps and optionally sends events to Gorilla /analytics/t.

    Modes (controlled by env vars):
    - Off (default): does nothing.
    - Dry-run (``MCP_SEGMENT_DRY_RUN=true``): logs mapped payloads without sending.
    - Live (``MCP_SEGMENT_FORWARD=true``): POSTs to the resolved W&B API URL.

    Live POSTs run in a daemon thread so they never block the MCP request path.

    Args:
        base_url: Explicit API URL override. If None, internal then public env wins.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.dry_run = os.environ.get("MCP_SEGMENT_DRY_RUN", "false").lower() == "true"
        self.live = os.environ.get("MCP_SEGMENT_FORWARD", "false").lower() == "true"
        self.base_url = (
            base_url
            or os.environ.get("WANDB_INTERNAL_BASE_URL")
            or os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
        ).rstrip("/")
        self._segment_logger = logging.getLogger("wandb_mcp_server.segment_dryrun")
        self._segment_logger.setLevel(logging.INFO)
        self._forwarded_payloads: Deque[Dict[str, Any]] = deque(maxlen=MCP_ANALYTICS_TEST_BUFFER_CAPACITY)
        self._executor: Optional[BoundedWorkerQueue[Dict[str, Any]]] = None
        self._executor_lock = threading.Lock()
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
            executor = self._get_executor()
            executor.submit(payload)
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

    @property
    def dropped_count(self) -> int:
        """Number of forwarding events dropped because the queue was full."""
        return self._executor.dropped_count if self._executor is not None else 0

    def _get_executor(self) -> BoundedWorkerQueue[Dict[str, Any]]:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = BoundedWorkerQueue(
                        self._post,
                        capacity=MCP_ANALYTICS_QUEUE_CAPACITY,
                        worker_count=4,
                        on_drop=lambda count: logger.warning(
                            "Segment forwarding queue full; dropped_count=%s",
                            count,
                        ),
                        thread_name_prefix="mcp-segment",
                    )
        return self._executor


# -- Singleton access -------------------------------------------------------

_segment_forwarder: Optional[SegmentForwarder] = None
_segment_forwarder_lock = threading.Lock()


def get_segment_forwarder() -> SegmentForwarder:
    """Get or create the global SegmentForwarder singleton."""
    global _segment_forwarder
    if _segment_forwarder is None:
        with _segment_forwarder_lock:
            if _segment_forwarder is None:
                _segment_forwarder = SegmentForwarder()
    return _segment_forwarder


def reset_segment_forwarder() -> None:
    """Reset the global SegmentForwarder (for testing)."""
    global _segment_forwarder
    with _segment_forwarder_lock:
        previous = _segment_forwarder
        _segment_forwarder = None
    if previous is not None and previous._executor is not None:
        previous._executor.shutdown(wait=False, cancel_futures=True)
