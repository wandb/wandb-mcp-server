"""Datadog HTTP Logs Intake forwarder.

Sends MCP analytics events directly to the Datadog Logs API via HTTP POST.
No Datadog Agent or sidecar required -- works on any platform including
serverless (Cloud Run, Lambda).

Enable with ``MCP_DATADOG_FORWARD=true`` + ``DD_API_KEY`` (from Secret Manager
on Cloud Run, or env var locally).

The forwarder POSTs to ``https://http-intake.logs.{DD_SITE}/api/v2/logs``
using the ``DD-API-KEY`` header. Each analytics event becomes one log entry
with structured JSON in the message field.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


def _build_retry_session() -> requests.Session:
    """Build a requests session with retry logic for Datadog POSTs."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _build_log_entry(event: Dict[str, Any], *, dd_env: str, dd_version: str, dd_service: str) -> Dict[str, Any]:
    """Convert an internal analytics event to a Datadog log entry.

    Args:
        event: Internal analytics event dict (as emitted by AnalyticsTracker).
        dd_env: Datadog environment tag (e.g. "staging", "production").
        dd_version: Service version tag (e.g. "0.3.0").
        dd_service: Service name tag.

    Returns:
        Dict suitable for the Datadog HTTP Logs Intake API.
    """
    event_type = event.get("event_type", "unknown")
    tool_name = event.get("tool_name", "")
    user_id = event.get("user_id", "anonymous")
    success = event.get("success")

    tags = [f"env:{dd_env}", f"service:{dd_service}", f"version:{dd_version}"]
    tags.append(f"event_type:{event_type}")
    if tool_name:
        tags.append(f"tool_name:{tool_name}")
    if success is not None:
        tags.append(f"success:{success}")

    status = "info"
    if event.get("error"):
        status = "error"

    summary = f"mcp.{event_type}"
    if tool_name:
        summary += f": {tool_name}"
    if user_id and user_id != "anonymous":
        summary += f" by {user_id}"

    return {
        "ddsource": "python",
        "ddtags": ",".join(tags),
        "hostname": os.environ.get("K_REVISION", os.environ.get("HOSTNAME", "unknown")),
        "service": dd_service,
        "message": json.dumps(event, default=str),
        "status": status,
    }


class DatadogForwarder:
    """Gated forwarder that sends analytics events to the Datadog HTTP Logs API.

    Controlled by env vars:
    - ``MCP_DATADOG_FORWARD=true``: enable forwarding (off by default).
    - ``DD_API_KEY``: Datadog API key (32-char hex, NOT an Application Key).
    - ``DD_SITE``: Datadog site (default ``datadoghq.com``; use ``us5.datadoghq.com`` for US5).
    - ``DD_ENV``: environment tag (default ``production``).
    - ``DD_VERSION``: version tag (default ``0.0.0``).
    - ``DD_SERVICE``: service name tag (default ``wandb-mcp-server``).

    Live POSTs run in a daemon thread so they never block the MCP request path.
    """

    def __init__(self):
        self.live = os.environ.get("MCP_DATADOG_FORWARD", "false").lower() == "true"
        self._api_key = os.environ.get("DD_API_KEY", "")
        self._site = os.environ.get("DD_SITE", "datadoghq.com")
        self._env = os.environ.get("DD_ENV", "production")
        self._version = os.environ.get("DD_VERSION", "0.0.0")
        self._service = os.environ.get("DD_SERVICE", "wandb-mcp-server")
        self._intake_url = f"https://http-intake.logs.{self._site}/api/v2/logs"
        self._forwarded_payloads: List[Dict[str, Any]] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._thread_local = threading.local()

        if self.live and not self._api_key:
            logger.warning("MCP_DATADOG_FORWARD=true but DD_API_KEY is empty -- forwarding disabled")
            self.live = False

    @property
    def enabled(self) -> bool:
        """True if forwarding is on and the API key is set."""
        return self.live

    def forward(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Forward an analytics event to Datadog.

        Returns the log entry if forwarding was attempted, None otherwise.
        POSTs are dispatched to a background thread.
        """
        if not self.enabled:
            return None

        entry = _build_log_entry(
            event,
            dd_env=self._env,
            dd_version=self._version,
            dd_service=self._service,
        )

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)
        self._executor.submit(self._post, entry)
        self._forwarded_payloads.append(entry)
        return entry

    def _post(self, entry: Dict[str, Any]) -> None:
        """POST the log entry to the Datadog HTTP Logs Intake (called from thread pool)."""
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = _build_retry_session()
            self._thread_local.session = session

        try:
            resp = session.post(
                self._intake_url,
                json=[entry],
                timeout=5,
                headers={
                    "Content-Type": "application/json",
                    "DD-API-KEY": self._api_key,
                },
            )
            if resp.status_code not in (200, 202):
                logger.warning(f"Datadog forward failed: {resp.status_code} {resp.text[:200]}")
        except Exception as exc:
            logger.warning(f"Datadog forward error (non-fatal): {exc}")

    def get_forwarded_payloads(self) -> List[Dict[str, Any]]:
        """Return all payloads that were forwarded (for testing/inspection)."""
        return list(self._forwarded_payloads)

    def clear_forwarded_payloads(self) -> None:
        """Clear the forwarded payloads buffer."""
        self._forwarded_payloads.clear()


_datadog_forwarder: Optional[DatadogForwarder] = None


def get_datadog_forwarder() -> DatadogForwarder:
    """Get or create the global DatadogForwarder singleton."""
    global _datadog_forwarder
    if _datadog_forwarder is None:
        _datadog_forwarder = DatadogForwarder()
    return _datadog_forwarder


def reset_datadog_forwarder() -> None:
    """Reset the global DatadogForwarder (for testing)."""
    global _datadog_forwarder
    _datadog_forwarder = None
