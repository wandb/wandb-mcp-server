"""Datadog HTTP Logs Intake forwarder.

Sends MCP analytics events to the Datadog Logs API via HTTP POST.
No Datadog Agent or sidecar required -- works on serverless (Cloud Run, Lambda).

Enable with ``MCP_DATADOG_FORWARD=true`` + ``DD_API_KEY`` (from Secret Manager
on Cloud Run, or env var locally).

The forwarder POSTs to ``https://http-intake.logs.{DD_SITE}/api/v2/logs``
using the ``DD-API-KEY`` header.  Each event is mapped to a Datadog log entry
with structured attributes (``@duration``, ``@http.status_code``,
``@error.kind``, ``@usr.id``) for automatic faceting, dashboards, and SLOs.

Segment receives *product analytics* (adoption, cohorts).
Datadog receives *operational observability* (errors, latency, alerting).
The mapper intentionally excludes ``params`` and ``api_key_hash`` to avoid
PII leakage into ops logs.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

_DATADOG_EVENT_PREFIX = "mcp"


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


def map_to_datadog_log(
    event: Dict[str, Any],
    *,
    dd_env: str,
    dd_version: str,
    dd_service: str,
) -> Dict[str, Any]:
    """Map an internal analytics event to a Datadog log entry with reserved attributes.

    Produces structured top-level attributes that Datadog auto-extracts for
    dashboards, monitors, and SLO definitions without custom Log Pipelines.

    PII policy: ``params``, ``api_key_hash``, ``metadata``, and ``email_domain``
    are intentionally excluded from the Datadog payload.  Only ``user_id``
    (which is already a non-PII identifier: username or domain) is forwarded
    as ``@usr.id``.

    Args:
        event: Internal analytics event dict (as emitted by AnalyticsTracker).
        dd_env: Datadog environment tag (e.g. "staging", "production").
        dd_version: Service version tag (e.g. "0.3.0").
        dd_service: Service name tag.

    Returns:
        Dict suitable for the Datadog HTTP Logs Intake API.
    """
    event_type = event.get("event_type", "unknown")

    status = _resolve_severity(event)

    tags = [
        f"env:{dd_env}",
        f"service:{dd_service}",
        f"version:{dd_version}",
        f"event_type:{event_type}",
    ]
    tool_name = event.get("tool_name")
    if tool_name:
        tags.append(f"tool_name:{tool_name}")
    success = event.get("success")
    if success is not None:
        tags.append(f"success:{str(success).lower()}")

    attributes: Dict[str, Any] = {"event_type": event_type}

    duration_ms = event.get("duration_ms")
    if duration_ms is not None:
        attributes["duration"] = int(duration_ms * 1_000_000)

    if event_type == "request":
        http_attrs: Dict[str, Any] = {}
        if event.get("status_code") is not None:
            http_attrs["status_code"] = event["status_code"]
        if event.get("method"):
            http_attrs["method"] = event["method"]
        if event.get("path"):
            http_attrs["url_details"] = {"path": event["path"]}
        if http_attrs:
            attributes["http"] = http_attrs

    error_str = event.get("error")
    if error_str:
        parts = str(error_str).split(": ", 1)
        attributes["error"] = {
            "kind": parts[0] if len(parts) > 1 else "Error",
            "message": parts[-1][:1000],
        }

    user_id = event.get("user_id")
    if user_id:
        attributes["usr"] = {"id": user_id}

    if event_type == "tool_call":
        tool_attrs: Dict[str, Any] = {}
        if tool_name:
            tool_attrs["name"] = tool_name
        if success is not None:
            tool_attrs["success"] = success
        if tool_attrs:
            attributes["tool"] = tool_attrs

    if event.get("session_id"):
        attributes["session_id"] = event["session_id"]

    message = _build_message(event)

    return {
        "ddsource": "wandb-mcp-server",
        "ddtags": ",".join(tags),
        "hostname": os.environ.get("K_REVISION", os.environ.get("HOSTNAME", "unknown")),
        "service": dd_service,
        "status": status,
        "message": message,
        "attributes": attributes,
    }


def _resolve_severity(event: Dict[str, Any]) -> str:
    """Map event content to a Datadog log severity level."""
    if event.get("error"):
        return "error"
    event_type = event.get("event_type")
    if event_type == "tool_call" and event.get("success") is False:
        return "error"
    if event_type == "request":
        sc = event.get("status_code", 200)
        if isinstance(sc, int):
            if sc >= 500:
                return "error"
            if sc >= 400:
                return "warn"
    return "info"


def _build_message(event: Dict[str, Any]) -> str:
    """Build a human-readable summary line for the Datadog log message."""
    event_type = event.get("event_type", "unknown")
    parts = [f"{_DATADOG_EVENT_PREFIX}.{event_type}"]

    if event_type == "tool_call":
        tool_name = event.get("tool_name")
        if tool_name:
            parts[0] += f".{tool_name}"
        if event.get("error"):
            parts.append(f"ERROR: {event['error'][:200]}")
        elif event.get("success") is False:
            parts.append("FAILED")
        duration = event.get("duration_ms")
        if duration is not None:
            parts.append(f"({duration:.0f}ms)")
    elif event_type == "request":
        method = event.get("method", "")
        path = event.get("path", "")
        sc = event.get("status_code", "?")
        parts.append(f"{method} {path} -> {sc}")
        duration = event.get("duration_ms")
        if duration is not None:
            parts.append(f"({duration:.0f}ms)")
    elif event_type == "user_session":
        user_id = event.get("user_id")
        if user_id and user_id != "anonymous":
            parts.append(f"user={user_id}")

    return " ".join(parts)


_DD_SECRET_NAME = "mcp-server-datadog-api-key"


def _resolve_dd_api_key() -> str:
    """Resolve DD_API_KEY: env var first, then SecretsResolver (GCP Secret Manager)."""
    from_env = os.environ.get("DD_API_KEY", "")
    if from_env:
        return from_env
    try:
        from wandb_mcp_server.secrets_resolver import get_secrets_resolver_from_env

        resolver = get_secrets_resolver_from_env()
        if resolver is not None:
            key_bytes = resolver.fetch_secret(_DD_SECRET_NAME)
            if key_bytes:
                return key_bytes.decode("utf-8").strip()
    except Exception as exc:
        logger.debug(f"SecretsResolver failed for {_DD_SECRET_NAME}: {exc}")
    return ""


class DatadogForwarder:
    """Gated forwarder that sends analytics events to the Datadog HTTP Logs API.

    Controlled by env vars:
    - ``MCP_DATADOG_FORWARD=true``: enable forwarding (off by default).
    - ``DD_API_KEY``: Datadog API key -- or fetched from GCP Secret Manager
      via ``SecretsResolver`` when ``MCP_SERVER_SECRETS_PROVIDER=gcp`` is set
      (same pattern as the HMAC session key).
    - ``DD_SITE``: Datadog site (default ``datadoghq.com``).
    - ``DD_ENV``: environment tag (default ``production``).
    - ``DD_VERSION``: version tag (default ``0.0.0``).
    - ``DD_SERVICE``: service name tag (default ``wandb-mcp-server``).

    Live POSTs run in a daemon thread so they never block the MCP request path.
    """

    def __init__(self) -> None:
        self.live = os.environ.get("MCP_DATADOG_FORWARD", "false").lower() == "true"
        self._api_key = _resolve_dd_api_key() if self.live else ""
        self._site = os.environ.get("DD_SITE", "datadoghq.com")
        self._env = os.environ.get("DD_ENV", "production")
        self._version = os.environ.get("DD_VERSION", "0.0.0")
        self._service = os.environ.get("DD_SERVICE", "wandb-mcp-server")
        self._intake_url = f"https://http-intake.logs.{self._site}/api/v2/logs"
        self._forwarded_payloads: List[Dict[str, Any]] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._thread_local = threading.local()

        if self.live and not self._api_key:
            # In managed K8s, the DD Agent DaemonSet (on the node at DD_AGENT_HOST:8126)
            # already collects stdout logs and APM traces using its own API key. A pod
            # with MCP_DATADOG_FORWARD=true and no DD_API_KEY would legitimately happen
            # if a chart pre-0.42.2 toggled the forwarder flag without seeding a secret.
            # That's a misconfiguration in serverless, but the correct state in K8s --
            # demote to debug there so we don't log spam in every production pod.
            if os.environ.get("DD_AGENT_HOST"):
                logger.debug(
                    "MCP_DATADOG_FORWARD=true but DD_API_KEY empty; DD_AGENT_HOST is set, "
                    "assuming a local Datadog Agent is collecting logs/APM. Forwarder disabled."
                )
            else:
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

        entry = map_to_datadog_log(
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
