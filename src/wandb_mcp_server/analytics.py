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

import hashlib
import importlib.metadata
import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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

# ---------------------------------------------------------------------------
# Privacy levels
#
# MCP_LOG_PRIVACY_LEVEL controls how aggressively we redact customer-supplied
# content before it lands in any log sink (Cloud Logging, Segment, Datadog
# forwarder). Consumed here by _sanitise_params and the identifier-hashing
# helpers; also consumed by tools_utils/weave_api to demote verbose log sites
# at standard+ levels.
#
#   off       -- today's behavior. Redact sensitive-looking keys, truncate long
#                strings. Free-text values pass through. Cloud Run uses this
#                to preserve the BigQuery analytics contract.
#   standard  -- additionally redact free-text value keys (query, prompt,
#                description, etc). Customer K8s chart default.
#   strict    -- additionally hash entity/project/user identifiers. For
#                regulated / privacy-sensitive deployments.
#
# Default is "off" so the image behaves identically when run outside the chart.
# Customer K8s installs flip to "standard" via the helm chart's env injection.
# ---------------------------------------------------------------------------

_PRIVACY_LEVEL_OFF = "off"
_PRIVACY_LEVEL_STANDARD = "standard"
_PRIVACY_LEVEL_STRICT = "strict"
_VALID_PRIVACY_LEVELS = frozenset({_PRIVACY_LEVEL_OFF, _PRIVACY_LEVEL_STANDARD, _PRIVACY_LEVEL_STRICT})

# Keys whose values are free-form customer text. At standard+ levels the
# value is replaced with "<redacted: text len=N>" so we can still analyse
# call-volume / length distributions without logging the content itself.
_FREE_TEXT_KEYS: frozenset = frozenset(
    {
        "query",
        "question",
        "prompt",
        "description",
        "title",
        "text",
        "content",
        "body",
        "eval_name",
        "analysis_name",
        "message",
    }
)

# Keys whose values identify a specific customer entity/project/artifact.
# At strict level we hash these to a 12-char sha256 prefix so cardinality
# for cohort analytics is preserved while plaintext is not retained.
_IDENTIFIER_KEYS_FOR_HASHING: frozenset = frozenset(
    {
        "entity_name",
        "project_name",
        "entity",
        "project",
        "organization",
        "registry_name",
        "collection_name",
        "artifact_name",
        "artifact_name_a",
        "artifact_name_b",
        "run_id",
        "run_id_a",
        "run_id_b",
    }
)

# Once-per-process latch so an invalid MCP_LOG_PRIVACY_LEVEL doesn't spam logs
# on every analytics emit. Operators see one WARNING in their first scrape.
_warned_invalid_privacy_level = False


def _resolve_privacy_level() -> str:
    """Return the active privacy level, defaulting to ``off``.

    Read lazily (not cached) so tests and runtime env-var toggles work
    without reloading the module.

    Invalid values fall back to ``off`` (most permissive -- preserves
    availability) but emit a single WARNING so a typo like ``stict`` is
    visible to operators rather than silently downgrading their privacy
    posture.
    """
    raw = os.environ.get("MCP_LOG_PRIVACY_LEVEL", _PRIVACY_LEVEL_OFF).strip().lower()
    if raw and raw not in _VALID_PRIVACY_LEVELS:
        global _warned_invalid_privacy_level
        if not _warned_invalid_privacy_level:
            _warned_invalid_privacy_level = True
            logger.warning(
                "MCP_LOG_PRIVACY_LEVEL=%r is not one of %s; falling back to 'off'. "
                "Set a valid level to silence this warning.",
                raw,
                sorted(_VALID_PRIVACY_LEVELS),
            )
        return _PRIVACY_LEVEL_OFF
    return raw or _PRIVACY_LEVEL_OFF


def _hash_identifier(value: Any) -> str:
    """Hash an identifier to a short sha256 prefix for strict-mode analytics.

    Non-string inputs are coerced via ``str()``. Empty/None values return
    ``"<empty>"`` so downstream consumers can distinguish "no value" from
    "hashed value".
    """
    if value is None or value == "":
        return "<empty>"
    digest = hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
    return f"<h:{digest[:12]}>"


def is_verbose_log_site_gated() -> bool:
    """True when standard+ privacy levels demote verbose log sites to DEBUG.

    Exported so external emit sites (tools_utils, weave_api/client) can gate
    their INFO-level logs without importing the private level constants.
    """
    return _resolve_privacy_level() in (_PRIVACY_LEVEL_STANDARD, _PRIVACY_LEVEL_STRICT)


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

_ANALYTICS_STREAM_STDOUT = "stdout"
_ANALYTICS_STREAM_STDERR = "stderr"
_VALID_ANALYTICS_STREAMS = frozenset({_ANALYTICS_STREAM_STDOUT, _ANALYTICS_STREAM_STDERR})


def _resolve_analytics_stream_name(stream: Optional[str] = None) -> str:
    """Resolve the structured analytics log stream name."""
    raw = stream or os.environ.get("MCP_ANALYTICS_LOG_STREAM", _ANALYTICS_STREAM_STDOUT)
    stream_name = raw.strip().lower()
    if stream_name in _VALID_ANALYTICS_STREAMS:
        return stream_name

    logger.warning(
        "MCP_ANALYTICS_LOG_STREAM=%r is not one of %s; falling back to stdout.",
        raw,
        sorted(_VALID_ANALYTICS_STREAMS),
    )
    return _ANALYTICS_STREAM_STDOUT


def configure_analytics_logging(stream: Optional[str] = None) -> str:
    """Configure the structured analytics logger stream.

    HTTP/container deployments keep stdout so Cloud Logging can parse analytics
    JSON. Stdio transport must use stderr because stdout is the MCP JSON-RPC
    wire and any non-protocol line corrupts clients like Claude Desktop.

    Args:
        stream: Optional explicit stream name, "stdout" or "stderr". If omitted,
            MCP_ANALYTICS_LOG_STREAM is honored, then stdout is used.

    Returns:
        The resolved stream name.
    """
    stream_name = _resolve_analytics_stream_name(stream)
    target_stream = sys.stderr if stream_name == _ANALYTICS_STREAM_STDERR else sys.stdout

    analytics_logger.handlers.clear()
    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(_StructuredJsonFormatter())
    analytics_logger.addHandler(handler)
    analytics_logger.setLevel(logging.INFO)
    analytics_logger.propagate = False
    return stream_name


def configure_analytics_logging_for_transport(transport: str) -> str:
    """Configure analytics output for an MCP transport."""
    if os.environ.get("MCP_ANALYTICS_LOG_STREAM"):
        return configure_analytics_logging()
    if transport == "stdio":
        return configure_analytics_logging(_ANALYTICS_STREAM_STDERR)
    return configure_analytics_logging(_ANALYTICS_STREAM_STDOUT)


configure_analytics_logging()

_REQUIRED_BASE_FIELDS = frozenset({"schema_version", "event_type", "timestamp"})

# Surface the active privacy level once at module import so operators can
# verify their config by grepping pod logs (instead of needing kubectl describe).
# Triggers _resolve_privacy_level()'s WARNING for invalid values, so an env-var
# typo also lights up here at startup.
logger.info("Analytics ready: MCP_LOG_PRIVACY_LEVEL=%s", _resolve_privacy_level())


def _resolve_release_version() -> str:
    """Return release version for Datadog, Segment, BigQuery, and Hex joins."""
    explicit = os.environ.get("MCP_RELEASE_VERSION") or os.environ.get("DD_VERSION")
    if explicit:
        return explicit.strip()
    for package_name in ("wandb_mcp_server", "wandb-mcp-server"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.0.0"


def _utcnow_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_stripped(name: str) -> Optional[str]:
    """Return a stripped environment value or None when unset/empty."""
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _safe_wandb_base_host() -> Optional[str]:
    """Return a host-only W&B base URL dimension."""
    raw = _env_stripped("WANDB_BASE_URL")
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.netloc or parsed.path or None


def _resolve_transport() -> str:
    """Resolve the MCP transport dimension."""
    transport = (_env_stripped("MCP_TRANSPORT") or "unknown").lower()
    if transport in {"stdio", "http", "streamable-http", "sse"}:
        return "http" if transport == "streamable-http" else transport
    return transport


def _resolve_runtime_surface(transport: str) -> str:
    """Resolve where the MCP server is running."""
    explicit = _env_stripped("MCP_RUNTIME_SURFACE")
    if explicit:
        return explicit
    if os.environ.get("K_SERVICE") or os.environ.get("K_REVISION"):
        return "cloud_run"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "helm_k8s"
    if transport == "stdio":
        return "local_stdio"
    if transport == "http":
        return "local_http"
    return "unknown"


def _resolve_deployment_type(runtime_surface: str, transport: str) -> str:
    """Resolve the deployment type dimension."""
    explicit = _env_stripped("MCP_DEPLOYMENT_TYPE")
    if explicit:
        return explicit
    if _env_bool("MCP_HOSTED_MODE"):
        return "hosted"
    if runtime_surface.startswith("local") or transport == "stdio":
        return "local"
    return "unknown"


def _deployment_context() -> Dict[str, Any]:
    """Build low-cardinality deployment dimensions shared by all events."""
    transport = _resolve_transport()
    runtime_surface = _resolve_runtime_surface(transport)
    context: Dict[str, Any] = {
        "runtime_surface": runtime_surface,
        "transport": transport,
        "deployment_type": _resolve_deployment_type(runtime_surface, transport),
        "environment": _env_stripped("ENVIRONMENT") or _env_stripped("DD_ENV") or "unknown",
        "hosted_mode": _env_bool("MCP_HOSTED_MODE"),
    }
    wandb_base_host = _safe_wandb_base_host()
    if wandb_base_host:
        context["wandb_base_host"] = wandb_base_host
    return context


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
    def _sanitise_params(
        cls,
        params: Optional[Dict[str, Any]],
        *,
        _depth: int = 0,
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Strip sensitive keys and truncate large values.

        Recursively sanitises nested dicts and lists up to 3 levels deep.
        When ``level`` is ``standard`` or ``strict``, also redacts free-text
        value keys (query, prompt, description, etc). When ``level`` is
        ``strict``, additionally hashes identifier keys (entity_name,
        project_name, run_id, etc).

        ``level`` defaults to the value of ``MCP_LOG_PRIVACY_LEVEL`` at call
        time. Explicit ``level`` arg is used by tests and by the emit sites
        that need deterministic behavior across a single call tree.
        """
        if not params:
            return {}
        if level is None:
            level = _resolve_privacy_level()
        redact_free_text = level in (_PRIVACY_LEVEL_STANDARD, _PRIVACY_LEVEL_STRICT)
        hash_identifiers = level == _PRIVACY_LEVEL_STRICT
        safe: Dict[str, Any] = {}
        for key, value in params.items():
            key_lower = key.lower()
            if any(p in key_lower for p in _SENSITIVE_PARAM_PATTERNS):
                safe[key] = "<redacted>"
            elif redact_free_text and key_lower in _FREE_TEXT_KEYS and isinstance(value, str):
                safe[key] = f"<redacted: text len={len(value)}>"
            elif hash_identifiers and key_lower in _IDENTIFIER_KEYS_FOR_HASHING:
                safe[key] = _hash_identifier(value)
            elif isinstance(value, dict) and _depth < 3:
                safe[key] = cls._sanitise_params(value, _depth=_depth + 1, level=level)
            elif isinstance(value, list) and _depth < 3:
                safe[key] = cls._sanitise_list(value, _depth=_depth + 1, level=level)
            elif isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LENGTH:
                safe[key] = f"<truncated:{len(value)} chars>"
            else:
                safe[key] = value
        return safe

    @classmethod
    def _sanitise_list(
        cls,
        items: list,
        *,
        _depth: int = 0,
        level: Optional[str] = None,
    ) -> list:
        """Sanitise each element in a list, recursing into dicts and nested lists."""
        if level is None:
            level = _resolve_privacy_level()
        result = []
        for item in items:
            if isinstance(item, dict) and _depth < 3:
                result.append(cls._sanitise_params(item, _depth=_depth, level=level))
            elif isinstance(item, list) and _depth < 3:
                result.append(cls._sanitise_list(item, _depth=_depth + 1, level=level))
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
            "release_version": _resolve_release_version(),
            **_deployment_context(),
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

    @staticmethod
    def _apply_identity_privacy(
        user_id: Optional[str],
        email_domain: Optional[str],
        *,
        level: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Transform identity fields according to the active privacy level.

        - ``off`` / ``standard``: return values unchanged.
        - ``strict``: hash both to opaque sha256 prefixes so cardinality
          is preserved for cohort analytics without retaining plaintext.

        Returns ``(user_id, email_domain)`` tuple.
        """
        if level is None:
            level = _resolve_privacy_level()
        if level != _PRIVACY_LEVEL_STRICT:
            return user_id, email_domain
        hashed_user = _hash_identifier(user_id) if user_id else user_id
        hashed_domain = _hash_identifier(email_domain) if email_domain else email_domain
        return hashed_user, hashed_domain

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
            level = _resolve_privacy_level()
            user_id = self._extract_user_id(viewer_info)
            email_domain = self._extract_email_domain(viewer_info)
            user_id, email_domain = self._apply_identity_privacy(user_id, email_domain, level=level)
            event = {
                **self._base_event("user_session"),
                "session_id": session_id,
                "user_id": user_id,
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
        mcp_tool_name: Optional[str] = None,
    ) -> None:
        """Record an MCP tool invocation."""
        if not self.enabled:
            return
        try:
            level = _resolve_privacy_level()
            user_id = self._extract_user_id(viewer_info)
            email_domain = self._extract_email_domain(viewer_info)
            user_id, email_domain = self._apply_identity_privacy(user_id, email_domain, level=level)
            event = {
                **self._base_event("tool_call"),
                "session_id": session_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "tool_name": tool_name,
                "params": self._sanitise_params(params, level=level),
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
            }
            if mcp_tool_name:
                event["mcp_tool_name"] = mcp_tool_name
            labels = {
                "event_type": "tool_call",
                "tool_name": tool_name,
                "email_domain": email_domain or "unknown",
                "success": str(success),
            }
            if mcp_tool_name:
                labels["mcp_tool_name"] = mcp_tool_name
            self._emit(
                event,
                labels,
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
            user_id, email_domain = self._apply_identity_privacy(user_id, email_domain)
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
