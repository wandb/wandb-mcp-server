"""Analytics tracking for W&B MCP Server.

Structured logging pipeline: Cloud Run -> Cloud Logging -> BigQuery -> Hex.

Event types:
  user_session  -- user login / session start
  tool_call     -- one public MCP invocation (compact usage dimensions only)
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

from wandb_mcp_server.config import MCP_REQUEST_SUCCESS_SAMPLE_RATE
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

SCHEMA_VERSION = "1.1"

_SENSITIVE_PARAM_PATTERNS: List[str] = [
    "api_key",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
]

_MAX_PARAM_VALUE_LENGTH = 200
_MAX_PARAM_DEPTH = 4
_MAX_PARAM_KEYS = 20
_MAX_PARAM_LIST_ITEMS = 20
_MAX_USAGE_DIMENSIONS = 12
_MAX_EVENT_BYTES = 4096
_SLOW_REQUEST_MS = 2_000.0

_configured_transport: Optional[str] = None
_analytics_startup_logged = False

_USAGE_ENUM_VALUES: Dict[str, frozenset[str]] = {
    "resource": frozenset({"project", "run", "runs", "sweep", "sweeps", "reports"}),
    "source": frozenset({"project", "registry"}),
    "kind": frozenset({"slack", "webhook"}),
    "mode": frozenset({"sampled", "scan", "full"}),
    "cost_class": frozenset({"light", "expensive", "heavy"}),
    "admission_outcome": frozenset({"disabled", "admitted", "rejected", "cancelled"}),
}

_USAGE_COUNTABLE_NUMBER_KEYS = frozenset(
    {
        "limit",
        "max_items",
        "items_per_page",
        "samples",
        "sample_size",
        "sample_runs",
        "max_projects",
        "max_files",
        "max_evals",
        "history_samples",
        "top_n_values",
        "max_file_diff_entries",
        "request_timeout",
        "queue_ms",
    }
)

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

_MISSING_IDENTITY_VALUES: frozenset = frozenset(
    {
        "",
        "anonymous",
        "none",
        "null",
        "unknown",
    }
)


def _resolve_privacy_level() -> str:
    """Use the same fail-closed parser as startup, including after an env change."""
    from wandb_mcp_server.privacy import resolve_privacy_level

    return resolve_privacy_level()


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
    normalized = transport.strip().lower()
    if normalized == "stdio":
        # STDIO stdout is the JSON-RPC wire. An environment override must never
        # be allowed to inject analytics records into the protocol stream.
        stream_name = configure_analytics_logging(_ANALYTICS_STREAM_STDERR)
    elif os.environ.get("MCP_ANALYTICS_LOG_STREAM"):
        stream_name = configure_analytics_logging()
    else:
        stream_name = configure_analytics_logging(_ANALYTICS_STREAM_STDOUT)
    _log_analytics_startup_once()
    return stream_name


def _log_analytics_startup_once() -> None:
    """Log analytics/privacy configuration after the transport is known."""
    global _analytics_startup_logged
    if _analytics_startup_logged:
        return
    _analytics_startup_logged = True
    logger.info("Analytics ready: MCP_LOG_PRIVACY_LEVEL=%s", _resolve_privacy_level())


# Before construction selects a transport, stderr is the only protocol-safe
# destination. HTTP construction switches analytics back to stdout below.
configure_analytics_logging(_ANALYTICS_STREAM_STDERR)

_REQUIRED_BASE_FIELDS = frozenset({"schema_version", "event_type", "timestamp"})


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


def configure_analytics_runtime(transport: str) -> None:
    """Record the transport selected by the constructed MCP server.

    Deployment environment variables still describe externally hosted surfaces,
    while local servers no longer report an unknown transport merely because an
    operator did not duplicate a CLI argument in the environment.
    """
    global _configured_transport
    normalized = transport.strip().lower()
    _configured_transport = "http" if normalized == "streamable-http" else normalized
    configure_analytics_logging_for_transport(_configured_transport)


def _safe_wandb_base_host() -> Optional[str]:
    """Return a host-only W&B base URL dimension."""
    raw = _env_stripped("WANDB_BASE_URL")
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.netloc or parsed.path or None


def _resolve_transport() -> str:
    """Resolve the MCP transport dimension."""
    transport = (_env_stripped("MCP_TRANSPORT") or _configured_transport or "unknown").lower()
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
    workload_profile = (_env_stripped("MCP_WORKLOAD_PROFILE") or "local").lower()
    if workload_profile == "shared":
        return "hosted"
    if workload_profile == "dedicated":
        return "dedicated"
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
        "hosted_mode": (_env_stripped("MCP_WORKLOAD_PROFILE") or "local").lower() == "shared",
    }
    wandb_base_host = _safe_wandb_base_host()
    if wandb_base_host:
        context["wandb_base_host"] = wandb_base_host
    return context


def _harness_context() -> Dict[str, Any]:
    """Return the current request's low-cardinality MCP harness dimensions."""
    try:
        from wandb_mcp_server.harness import HarnessContext, current_harness_context

        context = current_harness_context.get()
        if context is None:
            context = HarnessContext()
        return context.analytics_fields()
    except Exception:
        return {
            "agent_harness": "unknown",
            "client_vendor": "unknown",
            "call_type": "unknown",
        }


def _compact_value(value: Any) -> Any:
    """Recursively remove optional empty values while preserving False and zero."""
    if isinstance(value, dict):
        compacted = {
            str(key): compacted_value
            for key, child in value.items()
            if (compacted_value := _compact_value(child)) not in (None, "", {}, [])
        }
        return compacted
    if isinstance(value, list):
        return [compacted for child in value if (compacted := _compact_value(child)) not in (None, "", {}, [])]
    return value


def _prepare_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Compact and hard-bound an analytics event to the 4 KiB contract."""
    from wandb_mcp_server.error_sanitizer import sanitize_sensitive_value

    event = sanitize_sensitive_value(event)
    compacted = _compact_value(event)
    if not isinstance(compacted, dict):
        return {}
    if len(json.dumps(compacted, default=str, separators=(",", ":")).encode()) <= _MAX_EVENT_BYTES:
        return compacted

    compacted["event_truncated"] = True
    for key in (
        "metadata",
        "usage_dimensions",
        "mcp_client_name",
        "mcp_client_version",
        "mcp_user_agent_product",
        "mcp_client_mismatch",
    ):
        compacted.pop(key, None)
    if "error" in compacted:
        compacted["error"] = str(compacted["error"])[:200]
    if len(json.dumps(compacted, default=str, separators=(",", ":")).encode()) <= _MAX_EVENT_BYTES:
        return compacted

    essential_keys = {
        "schema_version",
        "event_type",
        "timestamp",
        "release_version",
        "runtime_surface",
        "transport",
        "deployment_type",
        "environment",
        "hosted_mode",
        "agent_harness",
        "client_vendor",
        "call_type",
        "mcp_client_family",
        "mcp_client_app",
        "mcp_jsonrpc_method",
        "session_id",
        "actor_id",
        "tool_name",
        "mcp_tool_name",
        "success",
        "error",
        "duration_ms",
        "request_id",
        "method",
        "path",
        "status_code",
        "event_truncated",
    }
    essential = {key: value for key, value in compacted.items() if key in essential_keys}
    for key, value in list(essential.items()):
        if isinstance(value, str):
            limit = 256 if key in {"error", "path", "session_id"} else 128
            essential[key] = value[:limit]
    return essential


def _number_bucket(value: int | float) -> str:
    numeric = float(value)
    if numeric <= 0:
        return "0"
    if numeric == 1:
        return "1"
    for upper, label in (
        (5, "2-5"),
        (10, "6-10"),
        (25, "11-25"),
        (50, "26-50"),
        (100, "51-100"),
        (500, "101-500"),
    ):
        if numeric <= upper:
            return label
    return "501+"


def _usage_dimensions(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a small allowlisted product-analytics summary of tool arguments."""
    if not params:
        return {}
    dimensions: Dict[str, Any] = {}
    excluded = _FREE_TEXT_KEYS | _IDENTIFIER_KEYS_FOR_HASHING
    for raw_key in sorted(params):
        if len(dimensions) >= _MAX_USAGE_DIMENSIONS:
            break
        key = str(raw_key).lower()
        value = params[raw_key]
        if any(pattern in key for pattern in _SENSITIVE_PARAM_PATTERNS) or key in excluded:
            continue
        if key in {"filter", "filters"}:
            dimensions["has_filters"] = bool(value)
            if isinstance(value, dict) and len(dimensions) < _MAX_USAGE_DIMENSIONS:
                dimensions["filter_key_count"] = min(len(value), _MAX_PARAM_KEYS)
            continue
        if isinstance(value, bool):
            dimensions[key] = value
        elif isinstance(value, (int, float)) and key in _USAGE_COUNTABLE_NUMBER_KEYS:
            dimensions[f"{key}_bucket"] = _number_bucket(value)
        elif isinstance(value, (list, tuple, set)):
            dimensions[f"{key}_count"] = min(len(value), _MAX_PARAM_LIST_ITEMS)
        elif isinstance(value, dict):
            dimensions[f"{key}_key_count"] = min(len(value), _MAX_PARAM_KEYS)
        elif isinstance(value, str) and key in _USAGE_ENUM_VALUES:
            normalized = value.strip().lower()
            if normalized in _USAGE_ENUM_VALUES[key]:
                dimensions[key] = normalized
    return dict(list(dimensions.items())[:_MAX_USAGE_DIMENSIONS])


def actor_id_from_api_key_hash(api_key_hash: Optional[str]) -> Optional[str]:
    """Return a stable, non-secret analytics actor ID from an API-key digest."""
    if not api_key_hash:
        return None
    normalized = str(api_key_hash).strip().lower()
    if not normalized:
        return None
    return f"wandb_key:{normalized[:24]}"


def current_actor_id() -> Optional[str]:
    """Resolve the current request's actor without making a W&B API call."""
    try:
        from wandb_mcp_server.session_manager import current_api_key_hash

        digest = current_api_key_hash.get()
        if digest:
            return actor_id_from_api_key_hash(digest)
    except Exception:
        pass
    try:
        from wandb_mcp_server.api_client import WandBApiManager

        api_key = WandBApiManager.get_api_key()
        if api_key:
            return actor_id_from_api_key_hash(hashlib.sha256(api_key.encode()).hexdigest())
    except Exception:
        pass
    return None


def _request_should_be_emitted(
    *,
    request_id: str,
    path: str,
    status_code: int,
    duration_ms: Optional[float],
) -> bool:
    """Apply deterministic sampling to successful operational request events."""
    if path in {"/health", "/mcp/health", "/favicon.ico", "/favicon.png"}:
        return False
    if status_code >= 400 or (duration_ms is not None and duration_ms >= _SLOW_REQUEST_MS):
        return True
    rate = MCP_REQUEST_SUCCESS_SAMPLE_RATE
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(request_id.encode("utf-8", errors="replace")).digest()
    sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return sample < rate


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
                        text = str(val).strip()
                        if text.lower() not in _MISSING_IDENTITY_VALUES:
                            return text
            if isinstance(viewer_info, dict):
                for key in ("username", "entity"):
                    val = viewer_info.get(key)
                    if val:
                        text = str(val).strip()
                        if text.lower() not in _MISSING_IDENTITY_VALUES:
                            return text
            if hasattr(viewer_info, "email"):
                email = getattr(viewer_info, "email")
                if email and "@" in str(email):
                    domain = str(email).split("@", 1)[1].strip().lower()
                    return domain or None
            if isinstance(viewer_info, str):
                text = viewer_info.strip()
                if text.lower() in _MISSING_IDENTITY_VALUES:
                    return None
                if "@" in text:
                    domain = text.split("@", 1)[1].strip().lower()
                    return domain or None
                return text
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

        Recursively sanitises nested dicts and lists to a hard depth/size cap.
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
        if _depth >= _MAX_PARAM_DEPTH:
            return {
                "_truncated": "max_depth",
                "_item_count": min(len(params), _MAX_PARAM_KEYS),
            }
        redact_free_text = level in (_PRIVACY_LEVEL_STANDARD, _PRIVACY_LEVEL_STRICT)
        hash_identifiers = level == _PRIVACY_LEVEL_STRICT
        safe: Dict[str, Any] = {}
        entries = list(params.items())
        for raw_key, value in entries[:_MAX_PARAM_KEYS]:
            key = str(raw_key)[:_MAX_PARAM_VALUE_LENGTH]
            key_lower = key.lower()
            if any(p in key_lower for p in _SENSITIVE_PARAM_PATTERNS):
                safe[key] = "<redacted>"
            elif redact_free_text and key_lower in _FREE_TEXT_KEYS and isinstance(value, str):
                safe[key] = f"<redacted: text len={len(value)}>"
            elif hash_identifiers and key_lower in _IDENTIFIER_KEYS_FOR_HASHING:
                safe[key] = _hash_identifier(value)
            elif isinstance(value, dict):
                safe[key] = cls._sanitise_params(value, _depth=_depth + 1, level=level)
            elif isinstance(value, list):
                safe[key] = cls._sanitise_list(value, _depth=_depth + 1, level=level)
            elif isinstance(value, str) and len(value) > _MAX_PARAM_VALUE_LENGTH:
                safe[key] = f"<truncated:{len(value)} chars>"
            elif value is None or isinstance(value, (str, bool, int, float)):
                safe[key] = value
            else:
                safe[key] = f"<{type(value).__name__}>"
        if len(entries) > _MAX_PARAM_KEYS:
            safe["_truncated_keys"] = len(entries) - _MAX_PARAM_KEYS
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
        if _depth >= _MAX_PARAM_DEPTH:
            return [f"<truncated:max_depth items={min(len(items), _MAX_PARAM_LIST_ITEMS)}>"]
        result = []
        for item in items[:_MAX_PARAM_LIST_ITEMS]:
            if isinstance(item, dict):
                result.append(cls._sanitise_params(item, _depth=_depth + 1, level=level))
            elif isinstance(item, list):
                result.append(cls._sanitise_list(item, _depth=_depth + 1, level=level))
            elif isinstance(item, str) and len(item) > _MAX_PARAM_VALUE_LENGTH:
                result.append(f"<truncated:{len(item)} chars>")
            elif item is None or isinstance(item, (str, bool, int, float)):
                result.append(item)
            else:
                result.append(f"<{type(item).__name__}>")
        if len(items) > _MAX_PARAM_LIST_ITEMS:
            result.append(f"<truncated:{len(items) - _MAX_PARAM_LIST_ITEMS} items>")
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
            **_harness_context(),
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
        from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text

        event = _prepare_event(event)
        labels = {
            sanitize_sensitive_text(key): sanitize_sensitive_text(value)
            for key, value in labels.items()
            if value not in (None, "", {}, [])
        }
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
            logger.debug("Analytics emit failed (non-fatal; %s)", type(exc).__name__)

        try:
            from wandb_mcp_server.analytics_segment import get_segment_forwarder

            forwarder = get_segment_forwarder()
            if forwarder.enabled:
                forwarder.forward(event)
        except Exception as exc:
            logger.debug("Segment forwarding failed (non-fatal; %s)", type(exc).__name__)

        try:
            from wandb_mcp_server.analytics_datadog import get_datadog_forwarder

            dd_forwarder = get_datadog_forwarder()
            if dd_forwarder.enabled:
                dd_forwarder.forward(event)
        except Exception as exc:
            logger.debug("Datadog forwarding failed (non-fatal; %s)", type(exc).__name__)

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
            actor_id = actor_id_from_api_key_hash(api_key_hash) or current_actor_id()
            event = {
                **self._base_event("user_session"),
                "session_id": session_id,
                "actor_id": actor_id,
                "user_id": user_id,
                "email_domain": email_domain,
                "api_key_hash": api_key_hash[:16] if api_key_hash else None,
                "metadata": self._sanitise_params(metadata, level=level) if metadata else None,
            }
            try:
                from wandb_mcp_server.harness import current_harness_context

                harness = current_harness_context.get()
                if harness is not None:
                    event.update(harness.debug_fields())
            except Exception:
                pass
            self._emit(
                _prepare_event(event),
                {
                    "event_type": "user_session",
                    "agent_harness": event.get("agent_harness", "unknown"),
                    "call_type": event.get("call_type", "unknown"),
                },
            )
        except Exception as exc:
            logger.warning("Failed to track user session (%s)", type(exc).__name__)

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
            public_tool_name = mcp_tool_name or tool_name
            event = {
                **self._base_event("tool_call"),
                "session_id": session_id,
                "actor_id": current_actor_id(),
                "user_id": user_id,
                "email_domain": email_domain,
                "tool_name": public_tool_name,
                "mcp_tool_name": public_tool_name,
                "usage_dimensions": _usage_dimensions(params),
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
            }
            labels = {
                "event_type": "tool_call",
                "tool_name": public_tool_name,
                "success": str(success),
                "agent_harness": event.get("agent_harness", "unknown"),
                "call_type": event.get("call_type", "unknown"),
            }
            self._emit(
                _prepare_event(event),
                labels,
            )
        except Exception as exc:
            logger.warning("Failed to track tool call (%s)", type(exc).__name__)

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
            if not _request_should_be_emitted(
                request_id=request_id,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            ):
                return
            user_id, email_domain = self._apply_identity_privacy(user_id, email_domain)
            event = {
                **self._base_event("request"),
                "request_id": request_id,
                "session_id": session_id,
                "actor_id": current_actor_id(),
                "user_id": user_id,
                "email_domain": email_domain,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
            self._emit(
                _prepare_event(event),
                {
                    "event_type": "request",
                    "status_code": str(status_code),
                    "agent_harness": event.get("agent_harness", "unknown"),
                    "call_type": event.get("call_type", "unknown"),
                },
            )
        except Exception as exc:
            logger.warning("Failed to track request (%s)", type(exc).__name__)


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
