import os
import ipaddress
import re
from urllib.parse import urlsplit

from wandb_mcp_server.runtime_contract import CAPACITY_BOUNDS, WORKLOAD_LIMIT_BOUNDS, load_runtime_contract


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a strict boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Read and validate an integer environment variable.

    Safety limits must fail closed. Silently replacing malformed production
    configuration with a default can remove the intended bound without an
    operator noticing.
    """
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be an integer") from None
    return _validate_int(name, value, minimum=minimum, maximum=maximum)


def _validate_int(name: str, value: int, *, minimum: int, maximum: int | None) -> int:
    """Validate an environment or packaged-policy integer against one bound."""
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite, bounded floating-point environment variable."""
    import math

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_aria_base_url(value: str) -> str:
    """Validate the operator-controlled ARIA endpoint before forwarding credentials."""
    normalized = value.strip().rstrip("/")
    if not normalized or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise ValueError("WB_AGENT_BASE_URL must be a valid absolute HTTPS URL")
    try:
        parsed = urlsplit(normalized)
        # Accessing port forces urllib to reject malformed port declarations.
        parsed.port
    except ValueError as exc:
        raise ValueError("WB_AGENT_BASE_URL must be a valid absolute HTTPS URL") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("WB_AGENT_BASE_URL must be an absolute HTTPS URL")
    hostname = parsed.hostname
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
        except UnicodeError as exc:
            raise ValueError("WB_AGENT_BASE_URL must contain a valid hostname") from exc
        labels = ascii_hostname.split(".")
        if (
            not ascii_hostname
            or len(ascii_hostname) > 253
            or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or re.fullmatch(r"[A-Za-z0-9-]+", label) is None
                for label in labels
            )
        ):
            raise ValueError("WB_AGENT_BASE_URL must contain a valid hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("WB_AGENT_BASE_URL must not include credentials")
    if parsed.query or parsed.fragment or "?" in normalized or "#" in normalized:
        raise ValueError("WB_AGENT_BASE_URL must not include a query string or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("WB_AGENT_BASE_URL must not include a path")
    return normalized


# Centralized configuration for base URLs used across the project.
# Values are read from environment variables with production defaults.

# Public W&B URL used for customer-visible links and local credential lookup.
WANDB_BASE_URL: str = (os.getenv("WANDB_BASE_URL") or "https://api.wandb.ai").rstrip("/")

# Optional in-cluster URL for server-to-server W&B API traffic. When configured,
# failures remain failures; callers must never retry through the public ingress.
_internal_base_url = (os.getenv("WANDB_INTERNAL_BASE_URL") or "").strip().rstrip("/")
WANDB_INTERNAL_BASE_URL: str | None = _internal_base_url or None
WANDB_API_BASE_URL: str = WANDB_INTERNAL_BASE_URL or WANDB_BASE_URL

# Weave Trace server URL used by the Weave API client and services
WF_TRACE_SERVER_URL: str = (
    os.getenv("WF_TRACE_SERVER_URL") or os.getenv("WEAVE_TRACE_SERVER_URL") or "https://trace.wandb.ai"
)

# Hosted W&B Agent (ARIA) service URL. This is intentionally separate from
# WANDB_BASE_URL: ARIA is a distinct asynchronous service, even when the MCP
# server itself is run locally or points its other tools at W&B Dedicated.
WB_AGENT_BASE_URL: str = (os.getenv("WB_AGENT_BASE_URL") or "https://wb-agent.wandb.ai").strip().rstrip("/")


def resolve_aria_base_url(fallback: str | None = None) -> str:
    """Read and validate the effective ARIA URL after CLI dotenv loading."""
    configured = os.getenv("WB_AGENT_BASE_URL")
    return validate_aria_base_url(configured or fallback or WB_AGENT_BASE_URL)


# Tool selection is resolved later, immediately before registration. Workload
# limits are process-wide and come from the same packaged contract. Managed
# profiles reject low-level overrides so the attested profile remains exact;
# local mode retains bounded advanced tuning for standalone operators.
_RUNTIME_CONTRACT = load_runtime_contract()
MCP_WORKLOAD_PROFILE: str = (os.getenv("MCP_WORKLOAD_PROFILE") or "local").strip()
if MCP_WORKLOAD_PROFILE not in _RUNTIME_CONTRACT["workload_profiles"]:
    raise ValueError("MCP_WORKLOAD_PROFILE must be one of: shared, dedicated, local")
MCP_CAPACITY_CLASS: str = (os.getenv("MCP_CAPACITY_CLASS") or "small").strip()
if MCP_CAPACITY_CLASS not in _RUNTIME_CONTRACT["capacity_classes"]:
    raise ValueError("MCP_CAPACITY_CLASS must be one of: small, medium, large")

_workload_policy = _RUNTIME_CONTRACT["workload_profiles"][MCP_WORKLOAD_PROFILE]
_workload_defaults = _workload_policy["limits"]
_capacity_defaults = _RUNTIME_CONTRACT["capacity_classes"][MCP_CAPACITY_CLASS]
_managed_workload = bool(_workload_policy["managed"])

if "MCP_HOSTED_MODE" in os.environ:
    raise ValueError("MCP_HOSTED_MODE is internal; configure MCP_WORKLOAD_PROFILE")
# Compatibility name for internal call sites. It is derived, never configured.
MCP_HOSTED_MODE: bool = MCP_WORKLOAD_PROFILE != "local"


def _profile_int(name: str) -> int:
    default = int(_workload_defaults[name])
    minimum, maximum = WORKLOAD_LIMIT_BOUNDS[name]
    default = _validate_int(name, default, minimum=minimum, maximum=maximum)
    if _managed_workload:
        if name in os.environ:
            raise ValueError(f"{name} cannot override managed MCP_WORKLOAD_PROFILE={MCP_WORKLOAD_PROFILE}")
        return default
    return _env_int(name, default, minimum=minimum, maximum=maximum)


def _profile_bool(name: str) -> bool:
    default = bool(_workload_defaults[name])
    if _managed_workload:
        if name in os.environ:
            raise ValueError(f"{name} cannot override managed MCP_WORKLOAD_PROFILE={MCP_WORKLOAD_PROFILE}")
        return default
    return _env_bool(name, default)


def _capacity_int(name: str, contract_key: str) -> int:
    default = int(_capacity_defaults[contract_key])
    minimum, maximum = CAPACITY_BOUNDS[contract_key]
    default = _validate_int(name, default, minimum=minimum, maximum=maximum)
    if _managed_workload:
        if name in os.environ:
            raise ValueError(f"{name} cannot override managed MCP_CAPACITY_CLASS={MCP_CAPACITY_CLASS}")
        return default
    return _env_int(name, default, minimum=minimum, maximum=maximum)


MAX_RESPONSE_TOKENS: int = _profile_int("MAX_RESPONSE_TOKENS")
MAX_ACCUMULATED_BYTES: int = _profile_int("MAX_ACCUMULATED_BYTES")
MCP_TOOL_TIMEOUT_SECONDS: int = _profile_int("MCP_TOOL_TIMEOUT_SECONDS")
MCP_WANDB_REQUEST_TIMEOUT_SECONDS: int = _profile_int("MCP_WANDB_REQUEST_TIMEOUT_SECONDS")
MCP_ADMISSION_CONTROL_ENABLED: bool = _profile_bool("MCP_ADMISSION_CONTROL_ENABLED")
MCP_ADMISSION_ACTOR_CAPACITY: int = _capacity_int("MCP_ADMISSION_ACTOR_CAPACITY", "actor_capacity")
MCP_ADMISSION_PROCESS_CAPACITY: int = _capacity_int("MCP_ADMISSION_PROCESS_CAPACITY", "process_capacity")
MCP_ADMISSION_WAIT_MS: int = _profile_int("MCP_ADMISSION_WAIT_MS")
MCP_MAX_QUERY_LIMIT: int = _profile_int("MCP_MAX_QUERY_LIMIT")
MCP_MAX_FULL_TRACE_LIMIT: int = _profile_int("MCP_MAX_FULL_TRACE_LIMIT")
MCP_MAX_HISTORY_SAMPLES: int = _profile_int("MCP_MAX_HISTORY_SAMPLES")
MCP_MAX_HISTORY_KEYS: int = _profile_int("MCP_MAX_HISTORY_KEYS")
MCP_MAX_HISTORY_RANGE_STEPS: int = _profile_int("MCP_MAX_HISTORY_RANGE_STEPS")
MCP_MAX_WANDB_QUERY_ITEMS: int = _profile_int("MCP_MAX_WANDB_QUERY_ITEMS")
MCP_MAX_FULL_DETAIL_ITEMS: int = _profile_int("MCP_MAX_FULL_DETAIL_ITEMS")
MCP_MAX_PROJECT_FIELDS: int = _profile_int("MCP_MAX_PROJECT_FIELDS")
MCP_MAX_PROBE_RUNS: int = _profile_int("MCP_MAX_PROBE_RUNS")
MCP_MAX_EVALUATION_ROWS: int = _profile_int("MCP_MAX_EVALUATION_ROWS")
MCP_MAX_SCHEMA_SAMPLE_ROWS: int = _profile_int("MCP_MAX_SCHEMA_SAMPLE_ROWS")
MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE: int = _profile_int("MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE")
MCP_MAX_GQL_ITEMS: int = _profile_int("MCP_MAX_GQL_ITEMS")
MCP_MAX_GQL_ITEMS_PER_PAGE: int = _profile_int("MCP_MAX_GQL_ITEMS_PER_PAGE")
MCP_SYNC_TOOL_WORKERS: int = _capacity_int("MCP_SYNC_TOOL_WORKERS", "sync_workers")
MCP_COUNT_TOOL_WORKERS: int = _capacity_int("MCP_COUNT_TOOL_WORKERS", "count_workers")
MCP_ANALYTICS_QUEUE_CAPACITY: int = _env_int(
    "MCP_ANALYTICS_QUEUE_CAPACITY",
    256,
    maximum=256,
)
MCP_ANALYTICS_TEST_BUFFER_CAPACITY: int = _env_int(
    "MCP_ANALYTICS_TEST_BUFFER_CAPACITY",
    100,
    maximum=1_000,
)
MCP_REQUEST_SUCCESS_SAMPLE_RATE: float = _env_float(
    "MCP_REQUEST_SUCCESS_SAMPLE_RATE",
    0.1,
    minimum=0.0,
    maximum=1.0,
)
SESSION_TTL_SECONDS: int = _profile_int("SESSION_TTL_SECONDS")
MAX_SESSIONS_PER_KEY: int = _profile_int("MAX_SESSIONS_PER_KEY")

if MCP_ADMISSION_ACTOR_CAPACITY > MCP_ADMISSION_PROCESS_CAPACITY:
    raise ValueError("MCP_ADMISSION_ACTOR_CAPACITY must not exceed MCP_ADMISSION_PROCESS_CAPACITY")
if MCP_ADMISSION_CONTROL_ENABLED and MCP_ADMISSION_ACTOR_CAPACITY < 4:
    raise ValueError("MCP_ADMISSION_ACTOR_CAPACITY must be at least 4 when admission control is enabled")
if MCP_ADMISSION_CONTROL_ENABLED and MCP_ADMISSION_PROCESS_CAPACITY < 4:
    raise ValueError("MCP_ADMISSION_PROCESS_CAPACITY must be at least 4 when admission control is enabled")
if MCP_SYNC_TOOL_WORKERS > MCP_ADMISSION_PROCESS_CAPACITY:
    raise ValueError("MCP_SYNC_TOOL_WORKERS must not exceed MCP_ADMISSION_PROCESS_CAPACITY")
if MCP_COUNT_TOOL_WORKERS > MCP_ADMISSION_PROCESS_CAPACITY:
    raise ValueError("MCP_COUNT_TOOL_WORKERS must not exceed MCP_ADMISSION_PROCESS_CAPACITY")
MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS: bool = _env_bool(
    "MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS",
    False,
)
COST_SORT_FIELDS: frozenset[str] = frozenset({"total_cost", "completion_cost", "prompt_cost"})


class HostedLimitExceeded(ValueError):
    """Raised when a hosted-mode request exceeds configured safety limits."""

    def __init__(self, message: str, *, error: str = "quota_exceeded", **details: object) -> None:
        super().__init__(message)
        self.error = error
        self.details = details


def structured_error(error: str, message: str, **extra: object) -> dict[str, object]:
    """Build a standard MCP JSON error payload."""
    payload: dict[str, object] = {"error": error, "message": message}
    payload.update(extra)
    return payload
