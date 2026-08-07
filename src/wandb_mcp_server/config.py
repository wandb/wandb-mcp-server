import os


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
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be an integer") from None
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
WB_AGENT_BASE_URL: str = os.getenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai").rstrip("/")

# Token budget for response truncation. When a query result exceeds this
# budget, least-recent traces are dropped and a truncation note is appended.
MAX_RESPONSE_TOKENS: int = _env_int("MAX_RESPONSE_TOKENS", 30_000, maximum=100_000)

# Memory guard for trace queries. Stops accumulating trace data when this
# threshold is reached, returning a partial result instead of OOM-crashing.
MAX_ACCUMULATED_BYTES: int = _env_int(
    "MAX_ACCUMULATED_BYTES",
    1024 * 1024 * 1024,
    maximum=1024 * 1024 * 1024,
)

# Workload profiles keep the common deployment choices simple while retaining
# the existing per-setting environment overrides for advanced operators.
MCP_HOSTED_MODE: bool = _env_bool("MCP_HOSTED_MODE", False)
_default_workload_profile = "shared" if MCP_HOSTED_MODE else "local"
MCP_WORKLOAD_PROFILE: str = (os.getenv("MCP_WORKLOAD_PROFILE") or _default_workload_profile).strip().lower()
if MCP_WORKLOAD_PROFILE not in {"shared", "dedicated", "local"}:
    raise ValueError("MCP_WORKLOAD_PROFILE must be one of: shared, dedicated, local")

_PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "shared": {
        "collection_items": 100,
        "full_detail_items": 3,
        "history_samples": 500,
        "history_keys": 20,
        "history_range_steps": 5_000,
        "project_fields": 500,
        "probe_runs": 6,
        "evaluation_rows": 500,
        "schema_rows": 100,
        "actor_capacity": 4,
        "process_capacity": 16,
    },
    "dedicated": {
        "collection_items": 250,
        "full_detail_items": 10,
        "history_samples": 1_500,
        "history_keys": 50,
        "history_range_steps": 20_000,
        "project_fields": 2_000,
        "probe_runs": 12,
        "evaluation_rows": 2_000,
        "schema_rows": 250,
        "actor_capacity": 8,
        "process_capacity": 16,
    },
    "local": {
        "collection_items": 1_000,
        "full_detail_items": 25,
        "history_samples": 5_000,
        "history_keys": 100,
        "history_range_steps": 100_000,
        "project_fields": 5_000,
        "probe_runs": 24,
        "evaluation_rows": 5_000,
        "schema_rows": 500,
        "actor_capacity": 16,
        "process_capacity": 16,
    },
}
_profile_defaults = _PROFILE_DEFAULTS[MCP_WORKLOAD_PROFILE]

MCP_TOOL_TIMEOUT_SECONDS: int = _env_int("MCP_TOOL_TIMEOUT_SECONDS", 30, maximum=300)
MCP_WANDB_REQUEST_TIMEOUT_SECONDS: int = _env_int(
    "MCP_WANDB_REQUEST_TIMEOUT_SECONDS",
    20,
    maximum=120,
)
MCP_ADMISSION_CONTROL_ENABLED: bool = _env_bool(
    "MCP_ADMISSION_CONTROL_ENABLED",
    MCP_WORKLOAD_PROFILE != "local",
)
MCP_ADMISSION_ACTOR_CAPACITY: int = _env_int(
    "MCP_ADMISSION_ACTOR_CAPACITY",
    _profile_defaults["actor_capacity"],
    maximum=64,
)
MCP_ADMISSION_PROCESS_CAPACITY: int = _env_int(
    "MCP_ADMISSION_PROCESS_CAPACITY",
    _profile_defaults["process_capacity"],
    maximum=64,
)
MCP_ADMISSION_WAIT_MS: int = _env_int(
    "MCP_ADMISSION_WAIT_MS",
    2000,
    minimum=0,
    maximum=30_000,
)
MCP_MAX_QUERY_LIMIT: int = _env_int(
    "MCP_MAX_QUERY_LIMIT",
    _profile_defaults["collection_items"],
    maximum=10_000,
)
MCP_MAX_FULL_TRACE_LIMIT: int = _env_int(
    "MCP_MAX_FULL_TRACE_LIMIT",
    25 if MCP_WORKLOAD_PROFILE == "shared" else _profile_defaults["collection_items"],
    maximum=1_000,
)
MCP_MAX_HISTORY_SAMPLES: int = _env_int(
    "MCP_MAX_HISTORY_SAMPLES",
    _profile_defaults["history_samples"],
    maximum=10_000,
)
MCP_MAX_HISTORY_KEYS: int = _env_int(
    "MCP_MAX_HISTORY_KEYS",
    _profile_defaults["history_keys"],
    maximum=500,
)
MCP_MAX_HISTORY_RANGE_STEPS: int = _env_int(
    "MCP_MAX_HISTORY_RANGE_STEPS",
    _profile_defaults["history_range_steps"],
    maximum=1_000_000,
)
MCP_MAX_WANDB_QUERY_ITEMS: int = _env_int(
    "MCP_MAX_WANDB_QUERY_ITEMS",
    _profile_defaults["collection_items"],
    maximum=10_000,
)
MCP_MAX_FULL_DETAIL_ITEMS: int = _env_int(
    "MCP_MAX_FULL_DETAIL_ITEMS",
    _profile_defaults["full_detail_items"],
    maximum=100,
)
MCP_MAX_PROJECT_FIELDS: int = _env_int(
    "MCP_MAX_PROJECT_FIELDS",
    _profile_defaults["project_fields"],
    maximum=10_000,
)
MCP_MAX_PROBE_RUNS: int = _env_int(
    "MCP_MAX_PROBE_RUNS",
    _profile_defaults["probe_runs"],
    maximum=100,
)
MCP_MAX_EVALUATION_ROWS: int = _env_int(
    "MCP_MAX_EVALUATION_ROWS",
    _profile_defaults["evaluation_rows"],
    maximum=10_000,
)
MCP_MAX_SCHEMA_SAMPLE_ROWS: int = _env_int(
    "MCP_MAX_SCHEMA_SAMPLE_ROWS",
    _profile_defaults["schema_rows"],
    maximum=1_000,
)
MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE: int = _env_int(
    "MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE",
    50 if MCP_WORKLOAD_PROFILE == "shared" else 200,
    maximum=500,
)
MCP_MAX_GQL_ITEMS: int = _env_int(
    "MCP_MAX_GQL_ITEMS",
    _profile_defaults["collection_items"],
    maximum=1_000,
)
MCP_MAX_GQL_ITEMS_PER_PAGE: int = _env_int(
    "MCP_MAX_GQL_ITEMS_PER_PAGE",
    50 if MCP_WORKLOAD_PROFILE == "shared" else 200,
    maximum=200,
)
MCP_SYNC_TOOL_WORKERS: int = _env_int(
    "MCP_SYNC_TOOL_WORKERS",
    min(MCP_ADMISSION_PROCESS_CAPACITY, 16),
    maximum=16,
)
MCP_COUNT_TOOL_WORKERS: int = _env_int(
    "MCP_COUNT_TOOL_WORKERS",
    min(MCP_ADMISSION_PROCESS_CAPACITY, 8),
    maximum=16,
)
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
SESSION_TTL_SECONDS: int = _env_int("SESSION_TTL_SECONDS", 3600, maximum=86_400)
MAX_SESSIONS_PER_KEY: int = _env_int("MAX_SESSIONS_PER_KEY", 10, maximum=1_000)

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
WANDB_MCP_ENABLE_WEAVE_TOOLS: bool = _env_bool("WANDB_MCP_ENABLE_WEAVE_TOOLS", True)
WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS: bool = _env_bool(
    "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
    False,
)
WANDB_MCP_READ_ONLY: bool = _env_bool("WANDB_MCP_READ_ONLY", False)
WANDB_MCP_ENABLE_RAW_GRAPHQL: bool = _env_bool("WANDB_MCP_ENABLE_RAW_GRAPHQL", False)
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
