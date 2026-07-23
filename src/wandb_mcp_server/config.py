import os


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


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

# Token budget for response truncation. When a query result exceeds this
# budget, least-recent traces are dropped and a truncation note is appended.
try:
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "30000"))
except (ValueError, TypeError):
    MAX_RESPONSE_TOKENS: int = 30000

# Memory guard for trace queries. Stops accumulating trace data when this
# threshold is reached, returning a partial result instead of OOM-crashing.
try:
    MAX_ACCUMULATED_BYTES: int = int(os.getenv("MAX_ACCUMULATED_BYTES", str(1024 * 1024 * 1024)))
except (ValueError, TypeError):
    MAX_ACCUMULATED_BYTES: int = 1024 * 1024 * 1024  # 1GB

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

MCP_TOOL_TIMEOUT_SECONDS: int = _env_int("MCP_TOOL_TIMEOUT_SECONDS", 30)
MCP_WANDB_REQUEST_TIMEOUT_SECONDS: int = _env_int("MCP_WANDB_REQUEST_TIMEOUT_SECONDS", 20)
MCP_ADMISSION_CONTROL_ENABLED: bool = _env_bool(
    "MCP_ADMISSION_CONTROL_ENABLED",
    MCP_WORKLOAD_PROFILE != "local",
)
MCP_ADMISSION_ACTOR_CAPACITY: int = _env_int(
    "MCP_ADMISSION_ACTOR_CAPACITY",
    _profile_defaults["actor_capacity"],
)
MCP_ADMISSION_PROCESS_CAPACITY: int = _env_int(
    "MCP_ADMISSION_PROCESS_CAPACITY",
    _profile_defaults["process_capacity"],
)
MCP_ADMISSION_WAIT_MS: int = _env_int("MCP_ADMISSION_WAIT_MS", 2000)
MCP_MAX_QUERY_LIMIT: int = _env_int("MCP_MAX_QUERY_LIMIT", _profile_defaults["collection_items"])
MCP_MAX_FULL_TRACE_LIMIT: int = _env_int(
    "MCP_MAX_FULL_TRACE_LIMIT",
    25 if MCP_WORKLOAD_PROFILE == "shared" else _profile_defaults["collection_items"],
)
MCP_MAX_HISTORY_SAMPLES: int = _env_int("MCP_MAX_HISTORY_SAMPLES", _profile_defaults["history_samples"])
MCP_MAX_HISTORY_KEYS: int = _env_int("MCP_MAX_HISTORY_KEYS", _profile_defaults["history_keys"])
MCP_MAX_HISTORY_RANGE_STEPS: int = _env_int(
    "MCP_MAX_HISTORY_RANGE_STEPS",
    _profile_defaults["history_range_steps"],
)
MCP_MAX_WANDB_QUERY_ITEMS: int = _env_int(
    "MCP_MAX_WANDB_QUERY_ITEMS",
    _profile_defaults["collection_items"],
)
MCP_MAX_FULL_DETAIL_ITEMS: int = _env_int(
    "MCP_MAX_FULL_DETAIL_ITEMS",
    _profile_defaults["full_detail_items"],
)
MCP_MAX_PROJECT_FIELDS: int = _env_int(
    "MCP_MAX_PROJECT_FIELDS",
    _profile_defaults["project_fields"],
)
MCP_MAX_PROBE_RUNS: int = _env_int(
    "MCP_MAX_PROBE_RUNS",
    _profile_defaults["probe_runs"],
)
MCP_MAX_EVALUATION_ROWS: int = _env_int(
    "MCP_MAX_EVALUATION_ROWS",
    _profile_defaults["evaluation_rows"],
)
MCP_MAX_SCHEMA_SAMPLE_ROWS: int = _env_int(
    "MCP_MAX_SCHEMA_SAMPLE_ROWS",
    _profile_defaults["schema_rows"],
)
MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE: int = _env_int(
    "MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE",
    50 if MCP_WORKLOAD_PROFILE == "shared" else 200,
)
MCP_MAX_GQL_ITEMS: int = _env_int("MCP_MAX_GQL_ITEMS", _profile_defaults["collection_items"])
MCP_MAX_GQL_ITEMS_PER_PAGE: int = _env_int(
    "MCP_MAX_GQL_ITEMS_PER_PAGE",
    50 if MCP_WORKLOAD_PROFILE == "shared" else 200,
)
WANDB_MCP_ENABLE_WEAVE_TOOLS: bool = _env_bool("WANDB_MCP_ENABLE_WEAVE_TOOLS", True)
WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS: bool = _env_bool(
    "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
    False,
)
WANDB_MCP_READ_ONLY: bool = _env_bool("WANDB_MCP_READ_ONLY", False)
WANDB_MCP_ENABLE_RAW_GRAPHQL: bool = _env_bool("WANDB_MCP_ENABLE_RAW_GRAPHQL", False)
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
