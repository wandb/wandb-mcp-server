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

# W&B Public API base URL
WANDB_BASE_URL: str = os.getenv("WANDB_BASE_URL") or "https://api.wandb.ai"

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

# Hosted-mode limits let Cloud Run deployments use stricter guardrails without
# changing local/stdio defaults for users running the package themselves.
MCP_HOSTED_MODE: bool = _env_bool("MCP_HOSTED_MODE", False)
MCP_TOOL_TIMEOUT_SECONDS: int = _env_int("MCP_TOOL_TIMEOUT_SECONDS", 30)
MCP_WANDB_REQUEST_TIMEOUT_SECONDS: int = _env_int("MCP_WANDB_REQUEST_TIMEOUT_SECONDS", 20)
MCP_ADMISSION_CONTROL_ENABLED: bool = _env_bool("MCP_ADMISSION_CONTROL_ENABLED", MCP_HOSTED_MODE)
MCP_ADMISSION_ACTOR_CAPACITY: int = _env_int("MCP_ADMISSION_ACTOR_CAPACITY", 4)
MCP_ADMISSION_PROCESS_CAPACITY: int = _env_int("MCP_ADMISSION_PROCESS_CAPACITY", 16)
MCP_ADMISSION_WAIT_MS: int = _env_int("MCP_ADMISSION_WAIT_MS", 2000)
MCP_MAX_QUERY_LIMIT: int = _env_int("MCP_MAX_QUERY_LIMIT", 100 if MCP_HOSTED_MODE else 1000)
MCP_MAX_FULL_TRACE_LIMIT: int = _env_int("MCP_MAX_FULL_TRACE_LIMIT", 25 if MCP_HOSTED_MODE else 1000)
MCP_MAX_HISTORY_SAMPLES: int = _env_int("MCP_MAX_HISTORY_SAMPLES", 500 if MCP_HOSTED_MODE else 2000)
MCP_MAX_HISTORY_KEYS: int = _env_int("MCP_MAX_HISTORY_KEYS", 20 if MCP_HOSTED_MODE else 100)
MCP_MAX_HISTORY_RANGE_STEPS: int = _env_int(
    "MCP_MAX_HISTORY_RANGE_STEPS",
    5000 if MCP_HOSTED_MODE else 100_000,
)
MCP_MAX_WANDB_QUERY_ITEMS: int = _env_int("MCP_MAX_WANDB_QUERY_ITEMS", 100 if MCP_HOSTED_MODE else 1000)
MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE: int = _env_int(
    "MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE",
    50 if MCP_HOSTED_MODE else 200,
)
MCP_MAX_GQL_ITEMS: int = _env_int("MCP_MAX_GQL_ITEMS", 100 if MCP_HOSTED_MODE else 1000)
MCP_MAX_GQL_ITEMS_PER_PAGE: int = _env_int("MCP_MAX_GQL_ITEMS_PER_PAGE", 50 if MCP_HOSTED_MODE else 200)
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
