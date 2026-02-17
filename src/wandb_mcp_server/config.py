import os

# Centralized configuration for base URLs used across the project.
# Values are read from environment variables with production defaults.

# W&B Public API base URL
WANDB_BASE_URL: str = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")

# SaaS default host used to distinguish SaaS from dedicated/on-prem
_SAAS_API_HOST = "api.wandb.ai"
_SAAS_TRACE_HOST = "https://trace.wandb.ai"


def _resolve_trace_server_url() -> str:
    """Resolve the Weave trace server URL with auto-detection for dedicated/on-prem.

    Priority:
      1. Explicit WF_TRACE_SERVER_URL or WEAVE_TRACE_SERVER_URL env var -- use as-is.
      2. WANDB_BASE_URL is SaaS default (api.wandb.ai) -- use trace.wandb.ai.
      3. WANDB_BASE_URL is a dedicated/on-prem host -- append /traces suffix.
    """
    explicit = os.getenv("WF_TRACE_SERVER_URL") or os.getenv("WEAVE_TRACE_SERVER_URL")
    if explicit:
        return explicit

    if _SAAS_API_HOST in WANDB_BASE_URL:
        return _SAAS_TRACE_HOST

    # Dedicated / on-prem: trace server lives at {base_url}/traces
    return WANDB_BASE_URL.rstrip("/") + "/traces"


# Weave Trace server URL used by the Weave API client and services
WF_TRACE_SERVER_URL: str = _resolve_trace_server_url()
