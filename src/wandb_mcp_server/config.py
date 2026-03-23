import os

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
