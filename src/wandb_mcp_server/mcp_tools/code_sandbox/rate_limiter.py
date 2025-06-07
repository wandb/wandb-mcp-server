"""
Rate limiting for sandbox executions.
"""

import asyncio
import time
from typing import List

from wandb_mcp_server.utils import get_rich_logger
from .sandbox_utils import (
    get_validated_env_int,
    DEFAULT_RATE_LIMIT_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
)

logger = get_rich_logger(__name__)

# Get rate limit configuration from environment
RATE_LIMIT_REQUESTS = get_validated_env_int(
    "SANDBOX_RATE_LIMIT_REQUESTS", 
    DEFAULT_RATE_LIMIT_REQUESTS,
    min_val=1,
    max_val=1000
)
RATE_LIMIT_WINDOW_SECONDS = get_validated_env_int(
    "SANDBOX_RATE_LIMIT_WINDOW_SECONDS",
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    min_val=1,
    max_val=300  # 5 minutes max
)


class RateLimiter:
    """Simple rate limiter for sandbox executions."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
        logger.info(f"Rate limiter configured: {max_requests} requests per {window_seconds} seconds")
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = time.time()
            # Remove old requests
            self.requests = [r for r in self.requests if now - r < self.window_seconds]
            
            if len(self.requests) >= self.max_requests:
                return False
            
            self.requests.append(now)
            return True