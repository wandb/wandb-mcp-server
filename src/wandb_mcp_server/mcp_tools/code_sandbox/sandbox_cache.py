"""
Caching implementation for sandbox execution results.
"""

import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from wandb_mcp_server.utils import get_rich_logger
from .sandbox_utils import DEFAULT_CACHE_SIZE, DEFAULT_CACHE_TTL_SECONDS, get_validated_env_int

logger = get_rich_logger(__name__)

# Get cache TTL from environment or use default
CACHE_TTL_SECONDS = get_validated_env_int("E2B_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS, min_val=60, max_val=3600)


class ExecutionCache:
    """Simple LRU cache for code execution results."""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, code: str, sandbox_type: str, packages: Optional[List[str]]) -> str:
        """Generate cache key from code and parameters."""
        package_str = ",".join(sorted(packages)) if packages else ""
        content = f"{code}|{sandbox_type}|{package_str}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, code: str, sandbox_type: str, packages: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        key = self._get_cache_key(code, sandbox_type, packages)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Move to end to maintain LRU order
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for key {key[:8]}...")
                return entry
            else:
                # Expired
                del self.cache[key]
        return None
    
    def set(self, code: str, sandbox_type: str, result: Dict[str, Any], packages: Optional[List[str]] = None):
        """Cache execution result."""
        key = self._get_cache_key(code, sandbox_type, packages)
        self.cache[key] = (result, datetime.now())
        self.cache.move_to_end(key)
        
        # Evict oldest if over size limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)