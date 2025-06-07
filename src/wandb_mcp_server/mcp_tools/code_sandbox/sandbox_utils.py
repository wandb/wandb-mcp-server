"""
Shared utilities for sandbox implementations.
"""

import os
import re
from typing import List, Optional, Tuple

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

# Configuration constants
DEFAULT_CACHE_SIZE = 100
DEFAULT_CACHE_TTL_SECONDS = 900  # 15 minutes
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
E2B_STARTUP_TIMEOUT_SECONDS = 60

# Package installation security configuration
# Can be overridden via E2B_PACKAGE_ALLOWLIST and E2B_PACKAGE_DENYLIST env vars
DEFAULT_PACKAGE_ALLOWLIST = None  # None means allow all (unless denied)
DEFAULT_PACKAGE_DENYLIST = [
    # Packages that could be used maliciously
    "subprocess32",  # Subprocess with additional features
    "psutil",  # System and process utilities
    "pyautogui",  # GUI automation
    "pynput",  # Input control
]


def validate_timeout(timeout: int, param_name: str = "timeout") -> int:
    """Validate timeout value is within acceptable range."""
    if timeout < 1:
        raise ValueError(f"{param_name} must be at least 1 second, got {timeout}")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{param_name} must not exceed {MAX_TIMEOUT_SECONDS} seconds, got {timeout}")
    return timeout


def get_validated_env_int(env_var: str, default: int, min_val: int = 1, max_val: Optional[int] = None) -> int:
    """Get and validate integer environment variable."""
    try:
        value = int(os.getenv(env_var, str(default)))
        if value < min_val:
            logger.warning(f"{env_var}={value} is below minimum {min_val}, using {min_val}")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{env_var}={value} exceeds maximum {max_val}, using {max_val}")
            return max_val
        return value
    except ValueError:
        logger.warning(f"Invalid {env_var} value, using default {default}")
        return default


def get_package_filters() -> Tuple[Optional[List[str]], List[str]]:
    """Get package allowlist and denylist from environment or defaults."""
    # Get allowlist
    allowlist_env = os.getenv("E2B_PACKAGE_ALLOWLIST")
    if allowlist_env:
        allowlist = [pkg.strip() for pkg in allowlist_env.split(",") if pkg.strip()]
    else:
        allowlist = DEFAULT_PACKAGE_ALLOWLIST
    
    # Get denylist
    denylist_env = os.getenv("E2B_PACKAGE_DENYLIST")
    if denylist_env:
        denylist = [pkg.strip() for pkg in denylist_env.split(",") if pkg.strip()]
    else:
        denylist = DEFAULT_PACKAGE_DENYLIST
    
    return allowlist, denylist


def validate_packages(packages: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate and filter package names for security.
    
    Returns:
        Tuple of (valid_packages, denied_packages, invalid_packages)
    """
    allowlist, denylist = get_package_filters()
    
    valid_packages = []
    denied_packages = []
    invalid_packages = []
    
    for pkg in packages:
        pkg = pkg.strip()
        
        # First check format safety
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', pkg):
            invalid_packages.append(pkg)
            logger.warning(f"Package '{pkg}' has invalid characters")
            continue
        
        # Check against denylist
        if denylist and pkg.lower() in [d.lower() for d in denylist]:
            denied_packages.append(pkg)
            logger.warning(f"Package '{pkg}' is in denylist")
            continue
        
        # Check against allowlist (if specified)
        if allowlist and pkg.lower() not in [a.lower() for a in allowlist]:
            denied_packages.append(pkg)
            logger.warning(f"Package '{pkg}' is not in allowlist")
            continue
        
        valid_packages.append(pkg)
    
    return valid_packages, denied_packages, invalid_packages


class SandboxError(Exception):
    """Exception raised when sandbox execution fails."""
    pass