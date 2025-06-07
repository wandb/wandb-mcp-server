"""
Sandbox code execution tool for the MCP server.
Supports both E2B cloud sandboxes and local Pyodide execution.
"""

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from wandb_mcp_server.utils import get_rich_logger

# Import from new modules
from .e2b_sandbox import E2BSandbox
from .pyodide_sandbox import PyodideSandbox
from .sandbox_cache import ExecutionCache, CACHE_TTL_SECONDS
from .sandbox_utils import (
    validate_timeout,
    DEFAULT_TIMEOUT_SECONDS,
)
from .rate_limiter import RateLimiter

logger = get_rich_logger(__name__)


def check_sandbox_availability() -> tuple[bool, List[str], str]:
    """
    Check if any sandbox is available for code execution.
    
    Returns:
        tuple containing:
            - is_available (bool): Whether any sandbox is available
            - available_types (List[str]): List of available sandbox types ('e2b', 'pyodide')
            - reason (str): Explanation of availability status
    """
    # Check if sandbox is disabled
    if os.getenv("DISABLE_CODE_SANDBOX"):
        return (
            False, 
            [], 
            "Code sandbox is disabled via DISABLE_CODE_SANDBOX environment variable. "
            "Remove this variable to enable sandbox functionality."
        )
    
    available_types = []
    reasons = []
    
    # Check E2B availability
    if os.getenv("E2B_API_KEY"):
        available_types.append("e2b")
        reasons.append("E2B cloud sandbox available (API key found)")
    else:
        reasons.append("E2B not available (E2B_API_KEY not set)")
    
    # Check Pyodide/Deno availability
    try:
        result = subprocess.run(
            ["deno", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            available_types.append("pyodide")
            reasons.append("Pyodide sandbox available (Deno found)")
        else:
            reasons.append("Pyodide not available (Deno command failed)")
    except FileNotFoundError:
        reasons.append(
            "Pyodide not available (Deno not installed). "
            "Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
        )
    except subprocess.TimeoutExpired:
        reasons.append("Pyodide not available (Deno check timed out)")
    except Exception as e:
        reasons.append(f"Pyodide not available (Error checking Deno: {str(e)})")
    
    # Determine overall availability
    is_available = len(available_types) > 0
    
    # Construct reason message
    if is_available:
        reason = f"Sandbox available. {' '.join(reasons)}"
    else:
        reason = (
            "No sandboxes available. To enable code execution:\n"
            "1. For cloud sandbox: Set E2B_API_KEY environment variable (get key at https://e2b.dev)\n"
            "2. For local sandbox: Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
        )
        if not is_available:
            reason = f"No sandboxes available. {' '.join(reasons)}"
    
    return (is_available, available_types, reason)


EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION = """
Execute Python code in a secure, isolated code sandbox environment for data analysis on queried \
Weights & Biases data. The sandbox comes with pandas and numpy pre-installed. If there is a need for data \
transforms to help answer the users' question then python code can be passed to this tool.

<usage_guidelines>
- Perfect for data analysis, visualization, and computational tasks
- Supports common packages like numpy, pandas, matplotlib
- Ideal for exploratory data analysis and quick computations
- Can be used to process W&B data safely
</usage_guidelines>

<security_features>
- **E2B**: Fully isolated cloud VM with ~150ms startup time, complete system isolation
- **Pyodide**: WebAssembly sandbox using Deno's permission model for enhanced security:
  - Explicit network permission only for package downloads
  - No filesystem access (except node_modules)
  - Process-level isolation with Deno's security sandbox
</security_features>

Args
-------
code : str
    Python code to execute in the sandbox.
timeout : int, optional
    Maximum execution time in seconds (default: 30).
install_packages : list of str, optional
    Additional packages to install for analysis on top of numpy and pandas.
    - E2B sandbox: Supports dynamic package installation with security filters
      (configurable via E2B_PACKAGE_ALLOWLIST and E2B_PACKAGE_DENYLIST env vars)
    - Pyodide sandbox: Pre-loaded with numpy, pandas, matplotlib only. 
      Additional pure Python packages can be imported but not installed.

Returns
-------
dict
    Dictionary with the following keys:
    success : bool
        Whether execution succeeded.
    output : str
        Standard output from code execution.
    error : str
        Error message if execution failed.
    logs : list of str
        Execution logs.
    sandbox_used : str
        Type of sandbox that was used.

Example:
    ```python
    # Simple computation
    result = execute_sandbox_code('''
    import math
    result = math.sqrt(16)
    print(f"Square root of 16 is {result}")
    ''')
    
    # Data analysis with pandas
    result = execute_sandbox_code('''
    import pandas as pd
    data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    df = pd.DataFrame(data)
    print(df.describe())
    ''')
    ```
"""

# Global instances
_execution_cache = ExecutionCache(ttl_seconds=CACHE_TTL_SECONDS)
_rate_limiter = RateLimiter()


async def execute_sandbox_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sandbox_type: Optional[str] = None,
    install_packages: Optional[List[str]] = None,
    return_sandbox: bool = False,
) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment.
    
    Automatically selects the best available sandbox or uses the specified type.
    """
    start_time = time.time()
    
    # Check rate limit
    if not await _rate_limiter.check_rate_limit():
        return {
            "success": False,
            "output": "",
            "error": "Rate limit exceeded. Please try again later.",
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Validate timeout
    try:
        timeout = validate_timeout(timeout)
    except ValueError as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Validate sandbox type if specified
    valid_sandbox_types = ["e2b", "pyodide", "auto", None]
    if sandbox_type and sandbox_type not in valid_sandbox_types:
        return {
            "success": False,
            "output": "",
            "error": f"Invalid sandbox_type: '{sandbox_type}'. Must be one of: {', '.join(str(t) for t in valid_sandbox_types[:-1])}",
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Normalize "auto" to None for auto-selection
    if sandbox_type == "auto":
        sandbox_type = None
    
    # Pre-validate packages for E2B if specified
    if install_packages and (sandbox_type == "e2b" or (sandbox_type is None and os.getenv("E2B_API_KEY"))):
        # Pre-flight validation
        api_key = os.getenv("E2B_API_KEY")
        if api_key:
            sandbox = E2BSandbox(api_key)
            valid_packages, denied_packages, invalid_packages = sandbox.pre_validate_packages(install_packages)
            
            if denied_packages or invalid_packages:
                error_parts = []
                if denied_packages:
                    error_parts.append(f"Denied packages: {', '.join(denied_packages)}")
                if invalid_packages:
                    error_parts.append(f"Invalid packages: {', '.join(invalid_packages)}")
                
                if not valid_packages:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Package validation failed. {' '.join(error_parts)}",
                        "logs": [],
                        "sandbox_used": "none",
                    }
    
    # Determine which sandbox to use
    sandboxes_to_try = []
    
    if sandbox_type:
        # User specified a sandbox type
        if sandbox_type == "e2b":
            api_key = os.getenv("E2B_API_KEY")
            if api_key is not None:
                sandboxes_to_try.append(("e2b", E2BSandbox(api_key)))
        elif sandbox_type == "pyodide":
            pyodide = PyodideSandbox()
            if pyodide.available:
                sandboxes_to_try.append(("pyodide", pyodide))
    else:
        # Auto-select based on availability
        # Check cache first
        for sb_type in ["e2b", "pyodide"]:
            cached_result = _execution_cache.get(code, sb_type, install_packages)
            if cached_result:
                # Add execution time to cached result
                cached_result["execution_time_ms"] = 0  # Cached, so no execution time
                return cached_result
        
        # Try E2B first if available
        api_key = os.getenv("E2B_API_KEY")
        if api_key is not None:
            sandboxes_to_try.append(("e2b", E2BSandbox(api_key)))
        
        # Then try Pyodide
        pyodide = PyodideSandbox()
        if pyodide.available:
            sandboxes_to_try.append(("pyodide", pyodide))
        
    
    # Check if we have any sandboxes available
    if not sandboxes_to_try:
        return {
            "success": False,
            "output": "",
            "error": (
                "No sandboxes available. To enable code execution:\n"
                "1. For cloud sandbox: Set E2B_API_KEY environment variable (get key at https://e2b.dev)\n"
                "2. For local sandbox: Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            ),
            "logs": [],
            "sandbox_used": "none",
            "execution_time_ms": int((time.time() - start_time) * 1000),
        }
    
    # Try each sandbox in order
    last_error = None
    for sandbox_name, sandbox in sandboxes_to_try:
        try:
            logger.info(f"Attempting to execute code in {sandbox_name} sandbox")
            
            # Execute based on sandbox type
            if sandbox_name == "e2b":
                result = await sandbox.execute_code(code, timeout, install_packages)
                await sandbox.close_sandbox()
            else:
                # Pyodide doesn't support dynamic package installation
                if install_packages and sandbox_name == "pyodide":
                    logger.warning(
                        "Pyodide sandbox doesn't support dynamic package installation. "
                        "Pre-loaded packages: numpy, pandas, matplotlib. "
                        "Additional packages can be imported if they're pure Python."
                    )
                result = await sandbox.execute_code(code, timeout)
            
            # Add sandbox info and execution time
            result["sandbox_used"] = sandbox_name
            result["execution_time_ms"] = int((time.time() - start_time) * 1000)
            
            # Optionally include sandbox instance for file operations
            if return_sandbox:
                result["_sandbox_instance"] = sandbox
                result["_sandbox_type"] = sandbox_name
            
            # Cache successful results
            if result["success"]:
                _execution_cache.set(code, sandbox_name, result, install_packages)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute in {sandbox_name} sandbox: {e}")
            last_error = str(e)
            continue
    
    # All sandboxes failed
    return {
        "success": False,
        "output": "",
        "error": f"All sandbox types failed. Last error: {last_error}",
        "logs": [],
        "sandbox_used": "none",
        "execution_time_ms": int((time.time() - start_time) * 1000),
    }