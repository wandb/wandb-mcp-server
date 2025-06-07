"""
Code sandbox execution module for the W&B MCP server.
Provides secure Python code execution through multiple sandbox environments.
"""

from .execute_sandbox_code import (
    execute_sandbox_code,
    EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION,
    check_sandbox_availability,
)

from .e2b_sandbox import E2BSandbox
from .pyodide_sandbox import PyodideSandbox
from .sandbox_utils import SandboxError

from .sandbox_models import (
    SandboxType,
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxSession,
)

__all__ = [
    # Main function
    "execute_sandbox_code",
    "EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION",
    "check_sandbox_availability",
    
    # Exceptions
    "SandboxError",
    
    # Sandbox implementations
    "E2BSandbox",
    "PyodideSandbox",
    
    # Models
    "SandboxType",
    "SandboxExecutionRequest",
    "SandboxExecutionResult",
    "SandboxSession",
]