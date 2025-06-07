from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class SandboxType(str, Enum):
    """Available sandbox types."""
    E2B = "e2b"
    PYODIDE = "pyodide"


class SandboxExecutionRequest(BaseModel):
    """Request model for sandbox code execution."""
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=180, description="Maximum execution time in seconds", ge=1, le=300)
    sandbox_type: Optional[SandboxType] = Field(default=None, description="Force specific sandbox type")
    install_packages: Optional[List[str]] = Field(default=None, description="Packages to install (E2B only)")


class SandboxExecutionResult(BaseModel):
    """Result model for sandbox code execution."""
    success: bool = Field(..., description="Whether execution succeeded")
    output: str = Field(default="", description="Standard output from code execution")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    logs: List[str] = Field(default_factory=list, description="Execution logs")
    sandbox_used: str = Field(..., description="Type of sandbox that was used")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")


class SandboxSession(BaseModel):
    """Model representing a sandbox session."""
    session_id: str = Field(..., description="Unique session identifier")
    sandbox_type: SandboxType = Field(..., description="Type of sandbox")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    is_active: bool = Field(default=True, description="Whether session is active")