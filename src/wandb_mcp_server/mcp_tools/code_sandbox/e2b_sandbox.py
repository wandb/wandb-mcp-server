"""
E2B cloud sandbox implementation.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from wandb_mcp_server.utils import get_rich_logger
from .sandbox_utils import (
    SandboxError,
    validate_timeout,
    get_validated_env_int,
    validate_packages,
    DEFAULT_TIMEOUT_SECONDS,
    E2B_STARTUP_TIMEOUT_SECONDS,
)

logger = get_rich_logger(__name__)


class E2BSandbox:
    """E2B cloud sandbox implementation with a single persistent instance."""
    
    # Class-level persistent sandbox instance
    _shared_sandbox = None
    _sandbox_lock = asyncio.Lock()
    _api_key = None
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None  # Initialize the sandbox attribute
        E2BSandbox._api_key = api_key
    
    async def __aenter__(self):
        """Context manager entry - get sandbox reference."""
        await self.create_sandbox()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release sandbox reference."""
        await self.close_sandbox()
        return False
    
    @classmethod
    async def get_or_create_sandbox(cls):
        """Get the shared sandbox instance, creating it if necessary."""
        async with cls._sandbox_lock:
            if cls._shared_sandbox is None:
                if not cls._api_key:
                    raise ValueError("E2B API key not set")
                logger.info("Creating new E2B sandbox instance")
                from e2b_code_interpreter import AsyncSandbox
                
                os.environ["E2B_API_KEY"] = cls._api_key
                
                # Get timeout from environment or use E2B default (15 minutes)
                # E2B expects timeout in seconds, not milliseconds
                timeout_seconds = get_validated_env_int(
                    "E2B_SANDBOX_TIMEOUT_SECONDS", 
                    900,  # 15 minutes default
                    min_val=60,  # 1 minute minimum
                    max_val=3600  # 1 hour maximum
                )
                logger.info(f"Creating E2B sandbox with timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} minutes)")
                
                cls._shared_sandbox = await AsyncSandbox.create(timeout=timeout_seconds)
                logger.info("E2B sandbox instance created successfully")
            return cls._shared_sandbox
    
    async def create_sandbox(self):
        """Get reference to the shared sandbox instance."""
        try:
            self.sandbox = await self.get_or_create_sandbox()
            logger.debug("Using shared E2B sandbox instance")
        except Exception as e:
            logger.error(f"Failed to create/get E2B sandbox: {e}", exc_info=True)
            raise
    
    def pre_validate_packages(self, packages: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Pre-validate packages before installation.
        Returns (valid_packages, denied_packages, invalid_packages)
        """
        return validate_packages(packages)
    
    async def install_packages(self, packages: List[str]) -> bool:
        """Install packages in the sandbox."""
        if not packages or not self.sandbox:
            return True
        
        try:
            # Validate packages
            valid_packages, denied_packages, invalid_packages = self.pre_validate_packages(packages)
            
            if denied_packages:
                logger.warning(f"Denied packages: {', '.join(denied_packages)}")
            
            if invalid_packages:
                logger.warning(f"Invalid packages: {', '.join(invalid_packages)}")
            
            if not valid_packages:
                if denied_packages or invalid_packages:
                    return False  # All packages were denied or invalid
                return True  # No packages to install
            
            # Install packages
            package_str = " ".join(valid_packages)
            logger.info(f"Installing packages: {package_str}")
            
            result = await self.sandbox.commands.run(
                f"uv pip install --no-cache-dir {package_str}",
                timeout=E2B_STARTUP_TIMEOUT_SECONDS
            )
            
            success = result.exit_code == 0
            if not success:
                logger.error(f"Package installation failed: {result.stderr}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    async def execute_code(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, install_packages_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute Python code in the E2B sandbox."""
        # Validate timeout
        timeout = validate_timeout(timeout)
        
        logger.debug(f"execute_code called, self.sandbox is: {self.sandbox}")
        if not hasattr(self, 'sandbox') or self.sandbox is None:
            logger.debug("self.sandbox not set, calling create_sandbox()")
            await self.create_sandbox()
            logger.debug(f"After create_sandbox, self.sandbox is: {self.sandbox}")
        
        # Install packages if requested
        if install_packages_list:
            success = await self.install_packages(install_packages_list)
            if not success:
                return {
                    "success": False,
                    "output": "",
                    "error": "Failed to install requested packages",
                    "logs": [],
                }
        
        try:
            # Write code to a temporary file in the sandbox to avoid shell escaping issues
            file_path = "/tmp/code_to_execute.py"
            
            # Write the code to file
            await self.sandbox.files.write(file_path, code)
            
            # Execute the file
            execution = await self.sandbox.commands.run(
                f"python {file_path}",
                timeout=timeout
            )
            
            # Format the result
            output = execution.stdout if execution.stdout else ""
            error_output = execution.stderr if execution.stderr else ""
            
            success = execution.exit_code == 0
            error_msg = error_output if error_output and not success else None
            
            # Clean up the temporary file
            try:
                await self.sandbox.commands.run(f"rm {file_path}")
            except Exception:
                pass  # Ignore cleanup errors
            
            return {
                "success": success,
                "output": output,
                "error": error_msg,
                "logs": [output] if not error_output else [output, error_output],
            }
            
        except Exception as e:
            logger.error(f"E2B execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "output": "",
                "error": f"E2B execution failed: {str(e)}",
                "logs": [],
            }
    
    async def close_sandbox(self):
        """Release the reference to the sandbox (but keep it running)."""
        # Don't actually close the shared sandbox - just release our reference
        self.sandbox = None
        logger.debug("Released reference to shared E2B sandbox")
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the E2B sandbox using native file operations."""
        if not self.sandbox:
            await self.create_sandbox()
        
        try:
            await self.sandbox.files.write(path, content)
            logger.debug(f"Wrote file to E2B sandbox: {path}")
        except Exception as e:
            raise SandboxError(f"Failed to write file to E2B sandbox: {e}")
    
    @classmethod
    async def cleanup_shared_sandbox(cls):
        """Explicitly close the shared sandbox instance (for cleanup)."""
        async with cls._sandbox_lock:
            if cls._shared_sandbox is not None:
                try:
                    await cls._shared_sandbox.close()
                    logger.info("Closed shared E2B sandbox instance")
                except Exception as e:
                    logger.error(f"Error closing shared E2B sandbox: {e}")
                finally:
                    cls._shared_sandbox = None