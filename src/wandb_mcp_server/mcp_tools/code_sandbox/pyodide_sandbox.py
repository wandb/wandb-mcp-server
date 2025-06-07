"""
Pyodide local sandbox implementation using Deno for enhanced security.
"""

import asyncio
import asyncio.subprocess
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from wandb_mcp_server.utils import get_rich_logger
from .sandbox_utils import (
    SandboxError,
    validate_timeout,
    DEFAULT_TIMEOUT_SECONDS,
)

logger = get_rich_logger(__name__)


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Deno for enhanced security."""
    
    # Class-level persistent process
    _shared_process = None
    _process_lock = asyncio.Lock()
    _initialized = False
    _initialization_error = None
    
    def __init__(self):
        self.available = self._check_deno_available()
        self._pyodide_script_path = Path(__file__).parent / "pyodide_sandbox.ts"
    
    async def __aenter__(self):
        """Context manager entry - ensure process is ready."""
        if self.available:
            await self.get_or_create_process(self._pyodide_script_path)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - process cleanup handled at shutdown."""
        # Don't cleanup process here as it's shared across executions
        return False
    
    def _check_deno_available(self) -> bool:
        """Check if Deno is available."""
        try:
            result = subprocess.run(
                ["deno", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning(
                    "Deno is installed but returned an error. "
                    "Please ensure Deno is properly installed. "
                    "Visit https://deno.land/manual/getting_started/installation for installation instructions."
                )
                return False
            return True
        except FileNotFoundError:
            logger.warning(
                "Deno is not installed. Pyodide sandbox requires Deno for secure local execution. "
                "To install Deno:\n"
                "  - macOS/Linux: curl -fsSL https://deno.land/install.sh | sh\n"
                "  - Windows: irm https://deno.land/install.ps1 | iex\n"
                "  - Or visit: https://deno.land/manual/getting_started/installation"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Deno check timed out. Please ensure Deno is properly installed.")
            return False
        except Exception as e:
            logger.warning(f"Error checking for Deno: {e}")
            return False
    
    @classmethod
    async def get_or_create_process(cls, script_path: Path):
        """Get or create the shared Pyodide process."""
        async with cls._process_lock:
            # Check if we have a cached initialization error
            if cls._initialization_error:
                raise cls._initialization_error
            
            if cls._shared_process is None or cls._shared_process.returncode is not None:
                logger.info("Starting persistent Pyodide sandbox process")
                try:
                    cls._shared_process = await asyncio.create_subprocess_exec(
                        'deno', 'run',
                        '--allow-net',  # For downloading Pyodide and packages
                        '--allow-read',  # To read local files
                        '--allow-write',  # To write output files and cache
                        '--allow-env',  # For environment variables
                        str(script_path),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    cls._initialized = True
                    
                    # Read initialization messages
                    try:
                        # Wait for "ready" message with timeout
                        ready_task = asyncio.create_task(cls._read_until_ready())
                        await asyncio.wait_for(ready_task, timeout=60)
                        logger.info("Pyodide sandbox process initialized successfully")
                        cls._initialization_error = None  # Clear any previous error
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for Pyodide to initialize")
                        if cls._shared_process:
                            cls._shared_process.kill()
                            await cls._shared_process.wait()
                        cls._shared_process = None
                        error = SandboxError(
                            "Failed to initialize Pyodide sandbox. "
                            "This may be due to network issues downloading Pyodide. "
                            "Please check your internet connection and try again."
                        )
                        cls._initialization_error = error
                        raise error
                except FileNotFoundError:
                    error = SandboxError(
                        "Deno executable not found. Please install Deno to use the Pyodide sandbox.\n"
                        "Installation instructions: https://deno.land/manual/getting_started/installation"
                    )
                    cls._initialization_error = error
                    raise error
                except Exception as e:
                    error = SandboxError(f"Failed to start Pyodide sandbox process: {e}")
                    cls._initialization_error = error
                    raise error
                    
            return cls._shared_process
    
    @classmethod
    async def _read_until_ready(cls):
        """Read stderr until we see the ready message."""
        if not cls._shared_process:
            return
            
        while True:
            line = await cls._shared_process.stderr.readline()
            if not line:
                break
            line_str = line.decode('utf-8').strip()
            logger.debug(f"Pyodide init: {line_str}")
            if "Pyodide sandbox server ready" in line_str:
                break
            elif "Initializing Pyodide..." in line_str:
                logger.info("Downloading and initializing Pyodide (this may take a minute on first run)...")
    
    async def execute_code(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Execute Python code using our persistent Pyodide process."""
        # Validate timeout
        timeout = validate_timeout(timeout)
        
        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution.\n"
                "To install Deno:\n"
                "  - macOS/Linux: curl -fsSL https://deno.land/install.sh | sh\n"
                "  - Windows: irm https://deno.land/install.ps1 | iex\n"
                "  - Or visit: https://deno.land/manual/getting_started/installation"
            )
        
        if not self._pyodide_script_path.exists():
            raise SandboxError(f"Pyodide script not found at {self._pyodide_script_path}")
        
        try:
            # Get or create the persistent process
            process = await self.get_or_create_process(self._pyodide_script_path)
            
            # Create execution request
            request = {
                "type": "execute",
                "code": code,
                "timeout": timeout
            }
            
            # Send request as a single line JSON
            request_json = json.dumps(request) + '\n'
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=timeout + 5  # Give extra time for process overhead
                )
            except asyncio.TimeoutError:
                # Don't kill the process on timeout, just return error
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timed out after {timeout} seconds",
                    "logs": [],
                }
            
            # Parse the response
            try:
                if response_line:
                    result = json.loads(response_line.decode('utf-8'))
                    return result
                else:
                    return {
                        "success": False,
                        "output": "",
                        "error": "No response from Pyodide process",
                        "logs": [],
                    }
                    
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Failed to parse response: {e}",
                    "logs": [],
                }
            
        except SandboxError:
            raise
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Pyodide execution failed: {str(e)}",
                "logs": [],
            }
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the Pyodide sandbox using persistent process."""
        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution. "
                "Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            )
        
        try:
            # Get or create the persistent process
            process = await self.get_or_create_process(self._pyodide_script_path)
            
            # Create file write request
            request = {
                "type": "writeFile",
                "path": path,
                "content": content
            }
            
            # Send request as a single line JSON
            request_json = json.dumps(request) + '\n'
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                raise SandboxError(f"Timeout writing file to {path}")
            
            # Parse the response
            if response_line:
                result = json.loads(response_line.decode('utf-8'))
                if not result.get("success", False):
                    raise SandboxError(f"Failed to write file: {result.get('error', 'Unknown error')}")
            else:
                raise SandboxError("No response from Pyodide process")
                
        except Exception as e:
            if isinstance(e, SandboxError):
                raise
            raise SandboxError(f"Failed to write file to Pyodide: {str(e)}")
    
    @classmethod
    async def initialize_early(cls):
        """Initialize Pyodide early to download packages during server startup."""
        sandbox = cls()
        if not sandbox.available:
            logger.info("Deno not available, skipping Pyodide early initialization")
            return False
            
        logger.info("Starting early Pyodide initialization to download packages...")
        try:
            # Create the process and send a simple execute request to trigger initialization
            await sandbox.get_or_create_process(sandbox._pyodide_script_path)
            
            # Execute a simple command to trigger package loading
            result = await sandbox.execute("print('Pyodide initialized successfully')", timeout=60)
            
            if result.get("success"):
                logger.info("Pyodide packages downloaded and ready for use")
                return True
            else:
                logger.warning(f"Pyodide initialization completed with error: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to initialize Pyodide early: {e}")
            return False
    
    @classmethod
    async def cleanup_shared_process(cls):
        """Explicitly close the shared Pyodide process (for cleanup)."""
        async with cls._process_lock:
            if cls._shared_process is not None:
                try:
                    # First try graceful termination
                    cls._shared_process.terminate()
                    try:
                        # Wait for process to exit with timeout
                        await asyncio.wait_for(cls._shared_process.wait(), timeout=5.0)
                        logger.info("Pyodide process terminated gracefully")
                    except asyncio.TimeoutError:
                        # Force kill if termination times out
                        logger.warning("Pyodide process did not terminate gracefully, forcing kill")
                        cls._shared_process.kill()
                        await cls._shared_process.wait()
                        logger.info("Pyodide process killed forcefully")
                except Exception as e:
                    logger.error(f"Error closing shared Pyodide process: {e}")
                finally:
                    cls._shared_process = None
                    cls._initialized = False
                    cls._initialization_error = None