"""
Sandbox code execution tool for the MCP server.
Supports both E2B cloud sandboxes and local Pyodide execution.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from datetime import datetime, timedelta


from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION = """
Execute Python code in a secure, isolated sandbox environment. This tool provides safe code execution
using properly sandboxed environments to ensure security.

<sandbox_types>
The tool automatically selects the best available sandbox:
1. **E2B Cloud Sandbox** - If E2B_API_KEY is available (most secure, cloud-based VM isolation)
2. **Pyodide Local Sandbox** - If Node.js is available (WebAssembly-based isolation)

You can force a specific sandbox type using the sandbox_type parameter.
</sandbox_types>

<security_features>
- **E2B**: Fully isolated cloud VM with ~150ms startup time, complete system isolation
- **Pyodide**: WebAssembly sandbox with no filesystem access, runs in isolated memory space
</security_features>

<usage_guidelines>
- Perfect for data analysis, visualization, and computational tasks
- Supports common packages like numpy, pandas, matplotlib (depending on sandbox)
- Use for user-provided code that needs isolation from the host system
- Ideal for exploratory data analysis and quick computations
- Can be used to process W&B data safely
</usage_guidelines>

<debugging_tips>
- If E2B fails, check E2B_API_KEY environment variable
- If Pyodide fails, ensure Node.js is installed
- If both are unavailable, the tool will return an error
- Check the 'sandbox_used' field in results to see which sandbox was used
</debugging_tips>

Args:
    code (str): Python code to execute in the sandbox
    timeout (int, optional): Maximum execution time in seconds (default: 30)
    sandbox_type (str, optional): Force specific sandbox ('e2b', 'pyodide')
    install_packages (List[str], optional): Packages to install (E2B only)

Returns:
    Dict containing:
        - success (bool): Whether execution succeeded
        - output (str): Standard output from code execution
        - error (str): Error message if execution failed
        - logs (List[str]): Execution logs
        - sandbox_used (str): Type of sandbox that was used

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


class SandboxError(Exception):
    """Exception raised when sandbox execution fails."""
    pass


class ExecutionCache:
    """Simple LRU cache for code execution results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 900):  # 15 minutes TTL
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




class E2BSandboxPool:
    """Connection pool for E2B sandboxes."""
    
    def __init__(self, api_key: str, pool_size: int = 3):
        self.api_key = api_key
        self.pool_size = pool_size
        self.available = asyncio.Queue(maxsize=pool_size)
        self.all_sandboxes = []
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the sandbox pool."""
        async with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing E2B sandbox pool with {self.pool_size} sandboxes")
            
            # Pre-create sandboxes
            tasks = [self._create_sandbox() for _ in range(self.pool_size)]
            sandboxes = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(sandboxes):
                if isinstance(result, Exception):
                    logger.error(f"Failed to create sandbox {i}: {result}")
                else:
                    self.all_sandboxes.append(result)
                    await self.available.put(result)
            
            self._initialized = True
            logger.info(f"E2B sandbox pool initialized with {len(self.all_sandboxes)} sandboxes")
    
    async def _create_sandbox(self):
        """Create a single E2B sandbox."""
        from e2b_code_interpreter import AsyncSandbox
        
        os.environ["E2B_API_KEY"] = self.api_key
        sandbox = await AsyncSandbox.create()
        return sandbox
    
    async def acquire(self, timeout: float = 30.0):
        """Acquire a sandbox from the pool."""
        if not self._initialized:
            await self.initialize()
        
        try:
            sandbox = await asyncio.wait_for(self.available.get(), timeout=timeout)
            return sandbox
        except asyncio.TimeoutError:
            raise SandboxError("Timeout waiting for available sandbox")
    
    async def release(self, sandbox):
        """Release a sandbox back to the pool."""
        if sandbox in self.all_sandboxes:
            await self.available.put(sandbox)
    
    async def cleanup(self):
        """Clean up all sandboxes in the pool."""
        for sandbox in self.all_sandboxes:
            try:
                await sandbox.close()
            except Exception as e:
                logger.error(f"Error closing sandbox: {e}")
        
        self.all_sandboxes.clear()
        self._initialized = False


class E2BSandbox:
    """E2B cloud sandbox implementation using official SDK with pooling."""
    
    _pool: Optional[E2BSandboxPool] = None
    _pool_lock = asyncio.Lock()
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None
        self._acquired_from_pool = False
    
    @classmethod
    async def get_pool(cls, api_key: str) -> E2BSandboxPool:
        """Get or create the shared sandbox pool."""
        async with cls._pool_lock:
            if cls._pool is None:
                cls._pool = E2BSandboxPool(api_key)
                await cls._pool.initialize()
            return cls._pool
    
    async def create_sandbox(self):
        """Acquire a sandbox from the pool or create a new one."""
        try:
            pool = await self.get_pool(self.api_key)
            self.sandbox = await pool.acquire(timeout=5.0)
            self._acquired_from_pool = True
            logger.debug("Acquired sandbox from pool")
        except Exception as e:
            logger.warning(f"Failed to acquire from pool, creating new sandbox: {e}")
            # Fallback to creating a new sandbox
            from e2b_code_interpreter import AsyncSandbox
            
            os.environ["E2B_API_KEY"] = self.api_key
            self.sandbox = await AsyncSandbox.create()
            self._acquired_from_pool = False
    
    async def install_packages(self, packages: List[str]) -> bool:
        """Install packages in the sandbox."""
        if not packages or not self.sandbox:
            return True
        
        try:
            # Sanitize package names
            safe_packages = [pkg.strip() for pkg in packages if re.match(r'^[a-zA-Z0-9\-_.]+$', pkg.strip())]
            if len(safe_packages) != len(packages):
                logger.warning("Some package names were filtered for safety")
            
            if not safe_packages:
                return True
            
            # Install packages
            package_str = " ".join(safe_packages)
            logger.info(f"Installing packages: {package_str}")
            
            result = await self.sandbox.commands.run(
                f"pip install --no-cache-dir {package_str}",
                timeout=60  # Give more time for package installation
            )
            
            success = result.exit_code == 0
            if not success:
                logger.error(f"Package installation failed: {result.stderr}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    async def execute_code(self, code: str, timeout: int = 30, install_packages_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute Python code in the E2B sandbox."""
        if not self.sandbox:
            await self.create_sandbox()
        
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
        """Release or close the E2B sandbox."""
        if self.sandbox:
            try:
                if self._acquired_from_pool:
                    pool = await self.get_pool(self.api_key)
                    await pool.release(self.sandbox)
                    logger.debug("Released sandbox back to pool")
                else:
                    await self.sandbox.close()
                    logger.debug("Closed standalone sandbox")
            except Exception as e:
                logger.warning(f"Error during sandbox cleanup: {e}")
            finally:
                self.sandbox = None
                self._acquired_from_pool = False


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Node.js."""
    
    def __init__(self):
        self.available = self._check_nodejs_available()
    
    def _check_nodejs_available(self) -> bool:
        """Check if Node.js is available."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code using Pyodide in Node.js."""
        if not self.available:
            raise SandboxError("Node.js is not available for Pyodide execution")
        
        try:
            # Encode the Python code as base64 to avoid shell escaping issues
            encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
            
            # Get the path to our pyodide runner script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pyodide_runner_path = os.path.join(script_dir, 'pyodide_runner.js')
            
            # If the runner script doesn't exist, fall back to inline execution
            if not os.path.exists(pyodide_runner_path):
                logger.warning("Pyodide runner script not found, using inline execution")
                return await self._execute_inline(code, timeout)
            
            # Execute the pyodide runner script
            process = await asyncio.create_subprocess_exec(
                'node', pyodide_runner_path, encoded_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timed out after {timeout} seconds",
                    "logs": [],
                }
            
            # Parse the JSON output
            try:
                result = json.loads(stdout.decode('utf-8'))
                # Ensure all required fields are present
                return {
                    "success": result.get("success", False),
                    "output": result.get("output", ""),
                    "error": result.get("error"),
                    "logs": [result.get("output", "")] if result.get("output") else [],
                }
            except json.JSONDecodeError:
                # If we can't parse JSON, treat the output as an error
                error_msg = stderr.decode('utf-8', errors='replace') or stdout.decode('utf-8', errors='replace')
                return {
                    "success": False,
                    "output": "",
                    "error": f"Failed to parse Pyodide output: {error_msg}",
                    "logs": [error_msg],
                }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Pyodide execution failed: {str(e)}",
                "logs": [],
            }
    
    async def _execute_inline(self, code: str, timeout: int) -> Dict[str, Any]:
        """Fallback inline execution when pyodide_runner.js is not available."""
        # Encode the code to avoid escaping issues
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        
        # Create an inline Node.js script that runs Pyodide
        node_script = f'''
const {{ loadPyodide }} = await import('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.mjs');

async function main() {{
    const startTime = Date.now();
    try {{
        const pyodide = await loadPyodide({{
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
        }});
        
        // Pre-load packages
        try {{
            await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
        }} catch (e) {{
            // Continue even if some packages fail
        }}
        
        // Decode the base64 encoded code
        const encodedCode = "{encoded_code}";
        const codeBuffer = Uint8Array.from(atob(encodedCode), c => c.charCodeAt(0));
        const code = new TextDecoder().decode(codeBuffer);
        
        // Set up Python environment
        pyodide.runPython(`
import sys
from io import StringIO
import traceback

_stdout_capture = StringIO()
_original_stdout = sys.stdout
sys.stdout = _stdout_capture

_result = {{'success': False, 'output': '', 'error': None}}

# Store the code in a variable to avoid escaping issues
_code_to_exec = globals().get('_code_to_exec', '')

try:
    exec(_code_to_exec)
    _result['output'] = _stdout_capture.getvalue()
    _result['success'] = True
except Exception as e:
    _result['error'] = traceback.format_exc()
    _result['output'] = _stdout_capture.getvalue()
finally:
    sys.stdout = _original_stdout
`);
        
        // Pass the code to Python
        pyodide.globals.set('_code_to_exec', code);
        
        // Re-run to execute with the code
        pyodide.runPython('exec(_code_to_exec, {{}})');
        
        const result = pyodide.globals.get('_result').toJs();
        console.log(JSON.stringify({{
            success: result.get('success'),
            output: result.get('output') || '',
            error: result.get('error'),
            execution_time_ms: Date.now() - startTime
        }}));
        
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            output: '',
            error: `Pyodide error: ${{error.message}}`,
            execution_time_ms: Date.now() - startTime
        }}));
        process.exit(1);
    }}
}}

main();
'''
        
        try:
            # Write the Node.js script to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mjs', delete=False) as f:
                f.write(node_script)
                temp_file = f.name
            
            # Execute the Node.js script with experimental modules
            process = await asyncio.create_subprocess_exec(
                'node', '--experimental-modules', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise SandboxError(f"Execution timed out after {timeout} seconds")
            finally:
                # Clean up
                os.unlink(temp_file)
            
            # Parse the JSON output
            try:
                result = json.loads(stdout.decode('utf-8').strip().split('\n')[-1])
                return {
                    "success": result.get("success", False),
                    "output": result.get("output", ""),
                    "error": result.get("error"),
                    "logs": [result.get("output", "")] if result.get("output") else [],
                }
            except (json.JSONDecodeError, IndexError):
                error_msg = stderr.decode('utf-8', errors='replace')
                return {
                    "success": False,
                    "output": "",
                    "error": f"Failed to execute: {error_msg}",
                    "logs": [stdout.decode('utf-8', errors='replace'), error_msg],
                }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Inline Pyodide execution failed: {str(e)}",
                "logs": [],
            }




# Global cache instance
_execution_cache = ExecutionCache()

# Rate limiting
class RateLimiter:
    """Simple rate limiter for sandbox executions."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
    
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


_rate_limiter = RateLimiter()


async def execute_sandbox_code(
    code: str,
    timeout: int = 30,
    sandbox_type: Optional[str] = None,
    install_packages: Optional[List[str]] = None,
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
    
    
    # Determine which sandbox to use
    sandboxes_to_try = []
    
    if sandbox_type:
        # User specified a sandbox type
        if sandbox_type == "e2b":
            if os.getenv("E2B_API_KEY"):
                sandboxes_to_try.append(("e2b", E2BSandbox(os.getenv("E2B_API_KEY"))))
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
        if os.getenv("E2B_API_KEY"):
            sandboxes_to_try.append(("e2b", E2BSandbox(os.getenv("E2B_API_KEY"))))
        
        # Then try Pyodide
        pyodide = PyodideSandbox()
        if pyodide.available:
            sandboxes_to_try.append(("pyodide", pyodide))
        
    
    # Check if we have any sandboxes available
    if not sandboxes_to_try:
        return {
            "success": False,
            "output": "",
            "error": "No sandboxes available. Please set E2B_API_KEY environment variable or install Node.js for Pyodide.",
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
                # Pyodide doesn't support package installation
                if install_packages and sandbox_name == "pyodide":
                    logger.warning("Pyodide sandbox doesn't support package installation")
                result = await sandbox.execute_code(code, timeout)
            
            # Add sandbox info and execution time
            result["sandbox_used"] = sandbox_name
            result["execution_time_ms"] = int((time.time() - start_time) * 1000)
            
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