"""
Utility functions for writing files to sandbox environments.
Provides a clean interface for writing data to both E2B and Pyodide sandboxes.
"""

import json
from typing import Any, Dict, Union

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


async def write_json_to_sandbox(
    json_data: Union[str, Dict[str, Any]],
    filename: str,
    path_prefix: str = "/tmp/",
) -> None:
    """
    Write JSON data to a file in any available sandbox using native file operations.
    
    Args:
        json_data: JSON data as string or dict
        filename: Name of the file to create
        path_prefix: Directory prefix for the file (default: /tmp/)
    """
    try:
        from wandb_mcp_server.mcp_tools.code_sandbox import (
            check_sandbox_availability,
            E2BSandbox,
            PyodideSandbox,
        )
        
        available, sandbox_types, _ = check_sandbox_availability()
        if not available:
            logger.debug("No sandbox available for file writing")
            return
        
        if isinstance(json_data, dict):
            content = json.dumps(json_data, indent=2)
        else:
            content = str(json_data)
        
        if not path_prefix.endswith('/'):
            path_prefix += '/'
        
        full_path = f"{path_prefix}{filename}"
        
        # Try to write using available sandbox
        if "e2b" in sandbox_types:
            # Use E2B
            import os
            api_key = os.getenv("E2B_API_KEY")
            if api_key:
                sandbox = E2BSandbox(api_key)
                await sandbox.create_sandbox()
                await sandbox.writeFile(full_path, content)
                await sandbox.close_sandbox()  # Just releases the reference, doesn't close the sandbox
                logger.info(f"Wrote {filename} to E2B sandbox")
                return
        
        if "pyodide" in sandbox_types:
            sandbox = PyodideSandbox()
            await sandbox.writeFile(full_path, content)
            logger.info(f"Wrote {filename} to Pyodide sandbox")
            return
            
    except Exception as e:
        logger.error(f"Error writing {filename} to sandbox: {e}", exc_info=True)