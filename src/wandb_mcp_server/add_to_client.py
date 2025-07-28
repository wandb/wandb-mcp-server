import json
import logging
import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import simple_parsing

from wandb_mcp_server.utils import get_rich_logger

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = get_rich_logger(__name__)


@dataclass
class AddToClientArgs:
    """Add Weights & Biases MCP server to your client config."""

    config_path: str
    """Path to the MCP client config file"""

    wandb_api_key: Optional[str] = None
    """W&B API key for authentication"""

    write_env_vars: List[str] = field(default_factory=list)
    """Write additional environment variables to client config file (format: KEY=VALUE)"""

    def get_env_vars(self) -> Dict[str, str]:
        """Get all environment variables to include in the config."""
        env_vars = {}

        # Parse additional env vars from list
        for env_str in self.write_env_vars:
            if "=" in env_str:
                key, value = env_str.split("=", 1)
                env_vars[key] = value

        # Add specific environment variables if provided
        if self.wandb_api_key:
            env_vars["WANDB_API_KEY"] = self.wandb_api_key

        return env_vars


def get_new_config(env_vars: Optional[Dict[str, str]] = None) -> dict:
    """
    Get the new configuration to add to the client config.

    Args:
        env_vars: Optional environment variables to include in the config

    Returns:
        Dictionary with the MCP server configuration
    """
    config = {
        "mcpServers": {
            "wandb": {
                "command": "uvx",
                "args": [
                    "--from",
                    "git+https://github.com/wandb/wandb-mcp-server",
                    "wandb_mcp_server",
                ],
            }
        }
    }

    # Add environment variables if provided
    if env_vars:
        config["mcpServers"]["wandb"]["env"] = env_vars

    return config


def add_to_client(args: AddToClientArgs) -> None:
    """
    Add MCP server configuration to a client config file.

    Args:
        args: Command line arguments

    Raises:
        Exception: If there are errors reading/writing the config file
    """
    # Handle potential path parsing issues
    config_path = args.config_path
    
    # Debug: Log the raw config_path to help diagnose issues
    logger.debug(f"Raw config_path argument: '{config_path}'")
    
    # Check if config_path looks malformed (starts with --)
    if config_path.startswith("--"):
        logger.error(f"Invalid config path detected: '{config_path}'")
        logger.error("This usually happens when command line arguments are not properly parsed.")
        logger.error("Try running the command on a single line or check for syntax errors.")
        sys.exit(1)
    
    # Expand user path and resolve to absolute path
    config_path = os.path.expanduser(config_path)
    config_path = os.path.abspath(config_path)
    
    logger.info(f"Using config path: {config_path}")

    # Read existing config file or initialize a default structure
    config = {"mcpServers": {}}  # Start with a default, ensures mcpServers key exists
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                # Attempt to load. If file is empty or has invalid JSON,
                # json.load will raise JSONDecodeError.
                loaded_config = json.load(f)
                # If load is successful, check if it's a dictionary (top-level JSON should be an object)
                if isinstance(loaded_config, dict):
                    config = loaded_config  # Use the loaded config
                    logger.info(f"Loaded existing config from {config_path}")
                else:
                    # Loaded JSON is not a dictionary (e.g. `null`, `[]`, `true`)
                    # This is unexpected for a config file that should hold mcpServers.
                    logger.warning(
                        f"Config file {config_path} did not contain a JSON object. Using default config."
                    )
                    # config remains the default {"mcpServers": {}}
        else:
            logger.info(
                f"Config file {config_path} doesn't exist. Will create new file."
            )
            # config remains the default {"mcpServers": {}}
    except json.JSONDecodeError as e:
        # This handles empty file or malformed JSON.
        logger.warning(
            f"Config file {config_path} is empty or contains invalid JSON: {e}. Using default config."
        )
        # config remains the default {"mcpServers": {}}.
    except IOError as e:
        logger.error(
            f"Fatal error reading config file {config_path}: {e}. Cannot proceed."
        )
        sys.exit(f"Fatal error reading config file: {e}")  # Exit if we can't read

    if not isinstance(config.get("mcpServers"), dict):
        if os.path.exists(config_path):
            logger.warning(
                f"Warning: 'mcpServers' key in the loaded config from {config_path} was missing or not a dictionary. Initializing it."
            )
        config["mcpServers"] = {}  # Ensure it's a dictionary

    # Get the new configuration with environment variables
    env_vars = args.get_env_vars()
    new_config = get_new_config(env_vars)

    # Check for key overlaps
    existing_keys = set(config["mcpServers"].keys())
    new_keys = set(new_config["mcpServers"].keys())
    overlapping_keys = existing_keys.intersection(new_keys)

    if overlapping_keys:
        logger.info(
            "The following tools already exist in your config and will be overwritten:"
        )
        for key in overlapping_keys:
            logger.info(f"- {key}")

        # Ask for confirmation
        answer = input("Do you want to overwrite them? (y/N): ").lower()
        if answer != "y":
            logger.info("Operation cancelled.")
            sys.exit(0)

    # Update config with new servers
    config["mcpServers"].update(new_config["mcpServers"])

    # Create directory if it doesn't exist
    config_dir = os.path.dirname(config_path)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Save the updated config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Successfully updated config at {config_path}")
    

def add_to_client_cli():
    args = simple_parsing.parse(AddToClientArgs)
    add_to_client(args)


if __name__ == "__main__":
    add_to_client_cli()
