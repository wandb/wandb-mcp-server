import json
import logging
import os
import sys

from wandb_mcp_server.utils import get_rich_logger

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = get_rich_logger(__name__)

new_config = {
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


def add_to_client(pathname: str | None = None) -> None:
    """
    Add MCP server configuration to a client config file.

    Args:
        pathname: Path to the MCP client config file

    Raises:
        ValueError: If pathname is not provided
        Exception: If there are errors reading/writing the config file
    """
    if not pathname:
        raise ValueError("Please provide the path to your MCP client config")

    config_path = os.path.abspath(pathname)

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

    # Ensure the 'mcpServers' key exists and is a dictionary.
    # This is a safeguard if the loaded_config was a dict but missed mcpServers or had it as a wrong type.
    if not isinstance(config.get("mcpServers"), dict):
        if os.path.exists(
            config_path
        ):  # Only print warning if file existed and was loaded
            logger.warning(
                f"Warning: 'mcpServers' key in the loaded config from {config_path} was missing or not a dictionary. Initializing it."
            )
        config["mcpServers"] = {}  # Ensure it's a dictionary

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
    if len(sys.argv) > 1:
        add_to_client(sys.argv[1])
    else:
        logger.error(
            "Please provide the path to your MCP client config as a command line argument"
        )
        sys.exit(1)


if __name__ == "__main__":
    add_to_client_cli()
