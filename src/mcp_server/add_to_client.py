import os
import json
import sys

new_config = {
    "mcpServers": {
        "wandb": {
            "command": "uvx",
            "args": [
                "--from", 
                "git+https://github.com/wandb/mcp-server", 
                "mcp_server"
            ],
        }
    }
}

def add_to_client(pathname: str | None = None, api_key: str | None = None) -> None:
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
    
    # Read existing config file or create empty object if not exists
    config = {"mcpServers": {}}
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                print(f"Loaded existing config from {config_path}")
        else:
            print(f"Config file doesn't exist. Will create new file at {config_path}")
    except Exception as error:
        print(f"Error reading config file: {str(error)}")
        raise
    
    # Ensure mcpServers exists in config
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Check for key overlaps
    existing_keys = set(config["mcpServers"].keys())
    new_keys = set(new_config["mcpServers"].keys())
    overlapping_keys = existing_keys.intersection(new_keys)
    
    if overlapping_keys:
        print("The following tools already exist in your config and will be overwritten:")
        for key in overlapping_keys:
            print(f"- {key}")
        
        # Ask for confirmation
        answer = input("Do you want to overwrite them? (y/N): ").lower()
        if answer != "y":
            print("Operation cancelled.")
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
    print(f"Successfully updated config at {config_path}")

def add_to_client_cli():
    if len(sys.argv) > 1:
        add_to_client(sys.argv[1])
    else:
        print("Please provide the path to your MCP client config as a command line argument")
        sys.exit(1)

if __name__ == "__main__":
    add_to_client_cli()
