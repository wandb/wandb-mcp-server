#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for the Weights & Biases MCP Server.

This script starts the MCP server in HTTP mode, making it accessible
via Server-Sent Events (SSE) for remote MCP clients.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wandb_mcp_server.server import cli
from wandb_mcp_server.utils import get_rich_logger

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_rich_logger("huggingface-spaces-app")

def main():
    """Main entry point for HuggingFace Spaces."""
    logger.info("Starting Weights & Biases MCP Server on HuggingFace Spaces")
    
    # Ensure we're running in HTTP mode for HuggingFace Spaces
    # Override command line arguments to force HTTP transport
    original_argv = sys.argv.copy()
    sys.argv = [
        sys.argv[0],  # Keep the script name
        "--transport", "http",
        "--host", "0.0.0.0",  # Listen on all interfaces for HuggingFace Spaces
        "--port", str(os.environ.get("PORT", "7860"))  # Use PORT env var or default to 8080
    ]
    
    # Check for required environment variables
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        logger.error("WANDB_API_KEY environment variable is required!")
        logger.error("Please set your Weights & Biases API key in the Space's environment variables.")
        logger.error("You can get your API key from: https://wandb.ai/authorize")
        sys.exit(1)
    
    logger.info(f"WANDB_API_KEY configured: {'Yes' if wandb_api_key else 'No'}")
    logger.info(f"Starting HTTP server on port {os.environ.get('PORT', '8080')}")
    logger.info("MCP endpoint will be available at: /mcp")
    
    try:
        # Call the CLI function which will start the server
        cli()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()
