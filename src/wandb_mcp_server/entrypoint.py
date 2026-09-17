"""Side-effect-free console bootstrap that loads dotenv before runtime imports."""

from dotenv import find_dotenv, load_dotenv


def cli() -> None:
    """Load local configuration, then import and construct the MCP server."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
    from wandb_mcp_server.server import cli as run_server

    run_server()
