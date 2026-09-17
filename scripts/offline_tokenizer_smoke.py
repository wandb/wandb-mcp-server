"""Prove a fresh wheel can register tools without tokenizer downloads or caches."""

import asyncio
import os
from pathlib import Path
import socket
import sys
import tempfile


def main() -> None:
    def deny_network(*args, **kwargs):
        raise AssertionError("offline tokenizer smoke attempted a network connection")

    with tempfile.TemporaryDirectory(prefix="mcp-offline-") as temporary:
        cache = Path(temporary) / "cache"
        cache.mkdir(mode=0o500)
        os.environ.update({"HOME": temporary, "TIKTOKEN_CACHE_DIR": str(cache), "MCP_ANALYTICS_ENABLED": "false"})
        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network
        socket.create_connection = deny_network

        import wandb_mcp_server
        from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
        from wandb_mcp_server.runtime_contract import resolve_runtime_selection
        from wandb_mcp_server.server import register_tools
        from wandb_mcp_server.tokenizer import load_tokenizer

        assert Path(wandb_mcp_server.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
            "expected installed wheel"
        )
        assert load_tokenizer().encode("hello world") == [15339, 1917]
        server = InstrumentedFastMCP("offline-wheel")
        register_tools(server, resolve_runtime_selection())
        assert asyncio.run(server.list_tools())
        assert not list(cache.iterdir()), "tokenizer wrote cache files"
        cache.chmod(0o700)
    print("Offline wheel tokenizer and tool registration passed.")


if __name__ == "__main__":
    main()
