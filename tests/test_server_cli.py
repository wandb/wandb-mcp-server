from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import simple_parsing

from wandb_mcp_server import server
from wandb_mcp_server.api_client import WandBApiManager


class _DummyMCPServer:
    def __init__(self):
        self.run_transport = None

    def run(self, transport):
        self.run_transport = transport


def test_cli_runs_stdio_transport(monkeypatch):
    dummy_server = _DummyMCPServer()

    monkeypatch.setattr(
        simple_parsing,
        "parse",
        lambda _: SimpleNamespace(
            transport="stdio",
            host="localhost",
            port=None,
            wandb_api_key=None,
        ),
    )
    monkeypatch.setattr(server, "configure_wandb_logging", lambda: None)
    monkeypatch.setattr(server, "validate_and_get_api_key", lambda args: None)
    monkeypatch.setattr(server, "initialize_weave_tracing", lambda: False)
    monkeypatch.setattr(server, "create_mcp_server", lambda *args: dummy_server)

    server.cli()

    assert dummy_server.run_transport == "stdio"


def test_standalone_http_requires_explicit_auth_disable(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_DISABLED", raising=False)

    with pytest.raises(ValueError, match="MCP_AUTH_DISABLED"):
        server.create_mcp_server("http", host="127.0.0.1")


def test_standalone_http_refuses_non_loopback_binding(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")

    with pytest.raises(ValueError, match="loopback"):
        server.create_mcp_server("http", host="0.0.0.0")


def test_standalone_http_allows_explicit_loopback_development(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")

    mcp = server.create_mcp_server("http", host="::1")

    assert mcp.settings.host == "::1"


def test_standalone_http_normalizes_localhost_to_literal_loopback(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")

    mcp = server.create_mcp_server("http", host="localhost")

    assert mcp.settings.host == "127.0.0.1"


def test_cli_rejects_unsafe_http_before_network_initialization(monkeypatch):
    validate_api_key = MagicMock()
    initialize_weave = MagicMock()
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")
    monkeypatch.setattr(
        simple_parsing,
        "parse",
        lambda _: SimpleNamespace(
            transport="http",
            host="0.0.0.0",
            port=8080,
            wandb_api_key="test-key",
        ),
    )
    monkeypatch.setattr(server, "configure_wandb_logging", lambda: None)
    monkeypatch.setattr(server, "validate_api_key", validate_api_key)
    monkeypatch.setattr(server, "initialize_weave_tracing", initialize_weave)

    with pytest.raises(ValueError, match="loopback"):
        server.cli()

    validate_api_key.assert_not_called()
    initialize_weave.assert_not_called()


def test_standalone_http_requires_server_api_key(monkeypatch):
    args = SimpleNamespace(
        transport="http",
        wandb_api_key=None,
    )
    monkeypatch.setattr(
        server,
        "get_server_args",
        lambda: SimpleNamespace(wandb_api_key=None),
    )

    with pytest.raises(ValueError, match="standalone http"):
        server.validate_and_get_api_key(args)


def test_cli_sets_single_actor_context_for_loopback_http(monkeypatch):
    dummy_server = _DummyMCPServer()
    set_context = MagicMock()
    reset_context = MagicMock()
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")
    monkeypatch.setattr(
        simple_parsing,
        "parse",
        lambda _: SimpleNamespace(
            transport="http",
            host="127.0.0.1",
            port=8080,
            wandb_api_key="test-key",
        ),
    )
    monkeypatch.setattr(server, "configure_wandb_logging", lambda: None)
    monkeypatch.setattr(
        server,
        "validate_and_get_api_key",
        lambda args: "test-key",
    )
    monkeypatch.setattr(server, "validate_api_key", lambda _: True)
    monkeypatch.setattr(server, "initialize_weave_tracing", lambda: False)
    monkeypatch.setattr(server, "create_mcp_server", lambda *args: dummy_server)
    monkeypatch.setattr(WandBApiManager, "set_context_api_key", set_context)
    monkeypatch.setattr(WandBApiManager, "reset_context_api_key", reset_context)

    server.cli()

    assert [entry.args for entry in set_context.call_args_list] == [
        ("test-key",),
        ("test-key",),
    ]
    reset_context.assert_called_once_with(set_context.return_value)
    assert dummy_server.run_transport == "streamable-http"


def test_cli_validation_context_redacts_candidate_key_and_resets_on_failure(monkeypatch):
    candidate_key = "candidate-secret-key-123456"
    monkeypatch.setattr(
        simple_parsing,
        "parse",
        lambda _: SimpleNamespace(
            transport="stdio",
            host="localhost",
            port=None,
            wandb_api_key=candidate_key,
        ),
    )
    monkeypatch.setattr(server, "configure_wandb_logging", lambda: None)
    monkeypatch.setattr(
        server,
        "validate_and_get_api_key",
        lambda args: candidate_key,
    )

    def reject_key(api_key):
        from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text

        assert WandBApiManager.get_api_key() == api_key
        assert api_key not in sanitize_sensitive_text(f"upstream echoed {api_key}")
        return False

    monkeypatch.setattr(server, "validate_api_key", reject_key)

    with pytest.raises(ValueError, match="validation failed"):
        server.cli()

    assert WandBApiManager.get_api_key() is None
