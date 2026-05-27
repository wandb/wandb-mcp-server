from types import SimpleNamespace

import simple_parsing

from wandb_mcp_server import analytics
from wandb_mcp_server import server


class _DummyMCPServer:
    def __init__(self):
        self.run_transport = None

    def run(self, transport):
        self.run_transport = transport


def test_cli_configures_analytics_for_stdio_transport(monkeypatch):
    calls = []
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
    monkeypatch.setattr(
        analytics,
        "configure_analytics_logging_for_transport",
        lambda transport: calls.append(transport) or "stderr",
    )
    monkeypatch.setattr(server, "configure_wandb_logging", lambda: None)
    monkeypatch.setattr(server, "validate_and_get_api_key", lambda args: None)
    monkeypatch.setattr(server, "initialize_weave_tracing", lambda: False)
    monkeypatch.setattr(server, "create_mcp_server", lambda *args: dummy_server)

    server.cli()

    assert calls == ["stdio"]
    assert dummy_server.run_transport == "stdio"
