"""Unit tests for MCP harness classification."""

from wandb_mcp_server.harness import extract_harness_context


def test_initialize_client_info_identifies_codex() -> None:
    context = extract_harness_context(
        {"User-Agent": "python-httpx/0.28"},
        {
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "clientInfo": {"name": "codex-mcp-client", "version": "1.2.3"},
            },
        },
    )

    assert context.mcp_client_family == "openai"
    assert context.mcp_client_app == "codex_cli"
    assert context.mcp_client_source == "initialize_client_info"
    assert context.mcp_protocol_version == "2025-03-26"
    assert context.mcp_jsonrpc_method == "initialize"
    assert "mcp_client_name" not in context.analytics_fields(include_debug=False)


def test_meta_client_info_wins_on_initialize() -> None:
    context = extract_harness_context(
        {},
        {
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "cursor", "version": "1"},
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "claude-code",
                        "version": "2.1.89",
                    },
                    "io.modelcontextprotocol/protocolVersion": "2025-06-18",
                },
            },
        },
    )

    assert context.mcp_client_family == "claude"
    assert context.mcp_client_app == "claude_code"
    assert context.mcp_client_source == "meta_client_info"
    assert context.mcp_protocol_version == "2025-06-18"


def test_session_metadata_wins_over_later_spoofed_meta() -> None:
    context = extract_harness_context(
        {"User-Agent": "Cursor/1.0", "MCP-Protocol-Version": "2025-06-18"},
        {
            "method": "tools/call",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "cursor",
                        "version": "1",
                    }
                }
            },
        },
        session_metadata={
            "mcp_client_family": "openai",
            "mcp_client_app": "codex_cli",
            "mcp_client_source": "initialize_client_info",
            "mcp_protocol_version": "2025-03-26",
            "mcp_jsonrpc_method": "initialize",
        },
    )

    assert context.mcp_client_family == "openai"
    assert context.mcp_client_app == "codex_cli"
    assert context.mcp_client_source == "session_metadata"
    assert context.mcp_client_mismatch == "client_info"
    assert context.mcp_jsonrpc_method == "tools.call"


def test_user_agent_is_fallback_only() -> None:
    context = extract_harness_context(
        {"User-Agent": "claude-code/2.1.89 (cli)", "MCP-Protocol-Version": "2025-06-18"},
        {"method": "tools/list"},
    )

    assert context.mcp_client_family == "claude"
    assert context.mcp_client_app == "claude_code"
    assert context.mcp_client_source == "user_agent"
    assert context.mcp_protocol_version == "2025-06-18"
    assert context.mcp_jsonrpc_method == "tools.list"


def test_oversized_and_control_values_become_safe_debug_values() -> None:
    context = extract_harness_context(
        {"User-Agent": "bad\r\nHeader: injected"},
        {
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "x" * 500 + "\nsecret@example.com",
                    "version": "1.0\r\nbad",
                }
            },
        },
    )

    assert context.mcp_client_family == "unknown"
    assert context.mcp_client_app == "unknown"
    assert context.mcp_client_name == ("x" * 64)
    assert context.mcp_client_version == "1.0.bad"
