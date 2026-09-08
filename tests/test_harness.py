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
    assert context.mcp_client_app == "codex"
    assert context.agent_harness == "codex"
    assert context.client_vendor == "openai"
    assert context.mcp_client_source == "initialize_client_info"
    assert context.mcp_protocol_version == "2025-03-26"
    assert context.mcp_jsonrpc_method == "initialize"
    assert context.call_type == "initialize"
    assert "mcp_client_name" not in context.analytics_fields(include_debug=False)


def test_official_client_info_wins_on_initialize() -> None:
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

    assert context.mcp_client_family == "cursor"
    assert context.mcp_client_app == "cursor"
    assert context.mcp_client_source == "initialize_client_info"
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
            "mcp_client_app": "codex",
            "mcp_client_source": "initialize_client_info",
            "mcp_protocol_version": "2025-03-26",
            "mcp_jsonrpc_method": "initialize",
        },
    )

    assert context.mcp_client_family == "openai"
    assert context.mcp_client_app == "codex"
    assert context.mcp_client_source == "session_metadata"
    assert context.mcp_client_mismatch == "client_info"
    assert context.mcp_jsonrpc_method == "tools.call"
    assert context.call_type == "tools/call"


def test_session_metadata_wins_over_conflicting_user_agent() -> None:
    context = extract_harness_context(
        {"User-Agent": "claude-code/2.1.89"},
        {"method": "tools/call"},
        session_metadata={
            "agent_harness": "codex",
            "mcp_client_family": "openai",
        },
    )

    assert context.agent_harness == "codex"
    assert context.client_vendor == "openai"
    assert context.mcp_client_mismatch == "user_agent"


def test_initialize_client_info_wins_over_conflicting_lower_priority_signals() -> None:
    context = extract_harness_context(
        {"User-Agent": "Cursor/1.0"},
        {
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "codex-mcp-client", "version": "1"},
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "claude-code",
                        "version": "2",
                    }
                },
            },
        },
    )

    assert context.agent_harness == "codex"
    assert context.client_vendor == "openai"
    assert context.mcp_client_mismatch == "client_info"


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


def test_supported_client_products_have_canonical_harnesses() -> None:
    fixtures = {
        "claude-code": ("claude_code", "anthropic"),
        "claude-desktop": ("claude_desktop", "anthropic"),
        "claude-ai": ("claude_ai", "anthropic"),
        "cursor-vscode": ("cursor", "cursor"),
        "gemini-cli": ("gemini_cli", "google"),
        "lechat": ("lechat", "mistral"),
        "linear-agent": ("linear", "linear"),
        "visual-studio-code": ("vscode", "microsoft"),
        "mcp-inspector": ("mcp_inspector", "modelcontextprotocol"),
        "wandb-mcp-load-test": ("load_test", "internal"),
    }
    for client_name, expected in fixtures.items():
        context = extract_harness_context(
            {},
            {
                "method": "initialize",
                "params": {"clientInfo": {"name": client_name, "version": "1"}},
            },
        )
        assert (context.agent_harness, context.client_vendor) == expected


def test_meta_client_info_is_used_without_session_metadata() -> None:
    context = extract_harness_context(
        {},
        {
            "method": "tools/call",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "cursor-vscode",
                        "version": "1",
                    }
                }
            },
        },
    )

    assert context.agent_harness == "cursor"
    assert context.call_type == "tools/call"
    assert context.mcp_client_source == "meta_client_info"
