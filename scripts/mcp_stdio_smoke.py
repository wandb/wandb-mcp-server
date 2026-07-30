#!/usr/bin/env python3
"""Exercise an installed W&B MCP wheel through the real STDIO protocol.

The harness intentionally sets ``MCP_ANALYTICS_LOG_STREAM=stdout`` for one
session. Any non-JSON-RPC stdout line is surfaced by the official MCP client's
parser and fails the run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


_TEST_API_KEY = "k" * 40
_VIEWER = {
    "id": "stdio-smoke-user-id",
    "name": "STDIO Smoke User",
    "username": "stdio-smoke-user",
    "email": "stdio-smoke@example.com",
    "admin": False,
    "flags": "",
    "entity": "stdio-smoke-user",
    "deletedAt": None,
    "apiKeys": {"edges": []},
    "teams": {"edges": [{"node": {"name": "stdio-smoke-team"}}]},
}
_CLIENTS = (
    ("codex-mcp-client", "codex", "openai"),
    ("claude-code", "claude_code", "anthropic"),
    ("cursor-vscode", "cursor", "cursor"),
)


class _FakeWandBHandler(BaseHTTPRequestHandler):
    """Return the public SDK viewer shape needed by read-only smoke calls."""

    server_version = "WandBStdioSmoke/1.0"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        content_length = int(self.headers.get("content-length", "0"))
        self.rfile.read(content_length)
        if self.path != "/graphql":
            self.send_error(404)
            return
        payload = json.dumps({"data": {"viewer": _VIEWER}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *args: object) -> None:
        del args


class _ProtocolParseCapture(logging.Handler):
    """Capture malformed server stdout reported by the official MCP client."""

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Failed to parse JSONRPC message from server" in message:
            self.failures.append(message)


def _result_text(result: Any) -> str:
    return "".join(getattr(item, "text", "") for item in result.content)


def _analytics_events(stderr: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event_type"):
            events.append(payload)
    return events


def _contains_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def _server_environment(base_url: str, home: str, *, unsafe_stdout_override: bool) -> dict[str, str]:
    return {
        "HOME": home,
        "WANDB_API_KEY": _TEST_API_KEY,
        "WANDB_BASE_URL": base_url,
        "WANDB_INTERNAL_BASE_URL": "",
        "WANDB_SILENT": "True",
        "WEAVE_SILENT": "True",
        "MCP_ANALYTICS_DISABLED": "false",
        "MCP_ANALYTICS_LOG_STREAM": "stdout" if unsafe_stdout_override else "stderr",
        "MCP_SEGMENT_DRY_RUN": "true",
        "MCP_SEGMENT_FORWARD": "false",
        "MCP_DATADOG_FORWARD": "false",
        "WANDB_MCP_ENABLE_RAW_GRAPHQL": "false",
        "WANDB_MCP_READ_ONLY": "false",
    }


async def _exercise_profile(
    server_command: str,
    base_url: str,
    *,
    client_name: str,
    expected_harness: str,
    expected_vendor: str,
    unsafe_stdout_override: bool = False,
) -> None:
    parse_capture = _ProtocolParseCapture()
    protocol_logger = logging.getLogger("mcp.client.stdio")
    previous_level = protocol_logger.level
    protocol_logger.addHandler(parse_capture)
    protocol_logger.setLevel(logging.ERROR)

    with tempfile.TemporaryDirectory(prefix="wandb-mcp-stdio-") as temp_dir:
        params = StdioServerParameters(
            command=server_command,
            env=_server_environment(
                base_url,
                temp_dir,
                unsafe_stdout_override=unsafe_stdout_override,
            ),
            cwd=temp_dir,
        )
        with tempfile.TemporaryFile(mode="w+") as server_stderr:
            try:
                with anyio.fail_after(60):
                    async with stdio_client(params, errlog=server_stderr) as (read_stream, write_stream):
                        async with ClientSession(
                            read_stream,
                            write_stream,
                            client_info=types.Implementation(name=client_name, version="stdio-smoke"),
                        ) as session:
                            await session.initialize()
                            await session.send_ping()

                            listed = await session.list_tools()
                            tools_by_name = {tool.name: tool for tool in listed.tools}
                            assert "list_entities_tool" in tools_by_name
                            assert "query_wandb_tool" in tools_by_name
                            assert "query_wandb_graphql_tool" not in tools_by_name

                            query_schema = tools_by_name["query_wandb_tool"].inputSchema
                            query_properties = query_schema["properties"]
                            assert "resource" in query_properties
                            assert "query" not in query_properties

                            entity_result = await session.call_tool("list_entities_tool", {})
                            assert entity_result.isError is False
                            entity_payload = json.loads(_result_text(entity_result))
                            assert entity_payload == {
                                "entities": [
                                    {"name": "stdio-smoke-user", "type": "user"},
                                    {"name": "stdio-smoke-team", "type": "team"},
                                ],
                                "count": 2,
                            }

                            invalid_result = await session.call_tool(
                                "query_wandb_tool",
                                {
                                    "entity_name": "test-entity",
                                    "project_name": "test-project",
                                    "resource": "run",
                                },
                            )
                            invalid_payload = json.loads(_result_text(invalid_result))
                            assert invalid_payload["error"] == "invalid_request"

                            unknown_result = await session.call_tool("not_a_real_tool", {})
                            assert unknown_result.isError is True
                            assert "Unknown tool" in _result_text(unknown_result)

                            # Prove structured tool failures do not poison the session.
                            await session.send_ping()
            finally:
                protocol_logger.removeHandler(parse_capture)
                protocol_logger.setLevel(previous_level)

            server_stderr.seek(0)
            stderr = server_stderr.read()

    assert parse_capture.failures == [], parse_capture.failures
    assert stderr.count("Analytics ready:") == 1
    assert _TEST_API_KEY not in stderr

    matching_events = [
        event
        for event in _analytics_events(stderr)
        if event.get("event_type") == "tool_call" and event.get("tool_name") == "list_entities_tool"
    ]
    assert len(matching_events) == 1
    event = matching_events[0]
    assert event["agent_harness"] == expected_harness
    assert event["client_vendor"] == expected_vendor
    assert event["call_type"] == "tools/call"
    assert event["transport"] == "stdio"
    assert event["success"] is True
    assert not _contains_none(event)


async def _exercise_missing_credentials(server_command: str, base_url: str) -> None:
    parse_capture = _ProtocolParseCapture()
    protocol_logger = logging.getLogger("mcp.client.stdio")
    previous_level = protocol_logger.level
    protocol_logger.addHandler(parse_capture)
    protocol_logger.setLevel(logging.ERROR)

    with tempfile.TemporaryDirectory(prefix="wandb-mcp-stdio-no-key-") as temp_dir:
        environment = _server_environment(base_url, temp_dir, unsafe_stdout_override=True)
        environment["WANDB_API_KEY"] = ""
        params = StdioServerParameters(
            command=server_command,
            env=environment,
            cwd=temp_dir,
        )
        failed_as_expected = False
        with tempfile.TemporaryFile(mode="w+") as server_stderr:
            try:
                with anyio.fail_after(15):
                    async with stdio_client(params, errlog=server_stderr) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
            except Exception:
                failed_as_expected = True
            finally:
                protocol_logger.removeHandler(parse_capture)
                protocol_logger.setLevel(previous_level)

            server_stderr.seek(0)
            stderr = server_stderr.read()

    assert failed_as_expected, "STDIO unexpectedly started without WANDB_API_KEY"
    assert parse_capture.failures == [], parse_capture.failures
    assert "WANDB_API_KEY must be set for STDIO transport" in stderr
    assert _TEST_API_KEY not in stderr


async def _run(server_command: str) -> None:
    fake_wandb = ThreadingHTTPServer(("127.0.0.1", 0), _FakeWandBHandler)
    thread = threading.Thread(target=fake_wandb.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{fake_wandb.server_port}"
    try:
        for client_name, expected_harness, expected_vendor in _CLIENTS:
            await _exercise_profile(
                server_command,
                base_url,
                client_name=client_name,
                expected_harness=expected_harness,
                expected_vendor=expected_vendor,
            )
        await _exercise_profile(
            server_command,
            base_url,
            client_name="codex-mcp-client",
            expected_harness="codex",
            expected_vendor="openai",
            unsafe_stdout_override=True,
        )
        await _exercise_missing_credentials(server_command, base_url)
    finally:
        fake_wandb.shutdown()
        fake_wandb.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-command",
        required=True,
        help="Path to the installed wandb_mcp_server console entrypoint.",
    )
    args = parser.parse_args()
    server_command = str(Path(args.server_command).resolve())
    if not Path(server_command).is_file():
        parser.error(f"Server command does not exist: {server_command}")
    anyio.run(_run, server_command)
    print("STDIO protocol smoke passed")


if __name__ == "__main__":
    main()
