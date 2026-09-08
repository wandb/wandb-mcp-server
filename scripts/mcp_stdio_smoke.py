#!/usr/bin/env python3
"""Exercise an installed W&B MCP wheel through the real STDIO protocol.

The harness intentionally sets ``MCP_ANALYTICS_LOG_STREAM=stdout`` for one
session. Any non-JSON-RPC stdout line is surfaced by the official MCP client's
parser and fails the run.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version as installed_version
import json
import logging
from pathlib import Path
import platform
import tempfile
import threading
import time
from zipfile import BadZipFile, ZipFile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

if __package__:
    from scripts.public_release import (
        DEFAULT_CONTRACT,
        Profile,
        ReleaseError,
        canonical_json,
        exhaustive_profiles,
        load_contract,
        named_profile,
        runtime_contract_payload_sha256,
        runtime_contract_sha256,
        sha256_file,
    )
else:
    from public_release import (
        DEFAULT_CONTRACT,
        Profile,
        ReleaseError,
        canonical_json,
        exhaustive_profiles,
        load_contract,
        named_profile,
        runtime_contract_payload_sha256,
        runtime_contract_sha256,
        sha256_file,
    )


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
_AGENT_TOOL_NAMES = {
    "list_weave_agents_tool",
    "list_weave_agent_versions_tool",
    "query_weave_agent_spans_tool",
    "get_weave_agent_span_stats_tool",
    "list_weave_agent_custom_attributes_tool",
    "search_weave_agents_tool",
    "get_weave_agent_trace_tool",
    "get_weave_agent_conversation_tool",
}
_ARIA_TOOL_NAMES = {"aria_send_message", "aria_get_turn", "aria_get_turns"}
_WHEEL_RUNTIME_CONTRACT = "wandb_mcp_server/runtime-contract.json"


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


def _wheel_runtime_contract_sha256(wheel: Path) -> str:
    """Bind evidence to the packaged policy inside the exact installed wheel."""
    try:
        with ZipFile(wheel) as archive:
            matches = [name for name in archive.namelist() if name == _WHEEL_RUNTIME_CONTRACT]
            if len(matches) != 1:
                raise ValueError("wheel must contain exactly one packaged MCP runtime contract")
            try:
                payload = json.loads(archive.read(matches[0]))
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("wheel contains an unreadable MCP runtime contract") from error
    except BadZipFile as error:
        raise ValueError("wheel is not a valid wheel archive") from error
    return runtime_contract_payload_sha256(payload)


def _server_environment(
    base_url: str,
    home: str,
    *,
    unsafe_stdout_override: bool,
    profile_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    # The release harness is hermetic: all functional traffic targets the
    # loopback fake below, and every other HTTP(S) destination is forced to a
    # closed loopback port. This prevents SDK diagnostics or optional telemetry
    # from escaping a qualification run.
    environment = {
        "HOME": home,
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "all_proxy": "http://127.0.0.1:9",
        "no_proxy": "127.0.0.1,localhost",
        "WANDB_API_KEY": _TEST_API_KEY,
        "WANDB_BASE_URL": base_url,
        "WANDB_INTERNAL_BASE_URL": "",
        "WF_TRACE_SERVER_URL": base_url,
        "WANDB_SILENT": "True",
        "WEAVE_SILENT": "True",
        "MCP_ANALYTICS_DISABLED": "false",
        "MCP_ANALYTICS_LOG_STREAM": "stdout" if unsafe_stdout_override else "stderr",
        "MCP_SEGMENT_DRY_RUN": "true",
        "MCP_SEGMENT_FORWARD": "false",
        "MCP_DATADOG_FORWARD": "false",
        "WANDB_MCP_PROXY_DOCS": "true",
        "WANDB_MCP_TOOL_PROFILE": "models-weave",
        "WANDB_MCP_ACCESS_MODE": "read-write",
        "MCP_WORKLOAD_PROFILE": "local",
        "MCP_CAPACITY_CLASS": "small",
        "WB_AGENT_BASE_URL": "https://127.0.0.1:9",
    }
    if profile_environment:
        environment.update(profile_environment)
    return environment


async def _exercise_profile(
    server_command: str,
    base_url: str,
    *,
    client_name: str,
    expected_harness: str,
    expected_vendor: str,
    unsafe_stdout_override: bool = False,
    profile: Profile | None = None,
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
                profile_environment=profile.environment if profile else None,
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
                            if profile is not None:
                                assert set(tools_by_name) == set(profile.tools), (
                                    profile.name,
                                    sorted(set(profile.tools) - set(tools_by_name)),
                                    sorted(set(tools_by_name) - set(profile.tools)),
                                )
                                agent_tools_enabled = bool(_AGENT_TOOL_NAMES & set(profile.tools))
                                aria_tools_enabled = bool(_ARIA_TOOL_NAMES & set(profile.tools))
                            else:
                                agent_tools_enabled = False
                                aria_tools_enabled = False
                                assert len(tools_by_name) == 22
                            assert _AGENT_TOOL_NAMES.issubset(tools_by_name) is agent_tools_enabled
                            if aria_tools_enabled:
                                expected_aria_tools = _ARIA_TOOL_NAMES - (
                                    {"aria_send_message"} if profile and profile.read_only else set()
                                )
                                assert expected_aria_tools <= tools_by_name.keys()
                            else:
                                assert _ARIA_TOOL_NAMES.isdisjoint(tools_by_name)
                            assert "list_entities_tool" in tools_by_name
                            assert "query_wandb_tool" in tools_by_name
                            raw_graphql_enabled = bool(profile and "query_wandb_graphql_tool" in profile.tools)
                            assert ("query_wandb_graphql_tool" in tools_by_name) is raw_graphql_enabled

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

                            if "query_weave_traces_tool" in tools_by_name:
                                weave_result = await session.call_tool(
                                    "query_weave_traces_tool",
                                    {
                                        "entity_name": "test-entity",
                                        "project_name": "test-project",
                                        "detail_level": "invalid",
                                    },
                                )
                                assert weave_result.isError is True
                                assert "detail_level" in _result_text(weave_result)

                            if "list_weave_agents_tool" in tools_by_name:
                                agent_result = await session.call_tool(
                                    "list_weave_agents_tool",
                                    {"entity_name": "test-entity", "project_name": "test-project"},
                                )
                                assert agent_result.isError is False
                                agent_payload = json.loads(_result_text(agent_result))
                                assert agent_payload["error"] == "agents_api_unavailable"

                            if "aria_get_turn" in tools_by_name:
                                aria_result = await session.call_tool("aria_get_turn", {"turn_id": ""})
                                assert aria_result.isError is True
                                assert "invalid_request" in _result_text(aria_result)

                            if "query_wandb_graphql_tool" in tools_by_name:
                                graphql_result = await session.call_tool(
                                    "query_wandb_graphql_tool",
                                    {"query": "mutation Forbidden { deleteRun(id: 1) }"},
                                )
                                assert graphql_result.isError is False
                                graphql_payload = json.loads(_result_text(graphql_result))
                                assert graphql_payload["errors"][0]["error"] == "read_only_violation"

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


async def _run_contract_profiles(server_command: str, profiles: tuple[Profile, ...]) -> None:
    fake_wandb = ThreadingHTTPServer(("127.0.0.1", 0), _FakeWandBHandler)
    thread = threading.Thread(target=fake_wandb.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{fake_wandb.server_port}"
    try:
        for profile in profiles:
            await _exercise_profile(
                server_command,
                base_url,
                client_name="codex-mcp-client",
                expected_harness="codex",
                expected_vendor="openai",
                profile=profile,
            )
        # Exact-profile qualification must also preserve the real client
        # negotiation/telemetry checks that this harness historically covered.
        # One already-qualified profile is sufficient for those independent
        # client-identity assertions.
        representative_profile = profiles[0]
        for client_name, expected_harness, expected_vendor in _CLIENTS[1:]:
            await _exercise_profile(
                server_command,
                base_url,
                client_name=client_name,
                expected_harness=expected_harness,
                expected_vendor=expected_vendor,
                profile=representative_profile,
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
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Version-neutral public tool contract.",
    )
    parser.add_argument("--profile", action="append", default=[], help="Named profile to verify (repeatable).")
    parser.add_argument("--all-profiles", action="store_true", help="Verify every supported profile/access mode.")
    parser.add_argument("--evidence-out", type=Path, help="Write deterministic exact-profile evidence after success.")
    parser.add_argument("--source-sha", help="Exact source commit used to build the installed wheel.")
    parser.add_argument("--wheel", type=Path, help="Exact installed wheel, used to bind evidence to its digest.")
    args = parser.parse_args()
    server_command = str(Path(args.server_command).resolve())
    if not Path(server_command).is_file():
        parser.error(f"Server command does not exist: {server_command}")
    contract = load_contract(args.contract)
    if args.all_profiles:
        profiles = exhaustive_profiles(contract)
    elif args.profile:
        profiles = tuple(named_profile(contract, name) for name in args.profile)
    else:
        profiles = ()

    if args.evidence_out and (not args.all_profiles or not args.source_sha or not args.wheel):
        parser.error("--evidence-out requires --all-profiles, --source-sha, and --wheel")
    if args.wheel and not args.wheel.is_file():
        parser.error(f"wheel does not exist: {args.wheel}")
    wheel_runtime_contract_sha256 = None
    if args.wheel:
        try:
            wheel_runtime_contract_sha256 = _wheel_runtime_contract_sha256(args.wheel)
        except (ReleaseError, ValueError) as error:
            parser.error(str(error))
        if wheel_runtime_contract_sha256 != runtime_contract_sha256(contract):
            parser.error("wheel runtime contract does not match the reviewed checkout policy")

    started = time.monotonic()
    if profiles:
        anyio.run(_run_contract_profiles, server_command, profiles)
    else:
        anyio.run(_run, server_command)

    if args.evidence_out:
        evidence = {
            "schema_version": 2,
            "status": "passed",
            "version": installed_version("wandb_mcp_server"),
            "source_sha": args.source_sha,
            "wheel_sha256": sha256_file(args.wheel),
            "contract_sha256": sha256_file(args.contract),
            "runtime_contract_sha256": wheel_runtime_contract_sha256,
            "harness_sha256": sha256_file(Path(__file__)),
            "mcp_version": installed_version("mcp"),
            "locked_runtime": {
                "wandb": installed_version("wandb"),
                "wandb-workspaces": installed_version("wandb-workspaces"),
                "weave": installed_version("weave"),
            },
            "python_version": ".".join(platform.python_version_tuple()[:2]),
            "profiles": [profile.as_dict() for profile in profiles],
            "duration_ms": round((time.monotonic() - started) * 1000),
        }
        args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_out.write_bytes(canonical_json(evidence))
    print("STDIO protocol smoke passed")


if __name__ == "__main__":
    main()
