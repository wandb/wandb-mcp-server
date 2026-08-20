"""Smoke tests for import paths that must work after a fresh install."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest


_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys

    target = sys.argv[1]
    for name in list(sys.modules):
        if name == "wandb_gql" or name.startswith("wandb_gql."):
            del sys.modules[name]
        if name == "wandb_graphql" or name.startswith("wandb_graphql."):
            del sys.modules[name]

    sys.path = [
        path for path in sys.path
        if "/wandb/vendor" not in path
        and "gql-0.2.0" not in path
        and "graphql-core-1.1" not in path
    ]

    importlib.import_module(target)

    leaked_paths = [
        path for path in sys.path
        if "/wandb/vendor" in path
        or "gql-0.2.0" in path
        or "graphql-core-1.1" in path
    ]
    if leaked_paths:
        message = "W&B vendor paths leaked into sys.path: "
        raise AssertionError(message + repr(leaked_paths))

    print("ok")
    """
)


@pytest.mark.parametrize(
    "module_name",
    [
        "wandb_mcp_server.mcp_tools.create_report",
        "wandb_mcp_server.mcp_tools.query_wandb",
        "wandb_mcp_server.mcp_tools.query_wandb_gql",
        "wandb_mcp_server",
    ],
)
def test_fresh_install_imports_without_preloaded_vendor_path(
    module_name: str,
) -> None:
    env = os.environ.copy()
    env["WANDB_SILENT"] = "True"
    env["WEAVE_SILENT"] = "True"

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, module_name],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_default_server_import_does_not_load_raw_graphql_modules() -> None:
    script = """
import sys
import wandb_mcp_server.server

assert "wandb_mcp_server.mcp_tools.query_wandb_gql" not in sys.modules
assert "wandb_mcp_server.mcp_tools.query_wandb_graphql" not in sys.modules
print("ok")
"""
    env = os.environ.copy()
    env["WANDB_MCP_TOOL_PROFILE"] = "models-weave"
    env["WANDB_MCP_ACCESS_MODE"] = "read-write"
    env["WANDB_SILENT"] = "True"
    env["WEAVE_SILENT"] = "True"

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_console_bootstrap_loads_cwd_dotenv_before_server_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from wandb_mcp_server import entrypoint

    for name in ("WANDB_MCP_TOOL_PROFILE", "WANDB_MCP_ACCESS_MODE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "WANDB_MCP_TOOL_PROFILE=models-only\nWANDB_MCP_ACCESS_MODE=read-only\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    called: list[tuple[str | None, str | None]] = []
    fake_server = ModuleType("wandb_mcp_server.server")

    def fake_cli() -> None:
        called.append(
            (
                os.environ.get("WANDB_MCP_TOOL_PROFILE"),
                os.environ.get("WANDB_MCP_ACCESS_MODE"),
            )
        )

    fake_server.cli = fake_cli  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb_mcp_server.server", fake_server)

    try:
        entrypoint.cli()
        assert called == [("models-only", "read-only")]
    finally:
        # python-dotenv mutates os.environ directly, outside MonkeyPatch's
        # bookkeeping. Keep this process-level bootstrap test isolated from
        # every later profile-registration test in the same pytest worker.
        os.environ.pop("WANDB_MCP_TOOL_PROFILE", None)
        os.environ.pop("WANDB_MCP_ACCESS_MODE", None)
