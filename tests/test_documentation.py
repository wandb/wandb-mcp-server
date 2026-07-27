"""Regression checks for durable repository documentation."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mcp.server.fastmcp import FastMCP

import wandb_mcp_server.config as config
from wandb_mcp_server.server import register_tools


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
FEATURE_FLAGS = (
    "WANDB_MCP_ENABLE_RAW_GRAPHQL",
    "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
    "WANDB_MCP_ENABLE_WEAVE_TOOLS",
    "WANDB_MCP_READ_ONLY",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _durable_markdown_files() -> list[Path]:
    return [
        *sorted(REPOSITORY_ROOT.glob("*.md")),
        *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
    ]


def test_readme_documents_every_default_tool(monkeypatch):
    for feature_flag in FEATURE_FLAGS:
        monkeypatch.delenv(feature_flag, raising=False)
    importlib.reload(config)

    server = FastMCP("documentation-test")
    register_tools(server)
    tool_names = set(server._tool_manager._tools)
    readme = README.read_text()

    missing = sorted(name for name in tool_names if f"**{name}**" not in readme)
    assert not missing, f"README is missing default tools: {missing}"


def test_documentation_json_examples_are_valid():
    errors: list[str] = []
    for path in _durable_markdown_files():
        for index, example in enumerate(JSON_FENCE.findall(path.read_text()), start=1):
            try:
                json.loads(example)
            except json.JSONDecodeError as error:
                errors.append(f"{path.relative_to(REPOSITORY_ROOT)} block {index}: {error}")

    assert not errors, "Invalid JSON documentation examples:\n" + "\n".join(errors)


def test_relative_documentation_links_resolve():
    broken: list[str] = []
    for path in _durable_markdown_files():
        for raw_target in MARKDOWN_LINK.findall(path.read_text()):
            target = raw_target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:")):
                continue

            relative_path = unquote(parsed.path)
            if not relative_path:
                continue
            resolved = (path.parent / relative_path).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not broken, "Broken relative documentation links:\n" + "\n".join(broken)
