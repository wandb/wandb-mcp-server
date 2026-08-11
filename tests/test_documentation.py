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
QUERY_CAPABILITIES = REPOSITORY_ROOT / "docs" / "query-capabilities.md"
RELEASE_NOTES = REPOSITORY_ROOT / "docs" / "releases" / "v0.4.0.md"
FEATURE_FLAGS = (
    "WANDB_MCP_ENABLE_RAW_GRAPHQL",
    "WANDB_MCP_ENABLE_ARIA_TOOLS",
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


def test_every_former_named_graphql_example_has_a_typed_v040_route():
    text = QUERY_CAPABILITIES.read_text()
    route_rows = dict(
        re.findall(
            r"^\| `([^`]+)` \| `([^`]+)` \|",
            text,
            flags=re.MULTILINE,
        )
    )

    assert route_rows == {
        "MinimalRunIdVsDisplayName": "query_wandb_tool",
        "GetProjectInfo": "query_wandb_tool",
        "GetSortedRuns": "query_wandb_tool",
        "GetFilteredRuns": "query_wandb_tool",
        "GetRunByDisplayName": "query_wandb_tool",
    }


def test_query_capability_matrix_routes_typed_specialized_and_raw_reads():
    text = QUERY_CAPABILITIES.read_text()

    for typed_route in (
        'query_wandb_tool(resource="project")',
        'query_wandb_tool(resource="run", run_id=...)',
        'query_wandb_tool(resource="runs", filters=..., order=...)',
        'query_wandb_tool(resource="sweep"|"sweeps")',
        'query_wandb_tool(resource="reports", report_name=...)',
    ):
        assert f"`{typed_route}`" in text

    for specialized_tool in (
        "get_run_history_tool",
        "list_artifact_versions_tool",
        "get_artifact_details_tool",
        "list_registries_tool",
        "list_registry_collections_tool",
        "list_wandb_automations_tool",
        "list_wandb_integrations_tool",
    ):
        assert f"`{specialized_tool}`" in text

    raw_rows = [line for line in text.splitlines() if line.startswith("|") and "query_wandb_graphql_tool" in line]
    assert len(raw_rows) == 4
    assert all(line.rstrip().endswith("| Yes |") for line in raw_rows)


def test_history_safety_metadata_is_documented_truthfully():
    for path in (README, QUERY_CAPABILITIES, RELEASE_NOTES):
        text = path.read_text()
        assert "non_finite_counts" in text
        assert "key_counts_exact" in text

    capability_text = QUERY_CAPABILITIES.read_text()
    assert "source_truncated" in capability_text
    assert "step-window" in capability_text
    assert "post-protobuf" in capability_text


def test_removed_wandbot_is_not_documented_or_shipped():
    assert not (REPOSITORY_ROOT / "src" / "wandb_mcp_server" / "mcp_tools" / "query_wandbot.py").exists()
    for path in (*_durable_markdown_files(), REPOSITORY_ROOT / "env.example"):
        text = path.read_text().lower()
        assert "wandbot" not in text
        assert "supportbot" not in text
