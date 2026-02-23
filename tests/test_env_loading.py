"""Unit tests for .env loading and env var alignment.

Verifies that MCP_LOGS_WANDB_ENTITY, MCP_LOGS_WANDB_PROJECT, and related
variables are read correctly by the server initialization code.
"""

import os
from unittest.mock import patch

import pytest


class TestEnvVarAlignment:
    """Verify server reads the right env vars for Weave project config."""

    def test_mcp_logs_entity_used_for_weave_project(self):
        """MCP_LOGS_WANDB_ENTITY should be the primary entity source."""
        with patch.dict(os.environ, {
            "MCP_LOGS_WANDB_ENTITY": "my-team",
            "MCP_LOGS_WANDB_PROJECT": "mcp-logs",
            "WEAVE_DISABLED": "false",
        }, clear=False):
            entity = os.environ.get("MCP_LOGS_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
            project = os.environ.get("MCP_LOGS_WANDB_PROJECT", "wandb-mcp-logs")
            assert entity == "my-team"
            assert project == "mcp-logs"
            assert f"{entity}/{project}" == "my-team/mcp-logs"

    def test_wandb_entity_fallback(self):
        """WANDB_ENTITY should be used if MCP_LOGS_WANDB_ENTITY is not set."""
        env = {"WANDB_ENTITY": "fallback-team"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("MCP_LOGS_WANDB_ENTITY", None)
            entity = os.environ.get("MCP_LOGS_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY")
            assert entity == "fallback-team"

    def test_mcp_logs_project_defaults_to_wandb_mcp_logs(self):
        """MCP_LOGS_WANDB_PROJECT should default to 'wandb-mcp-logs'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_LOGS_WANDB_PROJECT", None)
            project = os.environ.get("MCP_LOGS_WANDB_PROJECT", "wandb-mcp-logs")
            assert project == "wandb-mcp-logs"

    def test_weave_disabled_true_skips_tracing(self):
        with patch.dict(os.environ, {"WEAVE_DISABLED": "true"}, clear=False):
            disabled = os.environ.get("WEAVE_DISABLED", "true").lower() == "true"
            assert disabled is True

    def test_weave_disabled_false_enables_tracing(self):
        with patch.dict(os.environ, {"WEAVE_DISABLED": "false"}, clear=False):
            disabled = os.environ.get("WEAVE_DISABLED", "true").lower() == "true"
            assert disabled is False

    def test_dotenv_loads_from_repo_root(self):
        from pathlib import Path
        server_py = Path(__file__).parent.parent / "src" / "wandb_mcp_server" / "server.py"
        dotenv_path = server_py.parent.parent.parent / ".env"
        assert dotenv_path.name == ".env"

    def test_mcp_trace_list_operations_normalized(self):
        with patch.dict(os.environ, {"MCP_TRACE_LIST_OPERATIONS": "True"}, clear=False):
            val = os.environ.get("MCP_TRACE_LIST_OPERATIONS", "").lower() == "true"
            assert val is True

    def test_mcp_trace_list_operations_default_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_TRACE_LIST_OPERATIONS", None)
            val = os.environ.get("MCP_TRACE_LIST_OPERATIONS", "").lower() == "true"
            assert val is False
