"""Tests for the search_wandb_docs_tool proxy."""

import os
from unittest.mock import patch

from wandb_mcp_server.mcp_tools.docs_search import (
    SEARCH_WANDB_DOCS_TOOL_DESCRIPTION,
    is_docs_proxy_enabled,
)


class TestDocsSearchDescription:
    def test_has_when_to_use(self):
        assert "<when_to_use>" in SEARCH_WANDB_DOCS_TOOL_DESCRIPTION
        assert "</when_to_use>" in SEARCH_WANDB_DOCS_TOOL_DESCRIPTION

    def test_mentions_documentation(self):
        assert "documentation" in SEARCH_WANDB_DOCS_TOOL_DESCRIPTION.lower()


class TestDocsProxyEnabled:
    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WANDB_MCP_PROXY_DOCS", None)
            assert is_docs_proxy_enabled() is True

    def test_enabled_when_true(self):
        with patch.dict(os.environ, {"WANDB_MCP_PROXY_DOCS": "true"}):
            assert is_docs_proxy_enabled() is True

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"WANDB_MCP_PROXY_DOCS": "false"}):
            assert is_docs_proxy_enabled() is False

    def test_disabled_case_insensitive(self):
        with patch.dict(os.environ, {"WANDB_MCP_PROXY_DOCS": "False"}):
            assert is_docs_proxy_enabled() is False

    def test_other_values_treated_as_enabled(self):
        with patch.dict(os.environ, {"WANDB_MCP_PROXY_DOCS": "yes"}):
            assert is_docs_proxy_enabled() is True
