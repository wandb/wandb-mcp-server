"""Tests for the search_wandb_docs_tool proxy."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wandb_mcp_server.mcp_tools.docs_search import (
    SEARCH_WANDB_DOCS_TOOL_DESCRIPTION,
    is_docs_proxy_enabled,
    search_wandb_docs,
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


class TestSearchWandBDocs:
    @pytest.mark.asyncio
    @patch("wandb_mcp_server.mcp_tools.docs_search.httpx.AsyncClient")
    async def test_successful_search(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "content": [
                    {"text": "To create a scorer, use weave.Scorer..."},
                    {"text": "Example: class MyScorer(weave.Scorer):..."},
                ]
            }
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await search_wandb_docs("How to create a Weave scorer")
        assert "weave.Scorer" in result
        assert "---" in result

    @pytest.mark.asyncio
    @patch("wandb_mcp_server.mcp_tools.docs_search.httpx.AsyncClient")
    async def test_empty_results(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": {"content": []}}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await search_wandb_docs("nonexistent topic xyz")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    @patch("wandb_mcp_server.mcp_tools.docs_search.httpx.AsyncClient")
    async def test_timeout_graceful(self, mock_client_cls):
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await search_wandb_docs("test query")
        parsed = json.loads(result)
        assert "timed out" in parsed["error"].lower()

    @pytest.mark.asyncio
    @patch("wandb_mcp_server.mcp_tools.docs_search.httpx.AsyncClient")
    async def test_http_error_graceful(self, mock_client_cls):
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await search_wandb_docs("test query")
        parsed = json.loads(result)
        assert "500" in parsed["error"]
