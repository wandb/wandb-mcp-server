"""Verify trace_utils.py is deleted and not importable (MCP-10)."""

import importlib

import pytest


class TestTraceUtilsDeletion:
    def test_trace_utils_not_importable(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("wandb_mcp_server.trace_utils")
