"""Unit tests for config.py -- trace server URL resolution (MCP-5).

Tests the 3-tier resolution: explicit env var > SaaS default > dedicated {base_url}/traces.
"""

import importlib
import os
from unittest.mock import patch

import pytest


def _resolve_with_env(env: dict) -> str:
    """Reload config module with given env overrides and return resolved URL."""
    import wandb_mcp_server.config as cfg

    clean = {k: v for k, v in env.items()}
    remove = {"WF_TRACE_SERVER_URL", "WEAVE_TRACE_SERVER_URL", "WANDB_BASE_URL"} - set(clean.keys())

    with patch.dict(os.environ, clean, clear=False):
        for key in remove:
            os.environ.pop(key, None)
        importlib.reload(cfg)
        return cfg.WF_TRACE_SERVER_URL


class TestResolveTraceServerUrl:
    """Tests for _resolve_trace_server_url() 3-tier resolution."""

    def test_saas_default_resolves_to_trace_wandb(self):
        url = _resolve_with_env({"WANDB_BASE_URL": "https://api.wandb.ai"})
        assert url == "https://trace.wandb.ai"

    def test_dedicated_url_appends_traces(self):
        url = _resolve_with_env({"WANDB_BASE_URL": "https://t-mobile.wandb.io"})
        assert url == "https://t-mobile.wandb.io/traces"

    def test_dedicated_url_strips_trailing_slash(self):
        url = _resolve_with_env({"WANDB_BASE_URL": "https://custom.wandb.io/"})
        assert url == "https://custom.wandb.io/traces"

    def test_explicit_wf_trace_server_url_overrides(self):
        url = _resolve_with_env({
            "WANDB_BASE_URL": "https://t-mobile.wandb.io",
            "WF_TRACE_SERVER_URL": "https://my-custom-trace.example.com",
        })
        assert url == "https://my-custom-trace.example.com"

    def test_explicit_weave_trace_server_url_overrides(self):
        url = _resolve_with_env({
            "WANDB_BASE_URL": "https://t-mobile.wandb.io",
            "WEAVE_TRACE_SERVER_URL": "https://alt-trace.example.com",
        })
        assert url == "https://alt-trace.example.com"

    def test_wf_takes_priority_over_weave(self):
        url = _resolve_with_env({
            "WF_TRACE_SERVER_URL": "https://wf-wins.example.com",
            "WEAVE_TRACE_SERVER_URL": "https://weave-loses.example.com",
        })
        assert url == "https://wf-wins.example.com"

    def test_on_prem_style_url(self):
        url = _resolve_with_env({"WANDB_BASE_URL": "https://wandb.internal.corp.net"})
        assert url == "https://wandb.internal.corp.net/traces"
