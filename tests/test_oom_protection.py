"""Tests for OOM protection: memory guard, error handling, and pre-query estimation."""

import json
import sys
from unittest.mock import patch

import pytest
from wandb_mcp_server.config import MAX_ACCUMULATED_BYTES


class TestMemoryGuardConfig:
    """Test the memory guard configuration."""

    def test_max_accumulated_bytes_has_default(self):
        assert MAX_ACCUMULATED_BYTES == 1024 * 1024 * 1024

    def test_max_accumulated_bytes_is_configurable(self):
        import os

        with patch.dict(os.environ, {"MAX_ACCUMULATED_BYTES": "1000"}):
            import importlib
            import wandb_mcp_server.config as cfg

            importlib.reload(cfg)
            assert cfg.MAX_ACCUMULATED_BYTES == 1000
            importlib.reload(cfg)


class TestMemoryGuardLogic:
    """Test the streaming accumulator logic pattern (without full service instantiation)."""

    def test_accumulator_stops_at_limit(self):
        """Simulate the memory guard pattern used in TraceService.query_traces."""
        max_bytes = 1000
        traces_gen = ({"id": f"t{i}", "data": "x" * 200} for i in range(100))

        all_traces = []
        accumulated_bytes = 0
        for trace in traces_gen:
            trace_size = sys.getsizeof(str(trace))
            if accumulated_bytes + trace_size > max_bytes:
                break
            all_traces.append(trace)
            accumulated_bytes += trace_size

        assert len(all_traces) < 100
        assert accumulated_bytes <= max_bytes + sys.getsizeof(str({"id": "t0", "data": "x" * 200}))


class TestErrorPayloads:
    """Test that error payloads are valid JSON with expected structure."""

    def test_oom_error_payload(self):
        payload = json.dumps(
            {
                "error": "out_of_memory",
                "message": "This query exceeded server memory limits. "
                "Try: detail_level='schema', smaller limit, or metadata_only=True.",
            }
        )
        parsed = json.loads(payload)
        assert parsed["error"] == "out_of_memory"
        assert "memory" in parsed["message"].lower()

    def test_query_failed_payload(self):
        payload = json.dumps(
            {
                "error": "query_failed",
                "message": "Some error occurred"[:500],
            }
        )
        parsed = json.loads(payload)
        assert parsed["error"] == "query_failed"

    def test_query_too_large_payload(self):
        payload = json.dumps(
            {
                "error": "query_too_large",
                "message": "Found 1000 matching traces.",
                "trace_count": 1000,
                "suggestions": [
                    "detail_level='schema'",
                    "limit=100",
                    "metadata_only=True",
                    "Add filters",
                ],
            }
        )
        parsed = json.loads(payload)
        assert parsed["error"] == "query_too_large"
        assert parsed["trace_count"] == 1000
        assert len(parsed["suggestions"]) == 4

    def test_structured_error_helper(self):
        from wandb_mcp_server.config import structured_error

        payload = structured_error("timeout", "Tool timed out", timeout_seconds=30)

        assert payload == {"error": "timeout", "message": "Tool timed out", "timeout_seconds": 30}


class TestHostedLimitConfig:
    """Test hosted-mode limit configuration."""

    def test_hosted_mode_uses_stricter_defaults(self):
        import importlib
        import os
        import wandb_mcp_server.config as cfg

        with patch.dict(os.environ, {"MCP_HOSTED_MODE": "true"}, clear=False):
            importlib.reload(cfg)
            assert cfg.MCP_HOSTED_MODE is True
            assert cfg.MCP_MAX_QUERY_LIMIT == 100
            assert cfg.MCP_MAX_FULL_TRACE_LIMIT == 25
            assert cfg.MCP_MAX_HISTORY_SAMPLES == 500
        importlib.reload(cfg)

    def test_hosted_limits_are_configurable(self):
        import importlib
        import os
        import wandb_mcp_server.config as cfg

        with patch.dict(os.environ, {"MCP_HOSTED_MODE": "true", "MCP_MAX_QUERY_LIMIT": "42"}, clear=False):
            importlib.reload(cfg)
            assert cfg.MCP_MAX_QUERY_LIMIT == 42
        importlib.reload(cfg)

    def test_dedicated_profile_uses_larger_bounded_defaults(self):
        import importlib
        import os
        import wandb_mcp_server.config as cfg

        with patch.dict(
            os.environ,
            {"MCP_HOSTED_MODE": "true", "MCP_WORKLOAD_PROFILE": "dedicated"},
            clear=False,
        ):
            importlib.reload(cfg)
            assert cfg.MCP_MAX_WANDB_QUERY_ITEMS == 250
            assert cfg.MCP_MAX_FULL_DETAIL_ITEMS == 10
            assert cfg.MCP_MAX_HISTORY_SAMPLES == 1_500
            assert cfg.MCP_MAX_HISTORY_KEYS == 50
            assert cfg.MCP_ADMISSION_ACTOR_CAPACITY == 8
            assert cfg.MCP_ADMISSION_PROCESS_CAPACITY == 16
        importlib.reload(cfg)

    def test_local_profile_disables_admission_by_default(self):
        import importlib
        import os
        import wandb_mcp_server.config as cfg

        with patch.dict(
            os.environ,
            {
                "MCP_HOSTED_MODE": "false",
                "MCP_WORKLOAD_PROFILE": "local",
                "MCP_ADMISSION_CONTROL_ENABLED": "",
            },
            clear=False,
        ):
            os.environ.pop("MCP_ADMISSION_CONTROL_ENABLED")
            importlib.reload(cfg)
            assert cfg.MCP_MAX_WANDB_QUERY_ITEMS == 1_000
            assert cfg.MCP_MAX_HISTORY_SAMPLES == 5_000
            assert cfg.MCP_ADMISSION_CONTROL_ENABLED is False
        importlib.reload(cfg)

    def test_invalid_profile_is_rejected(self):
        import importlib
        import os
        import wandb_mcp_server.config as cfg

        with patch.dict(os.environ, {"MCP_WORKLOAD_PROFILE": "unbounded"}, clear=False):
            with pytest.raises(ValueError, match="MCP_WORKLOAD_PROFILE"):
                importlib.reload(cfg)
        importlib.reload(cfg)
