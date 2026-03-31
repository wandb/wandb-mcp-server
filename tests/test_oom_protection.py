"""Tests for OOM protection: memory guard, error handling, and pre-query estimation."""

import json
import sys
from unittest.mock import patch


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
