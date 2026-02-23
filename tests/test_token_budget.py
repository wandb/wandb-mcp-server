"""Unit tests for enforce_token_budget() and estimate_tokens() -- response size limits (MCP-6)."""

import json

from wandb_mcp_server.weave_api.processors import TraceProcessor


class TestEstimateTokens:

    def test_basic_estimate(self):
        assert TraceProcessor.estimate_tokens("a" * 400) == 100

    def test_minimum_one_token(self):
        assert TraceProcessor.estimate_tokens("hi") == 1

    def test_empty_string(self):
        assert TraceProcessor.estimate_tokens("") == 1


class TestEnforceTokenBudget:

    @staticmethod
    def _make_traces(n: int, io_size: int = 200) -> list:
        return [
            {
                "id": f"trace-{i}",
                "op_name": f"op_{i}",
                "inputs": {"text": "x" * io_size},
                "output": "y" * io_size,
                "status": "success",
            }
            for i in range(n)
        ]

    def test_level0_under_budget(self):
        traces = self._make_traces(2, io_size=10)
        result_json = json.dumps(traces)
        budget = TraceProcessor.estimate_tokens(result_json) + 100
        truncated, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)
        assert level == 0
        assert warning is None
        assert truncated is traces

    def test_level1_truncates_strings(self):
        traces = self._make_traces(5, io_size=5000)
        result_json = json.dumps(traces)
        l1_json = json.dumps([
            {k: TraceProcessor.truncate_value(v, 100) for k, v in t.items()}
            for t in traces
        ])
        budget = TraceProcessor.estimate_tokens(l1_json) + 10
        truncated, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)
        assert level == 1
        assert "100 chars" in warning

    def test_level2_drops_io_columns(self):
        traces = self._make_traces(10, io_size=5000)
        result_json = json.dumps(traces)
        truncated, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, 50)
        assert level >= 2
        if level == 2:
            assert "Dropped" in warning
            for t in truncated:
                assert "inputs" not in t
                assert "output" not in t

    def test_level3_samples_traces(self):
        traces = self._make_traces(100, io_size=500)
        result_json = json.dumps(traces)
        truncated, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, 1)
        assert level == 3
        assert "Sampled" in warning
        assert len(truncated) == 50

    def test_returns_tuple_of_three(self):
        traces = self._make_traces(1)
        result_json = json.dumps(traces)
        result = TraceProcessor.enforce_token_budget(result_json, traces, 999999)
        assert isinstance(result, tuple) and len(result) == 3
