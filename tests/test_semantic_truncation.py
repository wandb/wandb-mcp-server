"""Unit tests for semantic-aware enforce_token_budget() and estimate_tokens() (MCP-6).

Tests the field-priority truncation strategy:
  L0: under budget (pass-through)
  L1: truncate LOW_SIGNAL (inputs, output) to 100 chars
  L2: drop LOW_SIGNAL, truncate MEDIUM_SIGNAL to 200 chars
  L3: keep only HIGH_SIGNAL fields
  L4: sample traces, HIGH_SIGNAL only
"""

import json

from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata
from wandb_mcp_server.weave_api.processors import TraceProcessor


class TestEstimateTokens:
    def test_basic_estimate(self):
        assert TraceProcessor.estimate_tokens("a" * 400) == 100

    def test_minimum_one_token(self):
        assert TraceProcessor.estimate_tokens("hi") == 1

    def test_empty_string(self):
        assert TraceProcessor.estimate_tokens("") == 1


class TestFieldPriorityConstants:
    def test_high_signal_contains_key_diagnostic_fields(self):
        for field in ("id", "op_name", "status", "latency_ms", "exception", "started_at"):
            assert field in TraceProcessor.HIGH_SIGNAL_FIELDS

    def test_low_signal_contains_large_payloads(self):
        assert "inputs" in TraceProcessor.LOW_SIGNAL_FIELDS
        assert "output" in TraceProcessor.LOW_SIGNAL_FIELDS

    def test_no_overlap_between_tiers(self):
        h = TraceProcessor.HIGH_SIGNAL_FIELDS
        m = TraceProcessor.MEDIUM_SIGNAL_FIELDS
        lo = TraceProcessor.LOW_SIGNAL_FIELDS
        assert not (h & m), "HIGH and MEDIUM overlap"
        assert not (h & lo), "HIGH and LOW overlap"
        assert not (m & lo), "MEDIUM and LOW overlap"


class TestSemanticTruncation:
    @staticmethod
    def _make_traces(n: int, io_size: int = 200) -> list:
        return [
            {
                "id": f"trace-{i}",
                "op_name": f"weave:///e/p/op/op_{i}:abc",
                "display_name": f"op_{i}",
                "trace_id": f"trace-{i}",
                "parent_id": None,
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:01:00Z",
                "status": "success",
                "latency_ms": 1000 + i,
                "exception": None,
                "project_id": "e/p",
                "inputs": {"text": "x" * io_size},
                "output": "y" * io_size,
                "attributes": {"model": "gpt-4", "extra": "z" * 300},
                "summary": {"tokens": 100},
                "costs": {},
                "feedback": {},
                "wb_run_id": None,
                "wb_user_id": None,
            }
            for i in range(n)
        ]

    def test_level0_under_budget(self):
        traces = self._make_traces(2, io_size=10)
        result_json = json.dumps(traces)
        budget = TraceProcessor.estimate_tokens(result_json) + 100
        truncated, warning, level = TraceProcessor.enforce_token_budget(
            result_json, traces, budget
        )
        assert level == 0
        assert warning is None
        assert truncated is traces

    def test_level1_truncates_low_signal_only(self):
        traces = self._make_traces(5, io_size=5000)
        result_json = json.dumps(traces)
        l1_json = json.dumps([
            {
                k: (TraceProcessor.truncate_value(v, 100) if k in TraceProcessor.LOW_SIGNAL_FIELDS else v)
                for k, v in t.items()
            }
            for t in traces
        ])
        budget = TraceProcessor.estimate_tokens(l1_json) + 10
        truncated, warning, level = TraceProcessor.enforce_token_budget(
            result_json, traces, budget
        )
        assert level == 1
        assert "inputs/output shortened" in warning
        for t in truncated:
            assert "op_name" in t
            assert "status" in t
            if "inputs" in t:
                assert len(json.dumps(t["inputs"])) < 200

    def test_level2_drops_low_truncates_medium(self):
        traces = self._make_traces(10, io_size=5000)
        result_json = json.dumps(traces)
        truncated, warning, level = TraceProcessor.enforce_token_budget(
            result_json, traces, 200
        )
        assert level >= 2
        if level == 2:
            assert "Dropped inputs/output" in warning
            for t in truncated:
                assert "inputs" not in t
                assert "output" not in t
                assert "op_name" in t

    def test_level3_keeps_high_signal_only(self):
        traces = self._make_traces(20, io_size=5000)
        result_json = json.dumps(traces)
        truncated, warning, level = TraceProcessor.enforce_token_budget(
            result_json, traces, 50
        )
        assert level >= 3
        if level == 3:
            assert "high-signal diagnostic fields" in warning
            for t in truncated:
                for k in t:
                    assert k in TraceProcessor.HIGH_SIGNAL_FIELDS

    def test_level4_samples_traces(self):
        traces = self._make_traces(100, io_size=500)
        result_json = json.dumps(traces)
        truncated, warning, level = TraceProcessor.enforce_token_budget(
            result_json, traces, 1
        )
        assert level == 4
        assert "sampled" in warning.lower()
        assert len(truncated) < len(traces)

    def test_returns_tuple_of_three(self):
        traces = self._make_traces(1)
        result_json = json.dumps(traces)
        result = TraceProcessor.enforce_token_budget(result_json, traces, 999999)
        assert isinstance(result, tuple) and len(result) == 3

    def test_high_signal_fields_always_preserved(self):
        """At every truncation level, HIGH_SIGNAL fields should be present."""
        traces = self._make_traces(5, io_size=5000)
        result_json = json.dumps(traces)
        for budget in [9999, 500, 100, 10, 1]:
            truncated, _, level = TraceProcessor.enforce_token_budget(
                result_json, traces, budget
            )
            if level == 0:
                continue
            for t in truncated:
                assert "id" in t, f"id missing at level {level}"
                assert "op_name" in t, f"op_name missing at level {level}"


class TestTruncationWarningModel:
    def test_truncation_warning_defaults_to_none(self):
        assert TraceMetadata(total_traces=5).truncation_warning is None

    def test_truncation_warning_can_be_set(self):
        m = TraceMetadata(total_traces=10, truncation_warning="Truncated.")
        assert m.truncation_warning == "Truncated."

    def test_truncation_warning_in_query_result(self):
        m = TraceMetadata(total_traces=3, truncation_warning="Level 2.")
        r = QueryResult(metadata=m)
        assert r.model_dump()["metadata"]["truncation_warning"] == "Level 2."
