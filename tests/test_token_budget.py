"""Tests for token-aware progressive truncation."""

import json

from wandb_mcp_server.trace_utils import enforce_token_budget, count_tokens


class TestEnforceTokenBudget:
    def _make_traces(self, n: int) -> list:
        return [
            {
                "id": f"t{i}",
                "op_name": f"op_{i}",
                "status": "success",
                "inputs": {"text": f"input data for trace {i} " * 50},
                "output": {"result": f"output data for trace {i} " * 50},
            }
            for i in range(n)
        ]

    def test_no_truncation_when_under_budget(self):
        traces = self._make_traces(2)
        result_json = json.dumps(traces)
        budget = count_tokens(result_json) + 1000

        new_json, dropped = enforce_token_budget(result_json, traces, budget)
        assert dropped == 0
        assert len(traces) == 2

    def test_drops_traces_when_over_budget(self):
        traces = self._make_traces(20)
        result_json = json.dumps(traces)
        full_tokens = count_tokens(result_json)
        budget = full_tokens // 3

        _, dropped = enforce_token_budget(result_json, traces, budget)
        assert dropped > 0

    def test_keeps_at_least_one_trace(self):
        traces = self._make_traces(5)
        result_json = json.dumps(traces)

        _, dropped = enforce_token_budget(result_json, traces, max_tokens=1)
        assert dropped == 4

    def test_drops_from_end(self):
        traces = self._make_traces(5)
        result_json = json.dumps(traces)
        full_tokens = count_tokens(result_json)

        new_json, dropped = enforce_token_budget(result_json, traces, max_tokens=full_tokens // 2)
        remaining = json.loads(new_json)
        assert remaining[0]["id"] == "t0"

    def test_returns_valid_json(self):
        traces = self._make_traces(10)
        result_json = json.dumps(traces)
        full_tokens = count_tokens(result_json)

        new_json, _ = enforce_token_budget(result_json, traces, max_tokens=full_tokens // 4)
        parsed = json.loads(new_json)
        assert isinstance(parsed, list)

    def test_does_not_mutate_input_list(self):
        """enforce_token_budget must not mutate the caller's trace list."""
        traces = self._make_traces(20)
        original_len = len(traces)
        result_json = json.dumps(traces)
        budget = count_tokens(result_json) // 3

        _, dropped = enforce_token_budget(result_json, traces, budget)
        assert dropped > 0
        assert len(traces) == original_len


class TestTraceMetadataTruncationFields:
    def test_truncation_fields_default_false(self):
        from wandb_mcp_server.weave_api.models import TraceMetadata

        meta = TraceMetadata()
        assert meta.truncation_applied is False
        assert meta.truncation_dropped_count == 0
        assert meta.truncation_note is None

    def test_truncation_fields_settable(self):
        from wandb_mcp_server.weave_api.models import TraceMetadata

        meta = TraceMetadata(
            truncation_applied=True,
            truncation_dropped_count=47,
            truncation_note="47 more traces match this query.",
        )
        assert meta.truncation_applied is True
        assert meta.truncation_dropped_count == 47
        assert "47" in meta.truncation_note

    def test_truncation_fields_in_query_result(self):
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata

        meta = TraceMetadata(
            total_traces=50,
            truncation_applied=True,
            truncation_dropped_count=30,
            truncation_note="30 traces dropped.",
        )
        result = QueryResult(metadata=meta, traces=[])
        dumped = result.model_dump()
        assert dumped["metadata"]["truncation_applied"] is True
        assert dumped["metadata"]["truncation_dropped_count"] == 30

    def test_truncation_fields_serialize_to_json(self):
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata

        meta = TraceMetadata(
            truncation_applied=True,
            truncation_dropped_count=10,
            truncation_note="10 more traces.",
        )
        result = QueryResult(metadata=meta, traces=[])
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["truncation_applied"] is True
        assert parsed["metadata"]["truncation_note"] == "10 more traces."


class TestSemanticTokenBudget:
    """Tests for TraceProcessor.enforce_token_budget (semantic 5-level truncation)."""

    def _make_traces(self, n: int) -> list:
        return [
            {
                "id": f"t{i}",
                "op_name": f"weave:///entity/proj/op/op_{i}:abc",
                "trace_id": f"tr{i}",
                "started_at": "2026-03-20T12:00:00Z",
                "ended_at": "2026-03-20T12:01:00Z",
                "status": "success",
                "parent_id": None,
                "display_name": f"op_{i}",
                "inputs": {"text": f"input data for trace {i} " * 100},
                "output": {"result": f"output data for trace {i} " * 100},
                "summary": {"weave": {"status": "success", "latency_ms": 500}},
                "attributes": {"key": "value"},
                "costs": {},
                "feedback": {},
            }
            for i in range(n)
        ]

    def test_l0_under_budget(self):
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(2)
        result_json = json.dumps(traces)
        big_budget = len(result_json)

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, big_budget)
        assert level == 0
        assert warning is None
        assert len(out) == 2

    def test_l1_preserves_high_signal(self):
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(10)
        result_json = json.dumps(traces)
        small_budget = len(result_json) // 8

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, small_budget)
        assert level >= 1
        for t in out:
            assert "id" in t
            assert "op_name" in t
            assert "status" in t

    def test_high_levels_drop_low_signal(self):
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(20)
        result_json = json.dumps(traces)
        tiny_budget = 5

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, tiny_budget)
        assert level >= 3
        for t in out:
            assert "id" in t
            assert "inputs" not in t
            assert "output" not in t

    def test_does_not_mutate_input(self):
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(5)
        original_len = len(traces)
        original_keys = set(traces[0].keys())
        result_json = json.dumps(traces)

        TraceProcessor.enforce_token_budget(result_json, traces, 10)
        assert len(traces) == original_len
        assert set(traces[0].keys()) == original_keys
