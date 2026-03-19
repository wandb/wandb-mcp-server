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
