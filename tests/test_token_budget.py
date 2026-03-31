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

    def test_l4_rechecks_budget_with_large_exception_fields(self):
        """Even after sampling, large HIGH_SIGNAL fields must still fit under budget."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(8)
        for i, trace in enumerate(traces):
            trace["exception"] = f"huge-exception-{i} " * 5000

        result_json = json.dumps(traces)
        budget = 200

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)
        serialized = json.dumps(out)

        assert level >= 4
        assert TraceProcessor.estimate_tokens(serialized) <= budget

    def test_full_response_pipeline_respects_budget(self):
        """Server-style metadata + traces response should stay within total budget."""
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(12)
        for i, trace in enumerate(traces):
            trace["exception"] = f"oversized-exception-{i} " * 3000

        metadata = TraceMetadata(
            total_traces=len(traces),
            op_distribution={f"op_{i}": i for i in range(10)},
        )
        result = QueryResult(metadata=metadata, traces=traces)
        total_budget = 400

        metadata_json = result.metadata.model_dump_json()
        metadata_tokens = TraceProcessor.estimate_tokens(metadata_json)
        trace_budget = max(1, total_budget - metadata_tokens)

        response_json = result.model_dump_json()
        truncated_traces, warning, level = TraceProcessor.enforce_token_budget(
            response_json,
            result.traces,
            trace_budget,
        )
        result.traces = truncated_traces
        final_json = result.model_dump_json()

        assert level >= 1
        assert TraceProcessor.estimate_tokens(final_json) <= total_budget

    # -- L1/L2 exception bypass tests ----------------------------------------

    def test_l1_leaves_large_exception_untouched(self):
        """L1 only truncates LOW_SIGNAL (inputs/output); exception is HIGH_SIGNAL
        and should pass through L1 unchanged even when very large."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(3)
        for t in traces:
            t["exception"] = "E" * 20_000  # 20KB exception per trace

        result_json = json.dumps(traces)
        # Budget tight enough to trigger L1 (inputs shrink) but loose enough
        # that the exception-heavy payload fits after inputs are trimmed.
        budget = TraceProcessor.estimate_tokens(result_json) // 2

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)

        if level == 1:
            for t in out:
                assert len(t.get("exception", "")) == 20_000

    def test_l2_still_over_budget_with_huge_exception(self):
        """When exception alone exceeds the budget, L2 cannot solve it and must
        escalate to L3+."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(5)
        for t in traces:
            t["inputs"] = {"x": "small"}
            t["output"] = {"y": "small"}
            t["exception"] = "X" * 100_000  # 100KB per trace

        result_json = json.dumps(traces)
        budget = 500  # much smaller than 5 * 25K est tokens

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)
        assert level >= 3, f"Expected L3+ but got L{level}"

    def test_l3_caps_exception_strings_to_500(self):
        """L3 keeps only HIGH_SIGNAL and caps strings > 500 chars."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(4)
        for t in traces:
            t["exception"] = "ERR " * 500  # 2000 chars

        result_json = json.dumps(traces)
        # Budget that forces L3 but not L4
        l2_est = TraceProcessor.estimate_tokens(result_json) // 6
        budget = l2_est

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)

        if level == 3:
            for t in out:
                exc = t.get("exception", "")
                assert len(exc) <= 510, f"exception len {len(exc)} exceeds L3 cap"

    def test_non_string_exception_bypasses_l3_string_cap(self):
        """L3's isinstance(v, str) check misses dict/list exceptions. This
        documents the known gap where structured exceptions stay full size."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(3)
        big_dict_exc = {
            "type": "ValueError",
            "message": "M" * 2000,
            "traceback": [{"file": f"f{i}.py", "line": i, "code": "x" * 500} for i in range(20)],
        }
        for t in traces:
            t["inputs"] = {"x": "s"}
            t["output"] = {"y": "s"}
            t["exception"] = big_dict_exc

        result_json = json.dumps(traces)
        budget = TraceProcessor.estimate_tokens(result_json) // 4

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)

        if level == 3:
            for t in out:
                exc = t.get("exception", {})
                # Dict exception survives L3 string cap because it's not a str
                assert isinstance(exc, dict), "dict exception should remain a dict at L3"

    def test_estimate_tokens_vs_real_tokens_divergence(self):
        """Quantify the gap between estimate_tokens (len//4) and tiktoken on
        exception-heavy payloads so we know the safety margin needed."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(5)
        for t in traces:
            t["exception"] = "KeyError: 'missing_field'\n" * 200

        payload = json.dumps(traces)
        estimated = TraceProcessor.estimate_tokens(payload)
        real = count_tokens(payload)

        ratio = real / max(1, estimated)
        # Typically tiktoken gives fewer tokens than len//4 for English-ish JSON,
        # but the ratio should stay within 0.5x-2.0x to be a useful proxy.
        assert 0.3 < ratio < 2.5, f"estimate/real ratio {ratio:.2f} is dangerously off"

    def test_l4_recheck_converges_for_many_large_exceptions(self):
        """50 traces each with 10KB exceptions and a tiny budget; the L4
        recheck while-loop must converge and fit within budget."""
        from wandb_mcp_server.weave_api.processors import TraceProcessor

        traces = self._make_traces(50)
        for t in traces:
            t["exception"] = "FATAL " * 2000  # ~12KB per trace

        result_json = json.dumps(traces)
        budget = 500

        out, warning, level = TraceProcessor.enforce_token_budget(result_json, traces, budget)
        serialized = json.dumps(out)

        assert level == 4
        assert TraceProcessor.estimate_tokens(serialized) <= budget
        assert len(out) >= 1, "should keep at least one trace"
