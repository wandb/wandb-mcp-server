"""Tests for the summarize_evaluation tool."""

import json
from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.summarize_evaluation import (
    _aggregate_eval,
    _extract_scores,
    summarize_evaluation,
)


class TestExtractScores:
    def test_extracts_mean_scores(self):
        summary = {
            "weave": {
                "correctness": {"true_count": 8, "true_fraction": 0.8},
                "latency": {"mean": 1.23},
            }
        }
        scores = _extract_scores(summary)

        assert "correctness" in scores
        assert scores["correctness"]["true_fraction"] == 0.8
        assert scores["latency"]["mean"] == 1.23

    def test_ignores_non_scorer_keys(self):
        summary = {
            "weave": {
                "status": "success",
                "some_string": "not a score",
                "count": 42,
            }
        }
        scores = _extract_scores(summary)
        assert scores == {}

    def test_empty_summary(self):
        assert _extract_scores({}) == {}

    def test_missing_weave_key(self):
        assert _extract_scores({"other": {"mean": 1.0}}) == {}


class TestAggregateEval:
    def test_success_and_error_counts(self):
        eval_trace = {
            "id": "eval-1",
            "op_name": "Evaluation.evaluate",
            "started_at": "2026-01-01T00:00:00Z",
            "summary": {"weave": {"accuracy": {"mean": 0.9}}},
        }
        children = [
            {"summary": {"weave": {"status": "success"}}},
            {"summary": {"weave": {"status": "success"}}},
            {"exception": "ValueError: bad input"},
            {"summary": {"weave": {"status": "error"}}},
        ]

        result = _aggregate_eval(eval_trace, children)

        assert result["total_predictions"] == 4
        assert result["errors"] == 2
        assert result["successes"] == 2
        assert result["error_rate"] == 0.5

    def test_all_successes(self):
        eval_trace = {"id": "e1", "op_name": "eval", "started_at": "", "summary": {}}
        children = [
            {"summary": {"weave": {"status": "success"}}},
            {"summary": {"weave": {"status": "success"}}},
        ]

        result = _aggregate_eval(eval_trace, children)
        assert result["errors"] == 0
        assert result["error_rate"] == 0.0

    def test_empty_children(self):
        eval_trace = {"id": "e1", "op_name": "eval", "started_at": "", "summary": {}}
        result = _aggregate_eval(eval_trace, [])

        assert result["total_predictions"] == 0
        assert result["error_rate"] == 0.0

    def test_token_usage_aggregation(self):
        eval_trace = {"id": "e1", "op_name": "eval", "started_at": "", "summary": {}}
        children = [
            {"summary": {"usage": {"gpt-4": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}}},
            {"summary": {"usage": {"gpt-4": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}}}},
        ]

        result = _aggregate_eval(eval_trace, children)
        assert result["token_usage"]["prompt_tokens"] == 300
        assert result["token_usage"]["completion_tokens"] == 150
        assert result["token_usage"]["total_tokens"] == 450

    def test_scores_extracted_from_eval_summary(self):
        eval_trace = {
            "id": "e1",
            "op_name": "eval",
            "started_at": "",
            "summary": {"weave": {"relevance": {"true_fraction": 0.75}}},
        }
        result = _aggregate_eval(eval_trace, [])
        assert result["scores"]["relevance"]["true_fraction"] == 0.75


class TestSummarizeEvaluation:
    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.TraceService")
    def test_no_evals_found(self, mock_trace_svc_cls, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api_key.return_value = "fake-key"

        mock_service = MagicMock()
        mock_trace_svc_cls.return_value = mock_service

        mock_query_result = MagicMock()
        mock_query_result.traces = []
        mock_service.query_traces.return_value = mock_query_result

        result = json.loads(summarize_evaluation("ent", "proj"))

        assert result["evaluations"] == []
        assert "No Evaluation.evaluate" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.TraceService")
    def test_with_eval_traces(self, mock_trace_svc_cls, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api_key.return_value = "fake-key"

        mock_service = MagicMock()
        mock_trace_svc_cls.return_value = mock_service

        eval_trace = {
            "id": "eval-abc",
            "trace_id": "trace-123",
            "op_name": "Evaluation.evaluate",
            "started_at": "2026-01-01T00:00:00Z",
            "summary": {"weave": {"correctness": {"true_fraction": 0.9}}},
        }
        child_1 = {"summary": {"weave": {"status": "success"}}}
        child_2 = {"exception": "TimeoutError"}

        eval_result = MagicMock()
        eval_result.traces = [eval_trace]

        child_result = MagicMock()
        child_result.traces = [child_1, child_2]

        mock_service.query_traces.side_effect = [eval_result, child_result]

        result = json.loads(summarize_evaluation("ent", "proj"))

        assert result["count"] == 1
        ev = result["evaluations"][0]
        assert ev["eval_id"] == "eval-abc"
        assert ev["total_predictions"] == 2
        assert ev["errors"] == 1
        assert ev["successes"] == 1
        assert ev["scores"]["correctness"]["true_fraction"] == 0.9

    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.TraceService")
    def test_query_failure_returns_error(self, mock_trace_svc_cls, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api_key.return_value = "fake-key"

        mock_service = MagicMock()
        mock_trace_svc_cls.return_value = mock_service
        mock_service.query_traces.side_effect = Exception("Connection refused")

        result = json.loads(summarize_evaluation("ent", "proj"))

        assert result["error"] == "evaluation_query_failed"
        assert "Connection refused" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.summarize_evaluation.TraceService")
    def test_project_path_in_response(self, mock_trace_svc_cls, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api_key.return_value = "fake-key"

        mock_service = MagicMock()
        mock_trace_svc_cls.return_value = mock_service

        eval_trace = {
            "id": "e1",
            "trace_id": "t1",
            "op_name": "Evaluation.evaluate",
            "started_at": "",
            "summary": {},
        }
        eval_result = MagicMock()
        eval_result.traces = [eval_trace]
        child_result = MagicMock()
        child_result.traces = []
        mock_service.query_traces.side_effect = [eval_result, child_result]

        result = json.loads(summarize_evaluation("my-team", "my-project"))

        assert result["project"] == "my-team/my-project"
