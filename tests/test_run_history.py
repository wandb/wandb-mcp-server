"""Tests for the get_run_history_tool."""

import asyncio
import json
import weakref
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import wandb
from mcp.server.fastmcp import FastMCP
from wandb.apis.public.history import HistoryScan
from wandb.proto import wandb_api_pb2 as pb

from wandb_mcp_server.mcp_tools.run_history import (
    GET_RUN_HISTORY_TOOL_DESCRIPTION,
    MAX_HISTORY_ROWS,
    get_run_history,
)
from wandb_mcp_server.server import register_tools
from wandb_mcp_server.wandb_selective_reads import SelectiveReadUnavailable


class _SampledHistoryServiceApi:
    """Fake the W&B ServiceApi while keeping sampledHistory's real envelope."""

    def __init__(self, series):
        self.series = series
        self.calls = []

    def execute_graphql(self, query, variables=None):
        self.calls.append((query, dict(variables or {})))
        return {"project": {"run": {"sampledHistory": self.series}}}


def _api_with_run_and_sampled_series(run, series):
    service_api = _SampledHistoryServiceApi(series)
    api = MagicMock()
    api.run.return_value = run
    api._service_api = service_api
    return api, service_api


class _HistoryScanService:
    """Fake the service transport below W&B 0.28's real HistoryScan."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.init_calls = []
        self.cleanup_calls = 0

    def send_api_request(self, request):
        history_request = request.read_run_history_request
        kind = history_request.WhichOneof("request")
        if kind == "scan_run_history_init":
            init = history_request.scan_run_history_init
            self.init_calls.append(
                {
                    "entity": init.entity,
                    "project": init.project,
                    "run_id": init.run_id,
                    "keys": list(init.keys),
                }
            )
            return pb.ApiResponse(
                read_run_history_response=pb.ReadRunHistoryResponse(
                    scan_run_history_init=pb.ScanRunHistoryInitResponse(request_id=7)
                )
            )
        if kind == "scan_run_history_cleanup":
            self.cleanup_calls += 1
            return pb.ApiResponse(
                read_run_history_response=pb.ReadRunHistoryResponse(
                    scan_run_history_cleanup=pb.ScanRunHistoryCleanupResponse()
                )
            )

        page = history_request.scan_run_history
        self.calls.append((page.min_step, page.max_step))
        history_rows = []
        for row in self.rows:
            if not (page.min_step <= row.get("_step", -1) < page.max_step):
                continue
            history_rows.append(
                pb.HistoryRow(
                    history_items=[
                        pb.ParquetHistoryItem(key=key, value_json=json.dumps(value)) for key, value in row.items()
                    ]
                )
            )
        return pb.ApiResponse(
            read_run_history_response=pb.ReadRunHistoryResponse(
                run_history=pb.RunHistoryResponse(history_rows=history_rows)
            )
        )

    def finalize(self, owner, request):
        """Mirror the lifecycle hook added to W&B's public ServiceApi."""
        weakref.finalize(owner, self.send_api_request, request)


def _wandb_history_scan(rows, *, min_step, max_step, keys, page_size):
    """Construct the actual public paginator over a fake protobuf transport."""
    transport = _HistoryScanService(rows)
    run = SimpleNamespace(entity="e", project="p", id="run1")
    scan = HistoryScan(
        run,
        service_api=transport,
        min_step=min_step,
        max_step=max_step,
        keys=keys,
        page_size=page_size,
        use_cache=False,
    )
    return scan, transport


class _GuardedLargeSequence(Sequence):
    """Sequence that fails if a sanitizer tries to materialize every item."""

    def __init__(self):
        self.accesses = 0

    def __len__(self):
        return 1_000_000

    def __getitem__(self, index):
        if index >= 101:
            raise AssertionError("history sanitizer read past its bounded prefix")
        self.accesses += 1
        return index


class TestRunHistoryDescription:
    def test_has_when_to_use(self):
        assert "<when_to_use>" in GET_RUN_HISTORY_TOOL_DESCRIPTION
        assert "</when_to_use>" in GET_RUN_HISTORY_TOOL_DESCRIPTION

    def test_mentions_training_curves(self):
        desc_lower = GET_RUN_HISTORY_TOOL_DESCRIPTION.lower()
        assert "training curves" in desc_lower or "metric trends" in desc_lower

    def test_public_schema_exposes_custom_axis_and_stream_controls(self):
        mcp = FastMCP("history-schema")
        register_tools(mcp)
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

        properties = tools["get_run_history_tool"].inputSchema["properties"]

        assert {"x_axis", "target_x", "tolerance", "stream"} <= properties.keys()
        assert properties["stream"]["enum"] == ["default", "system"]
        compare_properties = tools["compare_runs_tool"].inputSchema["properties"]
        diagnose_properties = tools["diagnose_run_tool"].inputSchema["properties"]
        assert {"config_keys", "summary_keys", "x_axis"} <= compare_properties.keys()
        assert {"config_keys", "summary_keys", "x_axis", "samples"} <= diagnose_properties.keys()


class TestGetRunHistory:
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_basic_history(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "my-run"
        mock_run.lastHistoryStep = 1000
        mock_api, service_api = _api_with_run_and_sampled_series(
            mock_run,
            [
                [
                    {"_step": 0, "loss": 2.5},
                    {"_step": 100, "loss": 1.0},
                    {"_step": 200, "loss": 0.5},
                ],
                [
                    {"_step": 0, "accuracy": 0.1},
                    {"_step": 100, "accuracy": 0.5},
                    {"_step": 200, "accuracy": 0.8},
                ],
            ],
        )
        mock_api_mgr.get_api.return_value = mock_api
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("entity", "project", "abc12345", keys=["loss", "accuracy"]))

        assert result["run_id"] == "abc12345"
        assert result["run_name"] == "my-run"
        assert result["sampled_points"] == 3
        assert len(result["rows"]) == 3
        assert "loss" in result["keys_returned"]
        assert "accuracy" in result["keys_returned"]
        mock_api_mgr.get_api.assert_called_once_with("fake_key_12345678901234567890")
        mock_wandb_mod.Api.assert_not_called()
        mock_run.history.assert_not_called()
        assert len(service_api.calls) == 1

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_filters_internal_keys(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [
            {"_step": 0, "_wandb": {"internal": True}, "loss": 1.0},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        row = result["rows"][0]
        assert "_step" in row
        assert "_wandb" not in row
        assert "loss" in row

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_filters_nan_values(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [
            {"_step": 0, "loss": float("nan"), "accuracy": 0.5},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        row = result["rows"][0]
        assert "loss" not in row
        assert "accuracy" in row

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_clamps_samples(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = []
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", samples=99999)
        call_kwargs = mock_run.history.call_args[1]
        assert call_kwargs["samples"] <= MAX_HISTORY_ROWS

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_missing_run_raises(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(side_effect=wandb.errors.CommError("Not found")))
        mock_wandb_mod.errors = wandb.errors

        with pytest.raises(ValueError, match="Run not found"):
            get_run_history("e", "p", "nonexistent")

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    def test_no_api_key_raises(self, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = None

        with pytest.raises(ValueError, match="API key"):
            get_run_history("e", "p", "run1")

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_step_range_uses_scan_history(self, mock_wandb_mod, mock_api_mgr):
        """min_step/max_step must use scan_history, not history (which doesn't support them)."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 1000
        mock_run.scan_history.return_value = [
            {"_step": 50, "loss": 1.5},
            {"_step": 100, "loss": 1.0},
            {"_step": 150, "loss": 0.7},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=50, max_step=200))

        mock_run.scan_history.assert_called_once()
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["min_step"] == 50
        assert call_kwargs["max_step"] == 201
        mock_run.history.assert_not_called()
        assert result["sampled_points"] == 3

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_no_step_range_uses_history(self, mock_wandb_mod, mock_api_mgr):
        """Without min_step/max_step, should use history() for sampled data."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 100
        mock_run.history.return_value = [{"_step": 0, "loss": 1.0}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", samples=100)

        mock_run.history.assert_called_once()
        mock_run.scan_history.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_samples_large_result(self, mock_wandb_mod, mock_api_mgr):
        """scan_history results should be client-side sampled to match the samples parameter."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10000
        mock_run.scan_history.return_value = [{"_step": i, "loss": float(i)} for i in range(5000)]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=0, samples=500))

        assert result["sampled_points"] <= 500

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_step_range_samples_across_full_window(self, mock_wandb_mod, mock_api_mgr):
        """Step-range sampling should cover the full requested range, not just a prefix."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 99
        rows = [{"_step": i, "loss": float(i)} for i in range(100)]
        mock_run.scan_history.return_value = rows
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=0, max_step=99, samples=5))
        steps = [row["_step"] for row in result["rows"]]

        assert len(steps) == 5
        assert steps == sorted(steps)
        assert min(steps) >= 0
        assert max(steps) <= 99
        assert len(set(steps)) == 5

    # -- Additional edge-case tests ------------------------------------------

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_reservoir_sampling_covers_full_range(self, mock_wandb_mod, mock_api_mgr):
        """Reservoir sampling should draw from the entire scan window, not
        just the first N rows. Over multiple runs, the max step in the sample
        should reach into the tail of the data."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        total_rows = 10_000
        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = total_rows - 1
        mock_run.scan_history.return_value = [{"_step": i, "loss": float(i)} for i in range(total_rows)]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        max_steps_seen = []
        for _ in range(10):
            result = json.loads(get_run_history("e", "p", "run1", min_step=0, samples=100))
            steps = [row["_step"] for row in result["rows"]]
            max_steps_seen.append(max(steps))

        mean_max = sum(max_steps_seen) / len(max_steps_seen)
        assert mean_max > total_rows * 0.8, (
            f"Reservoir sampling mean max step {mean_max:.0f} is too low — "
            f"expected > {total_rows * 0.8:.0f} for {total_rows} total rows"
        )

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_only_min_step(self, mock_wandb_mod, mock_api_mgr):
        """Setting only min_step (no max_step) should use scan_history."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 500
        mock_run.scan_history.return_value = [{"_step": 100, "loss": 1.0}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", min_step=100)

        mock_run.scan_history.assert_called_once()
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["min_step"] == 100
        assert "max_step" not in call_kwargs
        mock_run.history.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_only_max_step(self, mock_wandb_mod, mock_api_mgr):
        """Setting only max_step (no min_step) should use scan_history."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 500
        mock_run.scan_history.return_value = [{"_step": 50, "loss": 1.0}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", max_step=200)

        mock_run.scan_history.assert_called_once()
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["max_step"] == 201
        assert "min_step" not in call_kwargs

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_sparse_metrics_both_keys_returned(self, mock_wandb_mod, mock_api_mgr):
        """Rows with disjoint metric sets should still report all keys."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 3
        mock_run.history.return_value = [
            {"_step": 0, "loss": 1.0},
            {"_step": 1, "accuracy": 0.5},
            {"_step": 2, "loss": 0.5},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        assert "loss" in result["keys_returned"]
        assert "accuracy" in result["keys_returned"]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_response_shape(self, mock_wandb_mod, mock_api_mgr):
        """Response must always contain the documented top-level keys."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [{"_step": 0, "loss": 1.0}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        for key in ("rows", "run_id", "run_name", "total_steps", "sampled_points", "keys_returned"):
            assert key in result, f"Missing required key: {key}"

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_empty_history(self, mock_wandb_mod, mock_api_mgr):
        """Run with no history rows should return gracefully."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "empty-run"
        mock_run.lastHistoryStep = 0
        mock_run.history.return_value = []
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        assert result["rows"] == []
        assert result["sampled_points"] == 0
        assert result["keys_returned"] == []


class TestCrossCadenceHistory:
    """Regression coverage for metrics logged in separate wandb.log calls (#136)."""

    @staticmethod
    def _configure(mock_wandb_mod, mock_api_mgr, series, *, last_step=1000):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock()
        run.name = "cross-cadence"
        run.lastHistoryStep = last_step
        api, service_api = _api_with_run_and_sampled_series(run, series)
        mock_api_mgr.get_api.return_value = api
        mock_wandb_mod.errors = wandb.errors
        return run, service_api

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_different_cadences_are_outer_joined(self, mock_wandb_mod, mock_api_mgr):
        run, service_api = self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [{"_step": 0, "loss": 1.0}, {"_step": 2, "loss": 0.5}],
                [{"_step": 1, "eval/loss": 0.9}, {"_step": 3, "eval/loss": 0.4}],
            ],
            last_step=3,
        )

        result = json.loads(get_run_history("e", "p", "run1", keys=["loss", "eval/loss"], samples=500))

        assert result["rows"] == [
            {"_step": 0, "loss": 1.0},
            {"_step": 1, "eval/loss": 0.9},
            {"_step": 2, "loss": 0.5},
            {"_step": 3, "eval/loss": 0.4},
        ]
        assert result["requested_keys"] == ["loss", "eval/loss"]
        assert result["join"] == "outer"
        assert result["matching_rows"] == 4
        assert result["key_row_counts"] == {
            "loss": {"observed": 2, "returned": 2},
            "eval/loss": {"observed": 2, "returned": 2},
        }
        assert result["missing_keys"] == []
        assert result["keys_omitted_by_limits"] == []
        assert result["key_counts_exact"] is False
        mock_api_mgr.get_api.return_value.run.assert_called_once_with("e/p/run1")
        run.history.assert_not_called()
        assert len(service_api.calls) == 1

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_same_and_partially_overlapping_cadences_align_by_step(
        self,
        mock_wandb_mod,
        mock_api_mgr,
    ):
        self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [
                    {"_step": 0, "loss": 1.0},
                    {"_step": 1, "loss": 0.8},
                    {"_step": 2, "loss": 0.6},
                ],
                [
                    {"_step": 1, "accuracy": 0.5},
                    {"_step": 2, "accuracy": 0.7},
                    {"_step": 3, "accuracy": 0.9},
                ],
            ],
            last_step=3,
        )

        result = json.loads(get_run_history("e", "p", "run1", keys=["loss", "accuracy"], samples=10))

        assert result["rows"] == [
            {"_step": 0, "loss": 1.0},
            {"_step": 1, "loss": 0.8, "accuracy": 0.5},
            {"_step": 2, "loss": 0.6, "accuracy": 0.7},
            {"_step": 3, "accuracy": 0.9},
        ]
        assert result["matching_rows"] == 4

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_duplicate_axis_values_preserve_occurrence_order(self, mock_wandb_mod, mock_api_mgr):
        self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [{"_step": 0, "loss": 1.0}, {"_step": 0, "loss": 0.8}],
                [{"_step": 0, "accuracy": 0.4}, {"_step": 0, "accuracy": 0.6}],
            ],
            last_step=0,
        )

        result = json.loads(get_run_history("e", "p", "run1", keys=["loss", "accuracy"], samples=10))

        assert result["rows"] == [
            {"_step": 0, "loss": 1.0, "accuracy": 0.4},
            {"_step": 0, "loss": 0.8, "accuracy": 0.6},
        ]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_custom_axis_is_the_fallback_join_key(self, mock_wandb_mod, mock_api_mgr):
        self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5}],
                [{"epoch": 1, "accuracy": 0.4}, {"epoch": 3, "accuracy": 0.8}],
            ],
            last_step=3,
        )

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["loss", "accuracy"],
                samples=10,
                x_axis="epoch",
            )
        )

        assert result["rows"] == [
            {"epoch": 1, "loss": 1.0, "accuracy": 0.4},
            {"epoch": 2, "loss": 0.5},
            {"epoch": 3, "accuracy": 0.8},
        ]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_custom_axis_sorts_when_step_is_present_but_unobserved(
        self,
        mock_wandb_mod,
        mock_api_mgr,
    ):
        self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [
                    {"_step": None, "epoch": 2, "loss": 0.5},
                    {"_step": None, "epoch": 1, "loss": 1.0},
                ],
                [
                    {"_step": float("nan"), "epoch": 2, "accuracy": 0.8},
                    {"_step": float("nan"), "epoch": 1, "accuracy": 0.4},
                ],
            ],
            last_step=2,
        )

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["loss", "accuracy"],
                samples=10,
                x_axis="epoch",
            )
        )

        assert result["rows"] == [
            {"epoch": 1, "loss": 1.0, "accuracy": 0.4},
            {"epoch": 2, "loss": 0.5, "accuracy": 0.8},
        ]
        assert result["coverage"] == {
            "first_x": 1.0,
            "last_x": 2.0,
            "requested_min_step": None,
            "requested_max_step": None,
            "target_x": None,
            "tolerance": None,
        }

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_duplicate_and_missing_keys_have_truthful_metadata(self, mock_wandb_mod, mock_api_mgr):
        _, service_api = self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [[{"_step": 0, "loss": 1.0}], []],
            last_step=0,
        )

        result = json.loads(get_run_history("e", "p", "run1", keys=["loss", "missing", "loss"], samples=10))

        assert result["requested_keys"] == ["loss", "missing"]
        assert result["rows"] == [{"_step": 0, "loss": 1.0}]
        assert result["missing_keys"] == ["missing"]
        assert result["key_row_counts"] == {
            "loss": {"observed": 1, "returned": 1},
            "missing": {"observed": 0, "returned": 0},
        }
        assert len(service_api.calls) == 1
        assert len(service_api.calls[0][1]["specs"]) == 2

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_tiny_budget_reports_the_unrepresented_key(self, mock_wandb_mod, mock_api_mgr):
        self._configure(
            mock_wandb_mod,
            mock_api_mgr,
            [
                [{"_step": 0, "loss": 1.0}],
                [{"_step": 1, "accuracy": 0.5}],
            ],
            last_step=1,
        )

        result = json.loads(get_run_history("e", "p", "run1", keys=["loss", "accuracy"], samples=1))

        assert len(result["rows"]) == 1
        assert result["matching_rows"] == 2
        assert len(result["keys_omitted_by_limits"]) == 1
        assert set(result["keys_omitted_by_limits"]) < {"loss", "accuracy"}
        assert result["key_counts_exact"] is False

    @pytest.mark.parametrize("status_code", [429, 503])
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_overload_is_mapped_to_retryable_server_busy(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        status_code,
    ):
        from wandb_mcp_server.api_client import WandBServerBusy

        run, service_api = self._configure(mock_wandb_mod, mock_api_mgr, [[], []])
        error = RuntimeError("W&B capacity exhausted")
        error.response = SimpleNamespace(
            status_code=status_code,
            headers={"Retry-After": "2"},
            reason="capacity exhausted",
            text="capacity exhausted",
        )
        service_api.execute_graphql = MagicMock(side_effect=error)

        with pytest.raises(WandBServerBusy) as caught:
            get_run_history("e", "p", "run1", keys=["loss", "eval/loss"])

        assert caught.value.status_code == status_code
        assert caught.value.retry_after_ms == 2_000
        assert service_api.execute_graphql.call_count == 1
        run.history.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("HTTP 401 unauthorized"),
            TimeoutError("history request timed out"),
        ],
    )
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_authentication_and_timeout_fail_without_fallback(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        error,
    ):
        run, service_api = self._configure(mock_wandb_mod, mock_api_mgr, [[], []])
        service_api.execute_graphql = MagicMock(side_effect=error)

        with pytest.raises(ValueError, match="Failed to fetch history"):
            get_run_history("e", "p", "run1", keys=["loss", "eval/loss"])

        assert service_api.execute_graphql.call_count == 1
        run.history.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_cancellation_propagates_without_fallback(self, mock_wandb_mod, mock_api_mgr):
        run, service_api = self._configure(mock_wandb_mod, mock_api_mgr, [[], []])
        service_api.execute_graphql = MagicMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            get_run_history("e", "p", "run1", keys=["loss", "eval/loss"])

        assert service_api.execute_graphql.call_count == 1
        run.history.assert_not_called()


class TestSparseRangeHistory:
    @staticmethod
    def _configure(mock_wandb_mod, mock_api_mgr, rows, *, last_step):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock()
        run.name = "sparse-range"
        run.lastHistoryStep = last_step
        scan_state = {}

        def scan_history(**kwargs):
            scan, transport = _wandb_history_scan(
                rows,
                min_step=kwargs.get("min_step", 0),
                max_step=kwargs.get("max_step", last_step + 1),
                keys=kwargs.get("keys"),
                page_size=kwargs.get("page_size", 1_000),
            )
            scan_state["scan"] = scan
            scan_state["transport"] = transport
            return scan

        run.scan_history.side_effect = scan_history
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors
        return run, scan_state

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_range_filters_step_only_shells_before_sampling(self, mock_wandb_mod, mock_api_mgr):
        rows = [
            *({"_step": step, "loss": float(step)} for step in range(200)),
            *({"_step": step, "eval/loss": float(step)} for step in range(200, 400)),
            *({"_step": step} for step in range(400, 600)),
        ]
        run, scan_state = self._configure(mock_wandb_mod, mock_api_mgr, rows, last_step=599)

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["loss", "eval/loss"],
                samples=500,
                min_step=0,
                max_step=599,
            )
        )

        assert result["sampled_points"] == 400
        assert result["matching_rows"] == 400
        assert result["rows_scanned"] == 600
        assert all("loss" in row or "eval/loss" in row for row in result["rows"])
        assert result["key_row_counts"] == {
            "loss": {"observed": 200, "returned": 200},
            "eval/loss": {"observed": 200, "returned": 200},
        }
        assert result["key_counts_exact"] is True
        assert result["keys_omitted_by_limits"] == []
        assert run.scan_history.call_args.kwargs["keys"] == ["_step", "loss", "eval/loss"]
        assert scan_state["transport"].init_calls == [
            {
                "entity": "e",
                "project": "p",
                "run_id": "run1",
                "keys": ["_step", "loss", "eval/loss"],
            }
        ]
        assert scan_state["transport"].calls == [(0, 600)]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_sparse_series_survives_dense_series_sampling(self, mock_wandb_mod, mock_api_mgr):
        rows = []
        for step in range(1_000):
            row = {"_step": step, "loss": float(step)}
            if step in {250, 750}:
                row["rare/accuracy"] = step / 1_000
            rows.append(row)
        _, scan_state = self._configure(mock_wandb_mod, mock_api_mgr, rows, last_step=999)

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["loss", "rare/accuracy"],
                samples=20,
                min_step=0,
                max_step=999,
            )
        )

        assert result["sampled_points"] == 20
        assert [row["_step"] for row in result["rows"] if "rare/accuracy" in row] == [250, 750]
        assert result["key_row_counts"]["rare/accuracy"] == {"observed": 2, "returned": 2}
        assert result["keys_omitted_by_limits"] == []
        assert scan_state["transport"].calls == [(0, 1_000)]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_duplicate_internal_steps_make_range_counts_inexact(
        self,
        mock_wandb_mod,
        mock_api_mgr,
    ):
        rows = [
            {"_step": 0, "loss": 1.0},
            {"_step": 0, "eval/loss": 0.9},
            {"_step": 1, "loss": 0.8},
        ]
        _, scan_state = self._configure(mock_wandb_mod, mock_api_mgr, rows, last_step=1)

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["loss", "eval/loss"],
                samples=10,
                min_step=0,
                max_step=1,
            )
        )

        assert result["rows"] == rows[:2]
        assert result["rows_scanned"] == 3
        assert result["source_truncated"] is True
        assert result["key_counts_exact"] is False
        assert result["truncated"] is True
        assert "retained prefix" in result["compatibility_caveat"]
        assert scan_state["transport"].calls == [(0, 2)]

    def test_complete_sparse_series_is_retained_when_it_fits(self):
        from wandb_mcp_server.mcp_tools.run_history import _key_aware_sample

        rows = [
            *({"_step": step, "dense": float(step)} for step in range(100)),
            *({"_step": 100 + step, "sparse": float(step)} for step in range(10)),
        ]

        sampled = _key_aware_sample(rows, 11, ["dense", "sparse"])

        assert len(sampled) == 11
        assert sum("sparse" in row for row in sampled) == 10
        assert sum("dense" in row for row in sampled) == 1
        assert [row["_step"] for row in sampled] == sorted(row["_step"] for row in sampled)


class TestHistoryTruncation:
    """Tests for row-budget enforcement on history responses (M1)."""

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_large_response_truncated(self, mock_wandb_mod, mock_api_mgr):
        """History exceeding token budget should be downsampled."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        rows = [{"_step": i, "loss": 1.0 / (i + 1), "acc": i * 0.01, "lr": 0.001} for i in range(2000)]
        mock_run = MagicMock()
        mock_run.name = "big-run"
        mock_run.lastHistoryStep = 2000
        mock_run.history.return_value = rows
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        with patch("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 1_000):
            result = json.loads(get_run_history("e", "p", "run1", samples=2000))
        assert result["sampled_points"] < 2000
        assert "truncation_note" in result
        assert len(result["rows"]) > 0

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_small_response_not_truncated(self, mock_wandb_mod, mock_api_mgr):
        """History under budget passes through unchanged."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        rows = [{"_step": i, "loss": 0.5} for i in range(10)]
        mock_run = MagicMock()
        mock_run.name = "small-run"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = rows
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", samples=10))
        assert result["sampled_points"] == 10
        assert "truncation_note" not in result

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_hosted_history_samples_clamped(self, mock_wandb_mod, mock_api_mgr):
        """Hosted mode clamps requested samples to the configured hosted limit."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "hosted-run"
        mock_run.lastHistoryStep = 2000
        mock_run.history.return_value = [{"_step": i, "loss": 0.5} for i in range(5)]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        with (
            patch("wandb_mcp_server.mcp_tools.run_history.MCP_HOSTED_MODE", True),
            patch("wandb_mcp_server.mcp_tools.run_history.MCP_MAX_HISTORY_SAMPLES", 5),
        ):
            result = json.loads(get_run_history("e", "p", "run1", keys=["loss"], samples=100))

        mock_run.history.assert_called_once()
        assert mock_run.history.call_args.kwargs["samples"] == 5
        assert "hosted_limit_note" in result

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    def test_hosted_history_requires_explicit_keys(self, mock_api_mgr):
        with patch("wandb_mcp_server.mcp_tools.run_history.MCP_HOSTED_MODE", True):
            with pytest.raises(ValueError, match="explicit history keys"):
                get_run_history("e", "p", "run1")
        mock_api_mgr.get_api_key.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    def test_hosted_history_rejects_wide_step_range(self, mock_api_mgr):
        with (
            patch("wandb_mcp_server.mcp_tools.run_history.MCP_HOSTED_MODE", True),
            patch("wandb_mcp_server.mcp_tools.run_history.MCP_MAX_HISTORY_RANGE_STEPS", 5000),
        ):
            with pytest.raises(ValueError, match="cannot exceed 5000"):
                get_run_history("e", "p", "run1", keys=["loss"], min_step=0, max_step=5000)
        mock_api_mgr.get_api_key.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_truncation_preserves_step_ordering(self, mock_wandb_mod, mock_api_mgr):
        """Truncated rows must remain sorted by _step."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        rows = [{"_step": i, "val": i * 0.1} for i in range(2000)]
        mock_run = MagicMock()
        mock_run.name = "ordered-run"
        mock_run.lastHistoryStep = 2000
        mock_run.history.return_value = rows
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        with patch("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 1_000):
            result = json.loads(get_run_history("e", "p", "run1", samples=2000))
        steps = [r["_step"] for r in result["rows"]]
        assert steps == sorted(steps)

    def test_enforce_row_budget_noop_when_small(self):
        """_enforce_row_budget returns rows unchanged when under budget."""
        from wandb_mcp_server.mcp_tools.run_history import _enforce_row_budget

        rows = [{"_step": i, "v": 1.0} for i in range(5)]
        result = _enforce_row_budget(rows, budget_chars=100000)
        assert len(result) == 5

    def test_history_value_sanitizer_bounds_cycles_depth_and_large_sequences(self):
        from wandb_mcp_server.mcp_tools.run_history import _clean_history_rows

        cycle = []
        cycle.append(cycle)
        deep = {"value": 1}
        for _ in range(10):
            deep = {"nested": deep}
        large = _GuardedLargeSequence()

        rows = _clean_history_rows(
            [
                {
                    "_step": 0,
                    "cycle": cycle,
                    "deep": deep,
                    "large": large,
                    "nan": float("nan"),
                    "positive_infinity": float("inf"),
                    "negative_infinity": float("-inf"),
                }
            ]
        )

        serialized = json.dumps(rows, allow_nan=False)
        assert "<cycle>" in serialized
        assert "<max-depth>" in serialized
        assert "list entries omitted" in serialized
        assert large.accesses == 101
        assert "nan" not in rows[0]
        assert "positive_infinity" not in rows[0]
        assert "negative_infinity" not in rows[0]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_complete_response_envelope_respects_token_budget(self, mock_wandb_mod, mock_api_mgr):
        from wandb_mcp_server.trace_utils import count_tokens_conservative

        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        mock_run = MagicMock(name="large-value-run")
        mock_run.name = "large-value-run"
        mock_run.lastHistoryStep = 0
        mock_run.history.return_value = [{"_step": 0, "loss": {"payload": "x" * 100_000}}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        with patch("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 1_000):
            response = get_run_history("e", "p", "run1", keys=["loss"], samples=1)

        result = json.loads(response)
        assert count_tokens_conservative(response) <= 1_000
        assert result["truncated"] is True
        assert result["sampled_points"] == 1
        assert result["rows"] == [{"_step": 0, "loss": "<value-truncated>"}]
        assert result["keys_omitted_by_limits"] == []
        assert result["key_row_counts"]["loss"] == {"observed": 1, "returned": 1}

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_metadata_too_large_returns_bounded_error(self, mock_wandb_mod, mock_api_mgr):
        from wandb_mcp_server.trace_utils import count_tokens_conservative

        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        mock_run = MagicMock(name="wide-run")
        mock_run.name = "wide-run"
        mock_run.lastHistoryStep = 0
        keys = [f"metric_{index}_" + "x" * 100 for index in range(20)]
        api, _ = _api_with_run_and_sampled_series(
            mock_run,
            [[{"_step": 0, key: float(index)}] for index, key in enumerate(keys)],
        )
        mock_api_mgr.get_api.return_value = api
        mock_wandb_mod.errors = wandb.errors

        with patch("wandb_mcp_server.config.MAX_RESPONSE_TOKENS", 50):
            response = get_run_history("e", "p", "run1", keys=keys, samples=1)

        assert count_tokens_conservative(response) <= 50
        assert json.loads(response)["error"] == "response_too_large"


class TestTieredStepRangeFetch:
    """Tests for the tiered step-range strategy: scan_history -> history fallback."""

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_tried_first(self, mock_wandb_mod, mock_api_mgr):
        """scan_history is the first strategy attempted for step-range queries."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        scan_rows = [{"_step": i, "loss": 0.5} for i in range(50)]
        mock_run = MagicMock()
        mock_run.name = "scan-run"
        mock_run.lastHistoryStep = 100
        mock_run.scan_history.return_value = iter(scan_rows)
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=0, max_step=100))
        mock_run.scan_history.assert_called_once()
        mock_run.history.assert_not_called()
        assert result["sampled_points"] == 50

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_fallback_to_history_when_scan_empty(self, mock_wandb_mod, mock_api_mgr):
        """When scan returns empty with lastHistoryStep<=0, falls back to history()."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        history_rows = [{"_step": i, "acc": 0.9} for i in range(10)]
        mock_run = MagicMock()
        mock_run.name = "broken-step-run"
        mock_run.lastHistoryStep = -1
        mock_run.scan_history.return_value = iter([])
        mock_run.history.return_value = history_rows
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=0, max_step=100))
        mock_run.scan_history.assert_called_once()
        mock_run.history.assert_called_once()
        assert result["sampled_points"] == 10

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_passes_keys_and_range(self, mock_wandb_mod, mock_api_mgr):
        """scan_history receives keys, min_step, max_step."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "params-run"
        mock_run.lastHistoryStep = 200
        mock_run.scan_history.return_value = iter([{"_step": 10, "loss": 0.5}])
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", keys=["loss"], min_step=10, max_step=100)
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["keys"] == ["_step", "loss"]
        assert call_kwargs["min_step"] == 10
        assert call_kwargs["max_step"] == 101

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_no_history_fallback_when_last_step_positive(self, mock_wandb_mod, mock_api_mgr):
        """When lastHistoryStep > 0 and scan returns empty, do NOT fall back to history()."""
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "normal-run"
        mock_run.lastHistoryStep = 1000
        mock_run.scan_history.return_value = iter([])
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=5000, max_step=6000))
        mock_run.history.assert_not_called()
        assert result["sampled_points"] == 0


class TestCustomAxisHistory:
    @patch(
        "wandb_mcp_server.mcp_tools.run_history.fetch_metric_value_steps",
        side_effect=SelectiveReadUnavailable("unsupported metric"),
    )
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_compatibility_scan_stops_at_first_exact_match(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        _mock_steps,
    ):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        rows_yielded = 0

        def history_rows():
            nonlocal rows_yielded
            rows_yielded += 1
            yield {"_step": 0, "validation/step": 1000.0, "validation/loss": 0.2}
            rows_yielded += 1
            yield {"_step": 1, "validation/step": 1001.0, "validation/loss": 0.1}

        run = MagicMock()
        run.name = "custom-axis"
        run.lastHistoryStep = 100
        run.scan_history.return_value = history_rows()
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["validation/loss"],
                x_axis="validation/step",
                target_x=1000,
            )
        )

        assert result["exact"] is True
        assert result["rows_scanned"] == 1
        assert result["retrieval_method"] == "sdk_bounded_compatibility_scan"
        assert rows_yielded == 1

    @patch(
        "wandb_mcp_server.mcp_tools.run_history.fetch_metric_value_steps",
        side_effect=SelectiveReadUnavailable("unsupported metric"),
    )
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_compatibility_tolerance_scan_is_bounded_and_selects_nearest(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        _mock_steps,
    ):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        rows_yielded = 0

        def history_rows():
            nonlocal rows_yielded
            for row in (
                {"_step": 0, "validation/step": 999.6},
                {"_step": 1, "validation/step": 1000.2},
                {"_step": 2, "validation/step": 999.9},
                {"_step": 3, "validation/step": 1000.01},
            ):
                rows_yielded += 1
                yield row

        run = MagicMock()
        run.name = "custom-axis"
        run.lastHistoryStep = 100
        run.scan_history.return_value = history_rows()
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        with patch("wandb_mcp_server.mcp_tools.run_history.MCP_MAX_HISTORY_RANGE_STEPS", 3):
            result = json.loads(
                get_run_history(
                    "e",
                    "p",
                    "run1",
                    keys=["validation/loss"],
                    x_axis="validation/step",
                    target_x=1000,
                    tolerance=0.5,
                )
            )

        assert result["exact"] is False
        assert result["rows_scanned"] == 3
        assert result["rows"] == [{"_step": 2, "validation/step": 999.9}]
        assert rows_yielded == 3

    @patch("wandb_mcp_server.mcp_tools.run_history.fetch_metric_value_steps", return_value=[42])
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_exact_custom_axis_target_is_verified_from_history(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        mock_steps,
    ):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock(name="run")
        run.name = "custom-axis"
        run.lastHistoryStep = 100
        run.scan_history.return_value = [{"_step": 42, "validation/step": 1000.0, "validation/loss": 0.2}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["validation/loss"],
                x_axis="validation/step",
                target_x=1000,
            )
        )

        assert result["exact"] is True
        assert result["sampled"] is False
        assert result["retrieval_method"] == "steps_for_metric_values"
        assert result["rows"] == [{"_step": 42, "validation/step": 1000.0, "validation/loss": 0.2}]
        assert result["rows_scanned"] == 1
        mock_steps.assert_called_once()

    @patch("wandb_mcp_server.mcp_tools.run_history.fetch_metric_value_steps", return_value=[42])
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_missing_exact_custom_axis_target_is_honest(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        _mock_steps,
    ):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock()
        run.name = "custom-axis"
        run.scan_history.return_value = [{"_step": 42, "validation/step": 999.5}]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["validation/loss"],
                x_axis="validation/step",
                target_x=1000,
            )
        )

        assert result["error"] == "target_not_logged"
        assert result["exact"] is False

    @patch("wandb_mcp_server.mcp_tools.run_history.fetch_metric_value_steps", return_value=[42])
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_tolerance_permits_bounded_nearest_refinement(
        self,
        mock_wandb_mod,
        mock_api_mgr,
        _mock_steps,
    ):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock()
        run.name = "custom-axis"
        run.lastHistoryStep = 100
        run.scan_history.return_value = [
            {"_step": 40, "validation/step": 997.0},
            {"_step": 41, "validation/step": 999.75, "validation/loss": 0.21},
            {"_step": 42, "validation/step": 1001.0},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["validation/loss"],
                x_axis="validation/step",
                target_x=1000,
                tolerance=0.5,
            )
        )

        assert result["exact"] is False
        assert result["sampled"] is False
        assert result["rows"][0]["validation/step"] == 999.75
        assert result["rows_scanned"] == 3
        scan_kwargs = run.scan_history.call_args.kwargs
        assert scan_kwargs["min_step"] == 40
        assert scan_kwargs["max_step"] == 45

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_system_stream_uses_public_sdk_sampling(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"
        run = MagicMock()
        run.name = "system"
        run.lastHistoryStep = 10
        run.history.return_value = [
            {"_timestamp": 10, "system.cpu": 40.0, "system.memory": 75.0},
            {"_timestamp": 11, "system.memory": 76.0},
        ]
        mock_api_mgr.get_api.return_value = MagicMock(run=MagicMock(return_value=run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(
            get_run_history(
                "e",
                "p",
                "run1",
                keys=["system.cpu"],
                x_axis="_timestamp",
                stream="system",
                samples=10,
            )
        )

        assert result["stream"] == "system"
        assert result["retrieval_method"] == "sdk_sample"
        assert result["sampled"] is True
        assert result["rows"] == [{"_timestamp": 10, "system.cpu": 40.0}]
        assert result["matching_rows"] == 1
        assert result["key_row_counts"] == {"system.cpu": {"observed": 1, "returned": 1}}
        assert run.history.call_args.kwargs["stream"] == "system"
        assert run.history.call_args.kwargs["x_axis"] == "_timestamp"
        assert "keys" not in run.history.call_args.kwargs
