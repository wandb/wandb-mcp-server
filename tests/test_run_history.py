"""Tests for the get_run_history_tool."""

import json
from unittest.mock import MagicMock, patch

import pytest
import wandb

from wandb_mcp_server.mcp_tools.run_history import (
    GET_RUN_HISTORY_TOOL_DESCRIPTION,
    MAX_HISTORY_ROWS,
    get_run_history,
)


class TestRunHistoryDescription:
    def test_has_when_to_use(self):
        assert "<when_to_use>" in GET_RUN_HISTORY_TOOL_DESCRIPTION
        assert "</when_to_use>" in GET_RUN_HISTORY_TOOL_DESCRIPTION

    def test_mentions_training_curves(self):
        desc_lower = GET_RUN_HISTORY_TOOL_DESCRIPTION.lower()
        assert "training curves" in desc_lower or "metric trends" in desc_lower


class TestGetRunHistory:
    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_basic_history(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "my-run"
        mock_run.lastHistoryStep = 1000
        mock_run.history.return_value = [
            {"_step": 0, "loss": 2.5, "accuracy": 0.1},
            {"_step": 100, "loss": 1.0, "accuracy": 0.5},
            {"_step": 200, "loss": 0.5, "accuracy": 0.8},
        ]

        mock_api = MagicMock()
        mock_api.run.return_value = mock_run
        mock_wandb_mod.Api.return_value = mock_api
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("entity", "project", "abc12345", keys=["loss", "accuracy"]))

        assert result["run_id"] == "abc12345"
        assert result["run_name"] == "my-run"
        assert result["sampled_points"] == 3
        assert len(result["rows"]) == 3
        assert "loss" in result["keys_returned"]
        assert "accuracy" in result["keys_returned"]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_filters_internal_keys(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [
            {"_step": 0, "_wandb": {"internal": True}, "loss": 1.0},
        ]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        row = result["rows"][0]
        assert "_step" in row
        assert "_wandb" not in row
        assert "loss" in row

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_filters_nan_values(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [
            {"_step": 0, "loss": float("nan"), "accuracy": 0.5},
        ]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        row = result["rows"][0]
        assert "loss" not in row
        assert "accuracy" in row

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_clamps_samples(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = []
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", samples=99999)
        call_kwargs = mock_run.history.call_args[1]
        assert call_kwargs["samples"] <= MAX_HISTORY_ROWS

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_missing_run_raises(self, mock_wandb_mod, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(side_effect=wandb.errors.CommError("Not found")))
        mock_wandb_mod.errors = wandb.errors

        with pytest.raises(ValueError, match="Run not found"):
            get_run_history("e", "p", "nonexistent")

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    def test_no_api_key_raises(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = None

        with pytest.raises(ValueError, match="API key"):
            get_run_history("e", "p", "run1")

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_step_range_uses_scan_history(self, mock_wandb_mod, mock_api_mgr):
        """min_step/max_step must use scan_history, not history (which doesn't support them)."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 1000
        mock_run.scan_history.return_value = [
            {"_step": 50, "loss": 1.5},
            {"_step": 100, "loss": 1.0},
            {"_step": 150, "loss": 0.7},
        ]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=50, max_step=200))

        mock_run.scan_history.assert_called_once()
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["min_step"] == 50
        assert call_kwargs["max_step"] == 200
        mock_run.history.assert_not_called()
        assert result["sampled_points"] == 3

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_no_step_range_uses_history(self, mock_wandb_mod, mock_api_mgr):
        """Without min_step/max_step, should use history() for sampled data."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 100
        mock_run.history.return_value = [{"_step": 0, "loss": 1.0}]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", samples=100)

        mock_run.history.assert_called_once()
        mock_run.scan_history.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_scan_history_samples_large_result(self, mock_wandb_mod, mock_api_mgr):
        """scan_history results should be client-side sampled to match the samples parameter."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10000
        mock_run.scan_history.return_value = [{"_step": i, "loss": float(i)} for i in range(5000)]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1", min_step=0, samples=500))

        assert result["sampled_points"] <= 500

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_step_range_samples_across_full_window(self, mock_wandb_mod, mock_api_mgr):
        """Step-range sampling should cover the full requested range, not just a prefix."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 99
        rows = [{"_step": i, "loss": float(i)} for i in range(100)]
        mock_run.scan_history.return_value = rows
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
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
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        total_rows = 10_000
        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = total_rows - 1
        mock_run.scan_history.return_value = [{"_step": i, "loss": float(i)} for i in range(total_rows)]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
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
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 500
        mock_run.scan_history.return_value = [{"_step": 100, "loss": 1.0}]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
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
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 500
        mock_run.scan_history.return_value = [{"_step": 50, "loss": 1.0}]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        get_run_history("e", "p", "run1", max_step=200)

        mock_run.scan_history.assert_called_once()
        call_kwargs = mock_run.scan_history.call_args[1]
        assert call_kwargs["max_step"] == 200
        assert "min_step" not in call_kwargs

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_sparse_metrics_both_keys_returned(self, mock_wandb_mod, mock_api_mgr):
        """Rows with disjoint metric sets should still report all keys."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 3
        mock_run.history.return_value = [
            {"_step": 0, "loss": 1.0},
            {"_step": 1, "accuracy": 0.5},
            {"_step": 2, "loss": 0.5},
        ]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        assert "loss" in result["keys_returned"]
        assert "accuracy" in result["keys_returned"]

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_response_shape(self, mock_wandb_mod, mock_api_mgr):
        """Response must always contain the documented top-level keys."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "r1"
        mock_run.lastHistoryStep = 10
        mock_run.history.return_value = [{"_step": 0, "loss": 1.0}]
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        for key in ("rows", "run_id", "run_name", "total_steps", "sampled_points", "keys_returned"):
            assert key in result, f"Missing required key: {key}"

    @patch("wandb_mcp_server.mcp_tools.run_history.WandBApiManager")
    @patch("wandb_mcp_server.mcp_tools.run_history.wandb")
    def test_empty_history(self, mock_wandb_mod, mock_api_mgr):
        """Run with no history rows should return gracefully."""
        mock_api_mgr.get_api.return_value = MagicMock(viewer="test-user")
        mock_api_mgr.get_api_key.return_value = "fake_key_12345678901234567890"

        mock_run = MagicMock()
        mock_run.name = "empty-run"
        mock_run.lastHistoryStep = 0
        mock_run.history.return_value = []
        mock_wandb_mod.Api.return_value = MagicMock(run=MagicMock(return_value=mock_run))
        mock_wandb_mod.errors = wandb.errors

        result = json.loads(get_run_history("e", "p", "run1"))
        assert result["rows"] == []
        assert result["sampled_points"] == 0
        assert result["keys_returned"] == []
