"""Tests for the diagnose_run tool."""

import json
from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.diagnose_run import (
    _auto_detect_key,
    _compute_trend,
    _detect_overfit,
    diagnose_run,
)


class TestComputeTrend:
    def test_insufficient_data(self):
        assert _compute_trend([1.0, 2.0, 3.0]) == "insufficient_data"

    def test_decreasing_trend(self):
        values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8]
        assert _compute_trend(values) == "decreasing"

    def test_increasing_trend(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        assert _compute_trend(values) == "increasing"

    def test_plateaued_trend(self):
        values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert _compute_trend(values) == "plateaued"

    def test_decreasing_then_plateau(self):
        values = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = _compute_trend(values)
        assert result in ("decreasing", "plateaued")


class TestDetectOverfit:
    def test_insufficient_data(self):
        result = _detect_overfit([1.0, 2.0], [1.0, 2.0])
        assert result["detected"] is False
        assert result["reason"] == "insufficient_data"

    def test_overfit_pattern(self):
        train = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8]
        val = [10.0, 9.5, 9.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
        result = _detect_overfit(train, val)

        assert result["detected"] is True
        assert result["train_loss_trend"] == "decreasing"
        assert result["val_loss_trend"] == "increasing"

    def test_normal_training(self):
        train = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8]
        val = [10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 1.4, 1.3]
        result = _detect_overfit(train, val)

        assert result["train_loss_trend"] == "decreasing"
        assert result["val_loss_trend"] == "decreasing"

    def test_gap_fields_present(self):
        train = list(range(20, 0, -1))
        val = list(range(20, 0, -1))
        result = _detect_overfit(
            [float(x) for x in train],
            [float(x) for x in val],
        )
        assert "gap_early" in result
        assert "gap_late" in result
        assert "gap_ratio" in result


class TestAutoDetectKey:
    def test_finds_loss(self):
        keys = ["_step", "accuracy", "loss", "lr"]
        assert _auto_detect_key(keys, ["train_loss", "train/loss", "loss"]) == "loss"

    def test_finds_train_loss_first(self):
        keys = ["loss", "train_loss", "val_loss"]
        assert _auto_detect_key(keys, ["train_loss", "loss"]) == "train_loss"

    def test_returns_none_when_no_match(self):
        keys = ["accuracy", "precision", "recall"]
        assert _auto_detect_key(keys, ["loss"]) is None

    def test_case_insensitive(self):
        keys = ["TrainLoss", "ValLoss"]
        assert _auto_detect_key(keys, ["trainloss"]) == "TrainLoss"


class TestDiagnoseRun:
    def _make_mock_run(self, name="test-run", state="finished", history_rows=None):
        run = MagicMock()
        run.name = name
        run.state = state
        run.scan_history.return_value = history_rows or []
        return run

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_run_not_found(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api.return_value.run.side_effect = Exception("Run xyz not found")

        result = json.loads(diagnose_run("ent", "proj", "xyz"))

        assert result["error"] == "run_not_found"
        assert "xyz" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_no_history(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run = self._make_mock_run(history_rows=[])
        mock_api_mgr.get_api.return_value.run.return_value = run

        result = json.loads(diagnose_run("ent", "proj", "r1"))

        assert result["diagnosis"] == "no_history"

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_converging_run(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        history = [{"_step": i, "loss": 10.0 / (i + 1)} for i in range(50)]
        run = self._make_mock_run(history_rows=history)
        mock_api_mgr.get_api.return_value.run.return_value = run

        result = json.loads(diagnose_run("ent", "proj", "r1"))

        assert result["diagnosis"] in ("training", "converged")
        assert result["loss_stats"] is not None
        assert result["loss_stats"]["key"] == "loss"
        assert result["loss_stats"]["first_value"] > result["loss_stats"]["last_value"]

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_nan_detection(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        history = [{"_step": i, "loss": 1.0, "grad_norm": float("nan") if i % 5 == 0 else 1.0} for i in range(20)]
        run = self._make_mock_run(history_rows=history)
        mock_api_mgr.get_api.return_value.run.return_value = run

        result = json.loads(diagnose_run("ent", "proj", "r1"))

        assert result["nan_warnings"] is not None
        assert "grad_norm" in result["nan_warnings"]
        assert result["nan_warnings"]["grad_norm"]["nan_count"] == 4

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_detected_keys_returned(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        history = [{"_step": i, "loss": 1.0, "val_loss": 1.1} for i in range(20)]
        run = self._make_mock_run(history_rows=history)
        mock_api_mgr.get_api.return_value.run.return_value = run

        result = json.loads(diagnose_run("ent", "proj", "r1"))

        assert result["detected_keys"]["loss"] == "loss"
        assert result["detected_keys"]["val_loss"] == "val_loss"

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_recommendations_for_diverging(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        history = [{"_step": i, "loss": float(i)} for i in range(50)]
        run = self._make_mock_run(history_rows=history)
        mock_api_mgr.get_api.return_value.run.return_value = run

        result = json.loads(diagnose_run("ent", "proj", "r1"))

        assert result["diagnosis"] == "diverging"
        assert any("learning rate" in r for r in result["recommendations"])
