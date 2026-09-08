"""Tests for the diagnose_run tool."""

import json
from unittest.mock import patch


from wandb_mcp_server.mcp_tools.diagnose_run import (
    _auto_detect_key,
    _compute_trend,
    _detect_overfit,
    diagnose_run,
)
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge


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

    def test_disjoint_cadences_do_not_create_an_ordinal_gap_signal(self):
        train = [100.0 / (index + 1) for index in range(20)]
        validation = [100.0 - index * 2.0 for index in range(20)]

        result = _detect_overfit(train, validation, aligned_pairs=[])

        assert result["train_loss_trend"] == "decreasing"
        assert result["val_loss_trend"] == "decreasing"
        assert result["detected"] is False
        assert result["aligned_points"] == 0
        assert result["gap_ratio"] is None


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
    @staticmethod
    def _projected_run(*, summary=None, config=None, name="test-run", state="finished"):
        return {
            "id": "r1",
            "display_name": name,
            "state": state,
            "summary": summary or {},
            "config": config or {},
        }

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_projected_run_too_large_never_hydrates_sdk_run(
        self,
        mock_api_mgr,
        mock_projected,
    ):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = GraphQLResponseTooLarge("bounded")

        result = json.loads(
            diagnose_run(
                "ent",
                "proj",
                "r1",
                loss_key="loss",
                config_keys=["learning_rate"],
                summary_keys=["loss"],
            )
        )

        assert result["error"] == "response_too_large"
        assert result["retryable"] is False
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run", return_value=None)
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_run_not_found(self, mock_api_mgr, _mock_projected):
        api = mock_api_mgr.get_api.return_value

        result = json.loads(diagnose_run("ent", "proj", "xyz", config_keys=[], summary_keys=[]))

        assert result["error"] == "run_not_found"
        assert "not found" in result["message"].lower()
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history", return_value=json.dumps({"rows": []}))
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_no_history(self, mock_api_mgr, mock_projected, mock_history):
        api = mock_api_mgr.get_api.return_value
        mock_projected.return_value = self._projected_run(summary={"loss": 1.0})

        result = json.loads(
            diagnose_run(
                "ent",
                "proj",
                "r1",
                loss_key="loss",
                config_keys=[],
                summary_keys=["loss"],
            )
        )

        assert result["diagnosis"] == "no_history"
        mock_history.assert_called_once()
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_missing_metric_keys_does_not_fetch_unbounded_history(
        self,
        mock_api_mgr,
        mock_projected,
        mock_history,
    ):
        api = mock_api_mgr.get_api.return_value
        mock_projected.return_value = self._projected_run()

        result = json.loads(diagnose_run("ent", "proj", "r1", config_keys=[], summary_keys=[]))

        assert result["diagnosis"] == "no_loss_key"
        mock_history.assert_not_called()
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_converging_run(self, mock_api_mgr, mock_projected, mock_history):
        api = mock_api_mgr.get_api.return_value
        history = [{"_step": i, "loss": 10.0 / (i + 1)} for i in range(50)]
        mock_projected.return_value = self._projected_run(summary={"loss": history[-1]["loss"]})
        mock_history.return_value = json.dumps({"rows": history, "join": "outer"})

        result = json.loads(diagnose_run("ent", "proj", "r1", config_keys=[], summary_keys=["loss"]))

        assert result["diagnosis"] in ("training", "converged")
        assert result["loss_stats"] is not None
        assert result["loss_stats"]["key"] == "loss"
        assert result["loss_stats"]["first_value"] > result["loss_stats"]["last_value"]
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_nan_detection(self, mock_api_mgr, mock_projected, mock_history):
        api = mock_api_mgr.get_api.return_value
        history = [{"_step": i, "loss": 1.0, **({} if i % 5 == 0 else {"grad_norm": 1.0})} for i in range(20)]
        mock_projected.return_value = self._projected_run(summary={"loss": 1.0, "grad_norm": 1.0})
        mock_history.return_value = json.dumps({"rows": history, "non_finite_counts": {"grad_norm": 4}})

        result = json.loads(
            diagnose_run(
                "ent",
                "proj",
                "r1",
                config_keys=[],
                summary_keys=["loss", "grad_norm"],
            )
        )

        assert result["nan_warnings"] is not None
        assert "grad_norm" in result["nan_warnings"]
        assert result["nan_warnings"]["grad_norm"]["nan_count"] == 4
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_detected_keys_returned(self, mock_api_mgr, mock_projected, mock_history):
        api = mock_api_mgr.get_api.return_value
        history = [{"_step": i, "loss": 1.0, "val_loss": 1.1} for i in range(20)]
        mock_projected.return_value = self._projected_run(summary={"loss": 1.0, "val_loss": 1.1})
        mock_history.return_value = json.dumps({"rows": history})

        result = json.loads(
            diagnose_run(
                "ent",
                "proj",
                "r1",
                config_keys=[],
                summary_keys=["loss", "val_loss"],
            )
        )

        assert result["detected_keys"]["loss"] == "loss"
        assert result["detected_keys"]["val_loss"] == "val_loss"
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_recommendations_for_diverging(self, mock_api_mgr, mock_projected, mock_history):
        api = mock_api_mgr.get_api.return_value
        history = [{"_step": i, "loss": float(i)} for i in range(50)]
        mock_projected.return_value = self._projected_run(summary={"loss": history[-1]["loss"]})
        mock_history.return_value = json.dumps({"rows": history})

        result = json.loads(diagnose_run("ent", "proj", "r1", config_keys=[], summary_keys=["loss"]))

        assert result["diagnosis"] == "diverging"
        assert any("learning rate" in r for r in result["recommendations"])
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.diagnose_run.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.fetch_projected_run")
    @patch(
        "wandb_mcp_server.mcp_tools.diagnose_run._indexed_diagnosis_keys",
        return_value=(["learning_rate"], ["train/loss", "validation/loss"], False),
    )
    @patch("wandb_mcp_server.mcp_tools.diagnose_run.WandBApiManager")
    def test_uses_indexed_bounded_keys_and_discloses_sampling(
        self,
        mock_api_mgr,
        _mock_keys,
        mock_projected,
        mock_history,
    ):
        api = mock_api_mgr.get_api.return_value
        rows = []
        for index in range(10):
            rows.extend(
                [
                    {"epoch": float(index), "train/loss": 10.0 / (index + 1)},
                    {"epoch": float(index) + 0.5, "validation/loss": 11.0 / (index + 1)},
                ]
            )
        mock_history.return_value = json.dumps({"rows": rows, "join": "outer"})
        mock_projected.return_value = {
            "id": "r1",
            "display_name": "test-run",
            "state": "finished",
            "config": {"learning_rate": 0.01},
            "summary": {"train/loss": rows[-2]["train/loss"]},
        }

        result = json.loads(diagnose_run("ent", "proj", "r1", x_axis="epoch", samples=20))

        assert result["selection"]["source"] == "project_field_index"
        assert result["selection"]["config_keys"] == ["learning_rate"]
        assert result["coverage"] == {
            "sampled": True,
            "requested_samples": 20,
            "effective_samples": 20,
            "returned_rows": 20,
            "x_axis": "epoch",
            "project_exhaustive": False,
            "conclusions_apply_to_sample": True,
        }
        assert set(result["available_keys"]) == {"train/loss", "validation/loss"}
        assert result["overfit_signal"]["aligned_points"] == 0
        assert result["overfit_signal"]["gap_ratio"] is None
        mock_history.assert_called_once_with(
            "ent",
            "proj",
            "r1",
            keys=["train/loss", "validation/loss"],
            samples=20,
            x_axis="epoch",
        )
        api.run.assert_not_called()
