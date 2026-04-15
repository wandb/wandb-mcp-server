"""Tests for the compare_runs tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from wandb_mcp_server.mcp_tools.compare_runs import (
    _diff_dicts,
    _safe_val,
    compare_runs,
)


class TestSafeVal:
    def test_nan_becomes_string(self):
        assert _safe_val(float("nan")) == "nan"

    def test_inf_becomes_string(self):
        assert _safe_val(float("inf")) == "inf"

    def test_normal_float_passes_through(self):
        assert _safe_val(3.14) == 3.14

    def test_string_passes_through(self):
        assert _safe_val("hello") == "hello"


class TestDiffDicts:
    def test_added_removed_changed_keys(self):
        a = {"lr": 0.01, "epochs": 10, "model": "resnet"}
        b = {"lr": 0.001, "epochs": 10, "batch_size": 32}
        diff = _diff_dicts(a, b)

        assert diff["only_in_run_a"] == {"model": "resnet"}
        assert diff["only_in_run_b"] == {"batch_size": 32}
        assert "lr" in diff["changed"]
        assert diff["changed"]["lr"]["run_a"] == 0.01
        assert diff["changed"]["lr"]["run_b"] == 0.001
        assert diff["changed"]["lr"]["delta"] == round(0.001 - 0.01, 6)
        assert diff["identical_count"] == 1

    def test_identical_dicts(self):
        d = {"lr": 0.01, "epochs": 10}
        diff = _diff_dicts(d, d)

        assert diff["only_in_run_a"] == {}
        assert diff["only_in_run_b"] == {}
        assert diff["changed"] == {}
        assert diff["identical_count"] == 2

    def test_empty_dicts(self):
        diff = _diff_dicts({}, {})
        assert diff["identical_count"] == 0
        assert diff["only_in_run_a"] == {}

    def test_numeric_delta_for_ints(self):
        diff = _diff_dicts({"steps": 100}, {"steps": 200})
        assert diff["changed"]["steps"]["delta"] == 100

    def test_no_delta_for_strings(self):
        diff = _diff_dicts({"model": "bert"}, {"model": "gpt2"})
        assert "delta" not in diff["changed"]["model"]


class TestCompareRuns:
    def _make_mock_run(
        self,
        *,
        config=None,
        summary=None,
        name="run",
        state="finished",
        tags=None,
        group=None,
        created_at="2026-01-01",
        heartbeat_at="2026-01-02",
    ):
        run = MagicMock()
        run.config = config or {}
        run.summary = summary or {}
        run.name = name
        run.state = state
        run.tags = tags or []
        run.group = group
        run.created_at = created_at
        run.heartbeat_at = heartbeat_at
        return run

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_config_diff(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run_a = self._make_mock_run(config={"lr": 0.01, "model": "resnet"})
        run_b = self._make_mock_run(config={"lr": 0.001, "batch_size": 64})

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "run-a", "run-b"))

        assert result["config_diff"]["only_in_run_a"] == {"model": "resnet"}
        assert result["config_diff"]["only_in_run_b"] == {"batch_size": 64}
        assert "lr" in result["config_diff"]["changed"]

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_summary_metric_delta(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run_a = self._make_mock_run(summary={"accuracy": 0.85, "loss": 0.3})
        run_b = self._make_mock_run(summary={"accuracy": 0.92, "loss": 0.15})

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        acc_change = result["summary_diff"]["changed"]["accuracy"]
        assert acc_change["delta"] == pytest.approx(0.07, abs=1e-5)
        loss_change = result["summary_diff"]["changed"]["loss"]
        assert loss_change["delta"] == pytest.approx(-0.15, abs=1e-5)

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_metadata_comparison(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run_a = self._make_mock_run(tags=["baseline"], group="exp1")
        run_b = self._make_mock_run(tags=["tuned"], group="exp2")

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        assert result["metadata_diff"]["run_a"]["tags"] == ["baseline"]
        assert result["metadata_diff"]["run_b"]["group"] == "exp2"

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_identical_configs(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        shared_config = {"lr": 0.01, "epochs": 10}
        run_a = self._make_mock_run(config=shared_config)
        run_b = self._make_mock_run(config=shared_config)

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        assert result["config_diff"]["changed"] == {}
        assert result["config_diff"]["identical_count"] == 2

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_filters_internal_summary_keys(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run_a = self._make_mock_run(summary={"loss": 0.5, "_runtime": 100, "wandb/cpu": 0.8})
        run_b = self._make_mock_run(summary={"loss": 0.3, "_runtime": 200, "wandb/cpu": 0.9})

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        assert "_runtime" not in result["summary_diff"]["changed"]
        assert "wandb/cpu" not in result["summary_diff"]["changed"]
        assert "loss" in result["summary_diff"]["changed"]

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_run_not_found_error(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api.return_value.run.side_effect = Exception("Run not found: bad-id")

        result = json.loads(compare_runs("ent", "proj", "bad-id", "other"))

        assert result["error"] == "run_not_found"
        assert "bad-id" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_run_ids_in_result(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run_a = self._make_mock_run(name="alpha")
        run_b = self._make_mock_run(name="beta")

        mock_api = mock_api_mgr.get_api.return_value
        mock_api.run.side_effect = [run_a, run_b]

        result = json.loads(compare_runs("ent", "proj", "id-a", "id-b"))

        assert result["run_a"]["id"] == "id-a"
        assert result["run_a"]["name"] == "alpha"
        assert result["run_b"]["id"] == "id-b"
        assert result["run_b"]["name"] == "beta"
