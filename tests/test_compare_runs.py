"""Tests for the compare_runs tool."""

import json
from unittest.mock import patch

import pytest

from wandb_mcp_server.mcp_tools.compare_runs import (
    _diff_dicts,
    _safe_val,
    compare_runs,
)
from wandb_mcp_server.wandb_graphql import GraphQLResponseTooLarge


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
    @staticmethod
    def _projected_run(
        *,
        run_id="run",
        config=None,
        summary=None,
        name="run",
        state="finished",
        tags=None,
        group=None,
        created_at="2026-01-01",
        heartbeat_at="2026-01-02",
    ):
        return {
            "id": run_id,
            "display_name": name,
            "state": state,
            "config": config or {},
            "summary": summary or {},
            "tags": tags or [],
            "group": group,
            "created_at": created_at,
            "heartbeat_at": heartbeat_at,
        }

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_config_diff(self, mock_api_mgr, mock_projected):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="run-a", config={"lr": 0.01, "model": "resnet"}),
            self._projected_run(run_id="run-b", config={"lr": 0.001, "batch_size": 64}),
        ]

        result = json.loads(
            compare_runs(
                "ent",
                "proj",
                "run-a",
                "run-b",
                config_keys=["lr", "model", "batch_size"],
                summary_keys=[],
            )
        )

        assert result["config_diff"]["only_in_run_a"] == {"model": "resnet"}
        assert result["config_diff"]["only_in_run_b"] == {"batch_size": 64}
        assert "lr" in result["config_diff"]["changed"]
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_summary_metric_delta(self, mock_api_mgr, mock_projected):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="a", summary={"accuracy": 0.85, "loss": 0.3}),
            self._projected_run(run_id="b", summary={"accuracy": 0.92, "loss": 0.15}),
        ]

        result = json.loads(
            compare_runs(
                "ent",
                "proj",
                "a",
                "b",
                config_keys=[],
                summary_keys=["accuracy", "loss"],
            )
        )

        acc_change = result["summary_diff"]["changed"]["accuracy"]
        assert acc_change["delta"] == pytest.approx(0.07, abs=1e-5)
        loss_change = result["summary_diff"]["changed"]["loss"]
        assert loss_change["delta"] == pytest.approx(-0.15, abs=1e-5)
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_metadata_comparison(self, mock_api_mgr, mock_projected):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="a", tags=["baseline"], group="exp1"),
            self._projected_run(run_id="b", tags=["tuned"], group="exp2"),
        ]

        result = json.loads(compare_runs("ent", "proj", "a", "b", config_keys=[], summary_keys=[]))

        assert result["metadata_diff"]["run_a"]["tags"] == ["baseline"]
        assert result["metadata_diff"]["run_b"]["group"] == "exp2"
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_identical_configs(self, mock_api_mgr, mock_projected):
        api = mock_api_mgr.get_api.return_value
        shared_config = {"lr": 0.01, "epochs": 10}
        mock_projected.side_effect = [
            self._projected_run(run_id="a", config=shared_config),
            self._projected_run(run_id="b", config=shared_config),
        ]

        result = json.loads(
            compare_runs(
                "ent",
                "proj",
                "a",
                "b",
                config_keys=["lr", "epochs"],
                summary_keys=[],
            )
        )

        assert result["config_diff"]["changed"] == {}
        assert result["config_diff"]["identical_count"] == 2
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch(
        "wandb_mcp_server.mcp_tools.compare_runs._indexed_comparison_keys",
        return_value=([], ["loss"], True),
    )
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_filters_internal_summary_keys(self, mock_api_mgr, _mock_keys, mock_projected):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="a", summary={"loss": 0.5}),
            self._projected_run(run_id="b", summary={"loss": 0.3}),
        ]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        assert "_runtime" not in result["summary_diff"]["changed"]
        assert "wandb/cpu" not in result["summary_diff"]["changed"]
        assert "loss" in result["summary_diff"]["changed"]
        assert result["selection"]["summary_keys"] == ["loss"]
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run", return_value=None)
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_run_not_found_error(self, mock_api_mgr, _mock_projected):
        api = mock_api_mgr.get_api.return_value

        result = json.loads(compare_runs("ent", "proj", "bad-id", "other", config_keys=[], summary_keys=[]))

        assert result["error"] == "run_not_found"
        assert "not found" in result["message"].lower()
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_run_ids_in_result(self, mock_api_mgr, mock_projected):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="id-a", name="alpha"),
            self._projected_run(run_id="id-b", name="beta"),
        ]

        result = json.loads(compare_runs("ent", "proj", "id-a", "id-b", config_keys=[], summary_keys=[]))

        assert result["run_a"]["id"] == "id-a"
        assert result["run_a"]["name"] == "alpha"
        assert result["run_b"]["id"] == "id-b"
        assert result["run_b"]["name"] == "beta"
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.get_run_history")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_history_overlap_preserves_cross_cadence_rows_without_full_run_hydration(
        self,
        mock_api_mgr,
        mock_projected,
        mock_history,
    ):
        api = mock_api_mgr.get_api.return_value
        mock_projected.side_effect = [
            self._projected_run(run_id="a", summary={"loss": 0.5, "eval/loss": 0.4}),
            self._projected_run(run_id="b", summary={"loss": 0.3, "eval/loss": 0.2}),
        ]
        mock_history.side_effect = [
            json.dumps(
                {
                    "rows": [
                        {"_step": 0, "loss": 1.0},
                        {"_step": 1, "eval/loss": 0.9},
                    ],
                    "join": "outer",
                }
            ),
            json.dumps(
                {
                    "rows": [
                        {"_step": 0, "loss": 0.8},
                        {"_step": 1, "eval/loss": 0.7},
                    ],
                    "join": "outer",
                }
            ),
        ]

        result = json.loads(
            compare_runs(
                "ent",
                "proj",
                "a",
                "b",
                include_history_overlap=True,
                history_keys=["loss", "eval/loss"],
                history_samples=10,
                config_keys=[],
                summary_keys=["loss", "eval/loss"],
            )
        )

        assert result["history_comparison"]["run_a_sample"] == [
            {"_step": 0, "loss": 1.0, "eval/loss": None},
            {"_step": 1, "loss": None, "eval/loss": 0.9},
        ]
        assert result["history_comparison"]["run_b_sample"] == [
            {"_step": 0, "loss": 0.8, "eval/loss": None},
            {"_step": 1, "loss": None, "eval/loss": 0.7},
        ]
        assert mock_history.call_count == 2
        assert all(call.kwargs["keys"] == ["loss", "eval/loss"] for call in mock_history.call_args_list)
        api.run.assert_not_called()

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch(
        "wandb_mcp_server.mcp_tools.compare_runs._indexed_comparison_keys",
        return_value=(["learning_rate"], ["validation/loss"], False),
    )
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_indexed_projection_avoids_full_sdk_runs_and_discloses_scope(
        self,
        mock_api_mgr,
        _mock_keys,
        mock_projected_run,
    ):
        api = mock_api_mgr.get_api.return_value
        mock_projected_run.side_effect = [
            {
                "id": "a",
                "display_name": "A",
                "state": "finished",
                "config": {"learning_rate": 0.01},
                "summary": {"validation/loss": 0.4},
            },
            {
                "id": "b",
                "display_name": "B",
                "state": "finished",
                "config": {"learning_rate": 0.001},
                "summary": {"validation/loss": 0.2},
            },
        ]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        api.run.assert_not_called()
        assert result["selection"] == {
            "source": "project_field_index",
            "config_keys": ["learning_rate"],
            "summary_keys": ["validation/loss"],
            "field_index_exhaustive": False,
        }
        assert result["coverage"]["full_run_fields_exhaustive"] is False
        assert result["summary_diff"]["changed"]["validation/loss"]["delta"] == -0.2

    @patch("wandb_mcp_server.mcp_tools.compare_runs.fetch_projected_run")
    @patch(
        "wandb_mcp_server.mcp_tools.compare_runs._indexed_comparison_keys",
        return_value=(["learning_rate"], ["validation/loss"], True),
    )
    @patch("wandb_mcp_server.mcp_tools.compare_runs.WandBApiManager")
    def test_second_projected_run_too_large_never_hydrates_sdk_runs(
        self,
        mock_api_mgr,
        _mock_keys,
        mock_projected_run,
    ):
        api = mock_api_mgr.get_api.return_value
        mock_projected_run.side_effect = [
            {
                "id": "a",
                "display_name": "A",
                "state": "finished",
                "config": {"learning_rate": 0.01},
                "summary": {"validation/loss": 0.4},
            },
            GraphQLResponseTooLarge("bounded"),
        ]

        result = json.loads(compare_runs("ent", "proj", "a", "b"))

        assert result["error"] == "response_too_large"
        assert result["retryable"] is False
        api.run.assert_not_called()
