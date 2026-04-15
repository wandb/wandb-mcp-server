"""Tests for the probe_project tool."""

import json
from unittest.mock import MagicMock, patch


from wandb_mcp_server.mcp_tools.probe_project import (
    _safe_sample_value,
    probe_project,
)


class TestSafeSampleValue:
    def test_short_string(self):
        assert _safe_sample_value("hello") == "hello"

    def test_long_string_truncated(self):
        long = "x" * 200
        result = _safe_sample_value(long)
        assert result.endswith("...")
        assert len(result) == 103

    def test_int_passthrough(self):
        assert _safe_sample_value(42) == 42

    def test_list_representation(self):
        result = _safe_sample_value([1, 2, 3])
        assert "list" in result
        assert "len=3" in result

    def test_dict_representation(self):
        result = _safe_sample_value({"a": 1, "b": 2})
        assert "dict" in result
        assert "keys=2" in result


class TestProbeProject:
    def _make_mock_run(
        self, *, config=None, summary=None, state="finished", tags=None, group=None, lastHistoryStep=100
    ):
        run = MagicMock()
        run.config = config or {}
        run.summary = summary or {}
        run.state = state
        run.tags = tags or []
        run.group = group
        run.lastHistoryStep = lastHistoryStep
        return run

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_key_extraction(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run1 = self._make_mock_run(
            config={"lr": 0.01, "model": "bert"},
            summary={"loss": 0.5, "accuracy": 0.9},
        )
        run2 = self._make_mock_run(
            config={"lr": 0.001, "batch_size": 32},
            summary={"loss": 0.3, "f1": 0.85},
        )

        mock_api_mgr.get_api.return_value.runs.return_value = iter([run1, run2])

        result = json.loads(probe_project("ent", "proj"))

        assert "lr" in result["config_keys"]
        assert "model" in result["config_keys"]
        assert "batch_size" in result["config_keys"]
        assert "loss" in result["metric_keys"]
        assert "accuracy" in result["metric_keys"]
        assert "f1" in result["metric_keys"]
        assert result["run_count"] == 2
        assert result["sampled_runs"] == 2

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_empty_project(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api.return_value.runs.return_value = iter([])

        result = json.loads(probe_project("ent", "empty-proj"))

        assert result["run_count"] == 0
        assert result["metric_keys"] == {}
        assert result["config_keys"] == {}
        assert result["has_history"] is False
        assert any("Small project" in r for r in result["recommendations"])

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_recommendations_for_large_project(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        runs = [
            self._make_mock_run(
                config={"lr": 0.01},
                summary={"loss": 0.5},
            )
            for _ in range(150)
        ]
        mock_api_mgr.get_api.return_value.runs.return_value = iter(runs)

        result = json.loads(probe_project("ent", "big-proj", sample_runs=5))

        assert result["run_count"] == 150
        assert result["sampled_runs"] == 5
        assert any("Large project" in r for r in result["recommendations"])

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_tag_and_group_collection(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run1 = self._make_mock_run(tags=["baseline", "v1"], group="experiment-1")
        run2 = self._make_mock_run(tags=["tuned", "v1"], group="experiment-2")

        mock_api_mgr.get_api.return_value.runs.return_value = iter([run1, run2])

        result = json.loads(probe_project("ent", "proj"))

        assert "baseline" in result["tags"]
        assert "tuned" in result["tags"]
        assert "v1" in result["tags"]
        assert "experiment-1" in result["groups"]
        assert "experiment-2" in result["groups"]
        assert any("Tags in use" in r for r in result["recommendations"])
        assert any("Run groups found" in r for r in result["recommendations"])

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_filters_internal_keys(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run = self._make_mock_run(
            config={"lr": 0.01, "_wandb": {}, "wandb_version": "0.1"},
            summary={"loss": 0.5, "_runtime": 100, "wandb/cpu": 0.8},
        )
        mock_api_mgr.get_api.return_value.runs.return_value = iter([run])

        result = json.loads(probe_project("ent", "proj"))

        assert "_wandb" not in result["config_keys"]
        assert "wandb_version" not in result["config_keys"]
        assert "lr" in result["config_keys"]
        assert "_runtime" not in result["metric_keys"]
        assert "wandb/cpu" not in result["metric_keys"]
        assert "loss" in result["metric_keys"]

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_project_not_found_error(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")
        mock_api_mgr.get_api.return_value.runs.side_effect = Exception("Project not found")

        result = json.loads(probe_project("ent", "no-such-proj"))

        assert result["error"] == "project_not_found"
        assert "Project not found" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_typical_steps_calculation(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run1 = self._make_mock_run(lastHistoryStep=1000)
        run2 = self._make_mock_run(lastHistoryStep=2000)
        mock_api_mgr.get_api.return_value.runs.return_value = iter([run1, run2])

        result = json.loads(probe_project("ent", "proj"))

        assert result["has_history"] is True
        assert result["typical_steps"] == 1500

    @patch("wandb_mcp_server.mcp_tools.probe_project.WandBApiManager")
    def test_run_states_counted(self, mock_api_mgr):
        mock_api_mgr.get_api.return_value = MagicMock(viewer="user")

        run1 = self._make_mock_run(state="finished")
        run2 = self._make_mock_run(state="finished")
        run3 = self._make_mock_run(state="crashed")
        mock_api_mgr.get_api.return_value.runs.return_value = iter([run1, run2, run3])

        result = json.loads(probe_project("ent", "proj"))

        assert result["run_states"]["finished"] == 2
        assert result["run_states"]["crashed"] == 1
