"""Tests for bounded project probing."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from wandb_mcp_server.mcp_tools import probe_project as probe_module
from wandb_mcp_server.wandb_selective_reads import (
    ProjectFieldPage,
    ProjectedRunPage,
    SelectiveReadUnavailable,
)


@contextmanager
def _tracking(*args, **kwargs):
    yield SimpleNamespace(mark_error=lambda error: None)


@pytest.fixture
def probe_fakes(monkeypatch):
    api = SimpleNamespace(runs=lambda *args, **kwargs: pytest.fail("SDK fallback must not run"))
    monkeypatch.setattr(probe_module.WandBApiManager, "get_api", lambda: api)
    monkeypatch.setattr(probe_module, "track_tool_execution", _tracking)
    monkeypatch.setattr(
        probe_module,
        "fetch_project_counts",
        lambda *args, **kwargs: {
            "all": 12_345,
            "finished": 12_000,
            "failed": 100,
            "crashed": 45,
            "running": 200,
        },
    )
    monkeypatch.setattr(
        probe_module,
        "fetch_project_fields",
        lambda *args, **kwargs: ProjectFieldPage(
            items=[
                {"path": "config.learning_rate", "type": "number"},
                {"path": "config.model.name", "type": "string"},
                {"path": "summary_metrics.validation/loss", "type": "number"},
                {"path": "summaryMetrics.accuracy", "type": "number"},
                {"path": "summary._runtime", "type": "number"},
            ],
            has_more=False,
            requests=1,
        ),
    )

    def projected(*args, order, limit, **kwargs):
        suffix = "new" if order.startswith("-") else "old"
        return ProjectedRunPage(
            items=[
                {
                    "id": f"{suffix}-{index}",
                    "state": "finished",
                    "group": "baseline",
                    "tags": ["production"],
                    "history_line_count": 100,
                }
                for index in range(limit)
            ],
            total_count=12_345,
            has_more=True,
            requests=1,
        )

    monkeypatch.setattr(probe_module, "fetch_projected_runs", projected)
    return api


def test_probe_uses_exact_counts_indexed_fields_and_recent_oldest_samples(probe_fakes):
    result = json.loads(probe_module.probe_project("entity", "project", sample_runs=6))

    assert result["run_count"] == 12_345
    assert result["state_counts"] == {
        "finished": 12_000,
        "failed": 100,
        "crashed": 45,
        "running": 200,
    }
    assert result["sampled_runs"] == 6
    assert {row["id"].split("-", 1)[0] for row in result["run_samples"]} == {"new", "old"}
    assert result["summary_fields"] == [
        {"path": "validation/loss", "type": "number"},
        {"path": "accuracy", "type": "number"},
    ]
    assert result["config_fields"] == [
        {"path": "learning_rate", "type": "number"},
        {"path": "model.name", "type": "string"},
    ]
    assert result["sample_scope"]["project_exhaustive"] is False
    assert result["field_inventory"]["project_exhaustive"] is True
    assert result["has_history_in_sample"] is True
    assert result["recommended_next_calls"][0]["tool"] == "query_wandb_tool"


def test_probe_forwards_field_pattern_and_caps_inventory(monkeypatch, probe_fakes):
    captured = {}

    def fields(*args, **kwargs):
        captured.update(kwargs)
        return ProjectFieldPage(
            items=[{"path": f"summary_metrics.metric_{index}", "type": "number"} for index in range(500)],
            has_more=True,
            requests=3,
        )

    monkeypatch.setattr(probe_module, "fetch_project_fields", fields)
    monkeypatch.setattr(probe_module, "MCP_MAX_PROJECT_FIELDS", 500)

    result = json.loads(probe_module.probe_project("entity", "project", field_pattern="validation"))

    assert captured["pattern"] == "validation"
    assert captured["limit"] == 500
    assert result["summary_field_count_returned"] == 500
    assert len(result["summary_fields"]) == 200
    assert result["field_inventory"]["has_more"] is True
    assert result["field_inventory"]["response_truncated"] is True


def test_probe_caps_requested_run_samples(monkeypatch, probe_fakes):
    monkeypatch.setattr(probe_module, "MCP_MAX_PROBE_RUNS", 4)

    result = json.loads(probe_module.probe_project("entity", "project", sample_runs=100))

    assert result["sampled_runs"] == 4
    assert result["sample_scope"] == {
        "strategy": "recent_and_oldest",
        "requested": 100,
        "effective": 4,
        "cap_applied": True,
        "project_exhaustive": False,
    }


def test_probe_optional_artifact_inventory(monkeypatch, probe_fakes):
    monkeypatch.setattr(
        probe_module,
        "fetch_artifact_inventory",
        lambda *args, **kwargs: {
            "types": [{"type": "model", "collection_count": 3}],
            "returned_count": 1,
            "has_more": False,
        },
    )

    result = json.loads(probe_module.probe_project("entity", "project", include_artifacts=True))

    assert result["artifact_inventory"]["types"][0]["type"] == "model"


def test_probe_uses_bounded_non_lazy_sdk_fallback(monkeypatch):
    calls = []

    class FakeRuns:
        def __init__(self, rows, total):
            self.rows = rows
            self.total = total

        def __iter__(self):
            return iter(self.rows)

        def __len__(self):
            return self.total

    run = SimpleNamespace(
        id="run-1",
        name="Run 1",
        state="finished",
        entity="entity",
        project="project",
        created_at="2026-01-01",
        group="group",
        job_type="train",
        tags=["tag"],
        lastHistoryStep=100,
        url="https://wandb.ai/entity/project/runs/run-1",
        config={"learning_rate": 0.1},
        summary={"loss": 0.2},
    )

    def runs(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("lazy") is False:
            return FakeRuns([run], 1)
        state = (kwargs.get("filters") or {}).get("state")
        return FakeRuns([], 1 if state == "finished" else 10 if state is None else 0)

    api = SimpleNamespace(runs=runs)
    monkeypatch.setattr(probe_module.WandBApiManager, "get_api", lambda: api)
    monkeypatch.setattr(probe_module, "track_tool_execution", _tracking)
    monkeypatch.setattr(
        probe_module,
        "fetch_project_counts",
        lambda *args, **kwargs: (_ for _ in ()).throw(SelectiveReadUnavailable("project fields unsupported")),
    )

    result = json.loads(probe_module.probe_project("entity", "project", sample_runs=2))

    assert result["source"] == "wandb_sdk_fallback"
    assert result["run_count"] == 10
    assert result["config_fields"] == [{"path": "learning_rate", "type": "float"}]
    assert result["summary_fields"] == [{"path": "loss", "type": "float"}]
    assert "bounded hydrated SDK runs" in result["compatibility_caveat"]
    assert any(call.get("lazy") is False for call in calls)


def test_probe_returns_structured_error(monkeypatch):
    monkeypatch.setattr(
        probe_module.WandBApiManager,
        "get_api",
        lambda: (_ for _ in ()).throw(ValueError("project not found")),
    )
    monkeypatch.setattr(probe_module, "track_tool_execution", _tracking)

    result = json.loads(probe_module.probe_project("entity", "missing"))

    assert result == {"error": "project_probe_failed", "message": "project not found"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entity_name": "", "project_name": "project"},
        {"entity_name": "entity", "project_name": ""},
        {"entity_name": "entity", "project_name": "project", "sample_runs": 0},
        {"entity_name": "entity", "project_name": "project", "field_pattern": ""},
        {"entity_name": "entity", "project_name": "project", "include_artifacts": "yes"},
    ],
)
def test_probe_validates_before_creating_api(monkeypatch, kwargs):
    monkeypatch.setattr(
        probe_module.WandBApiManager,
        "get_api",
        lambda: pytest.fail("API must not be constructed"),
    )

    with pytest.raises(ValueError):
        probe_module.probe_project(**kwargs)


def test_safe_sample_value_compatibility_helper():
    assert probe_module._safe_sample_value("x" * 200).endswith("...")
    assert probe_module._safe_sample_value([1, 2]) == "[list, len=2]"
