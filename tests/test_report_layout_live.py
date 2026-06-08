"""Opt-in live tests for MCP report layout creation.

These tests hit W&B. They are skipped unless `MCP_REPORT_LIVE=1` is set.
The harness loads `.env` files before checking credentials, so local runs can
use the same env files as manual MCP validation.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import scripts.report_layout_live_harness as harness


pytestmark = pytest.mark.integration


def test_env_file_candidates_deduplicates_explicit_path(tmp_path, monkeypatch):
    """The explicit .env path is first and duplicate paths are removed."""
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=test-key\n")
    monkeypatch.chdir(tmp_path)

    candidates = harness.env_file_candidates(str(env_file))

    assert candidates[0] == env_file.resolve()
    assert len(candidates) == len(set(candidates))


def test_load_harness_env_does_not_override_existing_shell_env(
    tmp_path,
    monkeypatch,
):
    """Shell env wins over .env values so callers can override safely."""
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=from-file\nMCP_REPORT_TEST_ENTITY=file-entity\n")
    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    monkeypatch.delenv("MCP_REPORT_TEST_ENTITY", raising=False)

    loaded = harness.load_harness_env(str(env_file))

    assert env_file.resolve() in loaded
    assert os.environ["WANDB_API_KEY"] == "from-shell"
    assert os.environ["MCP_REPORT_TEST_ENTITY"] == "file-entity"


def test_resolve_settings_prefers_explicit_entity_and_project(monkeypatch):
    """Explicit CLI-style args should win over .env-derived values."""
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("MCP_REPORT_TEST_ENTITY", "env-entity")
    monkeypatch.setenv("MCP_REPORT_TEST_PROJECT", "env-project")

    settings = harness.resolve_settings(
        entity="arg-entity",
        project="arg-project",
    )

    assert settings.entity == "arg-entity"
    assert settings.project == "arg-project"
    assert settings.api_key == "test-key"
    assert settings.data_mode == "api-shape"


def test_resolve_settings_accepts_renderable_data_mode(monkeypatch):
    """The harness can be switched to known-good visual render data."""
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("MCP_REPORT_TEST_ENTITY", "env-entity")
    monkeypatch.setenv("MCP_REPORT_TEST_PROJECT", "env-project")

    settings = harness.resolve_settings(data_mode="renderable")

    assert settings.data_mode == "renderable"


def test_resolve_settings_rejects_unknown_data_mode(monkeypatch):
    """Invalid data modes fail before seeding live runs."""
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("MCP_REPORT_TEST_ENTITY", "env-entity")
    monkeypatch.setenv("MCP_REPORT_TEST_PROJECT", "env-project")

    with pytest.raises(RuntimeError, match="data_mode must be one of"):
        harness.resolve_settings(data_mode="unknown")


def test_resolve_settings_requires_api_key(monkeypatch):
    """Missing credentials fail before any W&B network call is attempted."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(harness, "env_file_candidates", lambda *_: [])

    with pytest.raises(RuntimeError, match="WANDB_API_KEY is required"):
        harness.resolve_settings(entity="entity", project="project")


def test_report_layout_panels_cover_summary_and_history_tables():
    """The harness payload covers both summaryTable and historyTable paths."""
    panels = harness._report_layout_panels(
        ("run_a", "run_b"),
        data_mode="api-shape",
    )

    assert [panel["type"] for panel in panels] == [
        "heading",
        "markdown",
        "panel_grid",
        "heading",
        "markdown",
        "panel_grid",
    ]
    first_grid = panels[2]
    second_grid = panels[5]

    assert first_grid["run_ids"] == ["run_a", "run_b"]
    assert first_grid["hide_run_sets"] is False
    assert len(first_grid["panels"]) == 2
    assert all("summaryTable" in child["query"] for child in first_grid["panels"])
    assert {child["chart_name"] for child in first_grid["panels"]} == {
        "cruise/bar_chart/v2",
    }

    assert second_grid["run_ids"] == ["run_a"]
    assert second_grid["hide_run_sets"] is True
    assert len(second_grid["panels"]) == 1
    assert "historyTable" in second_grid["panels"][0]["query"]


def test_renderable_report_layout_panels_use_known_good_chart_tables():
    """Renderable mode uses built-in W&B line chart definitions."""
    panels = harness._report_layout_panels(
        ("run_a", "run_b"),
        data_mode="renderable",
    )

    first_grid = panels[2]
    second_grid = panels[5]

    assert first_grid["run_ids"] == ["run_a", "run_b"]
    assert first_grid["hide_run_sets"] is False
    assert [child["type"] for child in first_grid["panels"]] == [
        "custom_chart_table",
        "custom_chart_table",
    ]
    assert {child["chart_name"] for child in first_grid["panels"]} == {
        "wandb/line/v0",
    }
    assert [child["table_name"] for child in first_grid["panels"]] == [
        "loss_curve",
        "accuracy_curve",
    ]

    assert second_grid["run_ids"] == ["run_a"]
    assert second_grid["hide_run_sets"] is True
    assert second_grid["panels"][0]["table_name"] == "pr_curve_table"


def test_assert_run_filter_accepts_filter_v2_dict():
    """Filter assertions work against serialized filterV2 dictionaries."""
    runset = SimpleNamespace(
        search=SimpleNamespace(query=""),
        filters={
            "filterFormat": "filterV2",
            "filters": [
                {
                    "key": {"section": "run", "name": "name"},
                    "op": "IN",
                    "value": ["run_a", "run_b"],
                }
            ],
        },
    )

    harness._assert_run_filter(runset, ["run_a", "run_b"])


def test_live_report_layout_harness_creates_expected_report():
    """Create, reload, and validate a full report layout in W&B."""
    if os.getenv("MCP_REPORT_LIVE") != "1":
        pytest.skip("Set MCP_REPORT_LIVE=1 to run live W&B report layout tests.")

    settings = harness.resolve_settings(
        explicit_env_file=os.getenv("MCP_REPORT_TEST_ENV_FILE"),
    )
    result = harness.run_live_report_layout_verification(settings)

    assert result.report_url.startswith("https://")
    assert f"/{result.entity}/{result.project}/reports/" in result.report_url
    assert len(result.run_ids) == 2
