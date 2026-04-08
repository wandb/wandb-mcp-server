"""Tests for artifact tools (list_artifact_versions, get_artifact_details, compare_artifact_versions)."""

import json
from unittest.mock import MagicMock, PropertyMock, patch

from wandb_mcp_server.mcp_tools.query_artifacts import (
    COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION,
    LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    compare_artifact_versions,
    get_artifact_details,
    list_artifact_versions,
)


def _make_artifact(**overrides):
    art = MagicMock()
    art.id = overrides.get("id", "abc123")
    art.name = overrides.get("name", "my-model:v1")
    art.type = overrides.get("type", "model")
    art.version = overrides.get("version", "v1")
    art.state = overrides.get("state", "COMMITTED")
    art.description = overrides.get("description", "A test artifact")
    art.size = overrides.get("size", 1048576)
    art.file_count = overrides.get("file_count", 3)
    art.tags = overrides.get("tags", ["production"])
    art.aliases = overrides.get("aliases", ["latest"])
    art.metadata = overrides.get("metadata", {"framework": "pytorch"})
    art.created_at = overrides.get("created_at", "2025-01-01T00:00:00")
    art.digest = overrides.get("digest", "abc123def456")
    art.commit_hash = overrides.get("commit_hash", None)
    art.source_qualified_name = overrides.get("source_qualified_name", None)

    logged_by_run = overrides.get("logged_by_run", None)
    art.logged_by.return_value = logged_by_run

    used_by_runs = overrides.get("used_by_runs", [])
    art.used_by.return_value = used_by_runs

    source = overrides.get("source_artifact", art)
    type(art).source_artifact = PropertyMock(return_value=source)

    type(art).linked_artifacts = PropertyMock(return_value=overrides.get("linked_artifacts", []))

    files = overrides.get("files", [])
    art.files.return_value = iter(files)
    return art


def _make_run(run_id="run1", name="train-v1", project="my-project", entity="my-team"):
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.project = project
    run.entity = entity
    return run


def _make_file(name="model.pt", size=1000000, digest="aaa111"):
    f = MagicMock()
    f.name = name
    f.size = size
    f.digest = digest
    return f


class TestListArtifactVersions:
    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_project_source(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.artifacts.return_value = iter(
            [
                _make_artifact(version="v1"),
                _make_artifact(version="v2"),
            ]
        )
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_artifact_versions("team/project/my-model", type_name="model", source="project"))

        assert result["count"] == 2
        assert result["source"] == "project"
        assert result["versions"][0]["version"] == "v1"

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_registry_source(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_registry = MagicMock()
        mock_collections = MagicMock()
        mock_collections.versions.return_value = iter(
            [
                _make_artifact(version="v3"),
            ]
        )
        mock_registry.collections.return_value = mock_collections
        mock_api.registry.return_value = mock_registry
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_artifact_versions("my-model", registry_name="model-registry", source="registry"))

        assert result["count"] == 1
        assert result["source"] == "registry"

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_project_source_requires_type_name(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_artifact_versions("team/project/my-model", source="project"))

        assert result["error"] == "invalid_input"
        assert "type_name" in result["message"]

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_registry_source_requires_registry_name(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(list_artifact_versions("my-model", source="registry"))

        assert result["error"] == "invalid_input"
        assert "registry_name" in result["message"]


class TestGetArtifactDetails:
    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_basic(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        run = _make_run()
        art = _make_artifact(
            metadata={"accuracy": 0.95, "framework": "pytorch"},
            logged_by_run=run,
        )
        mock_api.artifact.return_value = art
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(get_artifact_details("team/proj/model:v1"))

        assert result["artifact"]["version"] == "v1"
        assert result["artifact"]["metadata"]["accuracy"] == 0.95
        assert result["lineage"]["logged_by"]["run_id"] == "run1"
        assert "files" not in result

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_with_files(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        files = [_make_file("model.pt", 1000), _make_file("config.json", 500)]
        art = _make_artifact(files=files)
        mock_api.artifact.return_value = art
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(get_artifact_details("team/proj/model:v1", include_files=True))

        assert len(result["files"]) == 2
        assert result["files"][0]["name"] == "model.pt"

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_lineage_failure_graceful(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        art = _make_artifact()
        art.logged_by.side_effect = Exception("Run deleted")
        art.used_by.side_effect = Exception("Run deleted")
        mock_api.artifact.return_value = art
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(get_artifact_details("team/proj/model:v1"))

        assert result["lineage"]["logged_by"] is None
        assert result["lineage"]["used_by"] is None

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_api_error_returns_json(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()
        mock_api.artifact.side_effect = Exception("Artifact not found")
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(get_artifact_details("team/proj/model:v99"))

        assert "error" in result
        assert "Artifact not found" in result["message"]


class TestCompareArtifactVersions:
    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_basic(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()

        art_a = _make_artifact(
            version="v1",
            size=1000,
            tags=["staging"],
            aliases=["old"],
            metadata={"accuracy": 0.90},
            digest="aaa",
            files=[_make_file("model.pt", 1000, "d1")],
        )
        art_b = _make_artifact(
            version="v2",
            size=2000,
            tags=["production"],
            aliases=["latest"],
            metadata={"accuracy": 0.95},
            digest="bbb",
            files=[_make_file("model.pt", 2000, "d2"), _make_file("vocab.json", 100, "d3")],
        )
        mock_api.artifact.side_effect = [art_a, art_b]
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(compare_artifact_versions("t/p/m:v1", "t/p/m:v2"))

        assert result["digest_match"] is False
        assert result["size_diff"]["delta"] == 1000
        assert result["metadata_diff"]["changed"]["accuracy"] == {"a": 0.90, "b": 0.95}
        assert "production" in result["tags_diff"]["added"]
        assert "staging" in result["tags_diff"]["removed"]

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_identical(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()

        art = _make_artifact(digest="same_digest")
        mock_api.artifact.side_effect = [art, art]
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(compare_artifact_versions("t/p/m:v1", "t/p/m:v1"))

        assert result["digest_match"] is True
        assert result["metadata_diff"]["added"] == {}
        assert result["metadata_diff"]["removed"] == {}
        assert result["metadata_diff"]["changed"] == {}

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_metadata_diff(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()

        art_a = _make_artifact(metadata={"a": 1, "b": 2, "c": 3})
        art_b = _make_artifact(metadata={"b": 2, "c": 99, "d": 4})
        mock_api.artifact.side_effect = [art_a, art_b]
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(compare_artifact_versions("t/p/m:v1", "t/p/m:v2"))

        diff = result["metadata_diff"]
        assert diff["added"] == {"d": 4}
        assert diff["removed"] == {"a": 1}
        assert diff["changed"] == {"c": {"a": 3, "b": 99}}

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_file_diff(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()

        art_a = _make_artifact(
            files=[_make_file("a.pt", 100, "d1"), _make_file("b.pt", 100, "d2")],
        )
        art_b = _make_artifact(
            files=[_make_file("b.pt", 100, "d2_changed"), _make_file("c.pt", 100, "d3")],
        )
        mock_api.artifact.side_effect = [art_a, art_b]
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(compare_artifact_versions("t/p/m:v1", "t/p/m:v2"))

        fd = result["file_diff"]
        assert "c.pt" in fd["added"]
        assert "a.pt" in fd["removed"]
        assert "b.pt" in fd["modified"]

    @patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager")
    def test_no_file_diff_when_disabled(self, mock_api_mgr):
        mock_api = MagicMock()
        mock_api.viewer = MagicMock()

        art_a = _make_artifact()
        art_b = _make_artifact()
        mock_api.artifact.side_effect = [art_a, art_b]
        mock_api_mgr.get_api.return_value = mock_api

        result = json.loads(compare_artifact_versions("t/p/m:v1", "t/p/m:v2", include_file_diff=False))

        assert "file_diff" not in result


class TestToolDescriptions:
    def test_list_artifact_versions_has_when_to_use(self):
        assert "<when_to_use>" in LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION
        assert "</when_to_use>" in LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION

    def test_get_artifact_details_has_when_to_use(self):
        assert "<when_to_use>" in GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION
        assert "</when_to_use>" in GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION

    def test_compare_artifact_versions_has_when_to_use(self):
        assert "<when_to_use>" in COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION
        assert "</when_to_use>" in COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION
