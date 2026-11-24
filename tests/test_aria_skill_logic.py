"""Tests for ARIA production skill logic ported to MCP tools in v0.4.7."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from wandb_mcp_server.mcp_tools import agent_lens, agents, automations, probe_project, query_artifacts, summarize_evaluation
from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects
from wandb_mcp_server.weave_api.query_builder import QueryBuilder


def test_list_entity_projects_uses_bounded_ordered_graphql_fast_path():
    mock_api = MagicMock()
    with (
        patch("wandb_mcp_server.api_client.get_wandb_api", return_value=mock_api),
        patch(
            "wandb_mcp_server.mcp_tools.list_wandb_entities_projects.fetch_entity_projects",
            return_value=(
                [
                    {
                        "name": "recent-proj",
                        "entity": "acme",
                        "description": "active",
                        "visibility": "private",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-11-24T00:00:00Z",
                        "run_count": 142,
                        "tags": [],
                    }
                ],
                True,
            ),
        ) as mock_fast_path,
    ):
        payload = json.loads(list_entity_projects(entity="acme", max_projects=10, order="-last_active"))

    mock_fast_path.assert_called_once_with(mock_api, entity="acme", limit=10, order="-lastActive")
    mock_api.projects.assert_not_called()
    assert payload["truncated"] is True
    assert payload["projects"]["acme"][0]["name"] == "recent-proj"
    assert payload["projects"]["acme"][0]["run_count"] == 142


def test_probe_project_include_traces_surfaces_weave_and_agents_planes():
    mock_api = MagicMock()
    mock_session = MagicMock()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "agents": [{"agent_name": "router"}, {"agent_name": "researcher"}],
        "total_count": 2,
    }
    mock_session.post.return_value = mock_resp

    with (
        patch.object(probe_project.WandBApiManager, "get_api", return_value=mock_api),
        patch.object(probe_project.WandBApiManager, "get_api_key", return_value="test-key"),
        patch.object(
            probe_project,
            "fetch_project_counts",
            return_value={"all": 0, "finished": 0, "failed": 0, "crashed": 0, "running": 0},
        ),
        patch.object(probe_project, "fetch_project_fields", return_value=SimpleNamespace(items=[], has_more=False)),
        patch.object(probe_project, "fetch_projected_runs", return_value=SimpleNamespace(items=[])),
        patch("wandb_mcp_server.mcp_tools.count_traces.count_traces", return_value=18),
        patch("wandb_mcp_server.mcp_tools.tools_utils.get_no_retry_session", return_value=mock_session),
    ):
        payload = json.loads(probe_project.probe_project("acme", "agent-app", include_traces=True))

    planes = payload["telemetry_planes"]
    assert planes["models_runs"] == {"status": "ok", "present": False, "run_count": 0}
    assert planes["weave_calls"] == {"status": "ok", "present": True, "root_trace_count": 18}
    assert planes["weave_agents"]["present"] is True
    assert planes["weave_agents"]["sample_agents"] == ["router", "researcher"]
    rec_tools = [entry["tool"] for entry in payload["recommended_next_calls"]]
    assert "query_weave_traces_tool" in rec_tools
    assert "list_weave_agents_tool" in rec_tools


def test_weave_query_builder_supports_has_error_and_status_lists():
    query = QueryBuilder.build_query_expression({"has_error": True})
    assert query is not None
    dumped = query.model_dump(by_alias=True)
    serialized = json.dumps(dumped)
    assert "descendant_error" in serialized
    assert "exception" in serialized

    status_list_query = QueryBuilder.build_query_expression({"status": ["error", "descendant_error"]})
    assert status_list_query is not None
    assert "$in" in json.dumps(status_list_query.model_dump(by_alias=True))


def test_summarize_evaluation_extracts_root_output_scores_and_descendant_errors():
    eval_trace = {
        "id": "eval-1",
        "op_name": "weave:///acme/proj/op/Evaluation.evaluate:v1",
        "started_at": "2026-11-24T00:00:00Z",
        "output": {
            "accuracy_scorer": {"true_count": 9, "true_fraction": 0.9},
            "model_latency": {"mean": 1.24},
        },
        "summary": {"weave": {}},
    }
    children = [
        {"id": "c1", "exception": None, "summary": {"weave": {"status": "success"}}},
        {"id": "c2", "exception": None, "summary": {"weave": {"status": "descendant_error"}}},
    ]
    agg = summarize_evaluation._aggregate_eval(eval_trace, children, total_matching_count=2)
    assert agg["scores"]["accuracy_scorer"]["true_fraction"] == 0.9
    assert agg["scores"]["model_latency"]["mean"] == 1.24
    assert agg["observed_errors"] == 1
    assert agg["observed_successes"] == 1


def test_list_artifact_versions_auto_resolves_project_type_name():
    mock_api = MagicMock()
    fake_art = SimpleNamespace(
        version="v0",
        name="ckpt:v0",
        aliases=["latest"],
        tags=[],
        state="COMMITTED",
        size=1024,
        file_count=1,
        description="",
        created_at="2026-11-24T00:00:00Z",
        digest="abc",
    )
    mock_api.artifacts.return_value = [fake_art]
    with (
        patch.object(query_artifacts.WandBApiManager, "get_api", return_value=mock_api),
        patch.object(
            query_artifacts,
            "fetch_artifact_inventory",
            return_value={
                "types": [
                    {"type": "dataset", "collection_count": 1, "sample_collections": [{"name": "train-data"}]},
                    {"type": "model", "collection_count": 1, "sample_collections": [{"name": "ckpt"}]},
                ]
            },
        ),
    ):
        payload = json.loads(
            query_artifacts.list_artifact_versions(
                collection_name="ckpt",
                entity_name="acme",
                project_name="proj",
                source="project",
            )
        )

    assert payload["type_name"] == "model"
    assert payload["returned_count"] == 1
    mock_api.artifacts.assert_called_once()
    assert mock_api.artifacts.call_args.kwargs["type_name"] == "model"


def test_list_automations_filters_registry_project_and_includes_integrations():
    mock_api = MagicMock()
    matching = MagicMock()
    non_matching = MagicMock()
    mock_api.automations.return_value = iter([non_matching, matching])
    mock_api.slack_integrations.return_value = iter([])

    def fake_jsonify(item):
        if item is matching:
            return {
                "id": "a1",
                "name": "prod-rule",
                "scope": {"type": "PROJECT", "id": "p1", "name": "wandb-registry-llm-models"},
                "event": {"type": "ADD_ARTIFACT_ALIAS", "filter": {}},
                "action": {"type": "NOTIFICATION", "integration_id": "i1"},
            }
        return {
            "id": "a2",
            "name": "other-rule",
            "scope": {"type": "PROJECT", "id": "p2", "name": "other-project"},
            "event": {"type": "RUN_STATE", "filter": {}},
            "action": {"type": "NO_OP"},
        }

    with (
        patch.object(automations.WandBApiManager, "get_api", return_value=mock_api),
        patch.object(automations, "_jsonify_automation", side_effect=fake_jsonify),
    ):
        payload = json.loads(
            automations.list_automations(
                entity="acme-org",
                registry_name="llm-models",
                include_integrations=True,
                integration_kind="slack",
            )
        )

    assert payload["project"] == "wandb-registry-llm-models"
    assert [row["id"] for row in payload["automations"]] == ["a1"]
    assert payload["integrations"] == []


def test_list_agents_normalizes_sort_aliases_and_embeds_versions():
    calls = []

    def fake_request(tool_name, path, body, track_params):
        calls.append((tool_name, path, body))
        if path == agents.AGENTS_QUERY_PATH:
            return json.dumps({"agents": [{"agent_name": "tutor", "invocation_count": 12}], "total_count": 1})
        return json.dumps({"versions": [{"agent_version": "v2", "invocation_count": 8}], "total_count": 1})

    with patch.object(agents, "_agents_request", side_effect=fake_request):
        payload = json.loads(
            agents.list_agents(
                "acme",
                "proj",
                agent_name="tutor",
                sort_by="-updated_at",
                include_versions=True,
            )
        )

    assert calls[0][2]["sort_by"] == [{"field": "last_seen", "direction": "desc"}]
    assert calls[1][1] == agents.AGENT_VERSIONS_QUERY_PATH
    assert payload["versions"] == [{"agent_version": "v2", "invocation_count": 8}]
    assert payload["versions_total_count"] == 1


def test_agent_lens_category_breakdowns_auto_window_and_empty_diagnostics():
    async def fake_fetch(path, **kwargs):
        if path == agent_lens.LATEST_WEEK_PATH:
            return {"data": {"first_week": "2026-08-01T00:00:00Z", "latest_week": "2026-08-10T00:00:00Z"}}
        if path == agent_lens.CLUSTERING_STATUS_PATH:
            return {"data": [{"signature_type": "failure", "completed_at": "2026-08-11T00:00:00Z"}]}
        return {"data": {"turn_count": 0, "categories": []}}

    with patch.object(agent_lens, "fetch_agent_lens_json", side_effect=fake_fetch):
        payload = json.loads(asyncio.run(agent_lens.get_category_breakdowns("acme", "proj", signature_type="failure")))

    assert payload["window"]["auto_resolved"] is True
    assert payload["window"]["end_at"] == "2026-08-17T00:00:00Z"
    assert payload["coverage"]["latest_week"] == "2026-08-10T00:00:00Z"
    assert payload["empty_diagnostics"]["reason"] == "no_classified_turns_in_window"


def test_agent_lens_list_tags_includes_iso_window_distribution():
    recorded = []

    async def fake_fetch(path, **kwargs):
        recorded.append((path, kwargs))
        if path == agent_lens.TAGS_PATH:
            return {"data": [{"id": "30201f95-1221-433a-9ea5-1e513081962f", "name": "hallucination"}]}
        return {"data": {"time_bucket_seconds": 86400, "buckets": [{"count": 3}]}}

    with patch.object(agent_lens, "fetch_agent_lens_json", side_effect=fake_fetch):
        payload = json.loads(
            asyncio.run(
                agent_lens.list_tags(
                    "acme",
                    "proj",
                    include_distribution=True,
                    start_at="2026-11-01T00:00:00Z",
                    end_at="2026-11-08T00:00:00Z",
                )
            )
        )

    assert payload["data"][0]["name"] == "hallucination"
    assert payload["distribution"]["buckets"] == [{"count": 3}]
    dist_call = [entry for entry in recorded if entry[0] == agent_lens.TAG_DISTRIBUTION_PATH][0]
    assert dist_call[1]["body"]["after_ms"] < dist_call[1]["body"]["before_ms"]
