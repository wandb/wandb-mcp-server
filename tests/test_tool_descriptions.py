"""Regression checks for compact, useful MCP tool descriptions."""

from pathlib import Path
import re

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from wandb_mcp_server.runtime_contract import load_runtime_contract, tools_for_profile
from wandb_mcp_server.server import create_mcp_server

from wandb_mcp_server.mcp_tools.query_weave import QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.count_traces import COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.query_wandb import QUERY_WANDB_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.query_wandb_gql import QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.create_report import CREATE_WANDB_REPORT_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.infer_schema import INFER_TRACE_SCHEMA_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.run_history import GET_RUN_HISTORY_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.docs_search import SEARCH_WANDB_DOCS_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.query_registry import (
    LIST_REGISTRIES_TOOL_DESCRIPTION,
    LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION,
)
from wandb_mcp_server.mcp_tools.query_artifacts import (
    LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION,
    COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
)
from wandb_mcp_server.mcp_tools.agents import (
    LIST_AGENTS_TOOL_DESCRIPTION,
    LIST_AGENT_VERSIONS_TOOL_DESCRIPTION,
    QUERY_AGENT_SPANS_TOOL_DESCRIPTION,
    GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION,
    LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION,
    SEARCH_AGENTS_TOOL_DESCRIPTION,
    GET_AGENT_TRACE_TOOL_DESCRIPTION,
    GET_AGENT_CONVERSATION_TOOL_DESCRIPTION,
)
from wandb_mcp_server.mcp_tools.aria import (
    ARIA_GET_TURN_TOOL_DESCRIPTION,
    ARIA_GET_TURNS_TOOL_DESCRIPTION,
    ARIA_SEND_MESSAGE_TOOL_DESCRIPTION,
)


ALL_DESCRIPTIONS = {
    "query_weave_traces": QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
    "count_weave_traces": COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    "query_wandb": QUERY_WANDB_TOOL_DESCRIPTION,
    "query_wandb_graphql": QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION,
    "create_wandb_report": CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    "list_entity_projects": LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    "infer_trace_schema": INFER_TRACE_SCHEMA_TOOL_DESCRIPTION,
    "get_run_history": GET_RUN_HISTORY_TOOL_DESCRIPTION,
    "search_wandb_docs": SEARCH_WANDB_DOCS_TOOL_DESCRIPTION,
    "list_registries": LIST_REGISTRIES_TOOL_DESCRIPTION,
    "list_registry_collections": LIST_REGISTRY_COLLECTIONS_TOOL_DESCRIPTION,
    "list_artifact_versions": LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    "get_artifact_details": GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION,
    "compare_artifact_versions": COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION,
    "list_agents": LIST_AGENTS_TOOL_DESCRIPTION,
    "list_agent_versions": LIST_AGENT_VERSIONS_TOOL_DESCRIPTION,
    "query_agent_spans": QUERY_AGENT_SPANS_TOOL_DESCRIPTION,
    "get_agent_span_stats": GET_AGENT_SPAN_STATS_TOOL_DESCRIPTION,
    "list_agent_custom_attributes": LIST_AGENT_CUSTOM_ATTRIBUTES_TOOL_DESCRIPTION,
    "search_agents": SEARCH_AGENTS_TOOL_DESCRIPTION,
    "get_agent_trace": GET_AGENT_TRACE_TOOL_DESCRIPTION,
    "get_agent_conversation": GET_AGENT_CONVERSATION_TOOL_DESCRIPTION,
    "aria_send_message": ARIA_SEND_MESSAGE_TOOL_DESCRIPTION,
    "aria_get_turn": ARIA_GET_TURN_TOOL_DESCRIPTION,
    "aria_get_turns": ARIA_GET_TURNS_TOOL_DESCRIPTION,
}

DESCRIPTION_LIMIT = 4_000
RUNTIME_CONTRACT = load_runtime_contract()
DISCOVERY_PROFILES = [
    (profile, workload, access)
    for profile, settings in RUNTIME_CONTRACT["tool_profiles"].items()
    for workload in ["local", *settings["managed_workloads"]]
    for access in RUNTIME_CONTRACT["selectors"]["access_mode"]["values"]
]


@pytest.mark.parametrize("tool_name,description", list(ALL_DESCRIPTIONS.items()))
def test_description_fits_enterprise_character_limit(tool_name, description):
    assert 0 < len(description) <= DESCRIPTION_LIMIT, (
        f"{tool_name}: {len(description)} characters exceeds {DESCRIPTION_LIMIT}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,workload,access", DISCOVERY_PROFILES)
async def test_mcp_discovery_descriptions_fit_enterprise_limit(monkeypatch, profile, workload, access):
    for name in RUNTIME_CONTRACT["legacy_tool_environment_variables"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", profile)
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", access)
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", workload)
    monkeypatch.setenv("MCP_CAPACITY_CLASS", "small")
    monkeypatch.setenv("WF_TRACE_SERVER_URL", "https://trace.wandb.ai")
    monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")
    monkeypatch.setenv("AGENT_LENS_BASE_URL", "https://agent-lens.example.com")
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")

    server = create_mcp_server("stdio")
    async with create_connected_server_and_client_session(server) as session:
        result = await session.list_tools()

    assert {tool.name for tool in result.tools} == set(tools_for_profile(RUNTIME_CONTRACT, profile, access))
    for tool in result.tools:
        assert tool.description and len(tool.description) <= DESCRIPTION_LIMIT, (
            f"{profile}/{workload}/{access}: {tool.name} has "
            f"{len(tool.description or '')} characters (limit {DESCRIPTION_LIMIT})"
        )


@pytest.mark.parametrize(
    "description",
    [
        QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
        CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
        QUERY_WANDB_TOOL_DESCRIPTION,
        GET_RUN_HISTORY_TOOL_DESCRIPTION,
    ],
)
def test_extended_guidance_links_resolve_to_documented_sections(description):
    match = re.search(
        r"https://github.com/wandb/wandb-mcp-server/blob/main/(docs/[^#\s]+)#([\w-]+)",
        description,
    )
    assert match, "Compact description must link to extended guidance"
    relative_path, anchor = match.groups()
    guidance = (Path(__file__).resolve().parents[1] / relative_path).read_text()
    headings = re.findall(r"^## (.+)$", guidance, re.MULTILINE)
    assert anchor in {heading.lower().replace(" ", "-") for heading in headings}


def test_compact_descriptions_preserve_safety_and_scope_guidance():
    for term in ("read-only", "never mix", "mutations", "subscriptions", "partial"):
        assert term in QUERY_WANDB_TOOL_DESCRIPTION
    for term in ("explicitly asks", "internal W&B run keys", "overrides run_ids"):
        assert term in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
    for term in ("NOT regex", "narrow", "parent_ids", "call_ids", "trace_id", "total_matching_count"):
        assert term in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
    for term in ("target_not_logged", "sampled", "non_finite_counts", "key_counts_exact", "truncation"):
        assert term in GET_RUN_HISTORY_TOOL_DESCRIPTION


class TestAllToolsHaveWhenToUse:
    @pytest.mark.parametrize("tool_name,description", list(ALL_DESCRIPTIONS.items()))
    def test_has_when_to_use_open_tag(self, tool_name, description):
        assert "<when_to_use>" in description, f"{tool_name} missing <when_to_use> tag"

    @pytest.mark.parametrize("tool_name,description", list(ALL_DESCRIPTIONS.items()))
    def test_has_when_to_use_close_tag(self, tool_name, description):
        assert "</when_to_use>" in description, f"{tool_name} missing </when_to_use> tag"

    @pytest.mark.parametrize("tool_name,description", list(ALL_DESCRIPTIONS.items()))
    def test_when_to_use_has_content(self, tool_name, description):
        start = description.index("<when_to_use>") + len("<when_to_use>")
        end = description.index("</when_to_use>")
        content = description[start:end].strip()
        assert len(content) > 20, f"{tool_name} <when_to_use> section is too short"


class TestDetailLevelInQueryWeave:
    def test_detail_level_documented(self):
        assert "detail_level" in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_schema_level_documented(self):
        assert '"schema"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_full_level_documented(self):
        assert '"full"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION


class TestQueryWandbSdkRouting:
    @pytest.mark.parametrize(
        "tool_name",
        [
            "query_wandb_entity_projects",
            "get_run_history_tool",
            "list_artifact_versions_tool",
            "get_artifact_details_tool",
            "list_registries_tool",
            "list_registry_collections_tool",
            "list_wandb_automations_tool",
            "list_wandb_integrations_tool",
        ],
    )
    def test_sdk_query_description_routes_dedicated_resources(self, tool_name):
        assert tool_name in QUERY_WANDB_TOOL_DESCRIPTION

    def test_documents_remaining_raw_graphql_use_cases(self):
        for use_case in ("introspection", "unmodeled", "cross-resource", "aliases", "exact GraphQL response shape"):
            assert use_case in QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION

    @pytest.mark.parametrize(
        "sdk_resource",
        ["projects", "run lookup", "sweeps", "reports", "artifacts", "registries", "automations", "integrations"],
    )
    def test_raw_graphql_description_rejects_sdk_parity_use_cases(self, sdk_resource):
        assert sdk_resource in QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION


class TestRunHistoryDescription:
    def test_documents_bounded_non_finite_counts(self):
        assert "non_finite_counts" in GET_RUN_HISTORY_TOOL_DESCRIPTION
        assert "key_counts_exact" in GET_RUN_HISTORY_TOOL_DESCRIPTION

    def test_documents_source_truncation_and_custom_ids(self):
        assert "source step-window/row cap reached" in GET_RUN_HISTORY_TOOL_DESCRIPTION
        assert "custom run IDs are also accepted" in GET_RUN_HISTORY_TOOL_DESCRIPTION


class TestPanelsInCreateReport:
    def test_panels_documented(self):
        assert "panels" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION

    def test_line_type_documented(self):
        assert "line" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()

    def test_bar_type_documented(self):
        assert "bar" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()

    def test_layout_types_documented(self):
        assert "panel_grid" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "heading" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "markdown" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION

    def test_custom_chart_sources_documented(self):
        assert "custom_chart" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "custom_chart_table" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "summaryTable" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "historyTable" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION

    def test_run_filtering_documented(self):
        assert "run_ids are converted to deterministic Reports v2 filters" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
        assert "filters may be passed as a Reports v2 expression string" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION
