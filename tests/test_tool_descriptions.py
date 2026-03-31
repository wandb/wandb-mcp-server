"""Tests verifying all tool descriptions contain <when_to_use> sections."""

import pytest

from wandb_mcp_server.mcp_tools.query_weave import QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.count_traces import COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.query_wandb_gql import QUERY_WANDB_GQL_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.create_report import CREATE_WANDB_REPORT_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.query_wandbot import WANDBOT_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.infer_schema import INFER_TRACE_SCHEMA_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.run_history import GET_RUN_HISTORY_TOOL_DESCRIPTION
from wandb_mcp_server.mcp_tools.docs_search import SEARCH_WANDB_DOCS_TOOL_DESCRIPTION


ALL_DESCRIPTIONS = {
    "query_weave_traces": QUERY_WEAVE_TRACES_TOOL_DESCRIPTION,
    "count_weave_traces": COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    "query_wandb_gql": QUERY_WANDB_GQL_TOOL_DESCRIPTION,
    "create_wandb_report": CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    "list_entity_projects": LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    "query_wandb_support_bot": WANDBOT_TOOL_DESCRIPTION,
    "infer_trace_schema": INFER_TRACE_SCHEMA_TOOL_DESCRIPTION,
    "get_run_history": GET_RUN_HISTORY_TOOL_DESCRIPTION,
    "search_wandb_docs": SEARCH_WANDB_DOCS_TOOL_DESCRIPTION,
}


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


class TestWandbotDeprecated:
    def test_wandbot_marked_deprecated(self):
        assert "[DEPRECATED]" in WANDBOT_TOOL_DESCRIPTION

    def test_wandbot_references_replacement(self):
        assert "search_wandb_docs_tool" in WANDBOT_TOOL_DESCRIPTION

    def test_wandbot_when_to_use_says_avoid(self):
        assert "AVOID" in WANDBOT_TOOL_DESCRIPTION or "avoid" in WANDBOT_TOOL_DESCRIPTION.lower()


class TestDetailLevelInQueryWeave:
    def test_detail_level_documented(self):
        assert "detail_level" in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_schema_level_documented(self):
        assert '"schema"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION

    def test_full_level_documented(self):
        assert '"full"' in QUERY_WEAVE_TRACES_TOOL_DESCRIPTION


class TestPanelsInCreateReport:
    def test_panels_documented(self):
        assert "panels" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION

    def test_line_type_documented(self):
        assert "line" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()

    def test_bar_type_documented(self):
        assert "bar" in CREATE_WANDB_REPORT_TOOL_DESCRIPTION.lower()
