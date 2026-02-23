"""Unit tests for _prefix_gql_operation_name() -- GQL operation naming (MCP-9).

Jira MCP-9 acceptance criteria: "Unit test validates prefix on 5 patterns."
"""

from wandb_mcp_server.mcp_tools.query_wandb_gql import _prefix_gql_operation_name


class TestPrefixGqlOperationName:

    def test_named_query_gets_prefix(self):
        query = "query GetRuns { project { runs { edges { node { name } } } } }"
        result = _prefix_gql_operation_name(query)
        assert "mcp_GetRuns" in result
        assert "query mcp_GetRuns" in result

    def test_named_mutation_gets_prefix(self):
        query = "mutation CreateReport($input: CreateReportInput!) { createReport(input: $input) { id } }"
        result = _prefix_gql_operation_name(query)
        assert "mcp_CreateReport" in result

    def test_already_prefixed_unchanged(self):
        query = "query mcp_GetRuns { project { runs { edges { node { name } } } } }"
        result = _prefix_gql_operation_name(query)
        assert "mcp_mcp_" not in result
        assert "mcp_GetRuns" in result

    def test_anonymous_query_unchanged(self):
        query = '{ project(name: "test") { runs { edges { node { name } } } } }'
        result = _prefix_gql_operation_name(query)
        assert "mcp_" not in result

    def test_malformed_query_returned_as_is(self):
        query = "this is not valid graphql {"
        result = _prefix_gql_operation_name(query)
        assert result == query

    def test_complex_query_with_variables(self):
        query = """query FilteredRuns($entity: String!, $project: String!, $limit: Int) {
            project(name: $project, entityName: $entity) {
                runs(first: $limit) { edges { node { name displayName state } } }
            }
        }"""
        result = _prefix_gql_operation_name(query)
        assert "mcp_FilteredRuns" in result

    def test_custom_prefix(self):
        query = "query MyOp { field }"
        result = _prefix_gql_operation_name(query, prefix="custom_")
        assert "custom_MyOp" in result
        assert "mcp_" not in result

    def test_subscription_gets_prefix(self):
        query = "subscription WatchRuns { runUpdated { id name } }"
        result = _prefix_gql_operation_name(query)
        assert "mcp_WatchRuns" in result

    def test_empty_string_returned_as_is(self):
        result = _prefix_gql_operation_name("")
        assert result == ""
