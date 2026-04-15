"""Tests for inputs/output content search filters in QueryBuilder."""

from wandb_mcp_server.weave_api.query_builder import QueryBuilder


class TestInputsContentSearch:
    """QueryBuilder.build_query_expression handles inputs filters."""

    def test_inputs_contains_produces_operation(self):
        filters = {"inputs": {"message": {"$contains": "hello world"}}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        expr = query.expr_
        assert hasattr(expr, "contains_")
        spec = expr.contains_
        assert spec.input.get_field_ == "inputs.message"
        assert spec.substr.literal_ == "hello world"

    def test_inputs_nested_path(self):
        filters = {"inputs": {"model.prompt": {"$contains": "summarize"}}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        spec = query.expr_.contains_
        assert spec.input.get_field_ == "inputs.model.prompt"

    def test_inputs_exact_match(self):
        filters = {"inputs": {"model": "gpt-4"}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None

    def test_inputs_comparison_operator(self):
        filters = {"inputs": {"token_count": {"$gt": 100}}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None

    def test_inputs_combined_with_other_filters(self):
        filters = {
            "inputs": {"message": {"$contains": "search term"}},
            "op_name_contains": "predict",
            "status": "success",
        }
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        assert hasattr(query.expr_, "and_")
        assert len(query.expr_.and_) == 3

    def test_inputs_multiple_fields(self):
        filters = {
            "inputs": {
                "message": {"$contains": "hello"},
                "model": "gpt-4",
            }
        }
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        assert hasattr(query.expr_, "and_")
        assert len(query.expr_.and_) == 2


class TestOutputContentSearch:
    """QueryBuilder.build_query_expression handles output filters."""

    def test_output_top_level_contains(self):
        filters = {"output": {"$contains": "error message"}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        spec = query.expr_.contains_
        assert spec.input.get_field_ == "output"
        assert spec.substr.literal_ == "error message"

    def test_output_nested_path(self):
        filters = {"output": {"result": {"$contains": "success"}}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        spec = query.expr_.contains_
        assert spec.input.get_field_ == "output.result"

    def test_output_combined_with_inputs(self):
        filters = {
            "inputs": {"message": {"$contains": "query"}},
            "output": {"$contains": "response"},
        }
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        assert hasattr(query.expr_, "and_")
        assert len(query.expr_.and_) == 2

    def test_output_combined_with_time_range(self):
        filters = {
            "output": {"$contains": "specific text"},
            "time_range": {
                "start": "2026-04-01T00:00:00Z",
                "end": "2026-04-13T00:00:00Z",
            },
        }
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        assert hasattr(query.expr_, "and_")
        assert len(query.expr_.and_) == 3


class TestContentSearchEdgeCases:
    """Edge cases for content search filters."""

    def test_empty_inputs_dict(self):
        filters = {"inputs": {}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is None

    def test_empty_output_dict(self):
        filters = {"output": {}}
        query = QueryBuilder.build_query_expression(filters)
        assert query is None

    def test_inputs_does_not_conflict_with_attributes(self):
        filters = {
            "inputs": {"message": {"$contains": "hello"}},
            "attributes": {"env": "production"},
        }
        query = QueryBuilder.build_query_expression(filters)
        assert query is not None
        assert hasattr(query.expr_, "and_")
        assert len(query.expr_.and_) == 2

    def test_case_insensitive_default(self):
        filters = {"inputs": {"text": {"$contains": "UPPER"}}}
        query = QueryBuilder.build_query_expression(filters)
        spec = query.expr_.contains_
        assert spec.case_insensitive is True
