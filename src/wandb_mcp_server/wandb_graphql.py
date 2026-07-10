"""Read-only GraphQL helpers for the supported W&B SDK transport."""

from __future__ import annotations

from typing import Any, Mapping

from graphql import parse
from graphql.language import ast as gql_ast


class GraphQLReadOnlyViolation(ValueError):
    """Raised when a GraphQL document contains a non-query operation."""

    def __init__(self, operation_types: set[str]) -> None:
        self.operation_types = tuple(sorted(operation_types))
        joined_types = ", ".join(self.operation_types)
        super().__init__(
            f"query_wandb_tool accepts GraphQL query operations only; rejected operation type(s): {joined_types}."
        )


def validate_read_only_graphql(query: str) -> gql_ast.DocumentNode:
    """Parse a GraphQL document and reject mutations and subscriptions."""

    document = parse(query.strip())
    disallowed_operations = {
        definition.operation.value
        for definition in document.definitions
        if isinstance(definition, gql_ast.OperationDefinitionNode)
        and definition.operation is not gql_ast.OperationType.QUERY
    }
    if disallowed_operations:
        raise GraphQLReadOnlyViolation(disallowed_operations)
    return document


def execute_graphql(
    api: Any,
    query: str,
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a read-only GraphQL document with W&B's service transport."""
    validate_read_only_graphql(query)
    variables_dict = dict(variables or {})
    service_api = getattr(api, "_service_api", None)
    execute = getattr(service_api, "execute_graphql", None)
    if not callable(execute):
        raise RuntimeError(
            "W&B SDK compatibility error: query_wandb_tool requires wandb>=0.28.0 with ServiceApi.execute_graphql."
        )
    return execute(query, variables=variables_dict)
