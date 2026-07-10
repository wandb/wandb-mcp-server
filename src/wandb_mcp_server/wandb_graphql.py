"""GraphQL helpers compatible across W&B SDK GraphQL transports."""

from __future__ import annotations

import importlib
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
    """Execute a read-only GraphQL document with the current or legacy W&B SDK transport."""
    validate_read_only_graphql(query)
    variables_dict = dict(variables or {})
    service_api = getattr(api, "__dict__", {}).get("_service_api")
    if service_api is not None and hasattr(service_api, "execute_graphql"):
        return service_api.execute_graphql(query, variables=variables_dict)

    gql = importlib.import_module("wandb_gql").gql
    return api.client.execute(gql(query), variable_values=variables_dict)
