"""Bounded, opt-in, query-only access to the W&B GraphQL API."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from graphql.language import ast as gql_ast
from graphql.language import printer as gql_printer

from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_graphql import (
    GraphQLReadOnlyViolation,
    GraphQLResponseTooLarge,
    execute_graphql,
    validate_read_only_graphql,
)

logger = get_rich_logger(__name__)

MAX_GRAPHQL_DOCUMENT_BYTES = 64 * 1024
MAX_GRAPHQL_DEPTH = 12
MAX_GRAPHQL_FIELDS = 200
MAX_GRAPHQL_FRAGMENTS = 32
MAX_GRAPHQL_VARIABLE_DEPTH = 12
MAX_GRAPHQL_VARIABLE_NODES = 5_000
MAX_GRAPHQL_EXPANDED_SELECTIONS = MAX_GRAPHQL_FIELDS * 4

QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION = """Execute a bounded, query-only GraphQL document against W&B Models.

This opt-in compatibility escape hatch is only for reads without public W&B API
parity: schema introspection, unmodeled/custom fields, aliases, an exact GraphQL response shape,
cross-resource nesting, sweep agents, report run sets, Launch resources, compound
reads, and backward-pagination shapes.

<when_to_use>
Use only when the requested read cannot be represented by a typed MCP tool and
the deployment administrator deliberately enabled raw GraphQL.
</when_to_use>

Do not use this tool for projects, run lookup/filtering/sorting, sweeps, reports,
artifacts, registries, automations, integrations, or run history. Those reads
have bounded typed MCP tools.

The document must contain exactly one query operation. Mutations, subscriptions,
mixed documents, multiple operations, nested/multiple paginated connections, and
unbounded connections are rejected before any W&B request. Pagination and MCP
output are bounded. On the supported ServiceApi path, response text is checked
before MCP JSON decoding and decoded structure is checked again; these checks
occur after the SDK receives the protobuf envelope and are not a wire-size or
backend-work limit.
"""


class GraphQLQueryValidationError(ValueError):
    """Raised before API construction when an opt-in raw query is unsafe."""

    def __init__(self, message: str, *, error: str = "invalid_request") -> None:
        self.error = error
        super().__init__(message)


@dataclass
class _ConnectionPlan:
    path: list[str]
    field: gql_ast.FieldNode
    first_variable: str | None
    after_variable: str | None
    initial_after: str | None
    initial_before: str | None
    edges_key: str | None
    page_info_key: str | None
    edge_cursor_key: str | None
    has_next_page_key: str | None
    end_cursor_key: str | None
    has_previous_page_key: str | None
    start_cursor_key: str | None
    backward: bool = False


@dataclass
class _SelectionExpansionBudget:
    visited: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.visited > MAX_GRAPHQL_EXPANDED_SELECTIONS:
            raise GraphQLQueryValidationError(
                f"GraphQL expanded selections are limited to {MAX_GRAPHQL_EXPANDED_SELECTIONS}",
                error="query_too_complex",
            )


def find_paginated_collections(obj: Dict, current_path: Optional[List[str]] = None) -> List[List[str]]:
    """Find W&B-style connection objects in a response."""
    path = [] if current_path is None else current_path
    collections: list[list[str]] = []
    if isinstance(obj, dict):
        if (
            isinstance(obj.get("edges"), list)
            and isinstance(obj.get("pageInfo"), dict)
            and "hasNextPage" in obj["pageInfo"]
            and "endCursor" in obj["pageInfo"]
        ):
            collections.append(list(path))
        for key, value in obj.items():
            path.append(key)
            collections.extend(find_paginated_collections(value, path))
            path.pop()
    elif isinstance(obj, list):
        for item in obj:
            collections.extend(find_paginated_collections(item, path))
    return collections


def get_nested_value(obj: Dict, path: list[str]) -> Optional[Any]:
    current: Any = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _field_name(node: gql_ast.FieldNode) -> str:
    return node.alias.value if node.alias else node.name.value


def _selection_field_names(
    selection_set: gql_ast.SelectionSetNode | None,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
    stack: tuple[str, ...] = (),
    budget: _SelectionExpansionBudget | None = None,
) -> set[str]:
    names: set[str] = set()
    if selection_set is None:
        return names
    budget = budget or _SelectionExpansionBudget()
    for selection in selection_set.selections:
        budget.consume()
        if isinstance(selection, gql_ast.FieldNode):
            names.add(selection.name.value)
        elif isinstance(selection, gql_ast.InlineFragmentNode):
            names.update(_selection_field_names(selection.selection_set, fragments, stack, budget))
        elif isinstance(selection, gql_ast.FragmentSpreadNode):
            fragment_name = selection.name.value
            if fragment_name in stack:
                raise GraphQLQueryValidationError("GraphQL fragment cycles are not allowed")
            fragment = fragments.get(fragment_name)
            if fragment is None:
                raise GraphQLQueryValidationError(f"Unknown GraphQL fragment: {fragment_name}")
            names.update(
                _selection_field_names(
                    fragment.selection_set,
                    fragments,
                    (*stack, fragment_name),
                    budget,
                )
            )
    return names


def _selection_fields(
    selection_set: gql_ast.SelectionSetNode | None,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
    stack: tuple[str, ...] = (),
    budget: _SelectionExpansionBudget | None = None,
) -> list[gql_ast.FieldNode]:
    """Expand one selection level, retaining response aliases."""
    fields: list[gql_ast.FieldNode] = []
    if selection_set is None:
        return fields
    budget = budget or _SelectionExpansionBudget()
    for selection in selection_set.selections:
        budget.consume()
        if isinstance(selection, gql_ast.FieldNode):
            fields.append(selection)
        elif isinstance(selection, gql_ast.InlineFragmentNode):
            fields.extend(_selection_fields(selection.selection_set, fragments, stack, budget))
        elif isinstance(selection, gql_ast.FragmentSpreadNode):
            fragment_name = selection.name.value
            if fragment_name in stack:
                raise GraphQLQueryValidationError("GraphQL fragment cycles are not allowed")
            fragment = fragments.get(fragment_name)
            if fragment is None:
                raise GraphQLQueryValidationError(f"Unknown GraphQL fragment: {fragment_name}")
            fields.extend(
                _selection_fields(
                    fragment.selection_set,
                    fragments,
                    (*stack, fragment_name),
                    budget,
                )
            )
    return fields


def _response_key(
    selection_set: gql_ast.SelectionSetNode | None,
    field_name: str,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
    budget: _SelectionExpansionBudget,
) -> str | None:
    """Return the unique response key selected for a schema field."""
    keys = {
        _field_name(field)
        for field in _selection_fields(selection_set, fragments, budget=budget)
        if field.name.value == field_name
    }
    if len(keys) > 1:
        raise GraphQLQueryValidationError(
            f"Paginated connection selects {field_name!r} through multiple aliases",
            error="query_too_complex",
        )
    return next(iter(keys), None)


def _nested_response_key(
    selection_set: gql_ast.SelectionSetNode | None,
    parent_field_name: str,
    child_field_name: str,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
    budget: _SelectionExpansionBudget,
) -> str | None:
    """Return a unique response key selected below a connection metadata field."""
    keys: set[str] = set()
    for field in _selection_fields(selection_set, fragments, budget=budget):
        if field.name.value != parent_field_name:
            continue
        key = _response_key(field.selection_set, child_field_name, fragments, budget)
        if key is not None:
            keys.add(key)
    if len(keys) > 1:
        raise GraphQLQueryValidationError(
            (f"Paginated connection selects {parent_field_name}.{child_field_name} through multiple aliases"),
            error="query_too_complex",
        )
    return next(iter(keys), None)


def _is_connection_field(
    node: gql_ast.FieldNode,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
    budget: _SelectionExpansionBudget,
) -> bool:
    argument_names = {argument.name.value for argument in node.arguments or ()}
    if argument_names & {"first", "last"}:
        return True
    return {"edges", "pageInfo"}.issubset(_selection_field_names(node.selection_set, fragments, budget=budget))


def _int_variable_defaults(operation: gql_ast.OperationDefinitionNode) -> dict[str, int]:
    defaults: dict[str, int] = {}
    for definition in operation.variable_definitions or ():
        if isinstance(definition.default_value, gql_ast.IntValueNode):
            defaults[definition.variable.name.value] = int(definition.default_value.value)
    return defaults


def _string_variable_defaults(operation: gql_ast.OperationDefinitionNode) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for definition in operation.variable_definitions or ():
        if isinstance(definition.default_value, gql_ast.StringValueNode):
            defaults[definition.variable.name.value] = definition.default_value.value
    return defaults


def _validate_variable_shape(value: Any) -> None:
    """Reject deeply nested or pathologically broad JSON variables."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_GRAPHQL_VARIABLE_NODES:
            raise GraphQLQueryValidationError(
                f"GraphQL variables may contain at most {MAX_GRAPHQL_VARIABLE_NODES} JSON values",
                error="query_too_complex",
            )
        if depth > MAX_GRAPHQL_VARIABLE_DEPTH:
            raise GraphQLQueryValidationError(
                f"GraphQL variables are limited to depth {MAX_GRAPHQL_VARIABLE_DEPTH}",
                error="query_too_complex",
            )
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def _replace_argument(
    field: gql_ast.FieldNode,
    name: str,
    value: gql_ast.ValueNode,
) -> None:
    arguments = list(field.arguments or ())
    for index, argument in enumerate(arguments):
        if argument.name.value == name:
            arguments[index] = gql_ast.ArgumentNode(name=argument.name, value=value)
            field.arguments = tuple(arguments)
            return
    arguments.append(gql_ast.ArgumentNode(name=gql_ast.NameNode(value=name), value=value))
    field.arguments = tuple(arguments)


def _ensure_variable_definition(
    operation: gql_ast.OperationDefinitionNode,
    variable_name: str,
    type_name: str,
) -> None:
    definitions = list(operation.variable_definitions or ())
    if any(definition.variable.name.value == variable_name for definition in definitions):
        return
    definitions.append(
        gql_ast.VariableDefinitionNode(
            variable=gql_ast.VariableNode(name=gql_ast.NameNode(value=variable_name)),
            type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value=type_name)),
        )
    )
    operation.variable_definitions = tuple(definitions)


def _analyze_and_bound_document(
    query: str,
    variables: Optional[Mapping[str, Any]],
    page_size: int,
) -> tuple[str, dict[str, Any], _ConnectionPlan | None]:
    if not isinstance(query, str) or not query.strip():
        raise GraphQLQueryValidationError("query must be a non-empty string")
    if len(query.encode("utf-8")) > MAX_GRAPHQL_DOCUMENT_BYTES:
        raise GraphQLQueryValidationError("GraphQL document exceeds the 64 KiB limit", error="query_too_complex")
    if variables is not None and not isinstance(variables, Mapping):
        raise GraphQLQueryValidationError("variables must be a mapping")
    try:
        serialized_variables = json.dumps(
            dict(variables or {}),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, ValueError):
        raise GraphQLQueryValidationError("variables must be finite JSON-compatible values") from None
    variables_bytes = len(serialized_variables.encode("utf-8"))
    if variables_bytes > MAX_GRAPHQL_DOCUMENT_BYTES:
        raise GraphQLQueryValidationError(
            "GraphQL variables exceed the 64 KiB limit",
            error="query_too_complex",
        )
    try:
        bounded_variables = json.loads(serialized_variables)
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
        raise GraphQLQueryValidationError("variables must be finite JSON-compatible values") from None
    _validate_variable_shape(bounded_variables)

    document = validate_read_only_graphql(query)
    operations = [
        definition for definition in document.definitions if isinstance(definition, gql_ast.OperationDefinitionNode)
    ]
    if len(operations) != 1:
        raise GraphQLQueryValidationError(
            "GraphQL documents must contain exactly one query operation",
            error="query_too_complex",
        )
    operation = operations[0]
    fragments = {
        definition.name.value: definition
        for definition in document.definitions
        if isinstance(definition, gql_ast.FragmentDefinitionNode)
    }
    if len(fragments) > MAX_GRAPHQL_FRAGMENTS:
        raise GraphQLQueryValidationError(
            f"GraphQL documents may contain at most {MAX_GRAPHQL_FRAGMENTS} fragments",
            error="query_too_complex",
        )

    connections: list[tuple[list[str], gql_ast.FieldNode]] = []
    field_count = 0
    max_depth = 0
    expansion_budget = _SelectionExpansionBudget()

    def walk(
        selection_set: gql_ast.SelectionSetNode | None,
        path: list[str],
        depth: int,
        connection_ancestor: bool,
        fragment_stack: tuple[str, ...] = (),
    ) -> None:
        nonlocal field_count, max_depth
        if selection_set is None:
            return
        for selection in selection_set.selections:
            expansion_budget.consume()
            if isinstance(selection, gql_ast.FieldNode):
                field_count += 1
                max_depth = max(max_depth, depth)
                if field_count > MAX_GRAPHQL_FIELDS or max_depth > MAX_GRAPHQL_DEPTH:
                    raise GraphQLQueryValidationError(
                        (
                            f"GraphQL documents are limited to {MAX_GRAPHQL_FIELDS} selected fields "
                            f"and depth {MAX_GRAPHQL_DEPTH}"
                        ),
                        error="query_too_complex",
                    )
                field_path = [*path, _field_name(selection)]
                is_connection = _is_connection_field(selection, fragments, expansion_budget)
                if is_connection:
                    if connection_ancestor:
                        raise GraphQLQueryValidationError(
                            f"Nested paginated connection is not allowed: {'/'.join(field_path)}",
                            error="query_too_complex",
                        )
                    connections.append((field_path, selection))
                    if len(connections) > 1:
                        raise GraphQLQueryValidationError(
                            "GraphQL documents may contain only one paginated connection",
                            error="query_too_complex",
                        )
                walk(
                    selection.selection_set,
                    field_path,
                    depth + 1,
                    connection_ancestor or is_connection,
                    fragment_stack,
                )
            elif isinstance(selection, gql_ast.InlineFragmentNode):
                walk(selection.selection_set, path, depth, connection_ancestor, fragment_stack)
            elif isinstance(selection, gql_ast.FragmentSpreadNode):
                fragment_name = selection.name.value
                if fragment_name in fragment_stack:
                    raise GraphQLQueryValidationError("GraphQL fragment cycles are not allowed")
                fragment = fragments.get(fragment_name)
                if fragment is None:
                    raise GraphQLQueryValidationError(f"Unknown GraphQL fragment: {fragment_name}")
                walk(
                    fragment.selection_set,
                    path,
                    depth,
                    connection_ancestor,
                    (*fragment_stack, fragment_name),
                )

    walk(operation.selection_set, [], 1, False)
    if not connections:
        return gql_printer.print_ast(document), bounded_variables, None

    connection_path, connection_field = connections[0]
    arguments = {argument.name.value: argument for argument in connection_field.arguments or ()}
    first = arguments.get("first")
    last = arguments.get("last")
    if first is not None and last is not None:
        raise GraphQLQueryValidationError("A paginated connection cannot use both first and last")
    if first is None and last is None:
        raise GraphQLQueryValidationError(
            f"Paginated connection must include first or last: {'/'.join(connection_path)}",
            error="query_too_complex",
        )

    first_variable: str | None = None
    pagination_argument = first or last
    assert pagination_argument is not None
    argument_name = "first" if first is not None else "last"
    if isinstance(pagination_argument.value, gql_ast.IntValueNode):
        requested_first = int(pagination_argument.value.value)
        if requested_first < 1:
            raise GraphQLQueryValidationError(f"GraphQL {argument_name} must be a positive integer")
        _replace_argument(
            connection_field,
            argument_name,
            gql_ast.IntValueNode(value=str(min(requested_first, page_size))),
        )
    elif isinstance(pagination_argument.value, gql_ast.VariableNode):
        first_variable = pagination_argument.value.name.value
        defaults = _int_variable_defaults(operation)
        requested_first = bounded_variables.get(first_variable, defaults.get(first_variable, page_size))
        if isinstance(requested_first, bool) or not isinstance(requested_first, int) or requested_first < 1:
            raise GraphQLQueryValidationError(
                f"GraphQL variable ${first_variable}, bound to {argument_name}, must be a positive integer"
            )
        bounded_variables[first_variable] = min(requested_first, page_size)
    else:
        raise GraphQLQueryValidationError(f"GraphQL {argument_name} must be an integer literal or variable")

    edges_key = _response_key(connection_field.selection_set, "edges", fragments, expansion_budget)
    page_info_key = _response_key(connection_field.selection_set, "pageInfo", fragments, expansion_budget)
    edge_cursor_key = _nested_response_key(
        connection_field.selection_set,
        "edges",
        "cursor",
        fragments,
        expansion_budget,
    )
    has_next_page_key = _nested_response_key(
        connection_field.selection_set,
        "pageInfo",
        "hasNextPage",
        fragments,
        expansion_budget,
    )
    end_cursor_key = _nested_response_key(
        connection_field.selection_set,
        "pageInfo",
        "endCursor",
        fragments,
        expansion_budget,
    )
    has_previous_page_key = _nested_response_key(
        connection_field.selection_set,
        "pageInfo",
        "hasPreviousPage",
        fragments,
        expansion_budget,
    )
    start_cursor_key = _nested_response_key(
        connection_field.selection_set,
        "pageInfo",
        "startCursor",
        fragments,
        expansion_budget,
    )

    if last is not None:
        before = arguments.get("before")
        initial_before: str | None = None
        if before is not None and isinstance(before.value, gql_ast.VariableNode):
            before_variable = before.value.name.value
            initial_before = bounded_variables.get(
                before_variable,
                _string_variable_defaults(operation).get(before_variable),
            )
            if initial_before is not None and not isinstance(initial_before, str):
                raise GraphQLQueryValidationError(f"GraphQL variable ${before_variable} must be a string or null")
        elif before is not None and isinstance(before.value, gql_ast.StringValueNode):
            initial_before = before.value.value
        elif before is not None and not isinstance(before.value, gql_ast.NullValueNode):
            raise GraphQLQueryValidationError("GraphQL before must be a string literal, variable, or null")
        return (
            gql_printer.print_ast(document),
            bounded_variables,
            _ConnectionPlan(
                path=connection_path,
                field=connection_field,
                first_variable=first_variable,
                after_variable=None,
                initial_after=None,
                initial_before=initial_before,
                edges_key=edges_key,
                page_info_key=page_info_key,
                edge_cursor_key=edge_cursor_key,
                has_next_page_key=has_next_page_key,
                end_cursor_key=end_cursor_key,
                has_previous_page_key=has_previous_page_key,
                start_cursor_key=start_cursor_key,
                backward=True,
            ),
        )

    after = arguments.get("after")
    after_variable = "__mcp_after"
    initial_after: str | None = None
    if after is not None and isinstance(after.value, gql_ast.VariableNode):
        after_variable = after.value.name.value
        supplied_after = bounded_variables.get(
            after_variable,
            _string_variable_defaults(operation).get(after_variable),
        )
        if supplied_after is not None and not isinstance(supplied_after, str):
            raise GraphQLQueryValidationError(f"GraphQL variable ${after_variable} must be a string or null")
        initial_after = supplied_after
    else:
        if after is not None:
            if isinstance(after.value, gql_ast.StringValueNode):
                initial_after = after.value.value
            elif not isinstance(after.value, gql_ast.NullValueNode):
                raise GraphQLQueryValidationError("GraphQL after must be a string literal, variable, or null")
        _ensure_variable_definition(operation, after_variable, "String")
        _replace_argument(
            connection_field,
            "after",
            gql_ast.VariableNode(name=gql_ast.NameNode(value=after_variable)),
        )
        bounded_variables[after_variable] = initial_after

    return (
        gql_printer.print_ast(document),
        bounded_variables,
        _ConnectionPlan(
            path=connection_path,
            field=connection_field,
            first_variable=first_variable,
            after_variable=after_variable,
            initial_after=initial_after,
            initial_before=None,
            edges_key=edges_key,
            page_info_key=page_info_key,
            edge_cursor_key=edge_cursor_key,
            has_next_page_key=has_next_page_key,
            end_cursor_key=end_cursor_key,
            has_previous_page_key=has_previous_page_key,
            start_cursor_key=start_cursor_key,
        ),
    )


def _estimate_tokens(payload: Mapping[str, Any]) -> int:
    from wandb_mcp_server.trace_utils import count_tokens_conservative

    serialized = json.dumps(payload, default=str, ensure_ascii=False, allow_nan=False)
    return count_tokens_conservative(serialized)


def _set_pagination_extension(
    result: dict[str, Any],
    *,
    returned_count: int,
    has_more: bool,
    next_cursor: str | None,
    truncated_by_budget: bool = False,
) -> None:
    extensions = result.setdefault("extensions", {})
    if not isinstance(extensions, dict):
        extensions = {}
        result["extensions"] = extensions
    extensions["wandb_mcp"] = {
        "returned_count": returned_count,
        "has_more": has_more,
        "next_cursor": next_cursor if has_more else None,
        "truncated_by_response_budget": truncated_by_budget,
    }


def _response_too_large(message: str) -> dict[str, Any]:
    return {
        "errors": [
            {
                "error": "response_too_large",
                "message": message,
            }
        ]
    }


def _edge_cursor(edge: Any, plan: _ConnectionPlan) -> str | None:
    if not isinstance(edge, Mapping) or plan.edge_cursor_key is None:
        return None
    value = edge.get(plan.edge_cursor_key)
    return value if isinstance(value, str) and value else None


def _pagination_error(error: str, message: str) -> dict[str, Any]:
    return {"errors": [{"error": error, "message": message}]}


def _is_non_advancing_cursor(cursor: str, plan: _ConnectionPlan) -> bool:
    initial_cursor = plan.initial_before if plan.backward else plan.initial_after
    return initial_cursor is not None and cursor == initial_cursor


def _bound_unpageable_connection(
    result: dict[str, Any],
    connection: dict[str, Any],
    edges: list[Any],
    plan: _ConnectionPlan,
    max_items: int,
) -> dict[str, Any]:
    """Enforce the local item cap when response pagination metadata is unusable."""
    if len(edges) <= max_items:
        return _fit_response_budget(result, plan)

    retained = edges[-max_items:] if plan.backward else edges[:max_items]
    boundary = retained[0] if plan.backward else retained[-1]
    resume_cursor = _edge_cursor(boundary, plan)
    if resume_cursor is None:
        return _pagination_error(
            "pagination_cursor_unavailable",
            ("The bounded GraphQL page exceeded the item limit but no selected edge cursor permits safe continuation"),
        )
    if _is_non_advancing_cursor(resume_cursor, plan):
        return _pagination_error(
            "pagination_cursor_non_advancing",
            "The GraphQL connection returned a repeated or non-advancing continuation cursor",
        )

    connection[plan.edges_key] = retained
    if plan.page_info_key is not None:
        selected_page_info = connection.get(plan.page_info_key)
        if isinstance(selected_page_info, Mapping):
            page_info = dict(selected_page_info)
            if plan.backward:
                if plan.has_previous_page_key is not None:
                    page_info[plan.has_previous_page_key] = True
                if plan.start_cursor_key is not None:
                    page_info[plan.start_cursor_key] = resume_cursor
            else:
                if plan.has_next_page_key is not None:
                    page_info[plan.has_next_page_key] = True
                if plan.end_cursor_key is not None:
                    page_info[plan.end_cursor_key] = resume_cursor
            connection[plan.page_info_key] = page_info
    _set_pagination_extension(
        result,
        returned_count=len(retained),
        has_more=True,
        next_cursor=resume_cursor,
    )
    result["extensions"]["wandb_mcp"]["limit_applied"] = max_items
    return _fit_response_budget(result, plan)


def _fit_response_budget(
    result: dict[str, Any],
    plan: _ConnectionPlan | None,
) -> dict[str, Any]:
    from wandb_mcp_server.config import MAX_RESPONSE_TOKENS

    try:
        if _estimate_tokens(result) <= MAX_RESPONSE_TOKENS:
            return result
    except (TypeError, ValueError):
        return _response_too_large("GraphQL response could not be serialized safely")
    if plan is None or plan.edges_key is None:
        return _response_too_large("GraphQL response exceeded the configured response budget")
    connection = get_nested_value(result, plan.path)
    if not isinstance(connection, dict):
        return _response_too_large("GraphQL response exceeded the configured response budget")
    edges = connection.get(plan.edges_key)
    if not isinstance(edges, list):
        return _response_too_large("GraphQL response exceeded the configured response budget")

    page_info: dict[str, Any] = {}
    if plan.page_info_key is not None:
        selected_page_info = connection.get(plan.page_info_key)
        if isinstance(selected_page_info, Mapping):
            page_info = dict(selected_page_info)

    while edges:
        if plan.backward:
            edges.pop(0)
        else:
            edges.pop()
        if not edges:
            break
        resume_cursor = _edge_cursor(edges[0] if plan.backward else edges[-1], plan)
        if resume_cursor is None:
            return _response_too_large(
                "GraphQL response exceeded the budget and no selected edge cursor permits safe continuation"
            )
        if _is_non_advancing_cursor(resume_cursor, plan):
            return _pagination_error(
                "pagination_cursor_non_advancing",
                "The GraphQL connection returned a repeated or non-advancing continuation cursor",
            )
        if plan.backward:
            if plan.has_previous_page_key is not None:
                page_info[plan.has_previous_page_key] = True
            if plan.start_cursor_key is not None:
                page_info[plan.start_cursor_key] = resume_cursor
        else:
            if plan.has_next_page_key is not None:
                page_info[plan.has_next_page_key] = True
            if plan.end_cursor_key is not None:
                page_info[plan.end_cursor_key] = resume_cursor
        if plan.page_info_key is not None:
            connection[plan.page_info_key] = page_info
        _set_pagination_extension(
            result,
            returned_count=len(edges),
            has_more=True,
            next_cursor=resume_cursor,
            truncated_by_budget=True,
        )
        try:
            if _estimate_tokens(result) <= MAX_RESPONSE_TOKENS:
                return result
        except (TypeError, ValueError):
            break
    return _response_too_large("GraphQL response exceeded the budget and could not retain one safely continuable edge")


def _validation_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, GraphQLReadOnlyViolation):
        return {
            "errors": [
                {
                    "error": "read_only_violation",
                    "message": str(exc),
                    "operation_types": list(exc.operation_types),
                }
            ]
        }
    return {
        "errors": [
            {
                "error": getattr(exc, "error", "invalid_request"),
                "message": str(exc),
            }
        ]
    }


def query_paginated_wandb_gql(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    max_items: int = 100,
    items_per_page: int = 20,
) -> Dict[str, Any]:
    """Execute one bounded read-only GraphQL operation."""
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
        return _validation_error(GraphQLQueryValidationError("max_items must be a positive integer"))
    if isinstance(items_per_page, bool) or not isinstance(items_per_page, int) or items_per_page < 1:
        return _validation_error(GraphQLQueryValidationError("items_per_page must be a positive integer"))

    from wandb_mcp_server.config import MCP_MAX_GQL_ITEMS, MCP_MAX_GQL_ITEMS_PER_PAGE

    applied_max_items = min(max_items, MCP_MAX_GQL_ITEMS)
    applied_page_size = min(items_per_page, MCP_MAX_GQL_ITEMS_PER_PAGE, applied_max_items)
    try:
        bounded_query, bounded_variables, plan = _analyze_and_bound_document(
            query,
            variables,
            applied_page_size,
        )
    except Exception as exc:
        return _validation_error(exc)

    from wandb_mcp_server.api_client import get_wandb_api

    with track_tool_execution(
        "query_paginated_wandb_gql",
        None,
        {
            "query_bytes": len(query.encode("utf-8")),
            "has_variables": bool(variables),
            "max_items": applied_max_items,
            "items_per_page": applied_page_size,
        },
        mcp_tool_name="query_wandb_graphql_tool",
    ) as ctx:
        try:
            api = get_wandb_api()
            initial = execute_graphql(api, bounded_query, bounded_variables)
            if not isinstance(initial, Mapping):
                raise TypeError("W&B returned a non-mapping GraphQL response")
            result = copy.deepcopy(dict(initial))
            if "errors" in result or plan is None:
                return _fit_response_budget(result, None)

            connection = get_nested_value(result, plan.path)
            if not isinstance(connection, dict):
                return _fit_response_budget(result, None)
            if plan.edges_key is None:
                return _fit_response_budget(result, plan)
            initial_edges = connection.get(plan.edges_key)
            if not isinstance(initial_edges, list):
                return _fit_response_budget(result, plan)
            page_info = connection.get(plan.page_info_key) if plan.page_info_key is not None else None
            page_flag_key = plan.has_previous_page_key if plan.backward else plan.has_next_page_key
            if (
                plan.page_info_key is None
                or not isinstance(page_info, dict)
                or page_flag_key is None
                or not isinstance(page_info.get(page_flag_key), bool)
            ):
                return _bound_unpageable_connection(
                    result,
                    connection,
                    initial_edges,
                    plan,
                    applied_max_items,
                )

            if plan.backward:
                cut_mid_page = len(initial_edges) > applied_max_items
                connection[plan.edges_key] = initial_edges[-applied_max_items:]
                has_more = (
                    bool(plan.has_previous_page_key is not None and page_info.get(plan.has_previous_page_key))
                    or cut_mid_page
                )
                next_cursor = page_info.get(plan.start_cursor_key) if plan.start_cursor_key is not None else None
                if cut_mid_page and connection[plan.edges_key]:
                    next_cursor = _edge_cursor(connection[plan.edges_key][0], plan)
                    if next_cursor is None:
                        return _pagination_error(
                            "pagination_cursor_unavailable",
                            (
                                "The bounded backward GraphQL page stopped mid-page but no selected "
                                "edge cursor permits safe continuation"
                            ),
                        )
                if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
                    next_cursor = None
                if next_cursor is not None and _is_non_advancing_cursor(next_cursor, plan):
                    return _pagination_error(
                        "pagination_cursor_non_advancing",
                        "The GraphQL connection returned a repeated or non-advancing continuation cursor",
                    )
                if has_more and next_cursor is None:
                    return _pagination_error(
                        "pagination_cursor_unavailable",
                        "The backward GraphQL response indicates more data but exposes no continuation cursor",
                    )
                updated_page_info = dict(page_info)
                if plan.has_previous_page_key is not None:
                    updated_page_info[plan.has_previous_page_key] = has_more
                if plan.start_cursor_key is not None:
                    updated_page_info[plan.start_cursor_key] = next_cursor
                connection[plan.page_info_key] = updated_page_info
                _set_pagination_extension(
                    result,
                    returned_count=len(connection[plan.edges_key]),
                    has_more=has_more,
                    next_cursor=next_cursor,
                )
                return _fit_response_budget(result, plan)

            aggregated: list[Any] = []

            def append_edges(edges: list[Any]) -> bool:
                cut_mid_page = False
                for edge in edges:
                    if len(aggregated) >= applied_max_items:
                        cut_mid_page = True
                        break
                    aggregated.append(edge)
                return cut_mid_page

            cut_mid_page = append_edges(initial_edges)
            current_page_info = dict(page_info)
            has_next = bool(plan.has_next_page_key is not None and current_page_info.get(plan.has_next_page_key))
            cursor = current_page_info.get(plan.end_cursor_key) if plan.end_cursor_key is not None else None
            if cursor is not None and (not isinstance(cursor, str) or not cursor):
                cursor = None
            seen_page_cursors = {plan.initial_after} if plan.initial_after else set()
            if has_next and cursor is not None:
                if cursor in seen_page_cursors:
                    return _pagination_error(
                        "pagination_cursor_non_advancing",
                        "The GraphQL connection returned a repeated or non-advancing continuation cursor",
                    )
                seen_page_cursors.add(cursor)
            max_page_requests = max(1, math.ceil(applied_max_items / applied_page_size) + 2)
            page_requests = 1
            partial_error = False
            last_page_had_edges = bool(initial_edges)

            while has_next and cursor and len(aggregated) < applied_max_items and page_requests < max_page_requests:
                request_cursor = cursor
                page_variables = dict(bounded_variables)
                assert plan.after_variable is not None
                page_variables[plan.after_variable] = request_cursor
                if plan.first_variable:
                    page_variables[plan.first_variable] = min(
                        applied_page_size,
                        applied_max_items - len(aggregated),
                    )
                try:
                    page = execute_graphql(api, bounded_query, page_variables)
                except GraphQLResponseTooLarge:
                    return _response_too_large(
                        "The W&B GraphQL response exceeded the safety limit; request fewer items or fields"
                    )
                except Exception as exc:
                    result.setdefault("errors", []).append(
                        {
                            "error": "upstream_error",
                            "message": f"W&B GraphQL pagination failed ({type(exc).__name__})",
                        }
                    )
                    partial_error = True
                    break
                page_requests += 1
                if not isinstance(page, Mapping):
                    partial_error = True
                    break
                if page.get("errors"):
                    result.setdefault("errors", []).extend(copy.deepcopy(page["errors"]))
                    partial_error = True
                    break
                page_connection = get_nested_value(dict(page), plan.path)
                if not isinstance(page_connection, Mapping):
                    partial_error = True
                    break
                page_edges = page_connection.get(plan.edges_key)
                next_page_info = page_connection.get(plan.page_info_key)
                if not isinstance(page_edges, list) or not isinstance(next_page_info, Mapping):
                    partial_error = True
                    break
                cut_mid_page = append_edges(page_edges) or cut_mid_page
                current_page_info = dict(next_page_info)
                if plan.has_next_page_key is None or not isinstance(
                    current_page_info.get(plan.has_next_page_key), bool
                ):
                    partial_error = True
                    last_page_had_edges = bool(page_edges)
                    break
                has_next = bool(plan.has_next_page_key is not None and current_page_info.get(plan.has_next_page_key))
                cursor = current_page_info.get(plan.end_cursor_key) if plan.end_cursor_key is not None else None
                if cursor is not None and (not isinstance(cursor, str) or not cursor):
                    cursor = None
                if has_next and cursor is not None:
                    if cursor == request_cursor or cursor in seen_page_cursors:
                        return _pagination_error(
                            "pagination_cursor_non_advancing",
                            "The GraphQL connection returned a repeated or non-advancing continuation cursor",
                        )
                    seen_page_cursors.add(cursor)
                last_page_had_edges = bool(page_edges)
                if not page_edges and has_next and cursor is None:
                    partial_error = True
                    break

            stopped_at_limit = len(aggregated) >= applied_max_items and (has_next or cut_mid_page)
            stopped_at_request_bound = page_requests >= max_page_requests and has_next
            has_more = bool(has_next or cut_mid_page or partial_error or stopped_at_request_bound)
            connection[plan.edges_key] = aggregated
            edge_cursor = _edge_cursor(aggregated[-1], plan) if aggregated else None
            if cut_mid_page and edge_cursor is None:
                return {
                    "errors": [
                        {
                            "error": "pagination_cursor_unavailable",
                            "message": (
                                "The bounded GraphQL page stopped mid-page but no selected edge cursor "
                                "permits safe continuation"
                            ),
                        }
                    ]
                }
            page_end_cursor = current_page_info.get(plan.end_cursor_key) if plan.end_cursor_key is not None else None
            last_cursor = (
                edge_cursor
                if cut_mid_page
                else page_end_cursor
                if not last_page_had_edges and isinstance(page_end_cursor, str) and page_end_cursor
                else edge_cursor or page_end_cursor
            )
            if has_more and last_cursor is None:
                return _pagination_error(
                    "pagination_cursor_unavailable",
                    "The GraphQL response indicates more data but exposes no continuation cursor",
                )
            if has_more and isinstance(last_cursor, str) and _is_non_advancing_cursor(last_cursor, plan):
                return _pagination_error(
                    "pagination_cursor_non_advancing",
                    "The GraphQL connection returned a repeated or non-advancing continuation cursor",
                )
            updated_page_info = dict(current_page_info)
            if plan.has_next_page_key is not None:
                updated_page_info[plan.has_next_page_key] = has_more
            if plan.end_cursor_key is not None:
                updated_page_info[plan.end_cursor_key] = last_cursor
            connection[plan.page_info_key] = updated_page_info
            _set_pagination_extension(
                result,
                returned_count=len(aggregated),
                has_more=has_more,
                next_cursor=last_cursor,
            )
            if stopped_at_limit:
                result["extensions"]["wandb_mcp"]["limit_applied"] = applied_max_items
            return _fit_response_budget(result, plan)
        except GraphQLResponseTooLarge:
            ctx.mark_error("response_too_large")
            return _response_too_large(
                "The W&B GraphQL response exceeded the safety limit; request fewer items or fields"
            )
        except Exception as exc:
            logger.error("Bounded GraphQL query failed (%s)", type(exc).__name__)
            ctx.mark_error(f"query_failed: {type(exc).__name__}")
            return {
                "errors": [
                    {
                        "error": "upstream_error",
                        "message": f"W&B GraphQL query failed ({type(exc).__name__})",
                    }
                ]
            }
