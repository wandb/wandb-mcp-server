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
    execute_graphql,
    validate_read_only_graphql,
)

logger = get_rich_logger(__name__)

MAX_GRAPHQL_DOCUMENT_BYTES = 64 * 1024
MAX_GRAPHQL_DEPTH = 12
MAX_GRAPHQL_FIELDS = 200
MAX_GRAPHQL_FRAGMENTS = 32

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
unbounded connections are rejected before any W&B request. Responses and
pagination are bounded by deployment limits.
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
    backward: bool = False


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
) -> set[str]:
    names: set[str] = set()
    if selection_set is None:
        return names
    for selection in selection_set.selections:
        if isinstance(selection, gql_ast.FieldNode):
            names.add(selection.name.value)
        elif isinstance(selection, gql_ast.InlineFragmentNode):
            names.update(_selection_field_names(selection.selection_set, fragments, stack))
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
                )
            )
    return names


def _is_connection_field(
    node: gql_ast.FieldNode,
    fragments: Mapping[str, gql_ast.FragmentDefinitionNode],
) -> bool:
    return {"edges", "pageInfo"}.issubset(_selection_field_names(node.selection_set, fragments))


def _int_variable_defaults(operation: gql_ast.OperationDefinitionNode) -> dict[str, int]:
    defaults: dict[str, int] = {}
    for definition in operation.variable_definitions or ():
        if isinstance(definition.default_value, gql_ast.IntValueNode):
            defaults[definition.variable.name.value] = int(definition.default_value.value)
    return defaults


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
        variables_bytes = len(json.dumps(dict(variables or {}), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError):
        raise GraphQLQueryValidationError("variables must be finite JSON-compatible values") from None
    if variables_bytes > MAX_GRAPHQL_DOCUMENT_BYTES:
        raise GraphQLQueryValidationError(
            "GraphQL variables exceed the 64 KiB limit",
            error="query_too_complex",
        )

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
                is_connection = _is_connection_field(selection, fragments)
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
    bounded_variables = dict(variables or {})
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

    if last is not None:
        return (
            gql_printer.print_ast(document),
            bounded_variables,
            _ConnectionPlan(
                path=connection_path,
                field=connection_field,
                first_variable=first_variable,
                after_variable=None,
                initial_after=None,
                backward=True,
            ),
        )

    after = arguments.get("after")
    after_variable = "__mcp_after"
    initial_after: str | None = None
    if after is not None and isinstance(after.value, gql_ast.VariableNode):
        after_variable = after.value.name.value
        supplied_after = bounded_variables.get(after_variable)
        if supplied_after is not None and not isinstance(supplied_after, str):
            raise GraphQLQueryValidationError(f"GraphQL variable ${after_variable} must be a string or null")
        initial_after = supplied_after
    else:
        if after is not None and isinstance(after.value, gql_ast.StringValueNode):
            initial_after = after.value.value
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
        ),
    )


def _estimate_tokens(payload: Mapping[str, Any]) -> int:
    return max(1, len(json.dumps(payload, default=str, ensure_ascii=False, allow_nan=False)) // 4)


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


def _fit_response_budget(
    result: dict[str, Any],
    connection_path: list[str] | None,
    *,
    backward: bool = False,
) -> dict[str, Any]:
    from wandb_mcp_server.config import MAX_RESPONSE_TOKENS

    try:
        if _estimate_tokens(result) <= MAX_RESPONSE_TOKENS:
            return result
    except (TypeError, ValueError):
        return {
            "errors": [
                {
                    "error": "response_too_large",
                    "message": "GraphQL response could not be serialized safely",
                }
            ]
        }
    connection = get_nested_value(result, connection_path or []) if connection_path else None
    if not isinstance(connection, dict) or not isinstance(connection.get("edges"), list):
        return {
            "errors": [
                {
                    "error": "response_too_large",
                    "message": "GraphQL response exceeded the configured response budget",
                }
            ]
        }
    edges = connection["edges"]
    dropped = 0
    while edges and _estimate_tokens(result) > MAX_RESPONSE_TOKENS:
        edges.pop()
        dropped += 1
    page_info = connection.setdefault("pageInfo", {})
    if backward:
        last_cursor = edges[0].get("cursor") if edges and isinstance(edges[0], dict) else None
        page_info["hasPreviousPage"] = True
        page_info["startCursor"] = last_cursor
    else:
        last_cursor = edges[-1].get("cursor") if edges and isinstance(edges[-1], dict) else None
        page_info["hasNextPage"] = True
        page_info["endCursor"] = last_cursor
    _set_pagination_extension(
        result,
        returned_count=len(edges),
        has_more=True,
        next_cursor=last_cursor,
        truncated_by_budget=True,
    )
    if _estimate_tokens(result) > MAX_RESPONSE_TOKENS:
        return {
            "errors": [
                {
                    "error": "response_too_large",
                    "message": "GraphQL response metadata exceeded the configured response budget",
                }
            ]
        }
    return result


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
            initial_edges = connection.get("edges")
            page_info = connection.get("pageInfo")
            if not isinstance(initial_edges, list) or not isinstance(page_info, dict):
                return _fit_response_budget(result, None)

            if plan.backward:
                cut_mid_page = len(initial_edges) > applied_max_items
                connection["edges"] = initial_edges[:applied_max_items]
                has_more = bool(page_info.get("hasPreviousPage")) or cut_mid_page
                next_cursor = page_info.get("startCursor")
                if cut_mid_page and connection["edges"] and isinstance(connection["edges"][0], Mapping):
                    next_cursor = connection["edges"][0].get("cursor") or next_cursor
                connection["pageInfo"] = {
                    **page_info,
                    "hasPreviousPage": has_more,
                    "startCursor": next_cursor,
                }
                _set_pagination_extension(
                    result,
                    returned_count=len(connection["edges"]),
                    has_more=has_more,
                    next_cursor=next_cursor,
                )
                return _fit_response_budget(result, plan.path, backward=True)

            aggregated: list[Any] = []
            seen_ids: set[Any] = set()

            def append_edges(edges: list[Any]) -> bool:
                cut_mid_page = False
                for edge in edges:
                    if len(aggregated) >= applied_max_items:
                        cut_mid_page = True
                        break
                    node = edge.get("node") if isinstance(edge, Mapping) else None
                    node_id = node.get("id") if isinstance(node, Mapping) else None
                    if node_id is not None and node_id in seen_ids:
                        continue
                    if node_id is not None:
                        seen_ids.add(node_id)
                    aggregated.append(edge)
                return cut_mid_page

            cut_mid_page = append_edges(initial_edges)
            current_page_info = dict(page_info)
            has_next = bool(current_page_info.get("hasNextPage"))
            cursor = current_page_info.get("endCursor")
            max_page_requests = max(1, math.ceil(applied_max_items / applied_page_size) + 2)
            page_requests = 1
            partial_error = False

            while has_next and cursor and len(aggregated) < applied_max_items and page_requests < max_page_requests:
                page_variables = dict(bounded_variables)
                assert plan.after_variable is not None
                page_variables[plan.after_variable] = cursor
                if plan.first_variable:
                    page_variables[plan.first_variable] = min(
                        applied_page_size,
                        applied_max_items - len(aggregated),
                    )
                try:
                    page = execute_graphql(api, bounded_query, page_variables)
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
                page_edges = page_connection.get("edges")
                next_page_info = page_connection.get("pageInfo")
                if not isinstance(page_edges, list) or not isinstance(next_page_info, Mapping):
                    partial_error = True
                    break
                cut_mid_page = append_edges(page_edges) or cut_mid_page
                current_page_info = dict(next_page_info)
                has_next = bool(current_page_info.get("hasNextPage"))
                cursor = current_page_info.get("endCursor")
                if not page_edges:
                    partial_error = has_next
                    break

            stopped_at_limit = len(aggregated) >= applied_max_items and (has_next or cut_mid_page)
            stopped_at_request_bound = page_requests >= max_page_requests and has_next
            has_more = bool(has_next or cut_mid_page or partial_error or stopped_at_request_bound)
            connection["edges"] = aggregated
            last_cursor = (
                aggregated[-1].get("cursor")
                if aggregated and isinstance(aggregated[-1], Mapping)
                else current_page_info.get("endCursor")
            )
            connection["pageInfo"] = {
                **current_page_info,
                "hasNextPage": has_more,
                "endCursor": last_cursor,
            }
            _set_pagination_extension(
                result,
                returned_count=len(aggregated),
                has_more=has_more,
                next_cursor=last_cursor,
            )
            if stopped_at_limit:
                result["extensions"]["wandb_mcp"]["limit_applied"] = applied_max_items
            return _fit_response_budget(result, plan.path)
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
