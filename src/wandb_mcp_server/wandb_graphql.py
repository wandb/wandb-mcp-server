"""Read-only GraphQL helpers for the supported W&B SDK transport."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Mapping

from graphql import parse
from graphql.language import ast as gql_ast
from wandb.proto.wandb_api_pb2 import ApiRequest, GraphQLRequest

from wandb_mcp_server.config import MCP_WANDB_REQUEST_TIMEOUT_SECONDS


class GraphQLReadOnlyViolation(ValueError):
    """Raised when a GraphQL document contains a non-query operation."""

    def __init__(self, operation_types: set[str]) -> None:
        self.operation_types = tuple(sorted(operation_types))
        joined_types = ", ".join(self.operation_types)
        super().__init__(
            "query_wandb_graphql_tool accepts GraphQL query operations only; "
            f"rejected operation type(s): {joined_types}."
        )


class GraphQLResponseTooLarge(ValueError):
    """Raised when a decoded GraphQL response exceeds the safe client budget."""


_MAX_DECODED_GRAPHQL_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_DECODED_GRAPHQL_RESPONSE_NODES = 500_000
_UTF8_SIZE_CHUNK_CHARACTERS = 4_096


def _bounded_utf8_size(value: str, limit: int) -> int:
    """Return the UTF-8 size without allocating a second full-size byte string."""
    if len(value) > limit:
        raise GraphQLResponseTooLarge("The W&B response exceeded the decoded response safety limit.")
    size = 0
    for offset in range(0, len(value), _UTF8_SIZE_CHUNK_CHARACTERS):
        size += len(value[offset : offset + _UTF8_SIZE_CHUNK_CHARACTERS].encode("utf-8"))
        if size > limit:
            raise GraphQLResponseTooLarge("The W&B response exceeded the decoded response safety limit.")
    return size


def _ensure_bounded_decoded_response(value: Any) -> None:
    """Reject structurally or textually oversized decoded JSON."""

    remaining = _MAX_DECODED_GRAPHQL_RESPONSE_BYTES
    nodes = 0
    # Tagged iterators keep traversal memory proportional to nesting depth,
    # rather than to the number of values in the response.
    pending: list[tuple[str, Any]] = [("value", value)]
    active_containers: set[int] = set()
    while pending:
        kind, current = pending.pop()
        if kind == "exit":
            active_containers.discard(current)
            continue
        if kind == "children":
            try:
                child = next(current)
            except StopIteration:
                continue
            pending.append(("children", current))
            pending.append(("value", child))
            continue
        nodes += 1
        remaining -= 1  # Charge structural work even for empty scalar values.
        if nodes > _MAX_DECODED_GRAPHQL_RESPONSE_NODES:
            raise GraphQLResponseTooLarge("The W&B response exceeded the decoded response safety limit.")
        if current is None or isinstance(current, (bool, int, float)):
            remaining -= len(str(current))
        elif isinstance(current, str):
            remaining -= _bounded_utf8_size(current, max(0, remaining))
        elif isinstance(current, (bytes, bytearray)):
            remaining -= len(current)
        elif isinstance(current, Mapping):
            identity = id(current)
            if identity in active_containers:
                remaining -= 1
                continue
            active_containers.add(identity)
            remaining -= 2
            children = (child for key, value in current.items() for child in (str(key), value))
            pending.append(("exit", identity))
            pending.append(("children", iter(children)))
        elif isinstance(current, Sequence):
            identity = id(current)
            if identity in active_containers:
                remaining -= 1
                continue
            active_containers.add(identity)
            remaining -= 2
            pending.append(("exit", identity))
            pending.append(("children", iter(current)))
        else:
            rendered = str(current)
            remaining -= _bounded_utf8_size(rendered, max(0, remaining))
        if remaining < 0:
            raise GraphQLResponseTooLarge("The W&B response exceeded the decoded response safety limit.")


def _execute_bounded_graphql(service_api: Any, query: str, variables: Mapping[str, Any]) -> Any:
    """Use W&B's service request while bounding its JSON text before decoding."""
    send = getattr(service_api, "send_api_request", None)
    if callable(send):
        request = ApiRequest(
            graphql_request=GraphQLRequest(
                query=query,
                variables_json=json.dumps(variables),
            )
        )
        response = send(request, timeout=MCP_WANDB_REQUEST_TIMEOUT_SECONDS)
        data_json = getattr(getattr(response, "graphql_response", None), "data_json", None)
        if not isinstance(data_json, str):
            raise RuntimeError("W&B SDK compatibility error: GraphQL service response contained no JSON data.")
        _bounded_utf8_size(data_json, _MAX_DECODED_GRAPHQL_RESPONSE_BYTES)
        result = json.loads(data_json)
        _ensure_bounded_decoded_response(result)
        return result

    # Lightweight service-compatible adapters used by callers may expose only
    # execute_graphql. The supported W&B 0.28+ ServiceApi always takes the
    # pre-decode branch above; this fallback still enforces structural limits.
    execute = getattr(service_api, "execute_graphql", None)
    if not callable(execute):
        raise RuntimeError(
            "W&B SDK compatibility error: query_wandb_graphql_tool requires "
            "wandb>=0.28.0 with ServiceApi GraphQL support."
        )
    result = execute(query, variables=dict(variables))
    _ensure_bounded_decoded_response(result)
    return result


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
    return _execute_bounded_graphql(service_api, query, variables_dict)
