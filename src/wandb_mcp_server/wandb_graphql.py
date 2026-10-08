"""Read-only GraphQL helpers for the supported W&B SDK transport."""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
import json
import logging
import time
from typing import Any, Mapping

import httpx
from graphql import parse
from graphql.language import ast as gql_ast
from wandb.proto.wandb_api_pb2 import ApiRequest, GraphQLRequest

from wandb_mcp_server.config import MAX_ACCUMULATED_BYTES, MCP_WANDB_REQUEST_TIMEOUT_SECONDS, WANDB_API_BASE_URL
from wandb_mcp_server.admission import ToolDeadlineExceeded, current_tool_deadline, raise_if_tool_deadline_exceeded
from wandb_mcp_server.api_client import WandBApiManager, WandBServerBusy


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
_app_http_active: ContextVar[bool] = ContextVar("wandb_app_graphql_http", default=False)


class _SuppressAppHTTPLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _app_http_active.get()


for _name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy"):
    _logger = logging.getLogger(_name)
    if not any(isinstance(item, _SuppressAppHTTPLog) for item in _logger.filters):
        _logger.addFilter(_SuppressAppHTTPLog())


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
        deadline = current_tool_deadline.get()
        timeout = MCP_WANDB_REQUEST_TIMEOUT_SECONDS
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ToolDeadlineExceeded("MCP tool execution deadline exceeded")
            timeout = min(timeout, remaining)
        response = send(request, timeout=timeout)
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
    raise_if_tool_deadline_exceeded()
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


def execute_app_graphql(query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded actor-authenticated transport for fixed application projections.

    Automation lists deliberately use an application User-Agent. W&B filters
    ARIA actions out of SDK requests because the SDK cannot deserialize them.
    Never use this boundary to accept caller-supplied GraphQL or an endpoint.
    """
    token = _app_http_active.set(True)
    try:
        return _execute_app_graphql(query, variables)
    finally:
        _app_http_active.reset(token)


def _execute_app_graphql(query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    validate_read_only_graphql(query)
    key = WandBApiManager.get_api_key()
    if not key:
        raise ValueError("W&B authentication is required.")
    deadline = time.monotonic() + MCP_WANDB_REQUEST_TIMEOUT_SECONDS
    outer = current_tool_deadline.get()
    if outer is not None:
        deadline = min(deadline, outer)
    if deadline <= time.monotonic():
        raise ToolDeadlineExceeded("MCP tool execution deadline exceeded")
    cap = min(MAX_ACCUMULATED_BYTES, _MAX_DECODED_GRAPHQL_RESPONSE_BYTES)
    with httpx.Client(timeout=max(0.001, deadline - time.monotonic()), follow_redirects=False) as client:
        with client.stream(
            "POST",
            WANDB_API_BASE_URL.rstrip("/") + "/graphql",
            auth=httpx.BasicAuth("api", key),
            headers={"User-Agent": "wandb-mcp-server", "Accept-Encoding": "identity"},
            json={"query": query, "variables": dict(variables)},
        ) as response:
            if response.status_code in {429, 503}:
                raise WandBServerBusy(status_code=response.status_code, retry_after_ms=1000)
            response.raise_for_status()  # Includes redirects; credentials never follow them.
            if response.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
                raise ValueError("W&B returned an unsupported response encoding.")
            length = response.headers.get("Content-Length")
            expected = None
            if length is not None:
                if not length.isascii() or not length.isdecimal():
                    raise ValueError("W&B returned an invalid response length.")
                length = length.lstrip("0") or "0"
                if len(length) > len(str(cap)) or int(length) > cap:
                    raise GraphQLResponseTooLarge("W&B response exceeded its download limit.")
                expected = int(length)
            raw = bytearray()
            for chunk in response.iter_raw():
                if time.monotonic() >= deadline:
                    raise ToolDeadlineExceeded("MCP tool execution deadline exceeded")
                if len(raw) + len(chunk) > cap:
                    raise GraphQLResponseTooLarge("W&B response exceeded its download limit.")
                raw.extend(chunk)
            if expected is not None and expected != len(raw):
                raise ValueError("W&B returned an inconsistent response length.")
    payload = json.loads(raw)
    _ensure_bounded_decoded_response(payload)
    if time.monotonic() >= deadline:
        raise ToolDeadlineExceeded("MCP tool execution deadline exceeded")
    if not isinstance(payload, dict) or payload.get("errors") or not isinstance(payload.get("data"), dict):
        raise ValueError("W&B returned an unsuccessful GraphQL response.")
    return payload["data"]
