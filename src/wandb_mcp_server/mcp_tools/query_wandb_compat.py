"""Input contract for the structured and legacy forms of query_wandb_tool."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from wandb_mcp_server.error_diagnostics import ToolInputValidationError

STRUCTURED_FIELDS = (
    "entity_name",
    "project_name",
    "resource",
    "run_id",
    "sweep_id",
    "report_name",
    "filters",
    "order",
    "limit",
    "include",
    "summary_keys",
    "config_keys",
    "response_mode",
    "cursor",
)
LEGACY_FIELDS = ("query", "variables", "max_items", "items_per_page")
REQUIRED_STRUCTURED_FIELDS = ("entity_name", "project_name", "resource")


def validate_query_interface(arguments: dict[str, Any]) -> None:
    """Validate original field presence before FastMCP applies defaults."""
    if not isinstance(arguments, dict):
        raise ToolInputValidationError("query_wandb_tool arguments must be an object.", code="dict_type")
    if "query" in arguments:
        query = arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise ToolInputValidationError(
                "Legacy GraphQL requires a nonempty query string.", field="query", code="string_too_short"
            )
        if any(field in arguments for field in STRUCTURED_FIELDS):
            raise ToolInputValidationError(
                "Choose one interface: query/variables or structured fields, not both.", field="query"
            )
        return
    if any(field in arguments for field in LEGACY_FIELDS):
        raise ToolInputValidationError(
            "Legacy variables, max_items, and items_per_page require a query; "
            "otherwise use entity_name, project_name, and resource.",
            field="query",
            code="missing",
        )
    for field in REQUIRED_STRUCTURED_FIELDS:
        if field not in arguments or arguments[field] is None:
            raise ToolInputValidationError(
                "The structured interface requires entity_name, project_name, and resource; "
                "legacy GraphQL instead requires query.",
                field=field,
                code="missing",
            )


def add_query_interface_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Describe the same conditional requirements using public MCP JSON Schema."""
    schema = deepcopy(schema)
    schema["oneOf"] = [
        {
            "required": ["query"],
            "properties": {"query": {"type": "string", "minLength": 1, "pattern": r"\S"}},
            "not": {"anyOf": [{"required": [field]} for field in STRUCTURED_FIELDS]},
        },
        {
            "required": list(REQUIRED_STRUCTURED_FIELDS),
            "properties": {field: {"not": {"type": "null"}} for field in REQUIRED_STRUCTURED_FIELDS},
            "not": {"anyOf": [{"required": [field]} for field in LEGACY_FIELDS]},
        },
    ]
    return schema
