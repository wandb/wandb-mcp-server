"""Infer trace schema for a project by sampling recent traces.

Returns field names, types, and top-N most common values so the agent
understands the data structure before querying.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

INFER_TRACE_SCHEMA_TOOL_DESCRIPTION = """Discover the schema of Weave traces in a project.

Returns field names, data types, and the most common values for each field,
plus total and root trace counts. Use this tool BEFORE querying traces to
understand what fields are available and what values to filter on.

<when_to_use>
Call this tool FIRST when working with a new project's Weave traces.
The output tells you exactly which columns and filter values to use in
query_weave_traces_tool and count_weave_traces_tool.
</when_to_use>

Parameters
----------
entity_name : str
    The Weights & Biases entity name (team or username).
project_name : str
    The Weights & Biases project name.
sample_size : int, optional
    Number of recent traces to sample for schema inference. Defaults to 20.
top_n_values : int, optional
    Number of most common values to return per field. Defaults to 5.

Returns
-------
JSON with:
  - fields: list of {path, type, top_values, non_null_count}
  - total_traces: total trace count in the project
  - root_traces: root-level trace count
  - sample_size: how many traces were sampled

Examples
--------
>>> infer_trace_schema_tool("wandb", "my-project")
# Returns field paths like "op_name", "status", "summary.usage.total_tokens"
# with their types and most common values
"""


def _flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dict into dot-notation paths."""
    items: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and len(v) <= 10:
                items.update(_flatten_dict(v, new_key))
            else:
                items[new_key] = v
    return items


def _infer_type(value: Any) -> str:
    """Infer a human-readable type string from a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        if len(value) >= 19:
            from datetime import datetime as _dt

            try:
                _dt.fromisoformat(value.replace("Z", "+00:00"))
                return "datetime"
            except (ValueError, TypeError):
                pass
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def infer_trace_schema(
    entity_name: str,
    project_name: str,
    sample_size: int = 20,
    top_n_values: int = 5,
) -> str:
    """Sample recent traces and infer the field schema."""
    from wandb_mcp_server.mcp_tools.count_traces import count_traces
    from wandb_mcp_server.mcp_tools.query_weave import get_trace_service

    if sample_size > 500:
        logger.warning(f"Large sample_size={sample_size} for schema inference; consider using a smaller value")

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "infer_trace_schema",
        api.viewer,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "sample_size": sample_size,
        },
    ) as ctx:
        try:
            total_traces = count_traces(entity_name, project_name)
            root_traces = count_traces(entity_name, project_name, filters={"trace_roots_only": True})

            service = get_trace_service()
            result = service.query_traces(
                entity_name=entity_name,
                project_name=project_name,
                filters={},
                sort_by="started_at",
                sort_direction="desc",
                limit=sample_size,
                include_costs=False,
                include_feedback=False,
                columns=[],
                expand_columns=[],
                truncate_length=100,
                return_full_data=False,
                metadata_only=False,
            )
        except Exception as e:
            logger.error(f"Failed to query traces for schema inference: {e}")
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": f"Failed to infer schema for {entity_name}/{project_name}: {type(e).__name__}"})

        traces = result.traces if hasattr(result, "traces") else []
        if not traces:
            return json.dumps(
                {
                    "fields": [],
                    "total_traces": total_traces,
                    "root_traces": root_traces,
                    "sample_size": 0,
                    "note": "No traces found in this project.",
                }
            )

        field_types: Dict[str, Counter] = defaultdict(Counter)
        field_values: Dict[str, Counter] = defaultdict(Counter)
        field_non_null: Dict[str, int] = defaultdict(int)

        for trace in traces:
            if isinstance(trace, dict):
                trace_dict = trace
            elif hasattr(trace, "model_dump"):
                trace_dict = trace.model_dump()
            else:
                trace_dict = {}
            flat = _flatten_dict(trace_dict)
            for path, value in flat.items():
                inferred = _infer_type(value)
                field_types[path][inferred] += 1
                if value is not None:
                    field_non_null[path] += 1
                    try:
                        val_str = str(value)[:100]
                        field_values[path][val_str] += 1
                    except Exception:
                        pass

        fields = []
        for path in sorted(field_types.keys()):
            type_counts = field_types[path]
            dominant_type = type_counts.most_common(1)[0][0]
            top_vals = [v for v, _ in field_values[path].most_common(top_n_values)]
            fields.append(
                {
                    "path": path,
                    "type": dominant_type,
                    "top_values": top_vals,
                    "non_null_count": field_non_null.get(path, 0),
                }
            )

        schema_result = json.dumps(
            {
                "fields": fields,
                "total_traces": total_traces,
                "root_traces": root_traces,
                "sample_size": len(traces),
            }
        )

        from wandb_mcp_server.config import MAX_RESPONSE_TOKENS
        from wandb_mcp_server.trace_utils import warn_if_response_large

        warn_if_response_large("infer_trace_schema", schema_result, MAX_RESPONSE_TOKENS)
        return schema_result
