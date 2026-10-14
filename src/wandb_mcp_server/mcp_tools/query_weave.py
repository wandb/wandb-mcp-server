from typing import Any, Dict, List, Optional

from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.service import TraceService
from wandb_mcp_server.weave_api.models import QueryResult
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution

logger = get_rich_logger(__name__)


def get_trace_service():
    """
    Get a TraceService instance with the current request's API key.

    This creates a new TraceService for each request to ensure
    the correct API key is used from the context.
    """
    # Get the API key from context (set by auth middleware) or environment
    api_key = WandBApiManager.get_api_key()
    return TraceService(api_key=api_key)


QUERY_WEAVE_TRACES_TOOL_DESCRIPTION = """Query Weave traces (LLM calls, evaluations, and classic agent traces) with filters and detail control.
For W&B runs/metrics use query_wandb_tool.

<when_to_use>
Use detail_level="schema" to browse, "summary" for analysis (default), and "full"
only for specific calls. Start with infer_trace_schema_tool for unfamiliar data;
use count_weave_traces_tool for counts without trace payloads.
</when_to_use>

Inputs:
- entity_name, project_name: required W&B scope.
- filters: display_name, op_name, op_name_contains, trace_roots_only, trace_id,
  call_ids, parent_ids, status ("success"/"error"), has_exception, time_range
  (start/end ISO timestamps), attributes, inputs, output.
  Prefer call_ids=["id"] for an ID from a Weave URL: a call/span ID identifies
  one operation; trace_id groups spans. Use parent_ids for evaluation children.
  attributes/inputs/output support dot-path fields, literal equality, $eq, $gt,
  $gte, $lt, $lte, and $contains (substring, NOT regex). For content search on
  large projects, narrow by time_range, op_name, or trace_roots_only first.
- sort_by: started_at by default; sort_direction: "desc" (default) or "asc".
  Hosted workloads reject cost sorting that requires a large scan.
- limit: maximum results; omitted limits and caps depend on workload. Use a
  small explicit limit, especially for full data.
- columns: request only needed fields, e.g. ["id", "op_name", "started_at",
  "summary"]; inputs/output can be large. expand_columns expands references.
- include_costs, include_feedback: both true by default.
- detail_level: "schema" gives structural fields; "summary" adds inputs/outputs
  truncated to 200 characters and summary/usage; "full" returns untruncated data.
- truncate_length: 1000 by default; return_full_data=true bypasses truncation.
  Prefer detail_level for normal queries. metadata_only=true omits trace data.

Returns a JSON string with traces and metadata. total_matching_count counts
traces matching the current filters before limit, not the project-wide total.
Use count_weave_traces_tool without filters for a project-wide count.
For evaluations, inspect a small sample and count children using parent_ids;
keep queries scoped to the evaluation and state how many traces were analyzed.
After finding child calls, use resolve_trace_roots_tool with their trace_ids
to retrieve root context in a single batched request.

<examples>
Browse: filters={"trace_roots_only": true}, detail_level="schema", limit=5.
Inspect one call: filters={"call_ids": ["span-id"]}, detail_level="full", limit=1.
Find errors: filters={"status": "error"}, columns=["id", "exception"], limit=10.
</examples>

Detailed filter, column, evaluation, and root-resolution guidance:
https://github.com/wandb/wandb-mcp-server/blob/main/docs/tool-guidance.md#query-weave-traces
"""


def query_traces(
    entity_name: str,
    project_name: str,
    filters: Dict[str, Any] = {},
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = 100,
    offset: int = 0,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: List[str] = [],
    expand_columns: List[str] = [],
    return_full_data: bool = True,
    api_key: str = "",
    query_expr: Any = None,  # We ignore this in the new implementation
    request_timeout: int = 10,
    retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    This maintains the original signature of query_traces from query_weave.py,
    but delegates to our new implementation.
    """
    service = get_trace_service()
    if api_key:
        service = TraceService(
            api_key=api_key,
            retries=retries,
            timeout=request_timeout,
        )

    with track_tool_execution(
        "query_traces",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": filters,
            "sort_by": sort_by,
            "sort_direction": sort_direction,
            "limit": limit,
            "offset": offset,
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": columns,
            "expand_columns": expand_columns,
            "return_full_data": return_full_data,
            "request_timeout": request_timeout,
            "retries": retries,
        },
    ):
        result = service.query_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=filters,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            include_costs=include_costs,
            include_feedback=include_feedback,
            columns=columns,
            expand_columns=expand_columns,
            return_full_data=return_full_data,
            metadata_only=False,
        )

        if result.traces:
            traces_as_dicts = []
            for trace in result.traces:
                if hasattr(trace, "model_dump"):
                    traces_as_dicts.append(trace.model_dump())
                elif isinstance(trace, dict):
                    traces_as_dicts.append(trace)
                else:
                    try:
                        traces_as_dicts.append(dict(trace))
                    except Exception:
                        traces_as_dicts.append({"error": f"Could not convert {type(trace)} to dict"})
            return traces_as_dicts
        else:
            return []


async def query_paginated_weave_traces(
    entity_name: str,
    project_name: str,
    chunk_size: int = 20,
    filters: Dict[str, Any] = {},
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    target_limit: Optional[int] = None,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: List[str] = [],
    expand_columns: List[str] = [],
    truncate_length: Optional[int] = 200,
    return_full_data: bool = True,
    metadata_only: bool = False,
    api_key: Optional[str] = None,
    retries: int = 3,
    debug_raw_traces: bool = False,
) -> QueryResult:
    """
    Query Weave traces with pagination and return results as a Pydantic model.

    This maintains the original signature of query_paginated_weave_traces from query_weave.py,
    but delegates to our new implementation and returns a Pydantic QueryResult model directly.

    Example:
        ```python
        result = await query_paginated_weave_traces(
            entity_name="my-entity",
            project_name="my-project"
        )

        # Access Pydantic model properties directly
        print(f"Total traces: {result.metadata.total_traces}")
        ```

    Args:
        entity_name: Weights & Biases entity name.
        project_name: Weights & Biands project name.
        chunk_size: Number of traces to retrieve in each chunk.
        filters: Dictionary of filter conditions.
        sort_by: Field to sort by.
        sort_direction: Sort direction ('asc' or 'desc').
        target_limit: Maximum total number of results to return.
        include_costs: Include tracked API cost information in the results.
        include_feedback: Include Weave annotations in the results.
        columns: List of specific columns to include in the results.
        expand_columns: List of columns to expand in the results.
        truncate_length: Maximum length for string values.
        return_full_data: Whether to include full untruncated trace data.
        metadata_only: Whether to only include metadata without traces.
        api_key: Optional API key to use for authentication.
        retries: Number of retry attempts for API calls.
        debug_raw_traces: Include raw traces in the response for debugging.

    Returns:
        QueryResult: A Pydantic model containing the query results
    """
    service = get_trace_service()
    if api_key:
        service = TraceService(
            api_key=api_key,
            retries=retries,
        )

    with track_tool_execution(
        "query_paginated_weave_traces",
        None,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "chunk_size": chunk_size,
            "filters": filters,
            "sort_by": sort_by,
            "sort_direction": sort_direction,
            "target_limit": target_limit,
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": columns,
            "expand_columns": expand_columns,
            "truncate_length": truncate_length,
            "return_full_data": return_full_data,
            "metadata_only": metadata_only,
            "retries": retries,
            "debug_raw_traces": debug_raw_traces,
        },
    ):
        from functools import partial

        from wandb_mcp_server.instrumented_server import run_sync_in_current_tool

        result = await run_sync_in_current_tool(
            partial(
                service.query_paginated_traces,
                entity_name=entity_name,
                project_name=project_name,
                chunk_size=chunk_size,
                filters=filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                target_limit=target_limit,
                include_costs=include_costs,
                include_feedback=include_feedback,
                columns=columns,
                expand_columns=expand_columns,
                truncate_length=truncate_length,
                return_full_data=return_full_data,
                metadata_only=metadata_only,
            )
        )

        if debug_raw_traces and result.traces:
            result_dict = result.model_dump()
            result_dict["raw_traces"] = result.traces
            result = QueryResult.model_validate(result_dict)

        assert isinstance(result, QueryResult), f"Result type must be a QueryResult, found: {type(result)}"
        return result
