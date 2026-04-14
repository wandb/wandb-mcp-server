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


QUERY_WEAVE_TRACES_TOOL_DESCRIPTION = """
Query Weave traces with filtering, sorting, and detail_level control.

For W&B runs/metrics, use query_wandb_tool instead. This tool is for Weave traces (LLM calls, evaluations, agent traces).

<when_to_use>
Call for Weave trace data. Use detail_level="schema" to browse, "summary" for analysis, "full" for specific traces only.
</when_to_use>

<usage_tips>
query_traces_tool can return a lot of data, below are some usage tips for this function
in order to avoid overwhelming a LLM's context window with too much data.

<managing_llm_context_window>

Returning all weave trace data can possibly result in overwhelming the LLM context window
if there are 100s or 1000s of logged weave traces (depending on how many child traces each has) as
well as resulting in a lot of data from or calls to the weave API.

So, depending on the user query, consider doing the following to return enough data to answer the user query
but not too much data that it overwhelms the LLM context window:

- return only the root traces using the `trace_roots_only` boolean filter if you only need the top-level/parent
traces and don't need the data from all child traces. For example, if a user wants to know the number of
successful traces in a project but doesn't need the data from all child traces. Or if a user
wants to visualise the number of parent traces over time.

- return only the truncated values of the trace data keys in order to first give a preview of the data that can then
inform more targeted weave trace queries from the user. in the extreme you can set `truncate_length` to 0 in order to
only return keys but not the values of the trace data.

- return only the metadata for all the traces (set `metadata_only = True`) if the query doesn't need to know anything
about the structure or content of the individual weave traces. Note that this still requires
requesting all the raw traces data from the weave API so can still result in a lot of data and/or a
lot of calls being made to the weave API.

- return only the columns needed using the `columns` parameter. In weave, the `inputs` and `output` columns of a
trace can contain a lot of data, so avoiding returning these columns can help. Note you have to explicitly specify
the columns you want to return if there are certain columns you don't want to return. Its almost always a good idea to
specficy the columns needed.

<returning_metadata_only>

If `metadata_only = True` this returns only metadata of the traces such as trace counts, token counts,
trace types, time range, status counts and distribution of op names. if `metadata_only = False` the
trace data is returned either in full or truncated to `truncate_length` characters depending if
`return_full_data = True` or `False` respectively.
</returning_metadata_only>

<truncating_trace_data_values>

If `return_full_data = False` the trace data is truncated to `truncate_length` characters,
default 1000 characters. Otherwise the trace data is returned in full.
</truncating_trace_data_values>

Remember, LLM context window is precious, only return the minimum amount of data needed to complete an analysis.
</managing_llm_context_window>

<usage_guidance>

- Exploratory queries: For generic exploratory or initial queries about a set of weave traces in a project it can
be a good idea to start with just returning metadata or truncated data. Consider asking the
user for clarification and warn them that returning a lot of weave traces data might
overwhelm the LLM context window. No need to warn them multiple times, just once is enough.

- Project size: Consider using the count_traces_tool to get an estimate of the number of traces in a project
before querying for them as query_trace_tool can return a lot of data.

- Partial op name matching: Use the `op_name_contains` filter if a users has only given a partial op name or if they
are unsure of the exact op name.

- Weave Evaluations: If asked about weave evaluations or evals traces:
    - Evals are complicated to query, prompt the user with follow up questions if needed.
    - First, always try and oritent yourself - pull a summary of the evaluation, get all of the top level column names in the eval \
and always get a count of the total number of child traces in this eval by filtering by parent_id and using the count_traces tool.
    - As part of orienting yourself, just pull a subset of child traces from the eval, maybe 3 to 5, to understand the column structure \
and values.
    - Always be explicit about the amount of data returned and limits used in your query - return to the user the count of traces \
analysed.
    - Always stay filterd on the evaluation id (filter by `parent_id`) unless specifically asked questions across different evaulations, e.g. \
if a parent id (or parentId) is provided then ensure to use that filter in the query.
    - filter for traces with `op_name_contains = "Evaluation.evaluate"` as a first step. These ops are parent traces that contain
    aggregated stats and scores about the evaluation. The child traces of these ops are the actual evaluation results
    for each sample in an evaluation dataset. If asked about individual rows in an evaluation then use the parent_ids
    filter to return the child traces.
    - for questions where both a child call name of an evaluation and an evaluation id or name are provided, always ensure \
that you first correctly get the evaluation id, and then use it as the parent_id in the query for the child traces. Otherwise \
there is a risk of returning traces that do not belong to the evaluation that was given.

- Weave nomenclature: Note that users might refer to weave ops as "traces" or "calls" or "traces" as "ops".

</usage_guidance>

Parameters
----------
entity_name : str
    The Weights & Biases entity name (team or username)
project_name : str
    The Weights & Biases project name
filters : dict
    Dict of filter conditions, supporting:

    - display_name : str or regex pattern
        Filter by display name seen in the Weave UI
    - op_name : str or regex pattern
        Filter by weave op name, a long URI starting with 'weave:///'
    - op_name_contains : str
        Filter for op_name containing this substring (easier than regex)
    - trace_roots_only : bool
        Boolean to filter for only top-level/parent traces. Useful when you don't need
        to return the data from all child traces.
    - trace_id : str
        Filter by a specific `trace_id` (e.g., "01958ab9-3c67-7c72-92bf-d023fa5a0d4d").
        A `trace_id` groups multiple calls/spans. Use if the user explicitly say they provided a "trace_id" for a group of operations.
        Always first try to filter by `call_ids` if a user provides an ID, before trying to filter by `trace_id`.
    - call_ids : str or list of str
        Filter by specific `call_id`s (also known as Span IDs) (string or list of strings, e.g., ["01958ab9-3c68-7c23-8ccd-c135c7037769"]).
        **GUIDANCE**: `call_id` (Span ID) identifies a *single* operation/span and is typically found in Weave UI URLs.
        If a user provides an ID for a specific item they're viewing, **prefer `call_ids`**.
        Format as a list: `{"call_ids": ["user_provided_id"]}`.
    - parent_ids : str or list of str
        Return traces that are children of the given parent trace ids (string or list of strings). Ensure you use this \
if given an evaluation trace id or name.
    - status : str
        Filter by trace status, defined as whether or not the trace had an exception or not. Can be
        `success` or `error`.
        NOTE: When users ask for "failed", "wrong", or "incorrect" traces, use `status:'error'` or
        `has_exception:True` as the filter.
    - time_range : dict
        Dict with "start" and "end" datetime strings. Datetime strings should be in ISO format
        (e.g. `2024-01-01T00:00:00Z`)
    - attributes : dict
        Dict of the weave attributes of the trace.
        Supports nested paths (e.g., "metadata.model_name") via dot notation.
        Value can be:
        *   A literal for exact equality (e.g., `"status": "success"`)
        *   A dictionary with a comparison operator: `$gt`, `$lt`, `$eq`, `$gte`, `$lte` (e.g., `{"token_count": {"$gt": 100}}`)
        *   A dictionary with the `$contains` operator for substring matching on string attributes (e.g., `{"model_name": {"$contains": "gpt-3"}}`)
        **Warning:** The `$contains` operator performs simple substring matching only, full regular expression matching (e.g., via `$regex`) is **not supported** for attributes. Do not attempt to use `$regex`.
    - has_exception : bool, optional
        Optional[bool] to filter traces by exception status:
        - None (or key not present): Show all traces regardless of exception status
        - True: Show only traces that have exceptions (exception field is not null)
        - False: Show only traces without exceptions (exception field is null)
sort_by : str, optional
    Field to sort by (started_at, ended_at, op_name, etc.). Defaults to 'started_at'
sort_direction : str, optional
    Sort direction ('asc' or 'desc'). Defaults to 'desc'
limit : int, optional
    Maximum number of results to return. Defaults to None
include_costs : bool, optional
    Include tracked api cost information in the results. Defaults to True
include_feedback : bool, optional
    Include weave annotations (human labels/feedback). Defaults to True
columns : list of str, optional
    List of specific columns to include in the results. Its almost always a good idea to specficy the
    columns needed. Defaults to None (all columns).
    Available columns are:
        id: <class 'str'>
        project_id: <class 'str'>
        op_name: <class 'str'>
        display_name: typing.Optional[str]
        trace_id: <class 'str'>
        parent_id: typing.Optional[str]
        started_at: <class 'datetime.datetime'>
        attributes: dict[str, typing.Any]
        inputs: dict[str, typing.Any]
        ended_at: typing.Optional[datetime.datetime]
        exception: typing.Optional[str]
        output: typing.Optional[typing.Any]
        summary: typing.Optional[SummaryMap] # Contains nested data like 'summary.weave.status' and 'summary.weave.latency_ms'
        status: typing.Optional[str] # Synthesized from summary.weave.status if requested
        latency_ms: typing.Optional[int] # Synthesized from summary.weave.latency_ms if requested
        wb_user_id: typing.Optional[str]
        wb_run_id: typing.Optional[str]
        deleted_at: typing.Optional[datetime.datetime]
expand_columns : list of str, optional
    List of columns to expand in the results. Defaults to None
truncate_length : int, optional
    Maximum length for string values in weave traces. Defaults to 1000
return_full_data : bool, optional
    Whether to include full untruncated trace data. If True, the `truncate_length` parameter is ignored. If  \
`False` returns truncation_length = 0, no values for the column keys are returned. Defaults to True.
metadata_only : bool, optional
    Return only metadata without traces. Defaults to False
detail_level : str, optional
    Controls how much data is returned per trace. Use this instead of manually tuning
    truncate_length/return_full_data. Defaults to "summary".
    - "schema": Structural fields only (op_name, trace_id, started_at, ended_at, status,
      parent_id, display_name). Fastest option, ideal for browsing and filtering large sets.
    - "summary": Schema fields plus truncated inputs/outputs (200 chars) and summary/usage
      data. Good default for understanding what traces contain.
    - "full": Everything untruncated. Use only when drilling into specific trace_ids, never
      for bulk queries as it can overwhelm the context window.

Returns
-------
str
    JSON string containing either full trace data or metadata only, depending on parameters.
    The response metadata includes `total_matching_count` -- the total traces matching your
    current filters before the limit is applied. Note: this reflects whatever filters you
    used, not the project-wide total. Use `count_weave_traces_tool` if you need a separate
    unfiltered count.

<examples>
    ```python
    # Get an overview of the traces in a project
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"trace_roots_only": True},
        metadata_only=True,
        return_full_data=False
    )

    # Get failed traces with costs and feedback
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"status": "error"},
        include_costs=True,
        include_feedback=True
    )

    # Get specific columns for traces who's op name (i.e. trace name) contains a specific substring
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"op_name_contains": "Evaluation.summarize"},
        columns=["id", "op_name", "started_at", "costs"]
    )

    # Schema-first workflow: browse traces quickly, then drill into specific ones
    # Step 1: Get structural overview
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"trace_roots_only": True},
        detail_level="schema"
    )
    # Step 2: Drill into a specific trace with full data
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
        detail_level="full"
    )
    ```
</examples>
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

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "query_traces",
        api.viewer,
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

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "query_paginated_weave_traces",
        api.viewer,
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
        result = service.query_paginated_traces(
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

        if debug_raw_traces and result.traces:
            result_dict = result.model_dump()
            result_dict["raw_traces"] = result.traces
            result = QueryResult.model_validate(result_dict)

        assert isinstance(result, QueryResult), f"Result type must be a QueryResult, found: {type(result)}"
        return result
