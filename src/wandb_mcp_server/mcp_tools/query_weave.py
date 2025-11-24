from typing import Any, Dict, List, Optional

from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.weave_api.service import TraceService
from wandb_mcp_server.weave_api.models import QueryResult
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call

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
Query Weave traces, trace metadata, and trace costs with filtering and sorting options.

---
**Cost Calculation and Sorting Enhancements:**
- For each model in the `costs` dictionary, a new field `total_cost` is computed as the sum of `completion_tokens_total_cost` and `prompt_tokens_total_cost`.
- You can post-hoc sort traces by any of: `total_cost`, `completion_cost`, or `prompt_cost` (across all models, summed if multiple).
---

<wandb_vs_weave_product_distinction>
**IMPORTANT PRODUCT DISTINCTION:**
W&B offers two distinct products with different purposes:

1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
    lifecycle management. Use `query_wandb_tool` for questions about:
    - Experiment runs, metrics, and performance comparisons
    - Artifact management and model registry
    - Hyperparameter optimization and sweeps
    - Project dashboards and reports

2. W&B Weave: A toolkit for LLM and GenAI application observability and evaluation. Use
    `query_weave_traces_tool` (this tool) for questions about:
    - Execution traces and paths of LLM operations
    - LLM inputs, outputs, and intermediate results
    - Chain of thought visualization and debugging
    - LLM evaluation results and feedback

FYI: The Weigths & Biases platform is owned by Coreweave. If there are queries related to W&B, wandb \
or weave and Coreweave, they might be related to W&B products or features that leverage Coreweave's \
GPU or compute infrastructure.
</wandb_vs_weave_product_distinction>

<use_case_selector>
**USE CASE SELECTOR - READ FIRST:**
- For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_tool
- For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

=====================================================================
⚠️ TOOL SELECTION WARNING ⚠️
This tool is ONLY for WEAVE TRACES (LLM operations), NOT for run metrics or experiments!
=====================================================================

**KEYWORD GUIDE:**
If user question contains:
- "runs", "experiments", "metrics" → Use query_wandb_tool
- "traces", "LLM calls" etc → Use this tool

**COMMON MISUSE CASES:**
❌ "Looking at metrics of my latest runs" - Do NOT use this tool, use query_wandb_tool instead
❌ "Compare performance across experiments" - Do NOT use this tool, use query_wandb_tool instead
</use_case_selector>

If the users asks for data about "runs" or "experiments" or anything about "experiment tracking"
then use the `query_wandb_tool` instead.
</use_case_selector>

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
default 200 characters. Otherwise the trace data is returned in full.
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
    Maximum length for string values in weave traces. Defaults to 200
return_full_data : bool, optional
    Whether to include full untruncated trace data. If True, the `truncate_length` parameter is ignored. If  \
`False` returns truncation_length = 0, no values for the column keys are returned. Defaults to True.
metadata_only : bool, optional
    Return only metadata without traces. Defaults to False

Returns
-------
str
    JSON string containing either full trace data or metadata only, depending on parameters

<examples>
    ```python
    # Get an overview of the traces in a project
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"root_traces_only": True},
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
    # If api_key was provided, create a new service with that key
    service = get_trace_service()
    if api_key:
        service = TraceService(
            api_key=api_key,
            retries=retries,
            timeout=request_timeout,
        )

    try:
        api = WandBApiManager.get_api()
        # Do not log api_key here
        log_tool_call(
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
        )
    except Exception:
        pass

    # Query traces
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
        return_full_data=return_full_data,  # Match original behavior
        metadata_only=False,
    )

    # Match the return type of the original function (List[Dict])
    if result.traces:
        # Convert WeaveTrace objects to dictionaries if needed
        traces_as_dicts = []
        for trace in result.traces:
            if hasattr(trace, "model_dump"):
                # Pydantic model - convert to dict
                traces_as_dicts.append(trace.model_dump())
            elif isinstance(trace, dict):
                # Already a dict
                traces_as_dicts.append(trace)
            else:
                # Unknown type, try to convert to dict
                try:
                    traces_as_dicts.append(dict(trace))
                except Exception:
                    # If all else fails, convert to string
                    traces_as_dicts.append(
                        {"error": f"Could not convert {type(trace)} to dict"}
                    )
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
    # If api_key was provided, create a new service with that key
    service = get_trace_service()
    if api_key:
        service = TraceService(
            api_key=api_key,
            retries=retries,
        )

    # Log tool call after service is obtained, with viewer from WandBApiManager
    try:
        api = WandBApiManager.get_api()
        log_tool_call(
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
        )
    except Exception:
        pass

    # Query traces with pagination
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

    # Add raw traces for debugging if requested
    if debug_raw_traces and result.traces:
        # Create a copy to avoid modifying the original result
        result_dict = result.model_dump()
        result_dict["raw_traces"] = result.traces
        # Convert back to QueryResult
        result = QueryResult.model_validate(result_dict)

    assert isinstance(result, QueryResult), (
        f"Result type must be a QueryResult, found: {type(result)}"
    )
    return result
