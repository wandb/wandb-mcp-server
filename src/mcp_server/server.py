#!/usr/bin/env python
"""
Weave MCP Server - A Model Context Protocol server for querying Weave traces.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_server.query_models import query_paginated_wandb_gql, list_entity_projects

# Import query_traces and our new utilities
from mcp_server.query_weave import count_traces, paginated_query_traces
from mcp_server.report import create_report
from mcp_server.trace_utils import DateTimeEncoder
from mcp_server.utils import get_server_args

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weave-mcp-server")

# Create an MCP server using FastMCP
mcp = FastMCP("weave-mcp-server")


@mcp.tool()
async def query_weave_traces_tool(
    entity_name: str,
    project_name: str,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = None,
    offset: int = 0,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: Optional[List[str]] = None,
    expand_columns: Optional[List[str]] = None,
    truncate_length: Optional[int] = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
) -> str:
    """
    Query Weave traces, trace metadata, and trace costs with filtering and sorting options.

    <wandb_vs_weave_product_distinction>
    **IMPORTANT PRODUCT DISTINCTION:**
    W&B offers two distinct products with different purposes:
    
    1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
       lifecycle management. Use `query_wandb_gql_tool` for questions about:
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
    </wandb_vs_weave_product_distinction>

    <use_case_selector>
    **USE CASE SELECTOR - READ FIRST:**
    - For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_gql_tool
    - For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

    =====================================================================
    ⚠️ TOOL SELECTION WARNING ⚠️
    This tool is ONLY for WEAVE TRACES (LLM operations), NOT for run metrics or experiments!
    =====================================================================

    **KEYWORD GUIDE:**
    If user question contains:
    - "runs", "experiments", "metrics" → Use query_wandb_gql_tool
    - "traces", "LLM calls" etc → Use this tool

    **COMMON MISUSE CASES:**
    ❌ "Looking at metrics of my latest runs" - Do NOT use this tool, use query_wandb_gql_tool instead
    ❌ "Compare performance across experiments" - Do NOT use this tool, use query_wandb_gql_tool instead
    </use_case_selector>

    If the users asks for data about "runs" or "experiments" or anything about "experiment tracking"
    then use the `query_wandb_gql_tool` instead.
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

    - Evaluations: If asked about weave evaluations or evals traces filter for traces with:
      `op_name_contains = "Evaluation.evaluate"` as a first step. These ops are parent traces that contain
      aggregated stats and scores about the evaluation. The child traces of these ops are the actual evaluation results
      for each sample in an evaluation dataset. If asked about individual rows in an evaluation then use the parent_ids
      filter to return the child traces.

    - Weave nomenclature: Note that users might refer to weave ops as "traces" or "calls" or "traces" as "ops".

    </usage_guidance>
    </usage_tips>

    Args:
        entity_name: The Weights & Biases entity name (team or username)
        project_name: The Weights & Biases project name
        filters: Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_roots_only: Boolean to filter for only top-level/parent traces. Useful when you don't need
                to return the data from all child traces.
            - trace_id: Filter by specific trace ID
            - call_ids: Filter by specific call IDs (string or list of strings). Note it is the call_id, not the
              trace id that is exposed in a weave url.
            - parent_ids: Return traces that are children of the given parent trace ids (string or list of strings)
            - status: Filter by trace status, defined as whether or not the trace had an exception or not. Can be
                `success` or `exception`.
            - time_range: Dict with "start" and "end" datetime strings. Datetime strings should be in ISO format
                (e.g. `2024-01-01T00:00:00Z`)
            - attributes: Dict of the weave attributes of the trace.
            - has_exception: Optional[bool] to filter traces by exception status:
                - None (or key not present): Show all traces regardless of exception status
                - True: Show only traces that have exceptions (exception field is not null)
                - False: Show only traces without exceptions (exception field is null)
        sort_by: Field to sort by (started_at, ended_at, op_name, etc.). Defaults to 'started_at'
        sort_direction: Sort direction ('asc' or 'desc'). Defaults to 'desc'
        limit: Maximum number of results to return. Defaults to None
        offset: Number of results to skip (for pagination). Defaults to 0
        include_costs: Include tracked api cost information in the results. Defaults to True
        include_feedback: Include weave annotations (human labels/feedback). Defaults to True
        columns: List of specific columns to include in the results. Its almost always a good idea to specficy the
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
                summary: typing.Optional[SummaryMap]
                wb_user_id: typing.Optional[str]
                wb_run_id: typing.Optional[str]
                deleted_at: typing.Optional[datetime.datetime]
        expand_columns: List of columns to expand in the results. Defaults to None
        truncate_length: Maximum length for string values in weave traces. Defaults to 200
        return_full_data: Whether to include full untruncated trace data. Defaults to False
        metadata_only: Return only metadata without traces. Defaults to False

    Returns:
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
    try:
        # Use paginated query with chunks of 20
        result = await paginated_query_traces(
            entity_name=entity_name,
            project_name=project_name,
            chunk_size=50,
            filters=filters,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            include_costs=include_costs,
            include_feedback=include_feedback,
            columns=columns,
            expand_columns=expand_columns,
            truncate_length=truncate_length,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )

        return json.dumps(result, cls=DateTimeEncoder)

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error querying traces: {str(e)}"


@mcp.tool()
async def count_weave_traces_tool(
    entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
) -> str:
    """Count Weave traces matching the given filters.

    Use this tool to query data from Weights & Biases Weave, an observability product for 
    tracing and evaluating LLMs and GenAI apps.

    This tool only provides COUNT information about traces, not actual metrics or run data.

    <tool_choice_guidance>
    <wandb_vs_weave_product_distinction>
    **IMPORTANT PRODUCT DISTINCTION:**
    W&B offers two distinct products with different purposes:
    
    1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
       lifecycle management. Use `query_wandb_gql_tool` for questions about:
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
    </wandb_vs_weave_product_distinction>

    <use_case_selector>
    **USE CASE SELECTOR - READ FIRST:**
    - For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_gql_tool
    - For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

    =====================================================================
    ⚠️ TOOL SELECTION WARNING ⚠️
    This tool is ONLY for WEAVE TRACES (LLM operations), NOT for run metrics or experiments!
    =====================================================================

    **KEYWORD GUIDE:**
    If user question contains:
    - "runs", "experiments", "metrics" → Use query_wandb_gql_tool
    - "traces", "LLM calls" etc → Use this tool

    **COMMON MISUSE CASES:**
    ❌ "Looking at metrics of my latest runs" - Do NOT use this tool, use query_wandb_gql_tool instead
    ❌ "Compare performance across experiments" - Do NOT use this tool, use query_wandb_gql_tool instead
    </use_case_selector>
    </tool_choice_guidance>
    
    Returns the total number of traces in a project and the number of root
    (i.e. "parent" or top-level) traces.

    This is more efficient than query_trace_tool when you only need the count.
    This can be useful to understand how many traces are in a project before
    querying for them as query_trace_tool can return a lot of data.

    Args:
        entity_name: The Weights & Biases entity name (team or username)
        project_name: The Weights & Biases project name
        filters: Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_id: Filter by specific trace ID
            - status: Filter by trace status (success, error, etc.)
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value to match
            - has_exception: Boolean to filter traces with/without exceptions

    Returns:
        JSON string containing the count of matching traces
    """
    try:
        # Call the synchronous count_traces function
        total_count = count_traces(
            entity_name=entity_name, project_name=project_name, filters=filters
        )

        # Create a copy of filters and ensure trace_roots_only is True
        root_filters = filters.copy() if filters else {}
        root_filters["trace_roots_only"] = True
        root_traces_count = count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=root_filters,
        )

        return json.dumps(
            {"total_count": total_count, "root_traces_count": root_traces_count}
        )

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error counting traces: {str(e)}"


@mcp.tool()
def query_wandb_gql_tool(
    query: str, 
    variables: Dict[str, Any] = None,
    max_items: int = 100,
    items_per_page: int = 20,
    ) -> Dict[str, Any]:
    """
    Execute an arbitrary GraphQL query against the Weights & Biases (W&B) Models API.

    Use this tool to query data from Weights & Biases Models features, including experiment tracking runs, 
    model registry, reports, artifacts, sweeps. 

    <wandb_vs_weave_product_distinction>
    **IMPORTANT PRODUCT DISTINCTION:**
    W&B offers two distinct products with different purposes:
    
    1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
       lifecycle management. Use `query_wandb_gql_tool` for questions about:
       - Experiment runs, metrics, and performance comparisons
       - Artifact management and model registry
       - Hyperparameter optimization and sweeps
       - Project dashboards and reports
    
    2. W&B Weave: A toolkit for LLM and GenAI application observability and evaluation. Use
       `query_weave_traces_tool` for questions about:
       - Execution traces and paths of LLM operations
       - LLM inputs, outputs, and intermediate results
       - Chain of thought visualization and debugging
       - LLM evaluation results and feedback
    </wandb_vs_weave_product_distinction>

    <use_case_selector>
    **USE CASE SELECTOR - READ FIRST:**
    - For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_gql_tool (this tool)
    - For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

    =====================================================================
    ⚠️ TOOL SELECTION WARNING ⚠️
    This tool is ONLY for WANDB MODELS DATA (MLOps), NOT for LLM TRACES or GENAI APPS!
    =====================================================================

    **KEYWORD GUIDE:**
    If user question contains:
    - "runs", "experiments", "metrics" → Use query_wandb_gql_tool (this tool)
    - "traces", "LLM calls" etc → Use query_weave_traces_tool

    **COMMON MISUSE CASES:**
    ❌ "Looking at performance of my latest weave evals" - Use query_weave_traces_tool 
    ❌ "what system prompt was used for my openai call" - Use query_weave_traces_tool 
    ❌ "Show me the traces for my weave evals" - Use query_weave_traces_tool
    </use_case_selector>

    This function allows interaction with W&B data (Projects, Runs, Artifacts, Sweeps, Reports, etc.)
    using the GraphQL query language.

    Args:
        query (str): The GraphQL query string. This defines the operation (query/mutation),
                     the data to fetch (selection set), and any variables used.
        variables (Dict[str, Any], optional): A dictionary of variables to pass to the query.
                                              Keys should match variable names defined in the query
                                              (e.g., $entity, $project). Values should match the
                                              expected types (String, Int, Float, Boolean, ID, JSONString).
                                              **Crucially, complex arguments like `filters` MUST be provided 
                                              as a JSON formatted *string*. Use `json.dumps()` in Python 
                                              to create this string.**
        max_items (int, optional): Maximum number of items to fetch across all pages (default: 100).
        items_per_page (int, optional): Number of items to request per page (default: 20).

    Returns:
        Dict[str, Any]: The raw dictionary result from the W&B GraphQL API.

    <required_pagination_structure>
    **⚠️ REQUIRED PAGINATION STRUCTURE ⚠️**
    
    All collection queries MUST include the complete W&B connection pattern with these elements:
    1. `edges` array containing nodes
    2. `node` objects inside edges containing your data fields
    3. `pageInfo` object with:
       - `endCursor` field (to enable pagination)
       - `hasNextPage` field (to determine if more data exists)
    
    This is a strict requirement enforced by the pagination system. Queries without this 
    structure will fail with the error "Query doesn't follow the W&B connection pattern."
    
    Example of required pagination structure for any collection:
    ```graphql
    runs(first: 10) {  # or artifacts, files, etc.
      edges {
        node {
          id
          name
          # ... other fields you need
        }
        # cursor # Optional: include cursor if needed for specific pagination logic
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
    ```
    </required_pagination_structure>
    
    <llm_context_window_management>
    **LLM CONTEXT WINDOW MANAGEMENT**
    
    The results of this tool are returned to a LLM. Be mindful of the context window of the LLM!
    
    <warning_about_open_ended_queries>
    **WARNING: AVOID OPEN-ENDED QUERIES!** 
    
    Open-ended queries should be strictly avoided when:
    - There are a lot of runs in the project (e.g., hundreds or thousands)
    - There are runs with large amounts of data (e.g., many metrics, large configs, etc.)
    
    Examples of problematic open-ended queries:
    - Requesting all runs in a project without limits
    - Requesting complete run histories without filtering specific metrics
    - Requesting all files from artifacts without specifying names/types
    
    Instead, always:
    - Use the `first` parameter to limit the number of items returned (start small, e.g., 5-10)
    - Apply specific filters to narrow down results (e.g., state, creation time, metrics)
    - Request only the specific fields needed, avoid selecting everything
    - Consider paginating results if necessary (don't request everything at once)
    
    Bad:
    ```graphql
    query AllRuns($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        # Potentially huge response: requests all fields for all runs
        runs { edges { node { id name state history summaryMetrics config files { edges { node { name size }}}}}}}
      }
    }
    ```
    
    Good:
    ```graphql
    query LimitedRuns($entity: String!, $project: String!) {
      project(name: $project, entityName: $entity) {
        # Limits runs, specifies filters, and selects only necessary fields
        runs(first: 5, filters: "{\\"state\\":\\"finished\\"}") {
          edges { 
            node { 
              id 
              name 
              createdAt 
              summaryMetrics # Get summary JSON, parse later if needed
            } 
          }
          pageInfo { endCursor hasNextPage } # Always include pageInfo for collections
        }
      }
    }
    ```
    </warning_about_open_ended_queries>
    
    Some tactics to consider to avoid exceeding the context window of the LLM when using this tool:
      - First return just metadata about the wandb project or run you will be returning.
      - Select only a subset of the data such as just particular columns or rows.
      - If you need to return a large amount of data consider using the `query_wandb_gql_tool` in a loop
      - Break up the query into smaller chunks.
    
    If you are returning just a sample subset of the data warn the user that this is a sample and that they should
    use the tool again with additional filters or pagination to get a more complete view.
    </llm_context_window_management>
    
    **Constructing GraphQL Queries:**

    1.  **Operation Type:** Start with `query` for fetching data or `mutation` for modifying data.
    2.  **Operation Name:** (Optional but recommended) A descriptive name (e.g., `ProjectInfo`).
    3.  **Variables Definition:** Define variables used in the query with their types (e.g., `($entity: String!, $project: String!)`). `!` means required.
    4.  **Selection Set:** Specify the fields you want to retrieve, nesting as needed based on the W&B schema.

    **W&B Schema Overview:**

    *   **Core Types:** `Entity`, `Project`, `Run`, `Artifact`, `Sweep`, `Report`, `User`, `Team`.
    *   **Relationships:** Entities contain Projects. Projects contain Runs, Sweeps, Artifacts. Runs use/are used by Artifacts. Sweeps contain Runs.
    *   **Common Fields:** `id`, `name`, `description`, `createdAt`, `config` (JSONString), `summaryMetrics` (JSONString - **Note:** use this field, 
          not `summary`, to access the run's summary dictionary as a JSON string), `historyKeys` (List of String), etc.
    *   **Connections (Lists):** Many lists (like `project.runs`, `artifact.files`) use a connection pattern:
        ```graphql
        runs(first: Int, after: String, filters: JSONString, order: String) {
          edges { node { id name ... } cursor }
          pageInfo { hasNextPage endCursor }
        }
        ```
        Use `first` for limit, `after` with `pageInfo.endCursor` for pagination, `filters` (as a JSON string) for complex filtering, and `order` for sorting.
    *   **Field Type Handling:**
        - Some fields require subfield selection (e.g., `tags { name }`) while others are scalar (e.g., `historyKeys`).
        - Check the schema if you get errors like "must have a selection of subfields" or "must not have a selection".

    **Query Examples:**

    <!-- WANDB_GQL_EXAMPLE_START name=GetProjectInfo -->
    *   **Get Project Info:** (Doesn't retrieve a collection, no pagination needed)
        ```graphql
        query ProjectInfo($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            id
            name
            entityName
            description
            runCount
          }
        }
        ```
        ```python
        variables = {"entity": "my-entity", "project": "my-project"}
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetProjectInfo -->

    <!-- WANDB_GQL_EXAMPLE_START name=GetSortedRuns -->
    *   **Get Sorted Runs:** (Retrieves a collection, requires pagination structure)
        ```graphql
        query SortedRuns($project: String!, $entity: String!, $limit: Int, $order: String) {
          project(name: $project, entityName: $entity) {
            runs(first: $limit, order: $order) {
              edges {
               node { id name displayName state createdAt summaryMetrics }
               cursor # Optional cursor
             }
             pageInfo { # Required for collections
               hasNextPage
               endCursor
             }
           }
         }
       }
       ```
       ```python
       variables = {
           "entity": "my-entity",
           "project": "my-project",
           "limit": 10,
           "order": "+summary_metrics.accuracy"  # Ascending order by accuracy
           # Use "-createdAt" for newest first (default if order omitted)
           # Use "+createdAt" for oldest first
       }
       ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetSortedRuns -->

    <!-- WANDB_GQL_EXAMPLE_START name=GetFilteredRuns -->
    *   **Get Runs with Pagination and Filtering:** (Requires pagination structure)
        ```graphql
        query FilteredRuns($project: String!, $entity: String!, $limit: Int, $cursor: String, $filters: JSONString, $order: String) {
          project(name: $project, entityName: $entity) {
            runs(first: $limit, after: $cursor, filters: $filters, order: $order) {
              edges {
                node { id name state createdAt summaryMetrics }
                cursor # Optional cursor
              }
              pageInfo { endCursor hasNextPage } # Required
            }
          }
        }
        ```
        ```python
        # Corrected: Show filters as the required escaped JSON string
        variables = {
            "entity": "my-entity",
            "project": "my-project",
            "limit": 10,
            "order": "-summary_metrics.accuracy",  # Optional: sort
            "filters": "{\"state\": \"finished\", \"summary_metrics.accuracy\": {\"$gt\": 0.9}}", # Escaped JSON string
            # "cursor": previous_pageInfo_endCursor # Optional for next page
        }
        # Note: The *content* of the `filters` JSON string must adhere to the specific 
        # filtering syntax supported by the W&B API (e.g., using operators like `$gt`, `$eq`, `$in`). 
        # Refer to W&B documentation for the full filter specification.
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetFilteredRuns -->

    <!-- WANDB_GQL_EXAMPLE_START name=GetRunHistoryKeys -->
    *   **Get Run History Keys:** (Run is not a collection, historyKeys is scalar)
        ```graphql
        query RunHistoryKeys($entity: String!, $project: String!, $runName: String!) {
          project(name: $project, entityName: $entity) {
            run(name: $runName) {
              id
              name
              historyKeys # Returns ["metric1", "metric2", ...]
            }
          }
        }
        ```
        ```python
        variables = {"entity": "my-entity", "project": "my-project", "runName": "run-abc"}
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetRunHistoryKeys -->
        
    <!-- WANDB_GQL_EXAMPLE_START name=GetRunHistorySampled -->
    *   **Get Specific Run History Data:** (Uses `sampledHistory` for specific keys)
        ```graphql
        # Corrected: Use specs argument
        query RunHistorySampled($entity: String!, $project: String!, $runName: String!, $specs: [JSONString!]!) {
          project(name: $project, entityName: $entity) {
            run(name: $runName) {
              id
              name
              # Use sampledHistory with specs to get actual values for specific keys
              sampledHistory(specs: $specs) { 
                 step # The step number
                 timestamp # Timestamp of the log
                 item # JSON string containing {key: value} for requested keys at this step
              } 
            }
          }
        }
        ```
        ```python
        # Corrected: Define specs variable with escaped JSON string literal for keys
        variables = {
            "entity": "my-entity", 
            "project": "my-project", 
            "runName": "run-abc", 
            "specs": ["{\"keys\": [\"loss\", \"val_accuracy\"]}}"] # List containing escaped JSON string
        }
        # Note: sampledHistory returns rows where *at least one* of the specified keys was logged.
        # The 'item' field is a JSON string, you'll need to parse it (e.g., json.loads(row['item'])) 
        # to get the actual key-value pairs for that step. It might not contain all requested keys
        # if they weren't logged together at that specific step.
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetRunHistorySampled -->

    <!-- WANDB_GQL_EXAMPLE_START name=GetArtifactDetails -->
    *   **Get Artifact Details:** (Artifact is not a collection, but `files` is)
        ```graphql
        query ArtifactDetails($entity: String!, $project: String!, $artifactName: String!) {
          project(name: $project, entityName: $entity) {
            artifact(name: $artifactName) { # Name format often 'artifact-name:version' or 'artifact-name:alias'
              id
              digest
              description
              state
              size
              createdAt
              metadata # JSON String
              aliases { alias } # Corrected: Use 'alias' field instead of 'name'
              files { # Files is a collection, requires pagination structure
                edges { 
                  node { name url digest } # Corrected: Removed 'size' from File fields
                } 
                pageInfo { endCursor hasNextPage } # Required for files collection
              } 
            }
          }
        }
        ```
        ```python
        variables = {"entity": "my-entity", "project": "my-project", "artifactName": "my-dataset:v3"}
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetArtifactDetails -->

    <!-- WANDB_GQL_EXAMPLE_START name=GetViewerInfo -->
    *   **Get Current User Info (Viewer):** (No variables needed)
        ```graphql
        query GetViewerInfo {
          viewer {
            id
            username
            email
            entity
          }
        }
        ```
        ```python
        # No variables needed for this query
        variables = {}
        ```
    <!-- WANDB_GQL_EXAMPLE_END name=GetViewerInfo -->

    **Troubleshooting Common Errors:**

    *   `"Cannot query field 'summary' on type 'Run'"`: Use the `summaryMetrics` field instead of `summary`. It returns a JSON string containing the summary dictionary.
    *   `"Argument 'filters' has invalid value ... Expected type 'JSONString'"`: Ensure the `filters` argument in your `variables` is a JSON formatted *string*, likely created using `json.dumps()`. Also check the *content* of the filter string for valid W&B filter syntax.
    *   `"400 Client Error: Bad Request"` (especially when using filters): Double-check the *syntax* inside your `filters` JSON string. Ensure operators (`$eq`, `$gt`, etc.) and structure are valid for the W&B API. Invalid field names or operators within the filter string can cause this.
    *   `"Unknown argument 'direction' on field 'runs'"`: Control sort direction using `+` (ascending) or `-` (descending) prefixes in the `order` argument string (e.g., `order: "-createdAt"`), not with a separate `direction` argument.
    *   Errors related to `history` (e.g., `"Unknown argument 'keys' on field 'history'"` or `"Field 'history' must not have a selection..."`): To get *available* metric keys, query the `historyKeys` field (returns `[String!]`). To get *time-series data* for specific keys, use the `sampledHistory(keys: [...])` field as shown in the examples; it returns structured data points. The simple `history` field might return raw data unsuitable for direct querying or is deprecated.
    *   `"Query doesn't follow the W&B connection pattern"`: Ensure any field returning a list/collection (like `runs`, `files`, `artifacts`, etc.) includes the full `edges { node { ... } } pageInfo { endCursor hasNextPage }` structure. This is mandatory for pagination.
    *   `"Field must not have a selection"` / `"Field must have a selection"`: Check if the field you are querying is a scalar type (like `String`, `Int`, `JSONString`, `[String!]`) which cannot have sub-fields selected, or an object type which requires you to select sub-fields.

    **Notes:**
    *   Refer to the official W&B GraphQL schema (via introspection or documentation) for the most up-to-date field names, types, and available filters/arguments.
    *   Structure your query to request only the necessary data fields to minimize response size and improve performance.
    *   **Sorting:** Use the `order` parameter string. Prefix with `+` for ascending, `-` for descending (default). 
            Common sortable fields: `createdAt`, `updatedAt`, `heartbeatAt`, `config.*`, `summary_metrics.*`.
    *   Handle potential errors in the returned dictionary (e.g., check for an 'errors' key in the response).
    """
    return query_paginated_wandb_gql(query, variables, max_items, items_per_page)



@mcp.tool()
async def create_wandb_report_tool(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: str = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
) -> str:
    """
    Create a new Weights & Biases Report and add text and HTML-rendered charts. Useful to save/document analysis and other findings.

    Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases. 
    
    Always provide the returned report link to the user.

    <plots_html_usage_guide>
    - If the analsis has generated plots then they can be logged to a Weights & Biases report via converting them to html.
    - All charts should be properly rendered in raw HTML, do not use any placeholders for any chart, render everything.
    - All charts should be beautiful, tasteful and well proportioned.
    - Plot html code should use SVG chart elements that should render properly in any modern browser.
    - Include interactive hover effects where it makes sense.
    - If the analysis contains multiple charts, break up the html into one section of html per chart.
    - Ensure that the axis labels are properly set and aligned for each chart.
    - Always use valid markdown for the report text.
    </plots_html_usage_guide>

    Args:
        entity_name: str, The W&B entity (team or username) - required
        project_name: str, The W&B project name - required
        title: str, Title of the W&B Report - required
        description: str, Optional description of the W&B Report
        markdown_report_text: str, beuatifully formatted markdown text for the report body
        plots_html: str, Optional dict of plot name and html string of any charts created as part of an analysis

    Returns:
        str, The url to the report

    Example:
        ```python
        # Create a simple report
        report = create_report(
            entity_name="my-team",
            project_name="my-project",
            title="Model Analysis Report",
            description="Analysis of our latest model performance",
            markdown_report_text='''
                # Model Analysis Report
                [TOC]
                ## Performance Summary
                Our model achieved 95% accuracy on the test set.
                ### Key Metrics
                Precision: 0.92
                Recall: 0.89
            '''
        )
        ```
    """
    # Handle plot_htmls if it's a JSON string
    if isinstance(plots_html, str):
        try:
            plots_html = json.loads(plots_html)
        except json.JSONDecodeError:
            # If it's not valid JSON, keep it as is (though this will likely cause other errors)
            pass

    report_link = create_report(
        entity_name=entity_name,
        project_name=project_name,
        title=title,
        description=description,
        markdown_report_text=markdown_report_text,
        plots_html=plots_html,
    )
    return f"The report was saved here: {report_link}"


@mcp.tool()
def query_wandb_entity_projects(entity: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all projects for a specific wandb or weave entity. Useful to use when 
    the user hasn't specified a project name or queries are failing due to a 
    missing or incorrect Weights & Biases project name.

    If no entity is provided, the tool will fetch all projects for the current user 
    as well as all the project in the teams they are part of.

    <critical_info>
    
    **Important:**
    
    Do not use this tool if the user has not specified a W&BB entity name. Instead ask
    the user to provide either their W&B username or W&B team name.
    </critical_info>

    <debugging_tips>
    
    **Error Handling:**

    If this function throws an error, it's likely because the W&B entity name is incorrect.
    If this is the case, ask the user to double check the W&B entity name given by the user, 
    either their personal user or their W&B Team name.

    **Expected Project Name Not Found:**

    If the user doesn't see the project they're looking for in the list of projects,
    ask them to double check the W&B entity name, either their personal W&B username or their 
    W&B Team name.
    </debugging_tips>
    
    Args:
        entity (str): The wandb entity (username or team name)
        
    Returns:
        List[Dict[str, Any]]: List of project dictionaries containing:
            - name: Project name
            - entity: Entity name
            - description: Project description
            - visibility: Project visibility (public/private)
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - tags: List of project tags
    """
    return list_entity_projects(entity)


def cli():
    """Command-line interface for starting the Weave MCP Server."""
    # Validate that we have the required API key
    if not get_server_args().wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY must be set either as an environment variable, in .env file, or as a command-line argument"
        )

    print(f"Starting Weights & Biases MCP Server for {get_server_args().product_name}")
    logger.info(
        f"API Key configured: {'Yes' if get_server_args().wandb_api_key else 'No'}"
    )

    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
