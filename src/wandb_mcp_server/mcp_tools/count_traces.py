import base64
import json
import os
from typing import Any, Dict

import requests

from wandb_mcp_server.weave_api.query_builder import QueryBuilder
from wandb_mcp_server.mcp_tools.tools_utils import get_retry_session
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call

logger = get_rich_logger(__name__)

COUNT_WEAVE_TRACES_TOOL_DESCRIPTION = """count Weave traces and return the total storage \
size in bytes for the given filters.

Use this tool to query data from Weights & Biases Weave, an observability product for 
tracing and evaluating LLMs and GenAI apps.

This tool only provides COUNT information and STORAGE SIZE (bytes) about traces, \
not actual logged traces data, metrics or run data.

<tool_choice_guidance>
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
</tool_choice_guidance>

Returns the total number of traces in a project and the number of root
(i.e. "parent" or top-level) traces.

This is more efficient than query_trace_tool when you only need the count.
This can be useful to understand how many traces are in a project before
querying for them as query_trace_tool can return a lot of data.

Parameters
----------
entity_name : str
    The Weights & Biases entity name (team or username).
project_name : str
    The Weights & Biases project name.
filters : Dict[str, Any], optional
    Dict of filter conditions, supporting:
        - display_name: Filter by display name (string or regex pattern)
        - op_name_contains: Filter for op_name containing a substring. Not a good idea to use in conjunction with trace_roots_only.
        - trace_id: Filter by specific trace ID
        - status: Filter by trace status ('success', 'error', etc.)
        - time_range: Dict with "start" and "end" datetime strings
        - latency: Filter by latency in milliseconds (summary.weave.latency_ms).
            Use a nested dict with operators: $gt, $lt, $eq, $gte, $lte.
            ($lt and $lte are implemented via logical negation on the backend).
            e.g., {"latency": {"$gt": 5000}}
        - attributes: Dict of attribute path and value/operator to match.
            Supports nested paths (e.g., "metadata.model_name") via dot notation.
            Value can be literal for equality or a dict with operator ($gt, $lt, $eq, $gte, $lte) for comparison
            (e.g., {"token_count": {"$gt": 100}}).
        - has_exception: Boolean to filter traces with/without exceptions
        - trace_roots_only: Boolean to filter for only top-level (aka parent) traces

Returns
-------
int
    The number of traces matching the query parameters.

Examples
--------
>>> # Count failed traces
>>> count = count_traces(
...     entity_name="my-team",
...     project_name="my-project",
...     filters={"status": "error"}
... )
>>> # Count traces faster than 500ms
>>> count = count_traces(
...     entity_name="my-team",
...     project_name="my-project",
...     filters={"latency": {"$lt": 500}}
... )
"""


def count_traces(
    entity_name: str,
    project_name: str,
    filters: dict = None,
    request_timeout: int = 30,
) -> int:
    """Count the number of traces matching the given filters.

    Counts without retrieving the full trace data, making it more efficient
    than `query_traces` when only the count is needed.

    Parameters
    ----------
    entity_name : str
        The Weights & Biases entity name (team or username).
    project_name : str
        The Weights & Biases project name.
    filters : Dict[str, Any], optional
        Dict of filter conditions, supporting:
            - display_name: Filter by display name (string or regex pattern)
            - op_name_contains: Filter for op_name containing a substring
            - trace_id: Filter by specific trace ID
            - status: Filter by trace status ('success', 'error', etc.)
            - latency: Filter by latency in milliseconds (summary.weave.latency_ms).
                Use a nested dict with operators: $gt, $lt, $eq, $gte, $lte.
                Note: $lt and $lte are implemented via logical negation.
                e.g., {"latency": {"$gt": 5000}}
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value/operator to match.
                Supports nested paths (e.g., "metadata.model_name") via dot notation.
                Value can be literal for equality or a dict with operator ($gt, $lt, $eq, $gte, $lte) for comparison
                (e.g., {"token_count": {"$gt": 100}}).
            - has_exception: Boolean to filter traces with/without exceptions
            - trace_roots_only: Boolean to filter for only top-level (aka parent) traces
    request_timeout : int, optional
        Timeout for the HTTP request in seconds. Defaults to 30.

    Returns
    -------
    int
        The number of traces matching the query parameters.

    Examples
    --------
    >>> # Count failed traces
    >>> count = count_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={"status": "error"}
    ... )
    >>> # Count traces matching an attribute and latency > 1s
    >>> count = count_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={
    ...         "attributes": {"metadata.environment": "production"},
    ...         "latency": {"$gt": 1000}
    ...     }
    ... )
    """
    project_id = f"{entity_name}/{project_name}"

    # Get API key from context (set by auth middleware) or environment
    api_key = WandBApiManager.get_api_key()
    if not api_key:
        logger.error("W&B API key not found in context or environment variables.")
        raise ValueError("W&B API key is required to query Weave traces count.")
    
    # Debug logging to diagnose API key issues
    logger.debug(f"Using W&B API key: length={len(api_key)}, is_40_chars={len(api_key) == 40}")

    try:
        api = WandBApiManager.get_api()
        log_tool_call(
            "count_traces",
            api.viewer,
            {
                "entity_name": entity_name,
                "project_name": project_name,
                "filters": filters,
                "request_timeout": request_timeout,
            },
        )
    except Exception:
        pass

    request_body: Dict[str, Any] = {"project_id": project_id}
    filter_payload: Dict[
        str, Any
    ] = {}  # For fields that go into the top-level 'filter' object
    complex_filters_for_query_expr: Dict[
        str, Any
    ] = {}  # For fields that go into query.$expr

    if filters:
        # Keys that belong inside the 'filter' object in the request body
        # as per https://weave-docs.wandb.ai/reference/service-api/calls-query-stats-calls-query_stats-post
        direct_filter_keys = {
            "op_names",
            "op_name",  # op_name will be converted to op_names list
            "input_refs",
            "output_refs",
            "parent_ids",
            "trace_ids",
            "trace_id",  # trace_id will be converted to trace_ids list
            "call_ids",
            "trace_roots_only",
            "wb_user_ids",
            "wb_run_ids",
        }

        temp_op_names = []
        if "op_name" in filters:
            temp_op_names.append(filters["op_name"])
        if "op_names" in filters:
            val = filters["op_names"]
            if isinstance(val, list):
                temp_op_names.extend(val)
            else:
                temp_op_names.append(val)
        if temp_op_names:
            filter_payload["op_names"] = list(set(temp_op_names))

        temp_trace_ids = []
        if "trace_id" in filters:
            temp_trace_ids.append(filters["trace_id"])
        if "trace_ids" in filters:
            val = filters["trace_ids"]
            if isinstance(val, list):
                temp_trace_ids.extend(val)
            else:
                temp_trace_ids.append(val)
        if temp_trace_ids:
            filter_payload["trace_ids"] = list(set(temp_trace_ids))

        # Handle other direct filter keys
        for key in [
            "input_refs",
            "output_refs",
            "parent_ids",
            "call_ids",
            "wb_user_ids",
            "wb_run_ids",
        ]:
            if key in filters:
                value = filters[key]
                filter_payload[key] = [value] if not isinstance(value, list) else value

        if "trace_roots_only" in filters:
            filter_payload["trace_roots_only"] = filters["trace_roots_only"]
        # Per docs, trace_roots_only is a boolean, not a list.
        # If not in filters, it's omitted, API default (false) should apply.

        # Populate complex_filters_for_query_expr for remaining keys
        for key, value in filters.items():
            # Skip keys already handled in direct_filter_keys or their singular versions
            if key not in direct_filter_keys and key not in ["op_name", "trace_id"]:
                complex_filters_for_query_expr[key] = value

    # Add the constructed filter_payload to the main request_body if it's not empty
    if filter_payload:
        request_body["filter"] = filter_payload

    # Build the query expression from remaining complex filters
    if complex_filters_for_query_expr:
        query_expr_obj = QueryBuilder.build_query_expression(
            complex_filters_for_query_expr
        )
        if query_expr_obj:
            dumped_query = query_expr_obj.model_dump(by_alias=True, exclude_none=True)
            if dumped_query and dumped_query.get("$expr"):
                request_body["query"] = dumped_query

    # Execute the HTTP query
    from wandb_mcp_server.config import WF_TRACE_SERVER_URL
    weave_server_url = WF_TRACE_SERVER_URL
    url = f"{weave_server_url}/calls/query_stats"

    auth_token = base64.b64encode(f":{api_key}".encode()).decode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",  # /calls/query_stats returns application/json
        "Authorization": f"Basic {auth_token}",
    }

    session = get_retry_session()

    logger.debug(f"Posting to {url} with body: {json.dumps(request_body)}")

    try:
        response = session.post(
            url,
            headers=headers,
            data=json.dumps(request_body),  # Ensure complex objects are serialized
            timeout=request_timeout,
        )

        if response.status_code != 200:
            error_msg = f"Error querying Weave trace count: {response.status_code} - {response.text}"
            logger.error(error_msg)
            # Log API key info for debugging
            logger.error(f"API key info: length={len(api_key)}, is_40_chars={len(api_key) == 40}")
            if "40 characters" in response.text:
                logger.error(f"W&B requires exactly 40 character API keys. Current key has {len(api_key)} characters.")
            # Log request body for easier debugging on error
            logger.debug(f"Failed request body: {json.dumps(request_body)}")
            raise Exception(error_msg)

        response_json = response.json()
        return response_json.get("count", 0)  # Default to 0 if count is not in response

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed for project {project_id}: {e}")
        if isinstance(e, requests.exceptions.RetryError):
            if e.__cause__ and hasattr(e.__cause__, "reason") and e.__cause__.reason:
                logger.error(
                    f"Specific reason for retry exhaustion: {e.__cause__.reason}"
                )
        logger.debug(
            f"Failed request body during exception for {project_id}: {json.dumps(request_body)}"
        )
        # traceback.print_exc() # Uncomment for detailed traceback during development
        raise Exception(
            f"Failed to query Weave trace count for {project_id} due to network error: {e}"
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response for {project_id}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}"
        )
        raise Exception(f"Failed to parse Weave API response for {project_id}: {e}")
