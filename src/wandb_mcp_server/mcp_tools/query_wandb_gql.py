"""Module for querying the W&B GraphQL API."""

import copy
import logging
import traceback
from typing import Any, Dict, List, Optional

import wandb
from graphql import parse
from graphql.language import ast as gql_ast
from graphql.language import printer as gql_printer
from graphql.language import visitor as gql_visitor
from wandb_gql import gql  # This must be imported after wandb
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call

logger = get_rich_logger(__name__)


QUERY_WANDB_GQL_TOOL_DESCRIPTION = """Execute an arbitrary GraphQL query against the Weights & Biases (W&B) Models API.

Use this tool to query data from Weights & Biases Models features, including experiment tracking runs, 
model registry, reports, artifacts, sweeps. 

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
    `query_weave_traces_tool` for questions about:
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
- For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_tool (this tool)
- For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

=====================================================================
⚠️ TOOL SELECTION WARNING ⚠️
This tool is ONLY for WANDB MODELS DATA (MLOps), NOT for LLM TRACES or GENAI APPS!
=====================================================================

**KEYWORD GUIDE:**
If user question contains:
- "runs", "experiments", "metrics" → Use query_wandb_tool (this tool)
- "traces", "LLM calls" etc → Use query_weave_traces_tool

**COMMON MISUSE CASES:**
❌ "Looking at performance of my latest weave evals" - Use query_weave_traces_tool 
❌ "what system prompt was used for my openai call" - Use query_weave_traces_tool 
❌ "Show me the traces for my weave evals" - Use query_weave_traces_tool

<query_analysis_step>
**STEP 1: ANALYZE THE USER QUERY FIRST!**
Before constructing the GraphQL query, determine how the user is referring to W&B entities, especially runs:
  - Is the user providing a short, 8-character **Run ID** (e.g., `gtng2y4l`, `h0fm5qp5`)?
  - Or are they providing a longer, human-readable **Display Name** (e.g., `transformer_train_run_123`, `eval_on_benchmark_v2`)?
Your choice of query structure depends heavily on this analysis (see Key Concepts and Examples below).
</query_analysis_step>

<key_concepts>
**KEY CONCEPTS - READ CAREFULLY:**

*   **Run ID vs. Display Name:**
    *   To fetch a **single, specific run** using its unique 8-character ID (e.g., `gtng2y4l`), \
use the `run(name: $runId)` field. The variable `$runId` MUST be the ID, not the display name.
    *   To **find runs based on their human-readable `displayName`** (e.g., `my-cool-experiment-1`), \
use the `runs` collection field with a `filters` argument like: `runs(filters: "{\\"displayName\\":\
{\\"$eq\\":\\"my-cool-experiment-1\\"}}")`. This might return multiple runs if display names are not unique.
*   **Filters require JSON Strings:** When using the `filters` argument (e.g., for `runs`, `artifacts`), \
the value provided in the `variables` dictionary MUST be a JSON formatted *string*. Use `json.dumps()` in Python to create it.
*   **Collections Require Pagination Structure:** Queries fetching lists/collections (like `project.runs`, \
`artifact.files`) MUST include the `edges { node { ... } } pageInfo { endCursor hasNextPage }` pattern.
*   **Summary Metrics:** Use the `summaryMetrics` field (returns a JSON string) to access a run's summary \
dictionary, not the deprecated `summary` field.
</key_concepts>

This function allows interaction with W&B data (Projects, Runs, Artifacts, Sweeps, Reports, etc.)
using the GraphQL query language.

Parameters
----------
query : str
   he GraphQL query string. This defines the operation (query/mutation),
                    the data to fetch (selection set), and any variables used.
variables : dict[str, Any] | None, optional
    A dictionary of variables to pass to the query.
                                            Keys should match variable names defined in the query
                                            (e.g., $entity, $project). Values should match the
                                            expected types (String, Int, Float, Boolean, ID, JSONString).
                                            **Crucially, complex arguments like `filters` MUST be provided 
                                            as a JSON formatted *string*. Use `json.dumps()` in Python 
                                            to create this string.**
max_items : int, optional
    Maximum number of items to fetch across all pages. Default is 100.
items_per_page : int, optional
    Number of items to request per page. Default is 50.

Returns
-------
Dict[str, Any]
    The aggregated GraphQL response dictionary.

<critical_warning>
**⚠️ CRITICAL WARNING: Run ID vs. Display Name ⚠️**
If the user query mentions a run using its **long, human-readable name** (Display Name), you **MUST** use the `runs(filters: ...)` approach shown in the examples.
**DO NOT** use `run(name: ...)` with a Display Name; it will fail because `name` expects the short Run ID. Use `run(name: ...)` **ONLY** when the user provides the 8-character Run ID.
Review the "Minimal Example: Run ID vs Display Name" and "Get Run by Display Name" examples carefully.
</critical_warning>

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
    - If you need to return a large amount of data consider using the `query_wandb_tool` in a loop
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

<!-- WANDB_GQL_EXAMPLE_START name=MinimalRunIdVsDisplayName -->
*   **Minimal Example: Run ID vs Display Name:**
    *   **A) User provides Run ID (e.g., "get info for run h0fm5qp5"):**
        ```graphql
        query GetRunById($entity: String!, $project: String!, $runId: String!) {
          project(name: $project, entityName: $entity) {
            # Use run(name: ...) with the Run ID
            run(name: $runId) {
              id
              name # This will be the Run ID
              displayName # This is the human-readable name
            }
          }
        }
        ```
        ```python
        variables = {"entity": "...", "project": "...", "runId": "h0fm5qp5"}
        ```
    *   **B) User provides Display Name (e.g., "get info for run transformer_train_123"):**
        ```graphql
        # Note: Querying *runs* collection and filtering
        query GetRunByDisplayNameMinimal($project: String!, $entity: String!, $displayNameFilter: JSONString) {
          project(name: $project, entityName: $entity) {
            # Use runs(filters: ...) with the Display Name
            runs(first: 1, filters: $displayNameFilter) {
              edges {
                node {
                  id
                  name # Run ID
                  displayName # Display Name provided by user
                }
              }
              pageInfo { endCursor hasNextPage } # Required for collections
            }
          }
        }
        ```
        ```python
        import json
        variables = {
            "entity": "...",
            "project": "...",
            "displayNameFilter": json.dumps({"displayName": {"$eq": "transformer_train_123"}})
        }
        ```
<!-- WANDB_GQL_EXAMPLE_END name=MinimalRunIdVsDisplayName -->

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

<!-- WANDB_GQL_EXAMPLE_START name=GetRunByDisplayName -->
*   **Get Run by Display Name:** (Requires filtering and pagination structure)
    ```graphql
    # Note: Querying *runs* collection and filtering, not the singular run(name:...) field
    query GetRunByDisplayName($project: String!, $entity: String!, $displayNameFilter: JSONString) {
        project(name: $project, entityName: $entity) {
        # Filter the runs collection by displayName
        runs(first: 1, filters: $displayNameFilter) {
            edges {
            # Select desired fields from the node (the run)
            node { id name displayName state createdAt summaryMetrics }
            }
            # Required pageInfo for collections
            pageInfo { endCursor hasNextPage }
        }
        }
    }
    ```
    ```python
    # Use json.dumps for the filters argument
    import json
    target_display_name = "my-experiment-run-123"
    variables = {
        "entity": "my-entity",
        "project": "my-project",
        # Filter for the specific display name
        "displayNameFilter": json.dumps({"displayName": {"$eq": target_display_name}})
        # W&B filter syntax might vary slightly, check docs if needed. Common is {"field": "value"} or {"field": {"$operator": "value"}}
    }
    # Note: This finds runs where displayName *exactly* matches.
    # It might return multiple runs if display names are not unique.
    # The `name` field (often the run ID like 'gtng2y4l') is guaranteed unique per project.
    # Use `run(name: $runId)` if you know the unique run ID ('name').
    ```
<!-- WANDB_GQL_EXAMPLE_END name=GetRunByDisplayName -->

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
*   `"Cannot query field 'step' on type 'Run'"`: The `Run` type does not have a direct `step` field. To find the maximum step count or total steps logged, query the `summaryMetrics` field (look for a key like `_step` or similar in the returned JSON string) or use the `historyLineCount` field which indicates the total number of history rows logged (often corresponding to steps).

**Notes:**
*   Refer to the official W&B GraphQL schema (via introspection or documentation) for the most up-to-date field names, types, and available filters/arguments.
*   Structure your query to request only the necessary data fields to minimize response size and improve performance.
*   **Sorting:** Use the `order` parameter string. Prefix with `+` for ascending, `-` for descending (default). 
        Common sortable fields: `createdAt`, `updatedAt`, `heartbeatAt`, `config.*`, `summary_metrics.*`.
*   Handle potential errors in the returned dictionary (e.g., check for an 'errors' key in the response).
"""


def find_paginated_collections(
    obj: Dict, current_path: Optional[List[str]] = None
) -> List[List[str]]:
    """Find collections in a response that follow the W&B connection pattern. Returns List[List[str]]."""
    # Ensure this implementation correctly builds and returns List[List[str]]
    if current_path is None:
        current_path = []
    collections = []
    if isinstance(obj, dict):
        if (
            "edges" in obj
            and "pageInfo" in obj
            and isinstance(obj.get("edges"), list)
            and isinstance(obj.get("pageInfo"), dict)
            and "hasNextPage" in obj.get("pageInfo", {})
            and "endCursor" in obj.get("pageInfo", {})
        ):
            collections.append(list(current_path))  # Correct: append list path
        # Recurse correctly
        for key, value in obj.items():
            current_path.append(key)
            collections.extend(find_paginated_collections(value, current_path))
            current_path.pop()
    elif isinstance(obj, list):
        for item in obj:
            collections.extend(find_paginated_collections(item, current_path))
    return collections


def get_nested_value(obj: Dict, path: list[str]) -> Optional[Any]:
    """Get a value from a nested dictionary using a list of keys (path)."""
    current = obj
    # Iterate directly over the list path
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def query_paginated_wandb_gql(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    max_items: int = 100,
    items_per_page: int = 50,
) -> Dict[str, Any]:
    """
    Execute a GraphQL query against the W&B API with pagination support using AST modification.
    Handles a single paginated field detected via the connection pattern.
    Modifies the result dictionary in-place.

    Args:
        query: The GraphQL query string. MUST include pageInfo{hasNextPage, endCursor} for paginated fields.
        variables: Variables to pass to the GraphQL query.
        max_items: Maximum number of items to fetch across all pages (default: 100).
        items_per_page: Number of items to request per page (default: 20).
        deduplicate: Whether to deduplicate nodes by ID across pages (default: True).

    Returns:
        The aggregated GraphQL response dictionary.
    """
    result_dict = {}
    api = None
    limit_key = None
    try:
        # Use API key from environment (set by auth middleware for HTTP, or by user for STDIO)
        # Get API instance with proper key handling
        from wandb_mcp_server.api_client import get_wandb_api
        api = get_wandb_api()
        try:
            log_tool_call(
                "query_paginated_wandb_gql",
                api.viewer,
                {
                    "query": query,
                    "variables": variables,
                    "max_items": max_items,
                    "items_per_page": items_per_page,
                },
            )
        except Exception:
            pass
        logger.info(
            "--- Inside query_paginated_wandb_gql: Step 0: Execute Initial Query ---"
        )

        # Determine limit key and set initial page vars
        page1_vars_func = variables.copy() if variables is not None else {}
        limit_key = None
        for k in page1_vars_func:
            if k.lower() in ["limit", "first", "count"]:
                limit_key = k
                break
        if limit_key:
            # Ensure first page uses items_per_page if limit is too high or missing
            page1_vars_func[limit_key] = min(
                items_per_page, page1_vars_func.get(limit_key) or items_per_page
            )
        else:
            limit_key = "limit"
            page1_vars_func[limit_key] = items_per_page
            logger.debug(
                f"No limit variable found in input, adding '{limit_key}={items_per_page}'"
            )

        # Parse for execution
        try:
            parsed_initial_query = gql(query.strip())
        except Exception as e:
            logger.error(f"Failed to parse initial query with wandb_gql: {e}")
            return {"errors": [{"message": f"Failed to parse initial query: {e}"}]}

        # Execute initial query
        try:
            result1 = api.client.execute(
                parsed_initial_query, variable_values=page1_vars_func
            )
            result_dict = copy.deepcopy(result1)  # Work on a copy
            if "errors" in result_dict:
                logger.error(
                    f"GraphQL errors in initial response: {result_dict['errors']}"
                )
                return result_dict  # Return errors if found
        except Exception as e:
            logger.error(f"Failed to execute initial GraphQL query: {e}", exc_info=True)
            return {"errors": [{"message": f"Failed to execute initial query: {e}"}]}

        # Find Collections
        detected_paths = find_paginated_collections(result_dict)
        if not detected_paths:
            logger.info("No paginated paths detected. Returning initial result.")
            return result_dict

        # --- Use the first detected path ---
        # TODO: Enhance to handle multiple paths if necessary
        path_to_paginate = detected_paths[0]
        logger.info(f"Using path for pagination: {'/'.join(path_to_paginate)}")

        # Extract page 1 data
        runs_data1 = get_nested_value(result_dict, path_to_paginate)
        if runs_data1 is None:
            logger.warning(
                f"Could not extract data for pagination path {'/'.join(path_to_paginate)}. Returning initial result."
            )
            return result_dict
        page_info1 = get_nested_value(runs_data1, ["pageInfo"])
        if page_info1 is None:
            logger.warning(
                f"Could not extract pageInfo for pagination path {'/'.join(path_to_paginate)}. Returning initial result."
            )
            return result_dict

        cursor = page_info1.get("endCursor")
        has_next = page_info1.get("hasNextPage")
        initial_edges = runs_data1.get("edges", [])
        logging.info(f"Page 1 Results: {len(initial_edges)} runs.")
        logging.info(f"Page 1 PageInfo: {page_info1}")

        # Deduplicate initial edges and update result_dict
        seen_ids = set()
        current_edge_count = 0
        temp_initial_edges = []
        if initial_edges:
            for edge in initial_edges:
                try:
                    # Check max items even on page 1 relative to the limit
                    if current_edge_count >= max_items:
                        break
                    node_id = edge["node"]["id"]
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        temp_initial_edges.append(edge)
                        current_edge_count += 1
                except (KeyError, TypeError):
                    if current_edge_count < max_items:
                        temp_initial_edges.append(edge)
                        current_edge_count += 1
            # Update the edges in the result_dict
            target_collection_dict = get_nested_value(result_dict, path_to_paginate)
            if target_collection_dict:
                target_collection_dict["edges"] = temp_initial_edges[
                    :max_items
                ]  # Ensure initial list respects max_items
                current_edge_count = len(target_collection_dict["edges"])
            logging.info(
                f"Stored {current_edge_count} unique edges after page 1 (max: {max_items})."
            )

        if not has_next or not cursor or current_edge_count >= max_items:
            logger.info(
                "No further pages needed based on page 1 info or max_items reached."
            )
            # Ensure final pageInfo reflects reality
            target_pi_dict = get_nested_value(
                result_dict, path_to_paginate + ["pageInfo"]
            )
            if target_pi_dict:
                target_pi_dict["hasNextPage"] = False
            return result_dict

        # Generate Paginated Query String
        logging.info("\n--- Generating Paginated Query String --- ")
        generated_paginated_query_string = None
        after_variable_name = "after"  # Standard name
        try:
            initial_ast = parse(query.strip())
            visitor = AddPaginationArgsVisitor(
                field_paths=detected_paths,
                first_variable_name=limit_key,
                after_variable_name=after_variable_name,
            )
            modified_ast = gql_visitor.visit(copy.deepcopy(initial_ast), visitor)
            generated_paginated_query_string = gql_printer.print_ast(modified_ast)
            logger.info("AST modification and printing successful.")
        except Exception as e:
            logger.error(f"Failed to generate query string via AST: {e}", exc_info=True)
            return result_dict  # Return what we have if generation fails

        if generated_paginated_query_string is None:
            return result_dict

        logging.info(
            "\n--- Loop: Execute, Deduplicate, Aggregate In-Place, Check Limit ---"
        )
        page_num = 1
        current_cursor = cursor
        current_has_next = has_next
        final_page_info = page_info1

        while current_has_next:
            if current_edge_count >= max_items:
                logging.info(f"Reached max_items ({max_items}). Stopping loop.")
                final_page_info = {**final_page_info, "hasNextPage": False}
                break

            page_num += 1
            logging.info(f"\nFetching Page {page_num}...")
            page_vars = (
                variables.copy() if variables is not None else {}
            )  # Start with original vars
            page_vars[limit_key] = items_per_page  # Set correct page size
            page_vars[after_variable_name] = current_cursor  # Set cursor

            try:
                # Parse and execute for the current page
                parsed_generated = gql(generated_paginated_query_string)
                logging.info(
                    f"Executing generated query for page {page_num} with vars: {page_vars}"
                )
                result_page = api.client.execute(
                    parsed_generated, variable_values=page_vars
                )

                if "errors" in result_page:
                    logger.error(
                        f"GraphQL errors on page {page_num}: {result_page['errors']}. Stopping pagination."
                    )
                    current_has_next = False
                    final_page_info = {
                        **final_page_info,
                        "hasNextPage": False,
                    }  # Update page info on error
                    continue  # Go to end of loop

                runs_data = get_nested_value(result_page, path_to_paginate)
                if runs_data is None:
                    logging.warning(
                        f"Could not get data for path {'/'.join(path_to_paginate)} on page {page_num}. Stopping."
                    )
                    current_has_next = False
                    continue
                else:
                    edges_this_page = get_nested_value(runs_data, ["edges"]) or []
                    page_info = get_nested_value(runs_data, ["pageInfo"]) or {}
                    final_page_info = page_info  # Store latest page info

                logging.info(
                    f"Result (Page {page_num}): {len(edges_this_page)} runs returned."
                )
                logging.info(f"Page Info (Page {page_num}): {page_info}")

                # Deduplicate & Find edges to append
                new_edges_for_aggregation = []
                duplicates_skipped = 0
                if edges_this_page:
                    for edge in edges_this_page:
                        if (
                            current_edge_count + len(new_edges_for_aggregation)
                            >= max_items
                        ):
                            logging.info(
                                f"Max items ({max_items}) reached mid-page {page_num}."
                            )
                            final_page_info = {**final_page_info, "hasNextPage": False}
                            current_has_next = False
                            break

                        try:
                            node_id = edge["node"]["id"]
                            if node_id not in seen_ids:
                                seen_ids.add(node_id)
                                new_edges_for_aggregation.append(edge)
                            else:
                                duplicates_skipped += 1
                        except (KeyError, TypeError):
                            new_edges_for_aggregation.append(edge)

                    if duplicates_skipped > 0:
                        logging.info(
                            f"Skipped {duplicates_skipped} duplicate edges on page {page_num}."
                        )

                    # Append new unique edges IN-PLACE
                    if new_edges_for_aggregation:
                        target_collection_dict_inplace = get_nested_value(
                            result_dict, path_to_paginate
                        )
                        if target_collection_dict_inplace and isinstance(
                            target_collection_dict_inplace.get("edges"), list
                        ):
                            target_collection_dict_inplace["edges"].extend(
                                new_edges_for_aggregation
                            )
                            current_edge_count = len(
                                target_collection_dict_inplace["edges"]
                            )
                            logging.info(
                                f"Appended {len(new_edges_for_aggregation)} new edges. Total unique edges: {current_edge_count}"
                            )
                        else:
                            logging.error(
                                "Could not find target edges list in result_dict to append in-place."
                            )
                            current_has_next = False
                    else:
                        if len(edges_this_page) > 0:
                            logging.info(
                                "No new unique edges found on page {page_num} after deduplication."
                            )
                        else:
                            logging.info(
                                "No edges returned on page {page_num} to aggregate."
                            )
                else:
                    logging.info("No edges returned on page {page_num} to aggregate.")

                # Update cursor and has_next for next loop iteration (or final state)
                current_cursor = final_page_info.get("endCursor")
                # Respect hasNextPage from API unless loop was broken early by max_items or errors
                if current_has_next:  # Only update if loop didn't break mid-page
                    current_has_next = final_page_info.get("hasNextPage", False)

                # Safety checks
                if current_has_next and not current_cursor:
                    logging.warning(
                        "hasNextPage is true but no endCursor received. Stopping loop."
                    )
                    current_has_next = False
                if not edges_this_page:
                    logging.warning(
                        f"No edges received for page {page_num}. Stopping loop."
                    )
                    current_has_next = False

            except Exception as e:
                logging.error(
                    f"Execution failed for page {page_num}: {e}", exc_info=True
                )
                current_has_next = False  # Stop loop on error

        logging.info(f"\n--- Pagination Loop Finished after page {page_num} ---")
        logging.info(f"Final aggregated edge count: {current_edge_count}")

        # Update the final pageInfo in the result dictionary
        target_collection_dict_final = get_nested_value(result_dict, path_to_paginate)
        if target_collection_dict_final:
            target_collection_dict_final["pageInfo"] = final_page_info
            logging.info(f"Updated final pageInfo: {final_page_info}")

        return result_dict  # Return the modified dictionary

    except Exception as e:
        error_message = f"Critical error in paginated GraphQL query function: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        # Return original dict if possible, else error structure
        if result_dict:
            if "errors" not in result_dict:
                result_dict["errors"] = []
            result_dict["errors"].append(
                {"message": "Pagination failed", "details": str(e)}
            )
            return result_dict
        else:
            return {
                "errors": [
                    {"message": "Pagination failed catastrophically", "details": str(e)}
                ]
            }


class AddPaginationArgsVisitor(gql_visitor.Visitor):
    """Adds first/after args and variables"""

    def __init__(
        self, field_paths, first_variable_name="limit", after_variable_name="after"
    ):
        super().__init__()
        self.field_paths = set(tuple(p) for p in field_paths)
        self.first_variable_name = first_variable_name
        self.after_variable_name = after_variable_name
        self.current_path = []
        self.modified_operation = False

    def enter_field(self, node, key, parent, path, ancestors):
        field_name = node.alias.value if node.alias else node.name.value
        self.current_path.append(field_name)
        current_path_tuple = tuple(self.current_path)
        if current_path_tuple in self.field_paths:
            existing_args = list(node.arguments)
            args_changed = False
            has_first = any(arg.name.value == "first" for arg in existing_args)
            if not has_first:
                # Defaulting variable name to 'limit' if not found, might need refinement
                limit_var_node = gql_ast.VariableNode(
                    name=gql_ast.NameNode(value=self.first_variable_name)
                )
                existing_args.append(
                    gql_ast.ArgumentNode(
                        name=gql_ast.NameNode(value="first"), value=limit_var_node
                    )
                )
                args_changed = True
            has_after = any(arg.name.value == "after" for arg in existing_args)
            if not has_after:
                existing_args.append(
                    gql_ast.ArgumentNode(
                        name=gql_ast.NameNode(value="after"),
                        value=gql_ast.VariableNode(
                            name=gql_ast.NameNode(value=self.after_variable_name)
                        ),
                    )
                )
                args_changed = True
            if args_changed:
                node.arguments = tuple(existing_args)

    def leave_field(self, node, key, parent, path, ancestors):
        if self.current_path:
            self.current_path.pop()

    def enter_operation_definition(self, node, key, parent, path, ancestors):
        if self.modified_operation:
            return
        existing_vars = {var.variable.name.value for var in node.variable_definitions}
        new_defs_list = list(node.variable_definitions)
        defs_changed = False
        # Determine limit variable name from existing vars if possible, else default
        current_limit_var = self.first_variable_name  # Default
        for var_name in existing_vars:
            if var_name.lower() in ["limit", "first", "count"]:
                current_limit_var = var_name
                break

        if current_limit_var not in existing_vars:
            new_defs_list.append(
                gql_ast.VariableDefinitionNode(
                    variable=gql_ast.VariableNode(
                        name=gql_ast.NameNode(value=current_limit_var)
                    ),
                    type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="Int")),
                )
            )
            defs_changed = True
        if self.after_variable_name not in existing_vars:
            new_defs_list.append(
                gql_ast.VariableDefinitionNode(
                    variable=gql_ast.VariableNode(
                        name=gql_ast.NameNode(value=self.after_variable_name)
                    ),
                    type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="String")),
                )
            )
            defs_changed = True
        if defs_changed:
            node.variable_definitions = tuple(new_defs_list)
        self.modified_operation = True
