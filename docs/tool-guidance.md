# Detailed MCP tool guidance

Tool descriptions returned by MCP discovery are kept below 4,000 characters for
enterprise client compatibility. They retain routing, essential inputs, safety
constraints, and result semantics. This reference preserves the extended guidance
and examples outside the discovery payload; clients may follow the description
links when more detail is needed. No additional tool or resource call is required.

Deployment limits take precedence over example sizes. The tool input schema is
the authoritative source for defaults and accepted types.

## Query Weave traces

Query Weave traces with filtering, sorting, and detail_level control.

For W&B runs/metrics, use query_wandb_tool instead. This tool is for Weave traces (LLM calls, evaluations, agent traces).

### when to use
Call for Weave trace data. Use detail_level="schema" to browse, "summary" for analysis, "full" for specific traces only.


### usage tips
query_weave_traces_tool can return a lot of data, below are some usage tips for this function
in order to avoid overwhelming a LLM's context window with too much data.

### managing llm context window

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

### returning metadata only

If `metadata_only = True` this returns only metadata of the traces such as trace counts, token counts,
trace types, time range, status counts and distribution of op names. if `metadata_only = False` the
trace data is returned either in full or truncated to `truncate_length` characters depending if
`return_full_data = True` or `False` respectively.


### truncating trace data values

If `return_full_data = False` the trace data is truncated to `truncate_length` characters,
default 1000 characters. Otherwise the trace data is returned in full.


Remember, LLM context window is precious, only return the minimum amount of data needed to complete an analysis.


### usage guidance

- Exploratory queries: For generic exploratory or initial queries about a set of weave traces in a project it can
be a good idea to start with just returning metadata or truncated data. Consider asking the
user for clarification and warn them that returning a lot of weave traces data might
overwhelm the LLM context window. No need to warn them multiple times, just once is enough.

- Project size: Consider using the count_weave_traces_tool to get an estimate of the number of traces in a project
before querying for them as query_weave_traces_tool can return a lot of data.

- Partial op name matching: Use the `op_name_contains` filter if a users has only given a partial op name or if they
are unsure of the exact op name.

- Weave Evaluations: If asked about weave evaluations or evals traces:
    - Evals are complicated to query, prompt the user with follow up questions if needed.
    - First, always try and oritent yourself - pull a summary of the evaluation, get all of the top level column names in the eval and always get a count of the total number of child traces in this eval by filtering by parent_ids and using the count_traces tool.
    - As part of orienting yourself, just pull a subset of child traces from the eval, maybe 3 to 5, to understand the column structure and values.
    - Always be explicit about the amount of data returned and limits used in your query - return to the user the count of traces analysed.
    - Always stay filterd on the evaluation id (filter by `parent_ids`) unless specifically asked questions across different evaulations, e.g. if a parent id (or parentId) is provided then ensure to use that filter in the query.
    - filter for traces with `op_name_contains = "Evaluation.evaluate"` as a first step. These ops are parent traces that contain
    aggregated stats and scores about the evaluation. The child traces of these ops are the actual evaluation results
    for each sample in an evaluation dataset. If asked about individual rows in an evaluation then use the parent_ids
    filter to return the child traces.
    - for questions where both a child call name of an evaluation and an evaluation id or name are provided, always ensure that you first correctly get the evaluation id, and then use it as the parent_id in the query for the child traces. Otherwise there is a risk of returning traces that do not belong to the evaluation that was given.

- Weave nomenclature: Note that users might refer to weave ops as "traces" or "calls" or "traces" as "ops".



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
        Return traces that are children of the given parent trace ids (string or list of strings). Ensure you use this if given an evaluation trace id or name.
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
    - inputs : dict, optional
        Filter on trace input fields using dot-path keys. Supports `$contains` for
        substring search and comparison operators.
        Examples:
        *   `"inputs": {"message": {"$contains": "search phrase"}}` -- find traces where input.message contains text
        *   `"inputs": {"model": "gpt-4"}` -- find traces where input.model equals "gpt-4"
        **PERFORMANCE WARNING:** Content search on inputs scans trace data server-side.
        For projects with >10k traces, ALWAYS combine with other filters (time_range,
        op_name, trace_roots_only) to narrow the scan.
    - output : dict, optional
        Filter on trace output fields. Supports the same operators as inputs.
        Examples:
        *   `"output": {"$contains": "error message"}` -- search the full output for a substring
        *   `"output": {"result": {"$contains": "success"}}` -- search output.result for a substring
        **PERFORMANCE WARNING:** Same as inputs -- combine with other filters for large projects.
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
    Whether to include untruncated trace data. If True, `truncate_length` is
    ignored. Defaults to False. With return_full_data=False and truncate_length=0,
    only column keys are returned, without their values.
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

### root span resolution
If you find child traces and need to know which root session they belong to,
use resolve_trace_roots_tool with the trace_ids from the results.
This resolves all roots in a single batched call (O(1), not N separate lookups).

Typical workflow:
1. query_weave_traces_tool(filters={...}) -> find child traces
2. resolve_trace_roots_tool(entity_name, project_name, trace_ids=[...]) -> get root context


Returns
-------
str
    JSON string containing either full trace data or metadata only, depending on parameters.
    The response metadata includes `total_matching_count` -- the total traces matching your
    current filters before the limit is applied. Note: this reflects whatever filters you
    used, not the project-wide total. Use `count_weave_traces_tool` if you need a separate
    unfiltered count.

### examples
    ```python
    # Get an overview of the traces in a project
    query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"trace_roots_only": True},
        metadata_only=True,
        return_full_data=False
    )

    # Get failed traces with costs and feedback
    query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"status": "error"},
        include_costs=True,
        include_feedback=True
    )

    # Get specific columns for traces who's op name (i.e. trace name) contains a specific substring
    query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"op_name_contains": "Evaluation.summarize"},
        columns=["id", "op_name", "started_at", "costs"]
    )

    # Schema-first workflow: browse traces quickly, then drill into specific ones
    # Step 1: Get structural overview
    query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"trace_roots_only": True},
        detail_level="schema"
    )
    # Step 2: Drill into a specific trace with full data
    query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]},
        detail_level="full"
    )

    # Content search + root resolution (two-step workflow):
    # Step 1: Find child traces matching content
    results = query_weave_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"inputs": {"message": {"$contains": "restaurant rush"}}},
        detail_level="schema"
    )
    # Step 2: Resolve root spans for the found traces
    resolve_trace_roots_tool(
        entity_name="my-team",
        project_name="my-project",
        trace_ids=["<unique trace_ids from step 1>"]
    )
    ```


## Create wandb report

Create a new Weights & Biases Report to document analysis and findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases.
Always provide the returned report link to the user.

### when to use
Call this tool AFTER completing analysis to create a shareable report. Combine
markdown text (for narrative, tables, and findings) with optional panels (for
native charts, custom Vega charts, and W&B Table-backed charts) to produce a
polished deliverable. If you have metric data from get_run_history_tool, use
native panels to visualize it in the report. If chart data already exists in a
W&B Table or summary table, use custom_chart_table.


### markdown generation guide
When generating the markdown_report_text parameter, structure your content using:

**Headers**: Organize content hierarchically
- # Main Title (H1)
- ## Section Title (H2)
- ### Subsection Title (H3)

**Paragraphs**: Write clear, informative text separated by blank lines

**Lists**: Present information clearly
- Bullet points: Use - or *
- Numbered lists: Use 1. 2. 3.

**Formatting**:
- **bold** for emphasis
- *italic* for subtle emphasis
- `inline code` for technical terms
- Links: [W&B documentation](https://docs.wandb.ai)

**Code blocks**: For code snippets or technical content
```language
code here
```

**Table of Contents**: Add [TOC] on its own line to auto-generate navigation

**Best Practices**:
- Start with a clear H1 title
- Use [TOC] after the title for easy navigation
- Structure content with logical sections (H2) and subsections (H3)
- Keep paragraphs concise and focused
- Use lists for multiple related items
- Include code blocks for technical examples


Args:
    entity_name: str, The W&B entity (team or username) - required
    project_name: str, The W&B project name - required
    title: str, Title of the W&B Report - required
    description: str, Optional brief description of the report
    markdown_report_text: str, Well-structured markdown content for the report body
    panels: list of dict, optional - Chart panels or ordered layout blocks to add after the markdown content.
        Each dict specifies a chart type and configuration:
        - {"type": "line", "x": "_step", "y": ["loss", "val_loss"], "title": "Training Loss"}
          Creates a LinePlot tracking metrics over steps.
        - {"type": "bar", "metrics": ["accuracy", "f1"], "title": "Metrics"}
          Creates a BarPlot comparing metrics across runs.
        - {"type": "run_comparison", "metrics": ["loss", "accuracy"], "run_ids": ["abc", "def"], "title": "Compare"}
          Creates a PanelGrid comparing specific runs on selected metrics.
        - {"type": "custom_chart", "title": "PR Curve", "query": {"summaryTable": {"tableKey": "pr_curve_table"}},
           "chart_name": "wandb/line/v0", "chart_fields": {"x": "recall", "y": "precision"},
           "chart_strings": {"title": "PR Curve"}, "run_ids": ["abc123"], "hide_run_sets": true}
          Creates a CustomChart from an explicit wandb-workspaces query.
        - {"type": "custom_chart_table", "title": "PR Curve", "table_name": "pr_curve_table",
           "chart_name": "wandb/line/v0", "chart_fields": {"x": "recall", "y": "precision"},
           "chart_strings": {"title": "PR Curve"}, "hide_run_sets": true}
          Creates a CustomChart from a W&B Table or summary table key via CustomChart.from_table().
        - {"type": "heading", "level": 2, "text": "Average precision by class"}
          Adds an H1/H2/H3 report heading at this position.
        - {"type": "markdown", "text": "These charts share the same filtered runset."}
          Adds markdown narrative at this position.
        - {"type": "panel_grid", "run_ids": ["run_a", "run_b"], "hide_run_sets": false,
           "panels": [{...chart panel...}, {...chart panel...}]}
          Creates one PanelGrid whose child panels share a single Runset.
        Use custom_chart_table for summary-table-backed charts. Use custom_chart with an explicit historyTable
        query for PR/ROC curves or other charts logged through run history.
        If panels only contains chart specs, the tool appends them under a Charts heading for backward compatibility.
        If panels contains heading, markdown, or panel_grid blocks, the list is treated as an ordered layout and no
        automatic Charts heading is added. If omitted, report is markdown-only.

### custom chart panel guide
Use native panel types for ordinary run metrics:
- line: metric history over _step
- bar: summary metric comparisons
- scatter: two summary/config fields

Use custom_chart_table when the source data already exists as a W&B Table saved
in run summary. This is the preferred path for final confusion matrices,
per-class AP tables, and other one-snapshot table-backed visualizations:
{
  "type": "custom_chart_table",
  "title": "Precision-Recall Curve",
  "table_name": "pr_curve_table",
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "recall", "y": "precision"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "hide_run_sets": true
}

Use custom_chart only when you know the exact wandb-workspaces query shape:
{
  "type": "custom_chart",
  "query": {"summaryTable": {"tableKey": "pr_curve_table"}},
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "recall", "y": "precision"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "run_ids": ["abc123"],
  "hide_run_sets": true
}

For PR curves, ROC curves, and other charts logged through run history, use an
explicit historyTable query. tableKey is the key passed to run.log(), not a
column name inside the table:
{
  "type": "custom_chart",
  "query": {"historyTable": {"tableKey": "pr_curve"}},
  "chart_name": "wandb/line/v0",
  "chart_fields": {"x": "r", "y": "p", "color": "c"},
  "chart_strings": {"title": "Precision-Recall Curve"},
  "run_ids": ["abc123"],
  "hide_run_sets": true
}

If the chart data is computed inside MCP rather than already stored as a W&B
Table, call log_analysis_to_wandb first, then reference the logged run/table
from this report tool.


### report layout guide
Use panel_grid when multiple panels should share one run selector / Runset. This is the correct structure for
reports such as H2 / markdown / panel-grid / H2 / markdown / panel-grid:
{
  "type": "panel_grid",
  "run_ids": ["run_a", "run_b"],
  "hide_run_sets": false,
  "panels": [
    {"type": "custom_chart", "query": {"summaryTable": {"tableKey": "car_ap"}},
     "chart_name": "cruise/bar_chart/v2", "chart_fields": {"x": "threshold", "y": "ap"},
     "chart_strings": {"title": "CAR AP"}},
    {"type": "custom_chart", "query": {"summaryTable": {"tableKey": "truck_ap"}},
     "chart_name": "cruise/bar_chart/v2", "chart_fields": {"x": "threshold", "y": "ap"},
     "chart_strings": {"title": "TRUCK AP"}}
  ]
}

Runset scoping:
- run_ids means W&B internal run keys (Python SDK run.id), not display names.
- run_ids are converted to deterministic Reports v2 filters, for example name in ["run_a", "run_b"].
- filters may be passed as a Reports v2 expression string and wins over generated run_ids filters.
- runset_query may be passed for explicit search behavior. Do not use custom_chart query for run filtering.
- chart_name maps to the Vega panelDefId such as wandb/line/v0 or cruise/bar_chart/v2. Put visible titles in chart_strings.


### manual validation recipe
To validate a table-backed custom chart manually:
1. Pick a run that has a logged W&B Table key, such as a summary table or PR/ROC history table.
2. Call create_wandb_report_tool with custom_chart_table for summary tables or custom_chart with historyTable for PR/ROC curves.
3. Open the returned report URL and confirm the custom Vega chart renders.
4. If the chart does not render, verify the table_name and chart_fields match the table columns and UI chart config.


Returns:
    The URL to the created report

Example markdown structure:
```markdown
# Analysis Report Title

[TOC]

## Executive Summary
Brief overview of the analysis and key findings.

## Methodology
Description of the approach used in the analysis.

### Data Collection
- Source 1: Description
- Source 2: Description

### Analysis Techniques
Technical details about methods used.

## Results
Key findings from the analysis.

### Performance Metrics
- Accuracy: 95%
- Precision: 92%
- Recall: 89%

## Conclusions
Summary of insights and recommendations.
```

## Query wandb

Query W&B Models data through bounded W&B read APIs.

Choose ONE input form. Prefer structured entity_name + project_name + resource
for new calls. Existing GraphQL clients may instead pass query + optional
variables, max_items (default 100), and items_per_page (default 20). GraphQL
works on hosted, Dedicated, and local servers using the caller's W&B access.
Do not mix the two forms, even when supplying a structured field's default.
The legacy form preserves GraphQL response structure and accepts one bounded
read-only query, including aliases and fragments. Mutations, subscriptions,
multiple operations, and nested/multiple paginated connections are rejected.
Deployment item/page limits, deadlines, and response budgets still apply.

Use this read-only tool for project metadata, individual runs, filtered or sorted
run collections, sweeps, and reports. For run history, artifacts, registries,
automations, and integrations, prefer the dedicated MCP tools.

For an unfamiliar project, call probe_project_tool first. Then pass only the
returned summary_keys/config_keys needed for the question. This avoids loading
every metric from wide runs and usually answers the question in one request.

Prefer the existing specialized tools when they match the request:
- entity/project discovery: list_entities_tool and query_wandb_entity_projects
- time-series metrics: get_run_history_tool
- artifact reads: list_artifact_versions_tool and get_artifact_details_tool
- registry reads: list_registries_tool and list_registry_collections_tool
- automations/integrations: list_wandb_automations_tool and list_wandb_integrations_tool

### when to use
Use this tool for run discovery, summary-metric analysis, project metadata,
sweep inspection, or report discovery. It is the normal W&B Models query path.


Parameters
----------
entity_name : str
    W&B entity or team name. Required for structured calls only.
project_name : str
    W&B project name. Required for structured calls only.
resource : "project" | "run" | "runs" | "sweep" | "sweeps" | "reports"
    Resource to read through the SDK. Required for structured calls only.
run_id : str, optional
    Required only for resource="run". This is the short W&B run ID, not its display name.
sweep_id : str, optional
    Required only for resource="sweep".
report_name : str, optional
    Optional exact report filter for resource="reports". Accepts either the
    internal report name or its user-visible display title.
filters : dict, optional
    W&B SDK Mongo-style run filters for resource="runs". Supported fields include
    createdAt, displayName, duration, group, host, jobType, name, state, tags,
    username, config.*, and summary_metrics.*. Operators include $and, $or, $eq,
    $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, and $regex.
order : str, optional
    Run ordering for resource="runs", such as -created_at or
    -summary_metrics.accuracy. Default: -created_at.
limit : int, optional
    Maximum collection items to return. Default: 50; deployment limits apply.
include : list[str], optional
    Additional resource details. run/runs accept summary, config, system_metrics,
    and sweep; sweep/sweeps accept config; reports accepts spec. Individual runs
    include summary metrics by default; run collections return metadata by default.
summary_keys : list[str], optional
    Specific summary metrics to include for run/runs. Supplying keys implies
    include=["summary"] and uses a server-side field projection.
config_keys : list[str], optional
    Specific config values to include for run/runs. Supplying keys implies
    include=["config"] and uses a server-side field projection.
response_mode : "items" | "count", optional
    "items" returns bounded resources. "count" is supported for resource="runs"
    and returns only the exact server-side matching count.
cursor : str, optional
    Opaque continuation cursor returned by a previous collection response. Reuse
    it only with the same resource, scope, filters, ordering, selector, and field
    projection. The next page may request a different limit.
query : str, optional
    Legacy GraphQL document, instead of all structured fields above.
variables : dict, optional
    GraphQL variables. Requires query; values must be bounded finite JSON.
max_items : int, optional
    Legacy GraphQL total item limit. Default 100, capped by the workload.
items_per_page : int, optional
    Legacy GraphQL page size. Default 20, capped by the workload.

Returns
-------
dict
    Collection results include returned_count, total_count, has_more, limit, and
    project_exhaustive. Single-resource results use item.
    Legacy calls retain the query's field/alias structure and bounded pagination
    metadata. A nonempty GraphQL errors envelope is an MCP tool failure, including
    partial responses; do not treat partial data as a complete result.

For schema introspection, unmodeled fields, aliases, cross-resource nesting, or an
exact GraphQL response shape, a local operator may select the explicit
models-weave-graphql-compat profile and call query_wandb_graphql_tool.

## Get run history

Retrieve bounded time-series metric data from a W&B run.

Use bounded independent-series sampling for explicit multi-key default-history
overviews and ranges, SDK scans for single-key ranges, and an exact point lookup
for a logged x-axis value. Every successful response states the
retrieval method, whether values are sampled or exact, rows scanned, coverage,
and any profile or response-budget truncation.

### when to use
Call this tool when the user asks about training curves, metric trends over time,
loss plots, or any time-series data logged to a W&B run. This is the only tool
that provides step-by-step metric history -- query_wandb_tool returns run-level
summary metrics but not the full training history.

Typical workflow:
1. For an unfamiliar project, use probe_project_tool to discover indexed keys.
2. Use query_wandb_tool with summary_keys/config_keys to select runs.
3. Use get_run_history_tool with explicit keys and x_axis for targeted curves.
4. Use create_wandb_report_tool to visualize the results.


Parameters
----------
entity_name : str
    The W&B entity (team or username).
project_name : str
    The W&B project name.
run_id : str
    The W&B run ID (often an 8-character generated ID such as "gtng2y4l";
    custom run IDs are also accepted). This is not the display name.
keys : list of str, optional
    Specific metric keys to retrieve (e.g., ["loss", "val_loss", "accuracy"]).
    For explicit multi-key default-history sampled and ranged collection reads,
    keys logged on different cadences are outer-unioned by step; a row may carry
    only the requested metrics logged at that point. target_x is an exact point
    lookup rather than a collection join.
    Shared hosted deployments require explicit keys up to the configured cap
    (20 by default; the Dedicated profile defaults to 50).
samples : int, optional
    Total merged-row budget shared across all requested keys. Defaults to 500.
    Use fewer samples for quick overviews, more for detailed analysis.
min_step : int, optional
    Inclusive non-negative minimum step to include. Defaults to None. Step
    ranges are supported only for the default history stream.
max_step : int, optional
    Inclusive non-negative maximum step to include. Defaults to None. Step
    ranges are supported only for the default history stream.
x_axis : str, optional
    History x-axis. Defaults to "_step". Set this to a logged monotonic metric
    such as "validation/step" for custom-axis projection or target lookup. For
    collection reads, the custom axis and metric must occur on the same row.
target_x : float, optional
    Retrieve the row where x_axis logged this exact value. If no exact value was
    logged, returns target_not_logged. Supported only for the default stream.
tolerance : float, optional
    When target_x was not logged exactly, permit a bounded nearest-value
    refinement within this absolute tolerance.
stream : "default" or "system", optional
    Select normal run history or system metrics. Defaults to "default".

Returns
-------
JSON with:
  - rows: list of {_step, key1, key2, ...} dicts
  - run_id: the queried run ID
  - run_name: the run's display name
  - total_steps: last logged step number
  - sampled_points: number of rows returned
  - keys_returned: list of metric keys in the response
  - requested_keys, optional join, matching_rows, per-key row counts, and
    unobserved/missing/omitted keys. Unobserved means no usable finite/non-null
    value appeared in a bounded result; missing is emitted only for exact counts.
  - non_finite_counts for NaN/Infinity observations seen in the bounded source;
    invalid JSON numeric values are counted rather than returned, and counts are
    exhaustive only when key_counts_exact is true.
  - retrieval_method, exact, sampled, rows_scanned, coverage, truncation,
    source_truncated (source step-window/row cap reached), and
    source_values_truncated (source rows containing an oversized value replaced
    by a bounded sentinel)

Examples
--------
>>> get_run_history_tool("my-team", "my-project", "gtng2y4l", keys=["loss", "val_loss"])
>>> get_run_history_tool(
...     "my-team", "my-project", "h0fm5qp5",
...     keys=["validation/loss"], x_axis="validation/step", target_x=1000,
... )
