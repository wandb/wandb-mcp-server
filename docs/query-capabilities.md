# W&B Query Capability Matrix

W&B MCP v0.4 does not expose caller-supplied GraphQL by default. Start with the
typed read tools below. An administrator can enable
`WANDB_MCP_ENABLE_RAW_GRAPHQL=true` only for read shapes the typed tools cannot
represent.

## v0.3 GraphQL example migration

Every named GraphQL example previously documented on `query_wandb_tool` has a
typed v0.4 route. These calls no longer require callers to construct a GraphQL
document or JSON-encode filter variables.

Collection `next_cursor` values are opaque and bound to the original resource,
entity/project, filters, ordering, selector, and field projection. Clients may
change only the page `limit` when continuing; mismatched reuse returns
`invalid_cursor` without contacting W&B.

| Former example | v0.4 route | Structured request |
|---|---|---|
| `MinimalRunIdVsDisplayName` | `query_wandb_tool` | Use `resource="run", run_id=...` for a short ID, or `resource="runs", filters={"displayName": {"$eq": ...}}` for a display name |
| `GetProjectInfo` | `query_wandb_tool` | `resource="project"` returns stable project metadata including `description` and `run_count` |
| `GetSortedRuns` | `query_wandb_tool` | `resource="runs", order=...` with optional `summary_keys` |
| `GetFilteredRuns` | `query_wandb_tool` | `resource="runs", filters=..., order=..., summary_keys=[...], cursor=...` |
| `GetRunByDisplayName` | `query_wandb_tool` | `resource="runs", filters={"displayName": {"$eq": ...}}, summary_keys=[...]` |

| Read requirement | Preferred MCP tool | Raw GraphQL needed? |
|---|---|---|
| Entity and project discovery | `list_entities_tool`, `query_wandb_entity_projects` | No |
| Project metadata | `query_wandb_tool(resource="project")` | No |
| Run by short ID | `query_wandb_tool(resource="run", run_id=...)` | No |
| Run by display name | `query_wandb_tool(resource="runs", filters={"displayName": ...})` | No |
| Filtered/sorted runs | `query_wandb_tool(resource="runs", filters=..., order=...)` | No |
| Selected or bounded full summary/config | `query_wandb_tool` with `summary_keys`, `config_keys`, or `include` | No |
| Run count | `query_wandb_tool(resource="runs", response_mode="count")` | No |
| Sweep lookup/list/config | `query_wandb_tool(resource="sweep"|"sweeps")` | No |
| Report lookup/list/spec | `query_wandb_tool(resource="reports", report_name=...)`; `report_name` accepts an exact internal name or display title | No |
| Run metric history, including sparse keys logged at different cadences | `get_run_history_tool` | No |
| Artifacts and registries | `list_artifact_versions_tool`, `get_artifact_details_tool`, `list_registries_tool`, and `list_registry_collections_tool` | No |
| Automations and integrations | `list_wandb_automations_tool` and `list_wandb_integrations_tool` | No |
| Schema introspection or unmodeled/custom fields | `query_wandb_graphql_tool` | Yes |
| Aliases or exact GraphQL response shape | `query_wandb_graphql_tool` | Yes |
| Cross-resource/compound nesting | `query_wandb_graphql_tool` | Yes |
| Sweep agents, report run sets, or Launch resources | `query_wandb_graphql_tool` | Yes |
| Backward pagination | Use a typed forward read when possible | Compatibility-only; raw tool returns one bounded `last` page |

## Run history guarantees

For explicit multi-key default-history sampled and ranged collection reads,
`get_run_history_tool` handles requested metrics as an outer union. It obtains
one bounded series per key with a fixed, application-owned, query-only
projection, batches at most eight series per upstream request, then merges the
series by `_step` or the selected x-axis. This avoids sending the combined-key
spec that triggers Server 0.82's within-spec all-keys filter, without one
network roundtrip per metric. Reads spanning multiple batches use one freshly
observed upper-step boundary. Metrics do not need to occur in the same
`wandb.log()` call or at
the same cadence. Duplicate axis values retain their per-series occurrence
order.

A custom x-axis remains in each metric's independent series specification. The
axis and metric must therefore occur on the same history row; the server does
not fabricate an alignment or interpolation policy. Exact `target_x` requests
remain point lookups and do not use collection outer-union semantics.

The `samples` argument is one total output-row budget across the merged result,
not a separate budget for every key. Range reads discard rows containing none
of the requested values before deterministic, key-aware sampling. All observed
sparse points in the bounded result are preserved when the budget permits; when
it does not, the response identifies affected keys rather than silently
presenting partial data as complete.

The existing response fields remain compatible. Additive diagnostics include
`requested_keys`, `join="outer"`, `matching_rows`, `matching_rows_exact`,
`key_row_counts`, `unobserved_keys`, `missing_keys`,
`keys_omitted_by_limits`, `key_counts_exact`, and `source_truncated`.
`unobserved_keys` means no usable finite/non-null value appeared for a key in the
bounded/sample result; `missing_keys` is emitted only when the count is exact.
The fixed projection is
revalidated as query-only and accepts no caller-selected GraphQL, so this
behavior is available with `WANDB_MCP_ENABLE_RAW_GRAPHQL=false`.

The compatibility tool accepts exactly one query operation and rejects
mutations, subscriptions, mixed/multiple operations, nested or multiple
paginated connections, and oversized documents. Its item, page, complexity, and
response limits apply in every deployment.
