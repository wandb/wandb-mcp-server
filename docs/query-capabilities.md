# W&B Query Capability Matrix

W&B MCP v0.4 does not expose caller-supplied GraphQL by default. Start with the
typed read tools below. An administrator can enable
`WANDB_MCP_ENABLE_RAW_GRAPHQL=true` only for read shapes the typed tools cannot
represent.

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
| Report lookup/list/spec | `query_wandb_tool(resource="reports")` | No |
| Run metric history | `get_run_history_tool` | No |
| Artifacts and registries | Artifact and registry tools | No |
| Automations and integrations | Automation and integration tools | No |
| Schema introspection or unmodeled/custom fields | `query_wandb_graphql_tool` | Yes |
| Aliases or exact GraphQL response shape | `query_wandb_graphql_tool` | Yes |
| Cross-resource/compound nesting | `query_wandb_graphql_tool` | Yes |
| Sweep agents, report run sets, or Launch resources | `query_wandb_graphql_tool` | Yes |
| Backward pagination | Use a typed forward read when possible | Compatibility-only; raw tool returns one bounded `last` page |

The compatibility tool accepts exactly one query operation and rejects
mutations, subscriptions, mixed/multiple operations, nested or multiple
paginated connections, and oversized documents. Its item, page, complexity, and
response limits apply in every deployment.
