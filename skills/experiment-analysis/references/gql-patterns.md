# GQL Query Patterns for Experiment Analysis

Tested patterns for common experiment analysis queries via `query_wandb_tool`.

## Compare Top Runs by Metric

```graphql
query TopRuns($entity: String!, $project: String!, $limit: Int!) {
  project(name: $project, entityName: $entity) {
    runs(first: $limit, order: "-summary_metrics.eval_loss") {
      edges {
        node {
          name
          displayName
          state
          summaryMetrics
          config
          createdAt
        }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
```

Variables: `{"entity": "my-team", "project": "my-project", "limit": 10}`

## Filter Runs by State and Date

```graphql
query FilteredRuns($entity: String!, $project: String!, $limit: Int!, $filters: JSONString!) {
  project(name: $project, entityName: $entity) {
    runs(first: $limit, filters: $filters, order: "-createdAt") {
      edges {
        node { name displayName state summaryMetrics config createdAt }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
```

Variables:
```json
{
  "entity": "my-team",
  "project": "my-project",
  "limit": 20,
  "filters": "{\"state\": \"finished\", \"createdAt\": {\"$gt\": \"2026-01-01\"}}"
}
```

## Get Run Count

```graphql
query RunCount($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runCount
  }
}
```

## Get History Keys (to discover available metrics)

```graphql
query HistoryKeys($entity: String!, $project: String!, $runName: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $runName) {
      historyKeys
      historyLineCount
    }
  }
}
```

## Common `order` Values

- `-summary_metrics.eval_loss` (best eval loss first)
- `-summary_metrics.accuracy` (highest accuracy first)
- `-createdAt` (newest first)
- `+createdAt` (oldest first)
- `-summary_metrics._step` (most training steps first)

## Common `filters` Patterns

- `{"state": "finished"}` -- only completed runs
- `{"tags": {"$in": ["baseline"]}}` -- runs with specific tag
- `{"config.learning_rate": {"$gt": 0.001}}` -- filter by config value
- `{"createdAt": {"$gt": "2026-02-01"}}` -- runs after date
