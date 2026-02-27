# MCP Server -> Gorilla/Segment Integration Spec

**Status:** RFC -- requires DS + platform team sign-off
**Depends on:** PR #19 (`hackathon/analytics`) for base event schema

## Context

The MCP hosted service currently emits structured analytics via Cloud Logging -> BigQuery -> Hex. The broader W&B analytics standard is Segment, forwarded through Gorilla's `/analytics/t` endpoint. This document specifies how to bridge the two.

### Gorilla `/analytics/t` contract

The handler in [`core/services/gorilla/api/handler/analytics.go`](https://github.com/wandb/core/blob/master/services/gorilla/api/handler/analytics.go#L144) expects a `segmentio/analytics-go/v3.Track` JSON body:

```json
{
  "userId": "<required -- event is dropped if empty>",
  "event": "<event name string>",
  "properties": { "<arbitrary key/value>" },
  "timestamp": "<optional ISO-8601>"
}
```

Events with empty `userId` are silently skipped. The handler forwards to the same `AnalyticsSink` (Segment) used by all other W&B telemetry.

## Proposed Event Taxonomy

| Internal event_type | Segment event name | Description |
|---|---|---|
| `user_session` | `mcp_server.session_start` | User authenticated and started an MCP session |
| `tool_call` | `mcp_server.tool_call` | An MCP tool was invoked |
| `request` | `mcp_server.http_request` | An HTTP request was served |

### Naming conventions

- Prefix: `mcp_server.` to namespace within the broader Segment event stream.
- Snake_case, dot-separated hierarchy, matching existing W&B patterns.

## Field Mapping

### `mcp_server.tool_call`

| Segment field | Source | Notes |
|---|---|---|
| `userId` | `user_id` (username > entity > email) | **Required by Gorilla** -- events without identity are dropped |
| `event` | `"mcp_server.tool_call"` | Static |
| `timestamp` | `timestamp` | ISO-8601 UTC |
| `properties.schema_version` | `schema_version` | `"1.0"` |
| `properties.source` | `"wandb-mcp-server"` | Static identifier |
| `properties.session_id` | `session_id` | MCP session or key-derived hash |
| `properties.tool_name` | `tool_name` | Canonical function name |
| `properties.params` | `params` (sanitised) | Redacted/truncated |
| `properties.success` | `success` | Boolean |
| `properties.error` | `error` | Error message if failed |
| `properties.duration_ms` | `duration_ms` | Wall-clock ms |

### `mcp_server.session_start`

| Segment field | Source | Notes |
|---|---|---|
| `userId` | `user_id` | Required |
| `event` | `"mcp_server.session_start"` | Static |
| `timestamp` | `timestamp` | ISO-8601 UTC |
| `properties.session_id` | `session_id` | |
| `properties.email_domain` | `email_domain` | Domain only |
| `properties.api_key_hash` | `api_key_hash` | 16 hex chars |
| `properties.metadata` | `metadata` | Arbitrary k/v |

### `mcp_server.http_request`

| Segment field | Source | Notes |
|---|---|---|
| `userId` | `user_id` | Required |
| `event` | `"mcp_server.http_request"` | Static |
| `timestamp` | `timestamp` | ISO-8601 UTC |
| `properties.request_id` | `request_id` | Short UUID |
| `properties.session_id` | `session_id` | |
| `properties.method` | `method` | HTTP method |
| `properties.path` | `path` | Request path |
| `properties.status_code` | `status_code` | Integer |
| `properties.duration_ms` | `duration_ms` | Wall-clock ms |

## Sample Mapped Payloads

### tool_call -> Segment Track

```json
{
  "userId": "alice",
  "event": "mcp_server.tool_call",
  "properties": {
    "schema_version": "1.0",
    "source": "wandb-mcp-server",
    "session_id": "abc123def456",
    "tool_name": "query_wandb_gql",
    "params": {
      "entity_name": "my-team",
      "project_name": "my-project",
      "api_key": "<redacted>"
    },
    "success": true,
    "error": null,
    "duration_ms": null
  },
  "timestamp": "2026-02-27T18:30:00+00:00"
}
```

### session_start -> Segment Track

```json
{
  "userId": "alice",
  "event": "mcp_server.session_start",
  "properties": {
    "schema_version": "1.0",
    "source": "wandb-mcp-server",
    "session_id": "abc123def456",
    "email_domain": "wandb.com",
    "api_key_hash": "a1b2c3d4e5f67890",
    "metadata": {}
  },
  "timestamp": "2026-02-27T18:29:55+00:00"
}
```

### http_request -> Segment Track

```json
{
  "userId": "alice",
  "event": "mcp_server.http_request",
  "properties": {
    "schema_version": "1.0",
    "source": "wandb-mcp-server",
    "request_id": "f47ac10b",
    "session_id": "abc123def456",
    "method": "POST",
    "path": "/mcp/sse",
    "status_code": 200,
    "duration_ms": 142.35
  },
  "timestamp": "2026-02-27T18:30:01+00:00"
}
```

## Identity Assumptions

1. **`userId` is the W&B username** (from viewer object). This requires successful viewer resolution during auth.
2. Events from unauthenticated or partially-authenticated requests (e.g. bad API keys) will be **dropped** by Gorilla because `userId` will be empty.
3. For STDIO transport (local usage without HTTP auth), no Segment events are emitted -- only Cloud Logging events.

## Rollout Plan

### Phase 1: Current (PR #19)
- Cloud Logging -> BigQuery -> Hex pipeline only.
- No Segment forwarding.

### Phase 2: This PR (stacked)
- `analytics_segment.py` provides pure mapping functions + gated forwarder.
- `MCP_SEGMENT_DRY_RUN=true` logs mapped payloads for DS inspection.
- `MCP_SEGMENT_FORWARD=true` enables live POST to `/analytics/t`.
- Both are **off by default**.

### Phase 3: Production enablement (post-DS approval)
- Enable `MCP_SEGMENT_FORWARD=true` in Cloud Run deployment.
- Confirm events appear in Segment debugger.
- Deprecate duplicate BigQuery pipeline if Segment covers all use cases.

## Open Questions for DS Review

1. **Event cardinality**: Should `mcp_server.http_request` be forwarded, or is `tool_call` + `session_start` sufficient for product analytics? HTTP requests are high-volume.
2. **Identity enrichment**: Should we send a `Segment.identify()` call on session start (with email_domain as a trait), or is userId in Track events sufficient?
3. **Param detail level**: The `params` property is sanitised but can still be large. Should we strip it entirely for Segment and keep it only in BigQuery?
4. **Deduplication**: Session events fire per-request (heartbeat). Should we deduplicate before forwarding, or let the warehouse handle it?
5. **Event naming**: Are `mcp_server.*` prefixes consistent with existing Segment event naming at W&B? (Dave/Taylor to confirm.)
