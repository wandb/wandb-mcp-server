# MCP Server Analytics Contract

**Schema version:** `1.0`
**Status:** Active (hackathon, pre-DS-review)
**Pipeline:** Cloud Run structured logs -> Cloud Logging -> BigQuery -> Hex

## Event Types

### `user_session`

Emitted once per authenticated request in the auth middleware (heartbeat semantics -- may fire multiple times per logical session).

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Always `"1.0"` |
| `event_type` | string | yes | Always `"user_session"` |
| `timestamp` | string | yes | ISO-8601, UTC (`+00:00`) |
| `session_id` | string | yes | MCP session header or SHA-256 prefix of API key |
| `user_id` | string | no | Best-effort: username > entity > email |
| `email_domain` | string | no | Domain part only (e.g. `wandb.com`) |
| `api_key_hash` | string | no | First 16 hex chars of SHA-256 hash |
| `metadata` | object | no | Arbitrary key/value pairs |

### `tool_call`

Emitted for every MCP tool invocation via `log_tool_call()`.

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Always `"1.0"` |
| `event_type` | string | yes | Always `"tool_call"` |
| `timestamp` | string | yes | ISO-8601, UTC |
| `session_id` | string | no | Session context if available |
| `user_id` | string | no | Best-effort identity |
| `email_domain` | string | no | Domain part only |
| `tool_name` | string | yes | Canonical tool function name |
| `params` | object | no | Sanitised invocation parameters |
| `success` | boolean | yes | Whether the call succeeded |
| `error` | string | no | Error message if `success=false` |
| `duration_ms` | number | no | Wall-clock duration in milliseconds |

### `request`

Emitted for every authenticated HTTP request to `/mcp/*`.

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Always `"1.0"` |
| `event_type` | string | yes | Always `"request"` |
| `timestamp` | string | yes | ISO-8601, UTC |
| `request_id` | string | yes | Short UUID for correlation |
| `session_id` | string | no | Session context if available |
| `user_id` | string | no | Username if viewer was resolved |
| `email_domain` | string | no | Domain part only |
| `method` | string | yes | HTTP method (`GET`, `POST`, ...) |
| `path` | string | yes | Request path (e.g. `/mcp/sse`) |
| `status_code` | integer | yes | HTTP response status code |
| `duration_ms` | number | no | Round-trip time in milliseconds |

## Security & Privacy

- **No full email addresses** -- only domain portion is logged.
- **API key hash prefix** -- 16 hex chars, irreversible.
- **Parameter sanitisation:**
  - Keys matching `api_key`, `token`, `secret`, `password`, `credential`, `auth` are replaced with `<redacted>`.
  - String values longer than 200 characters are replaced with `<truncated:N chars>`.
  - Nested dicts are sanitised recursively up to 3 levels deep; deeper levels pass through unsanitised.
- **Opt-out:** Set `MCP_ANALYTICS_DISABLED=true` to disable all tracking.

## Known Limitations (v1.0)

1. `user_session` events fire on every request, not just session start. Deduplication should happen at the query layer (BigQuery / Hex).
2. `session_id` falls back to an API-key-derived hash when no MCP session header is present, which may conflate sessions from the same user.
3. `track_request` does not yet capture non-`/mcp` paths.
4. `duration_ms` on tool calls is not populated by default -- callers would need to measure and pass it explicitly.

## Sample Payloads

### tool_call

```json
{
  "schema_version": "1.0",
  "event_type": "tool_call",
  "timestamp": "2026-02-27T18:30:00+00:00",
  "session_id": "abc123def456",
  "user_id": "alice",
  "email_domain": "wandb.com",
  "tool_name": "query_wandb_gql",
  "params": {
    "entity_name": "my-team",
    "project_name": "my-project",
    "query_string": "<truncated:1200 chars>",
    "api_key": "<redacted>"
  },
  "success": true,
  "error": null,
  "duration_ms": null
}
```

### user_session

```json
{
  "schema_version": "1.0",
  "event_type": "user_session",
  "timestamp": "2026-02-27T18:29:55+00:00",
  "session_id": "abc123def456",
  "user_id": "alice",
  "email_domain": "wandb.com",
  "api_key_hash": "a1b2c3d4e5f67890",
  "metadata": {}
}
```

### request

```json
{
  "schema_version": "1.0",
  "event_type": "request",
  "timestamp": "2026-02-27T18:30:01+00:00",
  "request_id": "f47ac10b",
  "session_id": "abc123def456",
  "user_id": "alice",
  "email_domain": null,
  "method": "POST",
  "path": "/mcp/sse",
  "status_code": 200,
  "duration_ms": 142.35
}
```
