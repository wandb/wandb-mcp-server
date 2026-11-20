# Secret resolution

GCP-backed session signing and Datadog forwarding use Application Default
Credentials. Install `google-cloud-secret-manager` in deployments that select
`MCP_SERVER_SECRETS_PROVIDER=gcp`, and set `MCP_SERVER_SECRETS_PROJECT`.
Use a runtime identity with access to only its required secrets.

| Setting | Default | Purpose |
|---|---|---|
| `MCP_SERVER_HMAC_SECRET_ID` | Existing built-in session secret | Session signing key |
| `MCP_SERVER_HMAC_SECRET_VERSION` | `latest` | Numeric version or `latest` |
| `MCP_DATADOG_SECRET_ID` | Existing built-in Datadog secret | Optional telemetry credential |
| `MCP_DATADOG_SECRET_VERSION` | `latest` | Numeric version or `latest` |

Secret IDs are short Secret Manager IDs, not resource paths. Versions must be
positive integers or `latest`. Empty values are configuration errors. These
settings do not enable HMAC signing or telemetry; their existing enable flags
still control those features. An explicit `DD_API_KEY` retains precedence over
the Datadog secret reference.

Each Secret Manager RPC has a three-second timeout with SDK retries disabled.
Timeouts, temporary unavailability and throttling receive one additional attempt.
Access denials, missing or disabled versions and empty payloads do not retry
within the same fetch. Session initialization shares one attempt per process
and waits five seconds after failure before another caller can retry. Errors
expose fixed categories rather than provider responses or secret references.

A worker caches its successfully loaded session key and Datadog credential for
its lifetime. Pin numeric versions in production, use different keys across
environments, and deploy new workers to rotate credentials. Confirm all serving
workers use the new version before disabling the old version. Retire old tagged
revisions that could still serve requests. An emergency revocation can require
immediate removal of old workers before normal migration finishes.

Changing the signing key invalidates existing sessions. Clients must initialize
new sessions; the server does not verify against an old key as a fallback.
Rollback must retain an approved signing key and may also require reconnecting.
Datadog credential failures disable or fail telemetry forwarding without making
session initialization depend on Datadog.
