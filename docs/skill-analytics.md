# Skill-Aware Analytics: Tracking Skill-Triggered Tool Usage

**Status:** Proposal for alignment meeting
**Depends on:** PR #19 (analytics schema), PR #25 (session management)

---

## Current analytics schema (PR #19)

Every MCP tool call emits a `tool_call` event:

```json
{
  "schema_version": "1.0",
  "event_type": "tool_call",
  "timestamp": "2026-03-03T18:30:00+00:00",
  "session_id": "sess_abc123",
  "user_id": "alice",
  "email_domain": "wandb.com",
  "tool_name": "query_weave_traces_tool",
  "params": { "entity_name": "wandb", "project_name": "weave-improver1", ... },
  "success": true,
  "duration_ms": 1420.5
}
```

Every session emits a `user_session` event on first request, and every HTTP round-trip emits a `request` event. Session IDs (PR #25) are server-issued UUIDs, so we can group tool calls per real client session.

**What's missing:** we don't know which skill (if any) triggered the tool call.

---

## The attribution problem

Claude Code skills work by injecting instructions into the agent's context window. When a skill tells Claude to call `query_weave_traces_tool`, the MCP server receives a normal tool call -- identical to an ad-hoc call the user prompted directly. The server has no way to distinguish them at the protocol level.

MCP has no standard header or parameter for "which skill triggered this call."

---

## Three approaches to skill attribution

### Approach 1: Pattern-based inference (no protocol changes)

Skills teach specific workflows. The trace-analyst skill teaches: count -> metadata query -> targeted query with specific columns. The experiment-analysis skill teaches: GQL for run comparison -> report creation.

Within a session, the tool call sequence is a fingerprint:

```
Session A (trace-analyst pattern):
  1. count_weave_traces_tool
  2. query_weave_traces_tool (metadata_only=true)
  3. query_weave_traces_tool (columns=["op_name","status","latency"])

Session B (experiment-analysis pattern):
  1. query_wandb_tool (GQL: runs query)
  2. query_wandb_tool (GQL: run history)
  3. create_wandb_report_tool

Session C (ad-hoc, no skill):
  1. query_weave_traces_tool (random params)
```

A BigQuery query over session tool sequences can classify sessions by skill pattern. This is lossy but requires zero protocol changes and works retroactively on existing data.

**Implementation:** BigQuery SQL + Hex dashboard. No server changes needed.

### Approach 2: Client context header (opt-in, non-breaking)

Add a custom header that MCP clients can optionally send:

```
X-MCP-Client-Context: skill=trace-analyst;client=cursor;version=1.2.3
```

The auth middleware parses this and includes it in analytics events:

```json
{
  "event_type": "tool_call",
  "tool_name": "query_weave_traces_tool",
  "client_context": {
    "skill": "trace-analyst",
    "client": "cursor",
    "version": "1.2.3"
  }
}
```

This requires agent/client cooperation -- the skill or client would need to set the header. Claude Code and Cursor don't do this natively today, but a skill could instruct the agent to include it (MCP allows arbitrary request headers in some transports).

**Implementation:** ~20 lines in `auth.py` to read the header, one new field in the analytics schema. Non-breaking: missing header means `client_context: null`.

### Approach 3: WBAF cross-reference (ground truth for benchmarks)

When WBAF runs a task with `codex-mcp` and a skill loaded, the Weave trace captures:
- Which skill was injected (in the agent's system prompt)
- Which MCP tool calls were made (in the agent's trajectory)
- Whether the task passed (scorer results)

Meanwhile, the MCP server logs the same tool calls to Cloud Logging -> BigQuery with `session_id` and timestamps.

Cross-joining WBAF Weave traces with MCP analytics by timestamp window gives us ground-truth skill attribution for benchmarked runs.

```
WBAF Weave trace                    MCP analytics (BigQuery)
┌───────────────────┐               ┌───────────────────┐
│ skill: trace-analyst               │ session: sess_xyz  │
│ task: count-evals  │               │ tool: count_weave  │
│ passed: true       │───timestamp──>│ tool: query_weave  │
│ agent: codex-mcp   │   window      │ tool: query_weave  │
└───────────────────┘               └───────────────────┘
```

**Implementation:** BigQuery join query. Requires WBAF eval runs against the live MCP server (not mocked).

---

## Recommended rollout

| Phase | Approach | Effort | Value |
|-------|----------|--------|-------|
| Now | Pattern-based inference | BigQuery SQL only | Retroactive, approximate |
| PR #19 merge | Client context header | ~20 lines server code | Explicit attribution when clients cooperate |
| WBAF integration | Cross-reference | BigQuery join | Ground-truth for benchmarked runs |

All three approaches are additive and non-exclusive. Start with pattern inference (zero code), add the header when PR #19 merges, and get ground truth once WBAF runs `codex-mcp` evals against the live server.

---

## Dashboard proposal

A Hex dashboard with three views:

1. **Tool usage by session** -- tool call sequences per session, clustered by inferred skill pattern. Shows which workflows users actually follow.
2. **Skill effectiveness** -- for WBAF eval runs: task pass rate by skill + agent backend. SDK vs MCP comparison.
3. **Tool health** -- per-tool success rate, latency P50/P95, error distribution. Alerts when a tool degrades.

Data sources: Cloud Logging -> BigQuery (MCP analytics), Weave API (WBAF eval results).
