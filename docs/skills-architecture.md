# Skills Architecture: How the Three Repos Relate

**Status:** Proposal for alignment meeting
**Audience:** MCP + Skills stakeholders (Zubin, Nico, Jason, Anish, Chander, Kimberly, Dave)

---

## The actual flow today

```
WandBAgentFactory (WBAF)               agent-skills (public)          wandb-mcp-server
┌──────────────────────┐               ┌──────────────────┐           ┌──────────────────┐
│ skills/              │  push-skills  │ skills/           │           │ MCP tools        │
│   wandb-data-analysis├──────────────>│   wandb/          │           │   query_weave_*  │
│   weave              │  .sh + PR     │   (more pending)  │           │   query_wandb_*  │
│   wandb-weave        │               │                   │           │   create_report   │
│   error-analysis     │               │                   │           │   count_traces    │
│                      │<──────────────┤ CI: eval-skills   │           │   list_entities   │
│ tasks/ (30+)         │  copies back  │ .yml copies into  │           │   query_wandbot   │
│ scorers/             │  for eval     │ WBAF and runs     │           │                   │
│ agents/              │               │ benchmarks        │           │                   │
│   codex/             │               │                   │           │                   │
│   codex-mcp/ ────────┼───────────────┼───────────────────┼──────────>│ /mcp endpoint     │
│                      │               │                   │           │                   │
└──────────────────────┘               └──────────────────┘           └──────────────────┘
```

- **WBAF** is where skills are authored, iterated, and benchmarked. It has 30+ tasks, multiple scorers, and agent backends (improver, codex, codex-mcp).
- **agent-skills** is the public distribution repo. `push-skills.sh` in WBAF publishes skills there via PR. Agent-skills CI evaluates PRs by running them back through WBAF.
- **wandb-mcp-server** exposes MCP tools. WBAF's `codex-mcp` agent config already points to `https://mcp.withwandb.com/mcp`.

The `registry/skills-publish.yaml` in WBAF controls what gets published:

```yaml
target_repo: wandb/agent-skills
skills:
  - local_name: wandb-data-analysis
    publish_as: wandb
  - local_name: weave
  - local_name: error-analysis
  - local_name: wandb-weave
```

---

## Two paradigms, not three duplicates

The Slack thread said "three parallel distributions of functionally the exact-same-skill(s)." That's not quite right. There are two paradigms that cover overlapping domains:

### SDK skills (WBAF + agent-skills)

The agent writes and executes Python in a sandbox. Skills teach `wandb.Api()` and `weave` SDK patterns.

```python
# What the agent does when following an SDK skill
api = wandb.Api()
runs = api.runs("entity/project", order="-created_at")
for run in runs[:20]:
    print(run.name, run.summary.get("loss"))
```

**Runtime:** Codex, Claude Code (with code execution). Agent needs `WANDB_API_KEY` in its environment.

### MCP skills (wandb-mcp-server)

The agent calls MCP tools via the tool-calling protocol. Skills teach tool-calling patterns.

```
# What the agent does when following an MCP skill
1. Call count_weave_traces_tool(entity, project)
2. Call query_weave_traces_tool(entity, project, metadata_only=True)
3. Call query_weave_traces_tool(entity, project, columns=[...], filters={...})
```

**Runtime:** Cursor, Claude Desktop, any MCP-connected IDE. Agent connects to the MCP server, no sandbox needed.

These are not the same thing. A Codex agent in a Modal sandbox benefits from SDK skills. A Cursor user with the MCP server connected benefits from MCP skills. The workflows overlap (both can "compare experiment runs") but the interface is fundamentally different.

---

## MCP server's role: tool provider, not skill host

The MCP server should not ship skills. It should ship **tools with descriptions good enough that skills (and agents without skills) can use them**.

Skills reference MCP tools by name in their instructions. The server doesn't need to know about skills -- it needs clear tool names and descriptions so that skill-authored workflows work correctly.

PR #20 in wandb-mcp-server prototyped 4 MCP-aware skills (`quickstart`, `experiment-analysis`, `failure-analysis`, `trace-analyst`). These were valuable for exploration but the right home is WBAF, where they can be benchmarked against the same 30+ tasks as SDK skills using the same scoring infrastructure.

---

## Where MCP-aware skills should live

**In WBAF**, as skill variants alongside SDK skills. The `push-skills.sh` pipeline publishes them to agent-skills. Example structure:

```
WBAF/skills/
  wandb-data-analysis/     # SDK skill (wandb.Api patterns)
  weave/                   # SDK skill (weave SDK patterns)
  wandb-mcp/               # MCP skill (tool-calling patterns)
  wandb-mcp-traces/        # MCP skill (trace analysis via MCP tools)
```

Each variant gets benchmarked against the same tasks. If the MCP skill passes `count-evals` at 90% and the SDK skill passes at 85%, we know MCP is competitive. If it's 40% vs 85%, we know MCP tools need work.

---

## Discoverability is description-driven, not tag-driven

Claude Code's skill system ([docs](https://code.claude.com/docs/en/skills)) discovers skills by matching the `description` field against user intent. There is no tag search, no metadata index. The description IS the discoverability mechanism.

This means Dave/Kimberly's review should focus on **crafting description strings** that contain the phrases users actually say:

```yaml
# Good: contains natural user phrases
description: >
  Analyze Weave traces, debug LLM failures, and summarize evaluations.
  Use when investigating traces, understanding eval results, or debugging
  agent behavior in a W&B Weave project.

# Bad: marketing terms that users don't type into an IDE
description: >
  Enterprise-grade ML observability analytics powered by
  Weights & Biases platform integration.
```

The test for a description: would a user's natural-language request match it? "Why is my agent failing?" should match a failure-analysis skill. "Compare my training runs" should match an experiment-analysis skill.
