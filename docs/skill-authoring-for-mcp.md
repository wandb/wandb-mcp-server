# Authoring Skills That Use MCP Tools

**Status:** Proposal for alignment meeting
**Reference:** [Claude Code Skills docs](https://code.claude.com/docs/en/skills), [agentskills.io](https://agentskills.io)

---

## How skill discovery actually works

Claude Code discovers skills by matching the `description` field against the user's request. There is no tag index, no metadata search, no keyword matching beyond description text. The description is the entire discoverability surface.

From the [Claude Code docs](https://code.claude.com/docs/en/skills):

> Skill descriptions are loaded into context so Claude knows what's available, but full skill content only loads when invoked.

This means:
- **The description must contain phrases users actually say.** "Analyze my training runs" beats "ML experiment analytics platform integration."
- **There is no tag system.** Adding `tags: [training, inference, wandb]` to frontmatter does nothing for discovery. The only standard frontmatter fields that affect behavior are `name`, `description`, `disable-model-invocation`, `user-invocable`, `allowed-tools`, `context`, and `agent`.
- **Dave/Kimberly's review** should focus on rewriting descriptions, not building metadata schemas.

---

## MCP skill vs SDK skill: what's different

An SDK skill teaches the agent to write Python:

```markdown
## Query traces
client = weave.init("entity/project")
calls = client.get_calls(filter=CallsFilter(...), limit=50)
```

An MCP skill teaches the agent to call tools:

```markdown
## Query traces
Use `query_weave_traces_tool` with:
- entity_name, project_name (required)
- filters: {"op_name": "predict", "status": "error"}
- metadata_only: true (for initial scan)
- columns: ["op_name", "status", "latency"] (for targeted retrieval)
```

Key difference: MCP skills should NOT import SDKs. If the agent has MCP tools available, the skill should use them. Mixing instructions ("call this tool OR write this code") confuses the agent.

---

## Skill template for MCP tools

### Directory structure

```
wandb-mcp-traces/
├── SKILL.md
└── references/
    └── field-priority.md
```

### SKILL.md

```yaml
---
name: wandb-trace-analyst
description: >
  Analyze Weave traces, debug LLM failures, and summarize evaluations.
  Use when investigating traces, understanding eval results, or
  debugging agent behavior in a W&B Weave project. Also use when
  asked to find errors, count calls, or drill into specific traces.
---

# Trace Analysis via MCP Tools

Analyze Weave traces using the W&B MCP server tools.
Always follow the count -> metadata -> targeted query pattern
to stay within context budget.

## Available tools

| Tool | Use for |
|------|---------|
| `count_weave_traces_tool` | Get total and root trace counts before querying |
| `query_weave_traces_tool` | Query traces with filters, columns, pagination |
| `query_wandb_tool` | GraphQL queries for run/experiment data |
| `create_wandb_report_tool` | Create shareable W&B Reports |
| `query_wandb_entity_projects` | Discover available entities and projects |
| `query_wandb_support_bot` | Ask W&B documentation questions |

## Workflow

### Step 1: Scope the project

Ask for or determine `entity_name` and `project_name`.
If unknown, call `query_wandb_entity_projects` to list available projects.

### Step 2: Count before you query

Always call `count_weave_traces_tool` first:

```
count_weave_traces_tool(entity_name, project_name, filters={})
```

This returns `total_count` and `root_traces_count`. Use these to decide
whether to paginate or apply tighter filters.

### Step 3: Metadata scan

Call with `metadata_only=true` to get aggregate statistics without
pulling full trace data:

```
query_weave_traces_tool(
  entity_name, project_name,
  metadata_only=true,
  limit=1000
)
```

### Step 4: Targeted query

Based on metadata, query specific columns and filters:

```
query_weave_traces_tool(
  entity_name, project_name,
  filters={"status": "error"},
  columns=["op_name", "status", "exception", "started_at"],
  limit=50,
  truncate_length=200
)
```

For field priority guidance, see [references/field-priority.md](references/field-priority.md).

### Step 5: Synthesize

Summarize findings. If the user wants a persistent artifact, create a report:

```
create_wandb_report_tool(
  entity_name, project_name,
  title="Trace Analysis: [topic]",
  markdown_report_text="## Findings\n..."
)
```
```

### references/field-priority.md

Reference files contain detailed information the agent loads only when needed. Keep SKILL.md under 500 lines; move lookup tables here.

```markdown
# Trace Field Priority

When selecting columns for `query_weave_traces_tool`, prioritize by tier:

## High priority (always include)
- op_name, status, started_at, ended_at

## Medium priority (include for analysis)
- latency, tokens (input/output/total), cost
- exception (when investigating errors)

## Low priority (include only when specifically asked)
- inputs, output (large; use truncate_length)
- attributes, summary
```

---

## Writing descriptions for discoverability

The description should answer: "if a user says X, should this skill activate?"

### Good patterns

```yaml
# Lists concrete user intents
description: >
  Analyze Weave traces, debug LLM failures, and summarize evaluations.
  Use when investigating traces, understanding eval results, or
  debugging agent behavior in a W&B Weave project.
```

```yaml
# Covers multiple phrasings of the same intent
description: >
  Compare W&B experiment runs and create reports. Use when asked to
  compare training runs, analyze metrics across experiments, find the
  best run, or create a report from run data.
```

### Bad patterns

```yaml
# Marketing-speak that users never type
description: Enterprise ML observability analytics suite for Weights & Biases.

# Too vague -- matches everything
description: Help with W&B data.

# Too specific -- misses paraphrases
description: Count the number of Evaluation.evaluate calls in a Weave project.
```

### Brand terms

Include "W&B", "Weights & Biases", and "Weave" in descriptions because users do type these. But they should be embedded in natural phrases, not listed as keywords.

---

## Testing MCP skills

1. **Manual test in Claude Code / Cursor**: Install the skill in `~/.claude/skills/`, connect the MCP server, and prompt with a task instruction from WBAF.
2. **WBAF eval**: Add the skill to a `codex-mcp` agent config and run `uv run python -m core.run_eval evals/mcp-vs-sdk.yaml`.
3. **Regression**: The agent-skills CI pipeline runs evals on every PR. Adding `codex-mcp` as a second agent catches MCP-specific regressions.
