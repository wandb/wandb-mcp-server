# Skills Index

This is the main entrypoint for the W&B feature-first skills package.

## Purpose

Use these skills to help any coding agent set up and run a reliable W&B + Weave evaluation loop, then scale to a self-improving agent workflow.

## How To Use

1. Start with this file.
2. Select only the feature skills you need.
3. Combine them with the orchestration skill for a full self-eval loop.

Recommended progression:
1. `wandb-projects`
2. `wandb-runs`
3. `wandb-traces`
4. `wandb-evals`
5. `wandb-reports`
6. `coding-agent-self-eval`

## Install Pattern

Use one skill at a time, for example:

```bash
npx skills add wandb/wandb-mcp-server --skill wandb-traces
npx skills add wandb/wandb-mcp-server --skill wandb-evals
```

Install full loop orchestration:

```bash
npx skills add wandb/wandb-mcp-server --skill coding-agent-self-eval
```

## Available Skills

1. [wandb-projects](./wandb-projects/SKILL.md)  
Resolve and validate W&B entity/project context using MCP before any automation.

2. [wandb-runs](./wandb-runs/SKILL.md)  
Standardize run naming, config, tags, and metrics logging for comparable experiments.

3. [wandb-traces](./wandb-traces/SKILL.md)  
Query and analyze Weave traces for debugging, RCA evidence, and trace-linked diagnostics.

4. [wandb-evals](./wandb-evals/SKILL.md)  
Set up benchmark/eval execution and scorer-aligned logging for agent-level outcomes.

5. [wandb-reports](./wandb-reports/SKILL.md)  
Publish dashboards and reports that compare runs, failures, and fix impact.

6. [coding-agent-self-eval](./coding-agent-self-eval/SKILL.md)  
Orchestrate end-to-end self-evaluation with human-gated fix decisions and version mapping.
