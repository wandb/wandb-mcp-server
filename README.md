# W&B MCP Server

Query and analyze your Weights & Biases data using natural language through the Model Context Protocol.

[![CI](https://github.com/wandb/wandb-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/wandb/wandb-mcp-server/actions/workflows/ci.yml)
<!-- BEGIN EVAL BADGES -->
[![SDK](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/wandb-mcp-server/main/.badges/sdk.json)](https://wandb.ai/wandb/mcp-server-ci/weave)
[![MCP](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wandb/wandb-mcp-server/main/.badges/mcp.json)](https://wandb.ai/wandb/mcp-server-ci/weave)
<!-- END EVAL BADGES -->

<div align="center">
  <a href="https://cursor.com/en/install-mcp?name=wandb&config=eyJ0cmFuc3BvcnQiOiJodHRwIiwidXJsIjoiaHR0cHM6Ly9tY3Aud2l0aHdhbmRiLmNvbS9tY3AiLCJoZWFkZXJzIjp7IkF1dGhvcml6YXRpb24iOiJCZWFyZXIge3tXQU5EQl9BUElfS0VZfX0iLCJBY2NlcHQiOiJhcHBsaWNhdGlvbi9qc29uLCB0ZXh0L2V2ZW50LXN0cmVhbSJ9fQ%3D%3D"><img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Cursor" height="28"/></a>
  <a href="#claude-desktop"><img src="https://img.shields.io/badge/Claude-6B5CE6?logo=anthropic&logoColor=white" alt="Claude" height="28"/></a>
  <a href="#openai-response-api"><img src="https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white" alt="OpenAI" height="28"/></a>
  <a href="#gemini-cli"><img src="https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white" alt="Gemini" height="28"/></a>
  <a href="#mistral-chat"><img src="https://img.shields.io/badge/LeChat-FF6B6B?logo=mistralai&logoColor=white" alt="LeChat" height="28"/></a>
  <a href="#vscode"><img src="https://img.shields.io/badge/VSCode-007ACC?logo=visualstudiocode&logoColor=white" alt="VSCode" height="28"/></a>
</div>

---

## v0.4.0 Release Highlights

Version 0.4.0 makes W&B reads SDK-first, bounded, and workload-aware:

- Run collections return lightweight metadata by default and support targeted
  `summary_keys` and `config_keys` instead of loading every metric.
- Explicit multi-key default-history reads use bounded outer-union semantics,
  so metrics logged at different cadences remain visible instead of requiring
  every key on the same history row.
- Shared and Dedicated workload profiles bound collection, history, evaluation,
  and schema reads. When admission is enabled, weighted in-flight work cannot
  exceed the configured per-actor and per-process cost capacities; overload
  returns retryable `server_busy`. Local STDIO leaves admission off by default.
- Dedicated and Self-Managed deployments can send W&B Models/API traffic over an
  internal Kubernetes service with `WANDB_INTERNAL_BASE_URL`, while public links
  continue to use `WANDB_BASE_URL`.
- Raw GraphQL is disabled by default and remains query-only when explicitly
  enabled. Mutations and subscriptions are rejected.
- Three opt-in ARIA tools support bounded asynchronous submission and polling
  when a deployment has an approved hosted W&B Agent path. Dedicated and
  Self-Managed deployments keep this integration disabled by default.
- Tool telemetry is bounded, excludes raw arguments and API keys, and correctly
  attributes supported clients such as Codex, Claude Code, and Cursor.

See the [v0.4.0 release notes](docs/releases/v0.4.0.md) for migration guidance,
deployment settings, and the complete customer-visible summary.

### Release availability

v0.4.0 is a release candidate until its signed source tag and each deployment
artifact are independently verified. The
[release index](docs/releases/README.md) is the source of truth for public
source, W&B-hosted, Dedicated/Self-Managed, and customer-container availability.
Do not infer availability from a branch, mutable tag, or version string in this
README.

The official artifacts are the verified signed source tag and GitHub Release,
plus channel-specific immutable container digests recorded there. PyPI and
mutable image tags such as `latest` are not supported release channels.

## What Can This Server Do?

<details open>
<summary><strong>Example Use Cases</strong> (click command to copy)</summary>

| **Analyze Experiments** | **Debug Traces** | **Create Reports** | **Get Help** |
|:---|:---|:---|:---|
| Show me the top 5 runs by eval/accuracy in wandb-smle/hiring-agent-demo-public? | How did the latency of my hiring agent predict traces evolve over the last months? | Generate a wandb report comparing the decisions made by the hiring agent last month | Search the official W&B docs for how to create a leaderboard in Weave. |

*"Go through the last 100 traces of my last training run in grpo-cuda/axolotl-grpo and tell me why rollout traces of my RL experiment were bad sometimes?"*
</details>

<details>
<summary><strong>Available Tools</strong></summary>

| Tool | Description | Example Query |
|------|-------------|---------------|
| **infer_trace_schema_tool** | Discover field names, types, and sample values | *"What fields are in my traces?"* |
| **query_weave_traces_tool** | Analyze LLM traces with `detail_level` control | *"Show failed traces with full data"* |
| **count_weave_traces_tool** | Count traces and get storage metrics | *"How many traces failed?"* |
| **resolve_trace_roots_tool** | Resolve spans to their root traces | *"Find the root traces for these calls"* |
| **query_wandb_tool** | Query projects, runs, sweeps, and reports through bounded W&B read APIs | *"Show me runs with loss < 0.1"* |
| **probe_project_tool** | Discover useful project fields and bounded samples | *"What metrics and config fields are available?"* |
| **get_run_history_tool** | Bounded time-series data with sparse, multi-key outer-union sampling | *"Show loss and validation-loss curves for run abc123"* |
| **compare_runs_tool** | Compare selected metrics and configuration across runs | *"Compare these three training runs"* |
| **diagnose_run_tool** | Diagnose a run using bounded metadata and history reads | *"Why did this run diverge?"* |
| **summarize_evaluation_tool** | Summarize bounded evaluation results with coverage metadata | *"Summarize this evaluation"* |
| **create_wandb_report_tool** | Create reports with markdown, charts, and panels | *"Create a report with loss plots"* |
| **log_analysis_to_wandb** | Log analysis metrics to W&B as a run | *"Log these latency stats to W&B"* |
| **search_wandb_docs_tool** | Search official W&B documentation | *"How do I create a Weave scorer?"* |
| **list_entities_tool** | List entities accessible to the current API key | *"Which W&B teams can I access?"* |
| **query_wandb_entity_projects** | List projects for an entity | *"What projects exist?"* |
| **list_registries_tool** | List model registries in an organization | *"What registries are available?"* |
| **list_registry_collections_tool** | List collections within a registry | *"What models are in the prod registry?"* |
| **list_artifact_versions_tool** | List versions of an artifact collection | *"Show versions of my model artifact"* |
| **get_artifact_details_tool** | Get full details of an artifact version | *"What's in model-v2 artifact?"* |
| **compare_artifact_versions_tool** | Diff two artifact versions | *"Compare model v1 vs v2"* |
| **list_wandb_automations_tool** | List W&B Automations | *"What automations alert on run metrics or status for my team's runs?"* |
| **list_wandb_integrations_tool** | List registered integrations for W&B automations (e.g. Slack, webhook) | *"Which Slack channels can my automations target?"* |
| **aria_send_message** *(opt-in)* | Start or continue async work with the hosted W&B agent | *"Ask ARIA to diagnose these failed evals"* |
| **aria_get_turn** *(opt-in)* | Poll an ARIA turn for progress or its final result | *"Check whether that ARIA analysis finished"* |
| **aria_get_turns** *(opt-in)* | Poll up to 20 ARIA turns concurrently | *"Check all of those ARIA analyses in one bounded wait"* |

**Read-only deployment mode:** Set `WANDB_MCP_READ_ONLY=true` to omit the three write-capable tools,
`create_wandb_report_tool`, `log_analysis_to_wandb`, and `aria_send_message`, while keeping every existing read tool.
`query_wandb_tool` is read-only in every mode. It normally uses documented W&B
APIs and may use fixed, application-owned query-only projections to avoid
per-result fan-out; callers cannot supply GraphQL to this tool.

**Advanced raw GraphQL:** Raw GraphQL is not registered by default. Set
`WANDB_MCP_ENABLE_RAW_GRAPHQL=true` to add the query-only `query_wandb_graphql_tool`
for schema introspection, unmodeled fields, cross-resource nesting, aliases, or exact
response shapes that the typed tool cannot represent. It accepts exactly one bounded
query operation; mutations and subscriptions are always rejected. This flag is
independent of `WANDB_MCP_READ_ONLY`. See the
[query capability matrix](docs/query-capabilities.md).

On the supported W&B service transport, raw and fixed GraphQL responses are
checked at 16 MiB before MCP JSON decoding and again after decoding with a
500,000-node structural ceiling. Oversized results return
`response_too_large`. These are MCP processing safeguards after the SDK has
received the protobuf envelope; they do not cap network bytes or backend work.

Recommended deployment presets:

| Deployment | Settings |
|---|---|
| W&B-hosted | Weave Agent tools enabled; ARIA and caller-supplied raw GraphQL disabled by default. ARIA is available only through a separately validated W&B-managed profile. |
| Strict customer read-only | `WANDB_MCP_READ_ONLY=true`, `WANDB_MCP_ENABLE_RAW_GRAPHQL=false`, `WANDB_MCP_ENABLE_ARIA_TOOLS=false` |
| Trusted read-only compatibility | `WANDB_MCP_READ_ONLY=true`, `WANDB_MCP_ENABLE_RAW_GRAPHQL=true`, `WANDB_MCP_ENABLE_ARIA_TOOLS=false`, `MCP_MAX_GQL_ITEMS=50`, `MCP_MAX_GQL_ITEMS_PER_PAGE=20` |

**Migration from v0.3.7:** `query_wandb_tool` now accepts structured SDK parameters
(`entity_name`, `project_name`, `resource`, filters, ordering, and identifiers) instead
of a GraphQL document. Existing raw-query callers must explicitly enable and call
`query_wandb_graphql_tool`.

**Run history semantics:** for explicit multi-key default-history sampled and
ranged collection reads, `get_run_history_tool` outer-joins requested metric
series by `_step`, falling back to the selected x-axis when `_step` is absent,
so keys logged in separate `wandb.log()` calls do not disappear. Multi-key
ranges use one independent bounded series per metric, in requests of at most
eight series. The independent-spec path is designed to cover active/newly
synced as well as exported history. Multi-request reads use the same freshly
observed upper-step boundary for every batch. A fixed snapshot query reads run
identity plus one complete, resume-oriented `historyTail` row; it avoids loading
the full config, summary, system metrics, or history-key index before the
bounded history read. `samples` is the
total final-row budget across all requested keys, not a per-key allowance. All
observed sparse points in the bounded result are
retained when the budget permits. Diagnostics report `unobserved_keys` when no
usable finite/non-null value appears in a bounded/sample result; `missing_keys`
is reported only when absence was checked exactly. A custom x-axis must occur
on the same row as its metric because the server does not invent interpolation.
`non_finite_counts` reports NaN or Infinity observations seen in the bounded
source; those invalid JSON numeric values are omitted from returned rows, and
their counts are exhaustive only when `key_counts_exact=true`.
Exact `target_x` remains a point lookup rather than an outer-union read. This
path uses a fixed application-owned query-only projection, accepts no
caller-supplied GraphQL, and works with `WANDB_MCP_ENABLE_RAW_GRAPHQL=false`.

**Weave Agents (OTel) tools** — these read the OpenTelemetry/GenAI agent-spans data plane (the **Agents** tab), which is separate from the classic Weave calls above:

These tools are disabled by default. Enable them with `WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS=true`.

| Tool | Description | Example Query |
|------|-------------|---------------|
| **list_weave_agents_tool** | List agents with aggregated stats (invocations, tokens, duration, errors) | *"Which agents ran this week and how many errors did each have?"* |
| **list_weave_agent_versions_tool** | Per-version stats for one agent | *"Did v2 of my agent regress on latency?"* |
| **query_weave_agent_spans_tool** | Query individual agent/LLM/tool spans (filter by agent, model, time) | *"Show failed tool spans for my-agent yesterday"* |
| **get_weave_agent_span_stats_tool** | Time-bucketed metric series (tokens, cost, latency, error rate) | *"Plot daily token usage by agent"* |
| **list_weave_agent_custom_attributes_tool** | Discover custom attribute keys on agent spans | *"What custom attributes do my agent spans have?"* |
| **search_weave_agents_tool** | Full-text / structured message search, grouped by conversation | *"Find conversations mentioning refunds"* |
| **get_weave_agent_trace_tool** | Structured chat/trajectory view for one trace (a turn) | *"What did the agent do in trace abc123?"* |
| **get_weave_agent_conversation_tool** | Multi-turn chat view for a conversation | *"Show the whole conversation conv-42"* |

**Schema-first workflow:** Call `infer_trace_schema_tool` first to discover fields, then `query_weave_traces_tool` with precise columns and `detail_level`:
- `"schema"` -- structural fields only (fast browsing)
- `"summary"` -- truncated inputs/outputs (default)
- `"full"` -- everything untruncated (drill into specific traces)

**Chart panels:** `create_wandb_report_tool` accepts a `panels` parameter for LinePlots, BarPlots, run comparisons, custom Vega charts, and ordered report layouts. Use `panel_grid` when multiple charts should share one runset, and use `heading` plus `markdown` blocks to interleave narrative sections with charts.

**Docs search:** `search_wandb_docs_tool` proxies [docs.wandb.ai](https://docs.wandb.ai) so you get data tools + documentation search from a single MCP connection. Disable with `WANDB_MCP_PROXY_DOCS=false` if you connect the docs MCP separately.

**ARIA tools are opt-in:** Set `WANDB_MCP_ENABLE_ARIA_TOOLS=true` only when the deployment has an approved HTTPS path to the hosted W&B Agent service. The default is `false`, including Dedicated/Self-Managed deployments, so customer credentials are never forwarded to the public ARIA service implicitly. `WANDB_MCP_READ_ONLY=true` additionally omits `aria_send_message` while retaining polling when the ARIA group is explicitly enabled.

**ARIA polling:** ARIA calls are asynchronous. `aria_send_message` returns a turn handle, and `aria_get_turn` polls one turn for up to 30 seconds. Use `aria_get_turns` for several outstanding turns so they are fetched concurrently within one shared polling window. Poll results are compact by default; pass `include_turn=true` only when a bounded raw service snapshot is needed.

**Registry organization resolution:** Registry tools use the authenticated
request's W&B client when `organization` is omitted. A single accessible
organization is selected automatically; callers with access to multiple
organizations receive `organization_required` with a bounded candidate list.
Collection listings intentionally return `aliases: null` and
`aliases_loaded: false` instead of loading every collection's version aliases.
Use `list_artifact_versions_tool` when aliases are needed.

</details>

<details>
<summary><strong>Usage Tips</strong> (best practices)</summary>

**→ Provide your W&B project and entity name**
LLMs are not mind readers, ensure you specify the W&B Entity and W&B Project to the LLM.

**→ Avoid asking overly broad questions**
Questions such as "what is my best evaluation?" are probably overly broad and you'll get to an answer faster by refining your question to be more specific such as: "what eval had the highest f1 score?"

**→ Check result coverage for broad questions**
Collection and evaluation tools report fields such as `total_count`,
`returned_count`, `has_more`, `project_exhaustive`, `sampled`, and `truncated`.
For broad questions, ask the client to explain these fields rather than assuming
that a bounded result represents the entire project.

</details>

---

## Quick Start

We recommend using our **hosted server** at `https://mcp.withwandb.com/mcp` - no installation required! <br>

> 🔑 Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize) <br>

> 🌐 For **W&B Dedicated / Self-Managed**, use the instance MCP endpoint when
> enabled by your operator chart (`https://<your-instance>/mcp`). Local STDIO
> remains available by setting `WANDB_BASE_URL=https://<your-instance>`.

### Cursor
<details>
<summary>One-click installation</summary>

  * Click on the button above to automatically add the config to Cursor
  * Then add your WANDB_API_KEY in the respective field `Bearer YOUR_API_KEY` and connect

For manual or local installation, see [Option 2](#general-installation-guide) below.
</details>

### OpenAI Response API
<details>
<summary>Python client setup</summary>

   ```python
from openai import OpenAI
import os

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_url": "https://mcp.withwandb.com/mcp",
        "authorization": os.getenv('WANDB_API_KEY'),
        "server_label": "WandB_MCP",
    }],
    input="How many traces are in my project?"
)
print(resp.output_text)
```

> **Note**: OpenAI's MCP is server-side, so localhost URLs won't work. Use the
> hosted W&B endpoint or an authenticated Helm deployment.
</details>

### Claude Code
<details>
<summary>One-command installation</summary>

```bash
# run in terminal
claude mcp add --transport http wandb https://mcp.withwandb.com/mcp --scope user --header "Authorization: Bearer <your-api-key-here>"
```

For local installation, see [Option 2](#general-installation-guide) below.
</details>

### OpenAI Codex
<details>
<summary>One-command installation</summary>

```bash
# run in terminal
export WANDB_API_KEY=<your-api-key>
codex mcp add wandb --url https://mcp.withwandb.com/mcp --bearer-token-env-var WANDB_API_KEY
```

For local installation, see [Option 2](#general-installation-guide) below.
</details>

### Gemini CLI
<details>
<summary>One-command installation</summary>

```bash
# Set your API key
export WANDB_API_KEY="your-api-key-here"

# Install the extension
gemini extensions install https://github.com/wandb/wandb-mcp-server
```

The extension will use the configuration from `gemini-extension.json` pointing to the hosted server.

For local installation, see [Option 2](#general-installation-guide) below.
</details>

### VSCode
<details>
<summary>Settings configuration</summary>

```bash
# Open settings
code ~/.vscode/mcp.json # or global mcp.json file
```

```json
{
  "servers": {
    "wandb": {
      "type": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY"
      }
    }
  }
}
```

For local installation, see [Option 2](#general-installation-guide) below.
</details>

### Mistral Chat
<details>
<summary>Configuration setup</summary>

Use the **Custom MCP Connector** flow:

1. Open Le Chat and go to **Connectors**.
2. Add a custom MCP connector.
3. Set the server URL to `https://mcp.withwandb.com/mcp`.
4. Select HTTP Bearer Token or API Key authentication.
5. Paste your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize).

If the UI asks for a token value, paste the raw W&B API key. If it asks for the full `Authorization` header value, use `Bearer <your-wandb-api-key>`.

</details>

### Claude Desktop
<details>
<summary>Configuration setup</summary>

For a local STDIO connection, add the server to the Claude Desktop
configuration. Use the absolute path to `uvx` if the desktop application cannot
resolve your shell `PATH`.

```bash
# macOS
open ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
notepad %APPDATA%\Claude\claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

Restart Claude Desktop to activate.
</details>

---

## General Installation Guide

<details>
<summary><strong>Option 1: Hosted Server (Recommended)</strong></summary>

The hosted server provides a managed, zero-installation experience. It uses
bounded workload profiles, request deadlines, and retryable overload responses
to protect W&B while serving multiple users.

### Using the Public Server

The easiest way is using our hosted server at `https://mcp.withwandb.com/mcp`.

**Benefits:**
- ✅ Zero installation
- ✅ Managed, reviewed release updates
- ✅ Managed workload limits
- ✅ No server maintenance

Simply use the configurations shown in [Quick Start](#quick-start).
</details>

<details>
<summary><strong>Option 2: Local Development (STDIO)</strong></summary>

Run the MCP server locally for development, testing, or when you need direct
control over its configuration. The local server runs on your machine with
STDIO transport for desktop clients or HTTP transport for web-based clients.
It still requires network access to the configured W&B instance and any enabled
documentation or telemetry endpoints. **See below for client-specific
installation.**

### Running the Server Locally

**Quick Start:**
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Replace vX.Y.Z with a signed, available release from the release index.
export WANDB_API_KEY="your-api-key"
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server
```

> 📖 For complete command line options and environment variables, see the [Command Line Reference](#command-line-reference) in the More Information section.

### Manual Configuration
Add to your MCP client config (for detailed client-specific configs see below):

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "YOUR_API_KEY",
        "WANDB_BASE_URL": "https://your-wandb-instance.example.com"
      }
    }
  }
}
```

### Cursor
1. Open Cursor Settings (`⌘,` or `Ctrl,`)
2. Navigate to **Features** → **Model Context Protocol**
3. Click **"Install from Registry"** or **"Add MCP Server"**
4. Search for "wandb" or enter:
   - **Name**: `wandb`
   - **URL**: `https://mcp.withwandb.com/mcp`
   - **API Key**: Your W&B API key

Manual hosted config in `mcp.json`:
```
"wandb": {
  "transport": "http",
  "url": "https://mcp.withwandb.com/mcp",
  "headers": {
    "Authorization": "Bearer YOUR-API_KEY",
    "Accept": "application/json, text/event-stream"
  }
}
```
Manual local (dedicated or on-prem) config in `mcp.json`:

```
"wandb": {
  "command": "uvx",
    "args": [
      "--from",
      "git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z",
      "wandb_mcp_server"
    ],
    "env": {
      "WANDB_API_KEY": "YOUR-API_KEY",
      "WANDB_BASE_URL": "https://your-wandb-instance.example.com"
    }
}
```


### Codex
```bash
codex mcp add wandb \
    --env WANDB_API_KEY=your_api_key_here \
    --env WANDB_BASE_URL=https://your-wandb-instance.example.com \
    -- uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server
```

### Claude Code
Add `--scope user` for global config.
```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -e WANDB_BASE_URL=your-base-url -- uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server
```

### Claude Desktop
Same as above.
```bash
# macOS
open ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
notepad %APPDATA%\Claude\claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "<your-api-key>",
        "WANDB_BASE_URL": "https://your-wandb-instance.example.com"
      }
    }
  }
}
```

Restart Claude Desktop to activate.

The standalone HTTP entrypoint is intentionally loopback-only and must not be
published through ngrok or another tunnel. Use the authenticated hosted wrapper
or the `operator-wandb` Helm deployment for a remotely accessible endpoint.

</details>

<details>
<summary><strong>Option 3: Local HTTP Development</strong></summary>

The package exposes an unauthenticated Streamable HTTP transport only for local
development. It requires an explicit acknowledgement and accepts loopback binds
only. Use the authenticated hosted wrapper or Helm image for production HTTP.

### Running HTTP Server Locally

For lightweight experimentation and testing, you can run the FastMCP HTTP transport directly:

```bash
# Basic loopback HTTP server
export MCP_AUTH_DISABLED=true
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server \
  --transport http \
  --host 127.0.0.1 \
  --port 8080

# With Weave tracing enabled
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server \
  --transport http \
  --host 127.0.0.1 \
  --port 8080 \
  --weave_entity your-entity \
  --weave_project mcp-server-logs
```

> 📖 For all available command line options, see the [Command Line Reference](#command-line-reference) in the More Information section.

**Note**: This entrypoint has no bearer-authentication middleware. It uses the
single server-side `WANDB_API_KEY` and refuses non-loopback binds.
</details>

<details>
<summary><strong>Option 4: Dedicated / On-Prem Deployment</strong></summary>

For W&B Dedicated and Self-Managed customers, the MCP server is available as an
optional component in the `operator-wandb` Helm chart. Enable it in your
`WeightsAndBiases` values:

```yaml
mcp-server:
  install: true
```

When an MCP-capable chart release is available, the server becomes accessible
at `https://<your-instance>/mcp`. The chart keeps this public URL for clients
and user-facing links while routing server-side W&B API calls to the
namespace-local API service.

**Requirements:**
- `weave-trace` must be installed (`weave-trace.install: true`)
- An `operator-wandb` release whose notes explicitly include MCP support
- The immutable MCP image digest recorded by that chart release

**Client configuration** for dedicated instances:

```json
{
  "mcpServers": {
    "wandb": {
      "url": "https://your-instance.wandb.io/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY"
      }
    }
  }
}
```

Contact your W&B account team to enable MCP on your dedicated deployment.
</details>

---

## More Information

### Command Line Reference

When running the server locally, you can customize its behavior with command line arguments:

#### Available Arguments

> **Note**: Arguments use underscores (e.g., `--wandb_api_key`), not dashes.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--transport` | string | `stdio` | Transport type: `stdio` for local MCP client communication or `http` for HTTP server |
| `--host` | string | `localhost` | Host to bind HTTP server to (only used with `--transport http`) |
| `--port` | integer | `8080` | Port to run the HTTP server on (only used with `--transport http`) |
| `--wandb_api_key` | string | None | Compatibility-only API key input; prefer `WANDB_API_KEY` so the key is not exposed in process arguments |
| `--weave_entity` | string | None | The W&B entity to log traced MCP server calls to |
| `--weave_project` | string | `weave-mcp-server` | The W&B project to log traced MCP server calls to |

#### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | Your W&B API key (alternative to `--wandb_api_key` flag) | Yes |
| `WANDB_BASE_URL` | Public W&B instance URL used for credentials and user-facing links | No |
| `WANDB_INTERNAL_BASE_URL` | Optional server-side W&B Models/API URL; Dedicated charts set this to the in-cluster API service. Weave trace routing remains controlled by `WF_TRACE_SERVER_URL`. | No |
| `WB_AGENT_BASE_URL` | Hosted ARIA service URL (default: `https://wb-agent.wandb.ai`) | No |
| `MCP_SERVER_LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | No |
| `WANDB_SILENT` | Set to `"True"` to suppress W&B SDK output (default: `true`) | No |
| `WEAVE_SILENT` | Set to `"True"` to suppress Weave SDK output (default: `true`) | No |
| `WANDB_DEBUG` | Set to `"true"` to enable detailed W&B logging | No |
| `MCP_AUTH_DISABLED` | Must be `true` to acknowledge unauthenticated loopback HTTP development | No |
| `WANDB_MCP_PROXY_DOCS` | Enable/disable docs search proxy (default: `true`) | No |
| `MCP_MAX_GQL_ITEMS` | Maximum items returned by the opt-in raw GraphQL tool (profile default) | No |
| `MCP_MAX_GQL_ITEMS_PER_PAGE` | Maximum raw GraphQL connection page size (profile default) | No |
| `MCP_HOSTED_MODE` | Marks an HTTP deployment as hosted; defaults the workload profile to `shared` | No |
| `MCP_WORKLOAD_PROFILE` | Bounded defaults for `shared`, `dedicated`, or `local` workloads (default: `shared` when hosted, otherwise `local`) | No |
| `MCP_ADMISSION_CONTROL_ENABLED` | Enable actor-aware weighted tool admission (default: enabled except for the `local` profile) | No |
| `MCP_ADMISSION_ACTOR_CAPACITY` | Maximum concurrent cost units per API-key actor (profile default: shared `4`, dedicated `8`, local `16`) | No |
| `MCP_ADMISSION_PROCESS_CAPACITY` | Maximum concurrent cost units per server process (default: `16`) | No |
| `MCP_ADMISSION_WAIT_MS` | Maximum queue wait before returning retryable `server_busy` (default: `2000`) | No |
| `MCP_TOOL_TIMEOUT_SECONDS` | Hosted public-tool response deadline (default: `30`); timed-out sync workers retain admission capacity until they finish | No |
| `MCP_WANDB_REQUEST_TIMEOUT_SECONDS` | Timeout for server-side W&B SDK and fixed read requests (default: `20`) | No |
| `MCP_SYNC_TOOL_WORKERS` | Bounded worker count for synchronous public tools (profile-derived default, capped at `16`) | No |
| `MCP_ANALYTICS_DISABLED` | Disable structured MCP analytics events | No |
| `MCP_ANALYTICS_QUEUE_CAPACITY` | Maximum outstanding events per optional Segment or Datadog forwarder (default: `256`) | No |
| `MCP_REQUEST_SUCCESS_SAMPLE_RATE` | Deterministic sample rate for successful HTTP request telemetry (default: `0.10`; failures and requests over two seconds are always retained). | No |
| `MCP_LOG_PRIVACY_LEVEL` | Telemetry privacy level: `off`, `standard`, or `strict` (default: `off`) | No |
| `MAX_RESPONSE_TOKENS` | Token budget for response truncation (default: `30000`) | No |

<!-- BEGIN GENERATED: PUBLIC FEATURE PROFILES -->
<!-- Generated by scripts/public_release.py docs. Do not edit this block. -->

Release-controlled feature variables:

| Variable | Default | Effect |
|---|---:|---|
| `WANDB_MCP_ENABLE_WEAVE_TOOLS` | `true` | Registers the `weave` group (5 tools). |
| `WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS` | `false` | Registers the `agents` group (8 tools). |
| `WANDB_MCP_ENABLE_ARIA_TOOLS` | `false` | Registers the `aria` group (3 tools). |
| `WANDB_MCP_ENABLE_RAW_GRAPHQL` | `false` | Registers the `raw_graphql` group (1 tools). |
| `WANDB_MCP_READ_ONLY` | `false` | Removes all tools classified as writes. |

Named exact tool profiles:

| Profile | Weave | Agents | ARIA | Raw GraphQL | Read-only | Exact tools |
|---|---:|---:|---:|---:|---:|---:|
| `default` | `true` | `false` | `false` | `false` | `false` | 22 |
| `models-only` | `false` | `false` | `false` | `false` | `false` | 17 |
| `agents` | `true` | `true` | `false` | `false` | `false` | 30 |
| `aria` | `true` | `false` | `true` | `false` | `false` | 25 |
| `full` | `true` | `true` | `true` | `true` | `false` | 34 |
| `strict-read-only` | `true` | `false` | `false` | `false` | `true` | 20 |

Use `python scripts/public_release.py profiles --all` for every exact feature/read-only combination and tool name.
<!-- END GENERATED: PUBLIC FEATURE PROFILES -->

For the standalone console entrypoint, credential resolution is command-line
compatibility input, then the `.netrc` entry for `WANDB_BASE_URL`, then
`WANDB_API_KEY` (including a repository-root `.env` loaded by the CLI). Prefer
the environment and remove stale `.netrc` entries for the same host; never pass
a key in release automation or a process argument.

Workload profiles provide one deployment-level choice while preserving the
individual `MCP_MAX_*` overrides for advanced operators:

| Profile | Collection rows | History samples | Metric keys | Range step span | Full-detail rows | Eval detail rows | Schema sample rows | Actor/process cost |
|---------|----------------:|----------------:|------------:|----------------:|-----------------:|-----------------:|-------------------:|-------------------:|
| `shared` | 100 | 500 | 20 | 5,000 | 3 | 500 | 100 | 4 / 16 |
| `dedicated` | 250 | 1,500 | 50 | 20,000 | 10 | 2,000 | 250 | 8 / 16 |
| `local` | 1,000 | 5,000 | 100 | 100,000 | 25 | 5,000 | 500 | admission disabled |

#### Usage Examples

**STDIO Transport (default for desktop clients):**
```bash
# Basic usage with environment variable
export WANDB_API_KEY="your-api-key"
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server
```

For stdio clients such as Claude Desktop, stdout is reserved for MCP JSON-RPC
messages. The server routes logs and analytics to stderr so clients do not parse
diagnostics as protocol messages.

**HTTP Transport (for testing and development):**
```bash
# Basic HTTP server on localhost:8080
export MCP_AUTH_DISABLED=true
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server \
  --transport http \
  --host 127.0.0.1 \
  --port 8080
```

**With Weave Tracing (log MCP calls to W&B):**
```bash
export MCP_AUTH_DISABLED=true
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server \
  --transport http \
  --host 127.0.0.1 \
  --port 8080 \
  --weave_entity my-team \
  --weave_project mcp-monitoring
```

**View all options:**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server@vX.Y.Z wandb_mcp_server --help
```

### Contributing & Releasing

- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- Development setup, testing, PR process, architecture overview
- **[RELEASING.md](RELEASING.md)** -- Version bumping, release checklist, deployment pipeline
- **[Release index](docs/releases/README.md)** -- Availability and immutable artifacts by channel
- **[Query capability matrix](docs/query-capabilities.md)** -- Typed reads and the opt-in compatibility path
- **[Observability](docs/OBSERVABILITY.md)** -- Logging, telemetry privacy, and supported collection modes
- **[Tool-development skill](.agents/skills/develop-wandb-mcp-tools/SKILL.md)** -- Safe implementation and validation workflow
- **[Release skill](.agents/skills/release-wandb-mcp-server/SKILL.md)** -- Exact-candidate release validation and handoff

### Key Resources

- **W&B Docs**: [docs.wandb.ai](https://docs.wandb.ai)
- **Weave Docs**: [weave-docs.wandb.ai](https://weave-docs.wandb.ai)
- **MCP Spec**: [modelcontextprotocol.io](https://modelcontextprotocol.io)

### Example Code

<details>
<summary>Complete OpenAI Example</summary>

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o",  # Use gpt-4o for larger context window
    tools=[
        {
            "type": "mcp",
            "server_label": "wandb",
            "server_description": "Query W&B data",
            "server_url": "https://mcp.withwandb.com/mcp",
            "authorization": os.getenv('WANDB_API_KEY'),
        },
    ],
    input="How many traces are in wandb-smle/hiring-agent-demo-public?",
)

print(resp.output_text)
```
</details>

### Development

#### Running Tests

Unit tests run without API keys or network access:

```bash
uv sync --frozen --extra test --extra http
uv run pytest tests/ -m "not integration" -v
```

CI runs automatically on every push and PR via GitHub Actions.

#### Release artifact model

The public Python source, W&B-managed deployment image, and Dedicated/Self-Managed
chart are independently versioned artifacts. Managed builds pin this repository
to a signed source tag, and the chart pins a verified image digest. Local
source installs should use `@vX.Y.Z` with a version marked available in the
[release index](docs/releases/README.md); an unqualified GitHub branch is a
development input, not an immutable release. See [RELEASING.md](RELEASING.md)
for the required attestation and digest handoff checks.

### Support

- [GitHub Issues](https://github.com/wandb/wandb-mcp-server/issues)
- Email support@wandb.com
