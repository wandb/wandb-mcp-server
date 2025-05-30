<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>

# Weights & Biases MCP Server

A Model Context Protocol (MCP) server for querying [Weights & Biases](https://www.wandb.ai/) data. This server allows a MCP Client to:

- query W&B Models runs, sweeps, artifacts and registry
- query W&B Weave traces, evaluations and datasets
- write text and charts to W&B Reports
- query [wandbot](https://github.com/wandb/wandbot), the W&B support bot, for general W&B feature questions

## Installation
We provide a helper utility for easily installing the Weights & Biases MCP Server into applications that use a JSON server spec. Please first [install `uv`](https://docs.astral.sh/uv/getting-started/installation/), typically by running `curl -LsSf https://astral.sh/uv/install.sh | sh` on your machine or running `brew install uv` on your mac.

From there, the `add_to_client` helper will add or update the required mcp json for popular MCP clients below - inspired by the OpenMCP Server Registry [`add-to-client` pattern](https://www.open-mcp.org/servers)

### Cursor

#### Cursor project (run from the project dir):
```
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client .cursor/mcp.json && uvx wandb login
```

#### Cursor global (applies to all projects):
```
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/.cursor/mcp.json && uvx wandb login
```

### Windsurf

```
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/.codeium/windsurf/mcp_config.json && uvx wandb login
```

### Claude Desktop:
First ensure `uv` is installed, you might have to use brew to install depite `uv` being available in your terminal:

```
brew install uv
```

then:

```
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client ~/Library/Application\ Support/Claude/claude_desktop_config.json && uvx wandb login
```

### Manual Installation
If you don't want to use the helper above, add the following to your MCP client config manually:

```
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server"
      ]
    }
  }
}
```

### Runing from Source

Run the server from source using:

```bash
wandb login && uv run src/wandb_mcp_server/server.py
```


## Available W&B tools

### wandb
-  **`query_wandb_gql_tool`** Execute an arbitrary GraphQL query against wandb experiment tracking data including Projects, Runs, Artifacts, Sweeps, Reports, etc.
  
### Weave
- **`query_weave_traces_tool`** Queries Weave traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **`count_weave_traces_tool`** Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.


### W&B Support bot
- **`query_wandb_support_bot`** Ask [wandbot](https://github.com/wandb/wandbot), our RAG-powered support agent for general help on how to use Weigths & Biases products and features. Powered by the W&B documentation.

### Saving Analysis
- **`create_wandb_report_tool`** Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

### General W&B helpers
- **`query_wandb_entity_projects`** List the available W&B entities and projects that can be accessed to give the LLM more context on how to write the correct queries for the above tools.

### Code Execution & Sandbox
- **`execute_sandbox_code_tool`** Execute Python code in secure, isolated sandbox environments:
  - **E2B Cloud Sandbox** - Most secure option with full VM isolation (requires `E2B_API_KEY`)
  - **Pyodide Local Sandbox** - WebAssembly-based execution (requires Node.js)


## Sandbox Requirements

### E2B Cloud Sandbox (Recommended)
For the most secure sandbox experience, set up an E2B API key:
1. Sign up at [e2b.dev](https://e2b.dev)
2. Get your API key from the dashboard
3. Set the environment variable: `export E2B_API_KEY=your_api_key_here`

### Local Pyodide Sandbox
For local code execution using WebAssembly:
1. Install Node.js (version 18 or higher): [nodejs.org](https://nodejs.org/)
2. The Pyodide runtime will be automatically loaded when needed

## Usage tips

#### Provide your W&B project and entity name

LLMs are not mind readers, ensure you specify the W&B Entity and W&B Project to the LLM. Example query for Claude Desktop:

```markdown
how many openai.chat traces in the wandb-applied-ai-team/mcp-tests weave project? plot the most recent 5 traces over time and save to a report
```

#### Avoid asking overly broad questions

Questions such as "what is my best evaluation?" are probably overly broad and you'll get to an answer faster by refining your question to be more specific such as: "what eval had the highest f1 score?"

#### Ensure all data was retrieved

When asking broad, general questions such as "what are my best performing runs/evaluations?" its always a good idea to ask the LLM to check that it retrieved all the available runs. The MCP tools are designed to fetch the correct amount of data, but sometimes there can be a tendency from the LLMs to only retrieve the latest runs or the last N runs.



## Troubleshooting

### Error: spawn uv ENOENT

If you encounter an error like this when starting the MCP server:
```
Error: spawn uv ENOENT
```

This indicates that the `uv` package manager cannot be found. Fix this with these steps:

1. Install `uv` using the official installation script:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. If the error persists after installation, create a symlink to make `uv` available system-wide:
   ```bash
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

3. Restart your application or IDE after making these changes.

This ensures that the `uv` executable is accessible from standard system paths that are typically included in the PATH for all processes.


## Testing

The tests include a mix of unit tests and integration tests that test the tool calling reliability of a LLM. For now the integration tets only use claude-sonnet-3.7.


####Set LLM provider API key

Set the appropriate api key in the `.env` file, e.g.

```
ANTHROPIC_API_KEY=<my_key>
```

####Run 1 test file

Run a single test using pytest with 10 workers
```
uv run pytest -s -n 10 tests/test_query_wandb_gql.py
```

####Test debugging

Turn on debug logging for a single sample in 1 test file

```
pytest -s -n 1 "tests/test_query_weave_traces.py::test_query_weave_trace[longest_eval_most_expensive_child]" -v --log-cli-level=DEBUG
```
