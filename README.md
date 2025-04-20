# Weights & Biases MCP Server

A Model Context Protocol (MCP) server for querying [Weights & Biases Weave](https://weave-docs.wandb.ai/) traces. This server allows a MCP Client to:

- query W&B Models runs, sweeps, artifacts and registry
- query W&B Weave traces, evaluations and datasets
- write text and charts to W&B Reports

## Installation
We provide a helper utility for easily installing the Weights and Biases MCP Server into applications that use a JSON server spec. Please first [install `uv`](https://docs.astral.sh/uv/getting-started/installation/), typically by running `curl -LsSf https://astral.sh/uv/install.sh | sh` on your machine.

From there, you can use the `add_to_client` helper to add the server to your MCP client - inspired by the OpenMCP Server Registry [`add-to-client` pattern](https://www.open-mcp.org/servers)

### Cursor project (run from the project dir):
`uvx --from git+https://github.com/wandb/mcp-server add_to_client .cursor/mcp.json && uvx wandb login`

### Cursor global (applies to all projects):
`uvx --from git+https://github.com/wandb/mcp-server add_to_client ~/.cursor/mcp.json && uvx wandb login`

### Claude desktop:
`uvx --from git+https://github.com/wandb/mcp-server add_to_client ~/Library/Application\ Support/Claude/claude_desktop_config.json && uvx wandb login`

### Manually
If you don't want to use the helper above, add the following to your MCP client config manually:

```
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/mcp-server",
        "mcp_server"
      ]
    }
  }
}
```

## Available tools


### wandb
-  **`query_wandb_gql_tool`**: Execute an arbitrary GraphQL query against wandb experiment tracking data including Projects, Runs, Artifacts, Sweeps, Reports, etc.
  
### Weave
- **`query_weave_traces_tool`**: Queries Weave traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **`count_weave_traces_tool`**: Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.

### Saving Anaysis
- **`create_wandb_report_tool`**: Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

## Usage Tips

- When asking broad, general questions such as "what are my best performing runs/evaluations?" its always a good idea to ask the LLM to check that it retrieved all the available runs. Sometimes there can be a tendency from the LLMs to only retrieve the latest runs or the last X runs.

## Usage

Ensure you specify the W&B Entity and W&B Project to the LLM/MCP Client.

Example query for Claude Desktop:

```markdown
how many openai.chat traces in the wandb-applied-ai-team/mcp-tests weave project? plot the most recent 5 traces over time and save to a report
```

## Running from Source

```bash
git clone https://github.com/wandb/mcp-server.git
cd mcp-server && uv venv && source .venv/bin/activate
uv pip install -e .
```

### Configuration

1. Create a `.env` file in the root directory with your Weights & Biases API key:
```
WANDB_API_KEY=your_api_key_here
```

### Running the Server

Run the server using:

```bash
uv run src/mcp_server/server.py
```

### Client Setup

#### Claude Desktop

```json
    "mcpServers": {
        "weights_and_biases": {
        "command": "uv",
        "args": [
            "--directory",
            "/ABSOLUTE/PATH/TO/PROJECT",
            "run",
            "src/mcp_server/server.py"
        ]
        }
    }
```

## TODOs

- [ ] Add W&B Models data
- [ ] Convert to run with npx
- [ ] Make more configurable: specify wandb URL
- [ ] Work on reports plots prompt for consistent visualizations
- [ ] Look into auth solutions

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
