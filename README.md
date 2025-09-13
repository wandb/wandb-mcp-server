<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>


# Weights & Biases MCP Server

A Model Context Protocol (MCP) server for querying [Weights & Biases](https://www.wandb.ai/) data. This server allows a MCP Client to:

- query W&B Models runs and sweeps
- query W&B Weave traces, evaluations and datasets
- query [wandbot](https://github.com/wandb/wandbot), the W&B support agent, for general W&B feature questions
- write text and charts to W&B Reports


## Installation

### 1. Install `uv`

Please first install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) with either:


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or 

```bash
brew install uv
```

### 2. Install on your MCP client of choice:

### Cursor, project-only
Enable the server for a specific project. Run the following in the root of your project dir:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path .cursor/mcp.json && uvx wandb login
```

### Cursor global
Enable the server for all Cursor projects, doesn't matter where this is run:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.cursor/mcp.json && uvx wandb login
```

### Windsurf 

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.codeium/windsurf/mcp_config.json && uvx wandb login
```

### Claude Code

```bash
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server && uvx wandb login
```

Passing an environment variable to Claude Code, e.g. api key:

```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```

### Claude Desktop
First ensure `uv` is installed, you might have to use `homebrew` to install depite `uv` being available in your terminal. Then run the below:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path "~/Library/Application Support/Claude/claude_desktop_config.json" && uvx wandb login
```

### Manual Installation
1. Ensure you have `uv` installed, see above installation instructions for uv.
2. Get your W&B api key [here](https://www.wandb.ai/authorize)
3. Add the following to your MCP client config manually.

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "<insert your wandb key>",
      }
    }
  }
}
```

These help utilities above are inspired by the OpenMCP Server Registry [add-to-client pattern](https://www.open-mcp.org/servers).

## Available MCP tools

### 1. wandb
-  **`query_wandb_tool`** Execute queries against wandb experiment tracking data including Runs & Sweeps.
  
### 2. weave
- **`query_weave_traces_tool`** Queries Weave evaluations and traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **`count_weave_traces_tool`** Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.

### 3. W&B Support agent
- **`query_wandb_support_bot`** Connect your client to [wandbot](https://github.com/wandb/wandbot), our RAG-powered support agent for general help on how to use Weigths & Biases products and features.

### 4. Saving Analysis
- **`create_wandb_report_tool`** Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

### 5. General W&B helpers
- **`query_wandb_entity_projects`** List the available W&B entities and projects that can be accessed to give the LLM more context on how to write the correct queries for the above tools.

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

## Advanced

### Writing environment variables to the config file

The `add_to_client` function accepts a number of flags to enable writing optional environment variables to the server's config file. Below is an example setting other env variables that don't have dedicated flags.

```bash
# Write the server config file with additional env vars
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client \
  --config_path ~/.codeium/windsurf/mcp_config.json \
  --write_env_vars MCP_LOGS_WANDB_ENTITY=my_wandb_entity

# Then login to W&B
uvx wandb login
```

Arguments passed to `--write_env_vars` must be space separated and the key and value of each env variable must be separated only by a `=`.

### Running from Source

Run the server from source by running the below in the root dir:

```bash
wandb login && uv run src/wandb_mcp_server/server.py
```

### Transport Options

The server supports two transport modes. You can specify these options when running from source:

#### Local MCP Client Communication (default)
For standard MCP client integration (Cursor, Claude Desktop, etc.), use the default stdio transport:

```bash
# Default - uses stdio transport (same as Running from Source section)
uv run src/wandb_mcp_server/server.py

# Explicit stdio transport
uv run src/wandb_mcp_server/server.py --transport stdio
```

#### HTTP Server Transport (SSE)
For remote access or web-based applications that need HTTP connectivity via Server-Sent Events:

```bash
# HTTP server on default port 8080
uv run src/wandb_mcp_server/server.py --transport http

# HTTP server on custom port
uv run src/wandb_mcp_server/server.py --transport http --port 9090

# HTTP server accessible from any IP
uv run src/wandb_mcp_server/server.py --transport http --host 0.0.0.0 --port 8080
```

**Available Options:**
- `--transport`: Choose `stdio` (default) for local MCP clients or `http` for HTTP server with SSE
- `--port`: Port number for HTTP server (defaults to 8080 when using HTTP transport)  
- `--host`: Host to bind HTTP server to (defaults to `localhost`)

**Note:** The HTTP transport uses the streamable HTTP protocol for bidirectional communication. No additional dependencies are required.

#### Using with Chat Applications via ngrok

To use the HTTP server with external chat applications like Mistral's le Chat, you can expose it publicly using ngrok:

1. **Install ngrok** (if not already installed):
   ```bash
   # macOS
   brew install ngrok
   
   # Or download from https://ngrok.com/download
   ```

2. **Start the MCP server** on HTTP transport:
   ```bash
   uv run src/wandb_mcp_server/server.py --transport http --port 8080 --wandb_api_key your_wandb_key
   ```

3. **In a new terminal, expose the server** with ngrok:
   ```bash
   ngrok http 8080
   ```

4. **Copy the public URL** from ngrok output (e.g., `https://abc123.ngrok.io`)

5. **Configure your chat application** to use the MCP server:
   - **Mistral le Chat**: Add the ngrok URL + `/mcp` as the MCP server endpoint
   - **Other chat apps**: Use the ngrok URL + `/mcp` for MCP connections
   - **Example**: `https://abc123.ngrok.io/mcp`

**Example ngrok output:**
```
Session Status                online
Account                       your-account (Plan: Free)
Version                       3.0.0
Region                        United States (us)
Forwarding                    https://abc123.ngrok.io -> http://localhost:8080
```

Use `https://abc123.ngrok.io/mcp` as your MCP server endpoint in chat applications.

### Environment Variables

The full list of environment variables used to control the server's settings can be found in the `.env.example` file.

## Troubleshooting

### Authentication

Ensure the machine running the MCP server is authenticated  to Weights & Biases, either by setting the `WANDB_API_KEY` or running the below to add the key to the .netrc file:

```bash
uvx wandb login
```

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

   or if using a Mac:

   ```
   brew install uv
   ```

2. If the error persists after installation, create a symlink to make `uv` available system-wide:
   ```bash
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

3. Restart your application or IDE after making these changes.

This ensures that the `uv` executable is accessible from standard system paths that are typically included in the PATH for all processes.

## Testing

The tests include a mix of unit tests and integration tests that test the tool calling reliability of a LLM. For now the integration tets only use claude-sonnet-3.7.


#### Set LLM provider API key

Set the appropriate api key in the `.env` file, e.g.

```
ANTHROPIC_API_KEY=<my_key>
```

#### Run 1 test file

Run a single test using pytest with 10 workers
```
uv run pytest -s -n 10 tests/test_query_wandb_gql.py
```

#### Test debugging

Turn on debug logging for a single sample in 1 test file

```
pytest -s -n 1 "tests/test_query_weave_traces.py::test_query_weave_trace[longest_eval_most_expensive_child]" -v --log-cli-level=DEBUG
```