# W&B MCP Server

Query and analyze your Weights & Biases data using natural language through the Model Context Protocol.

<div align="center">
  <a href="https://cursor.com/en/install-mcp?name=wandb&config=eyJ0cmFuc3BvcnQiOiJodHRwIiwidXJsIjoiaHR0cHM6Ly9tY3Aud2l0aHdhbmRiLmNvbS9tY3AiLCJoZWFkZXJzIjp7IkF1dGhvcml6YXRpb24iOiJCZWFyZXIge3tXQU5EQl9BUElfS0VZfX0iLCJBY2NlcHQiOiJhcHBsaWNhdGlvbi9qc29uLCB0ZXh0L2V2ZW50LXN0cmVhbSJ9fQ%3D%3D"><img src="https://cursor.com/deeplink/mcp-install-dark.svg" alt="Cursor" height="28"/></a>
  <a href="#claude-desktop"><img src="https://img.shields.io/badge/Claude-6B5CE6?logo=anthropic&logoColor=white" alt="Claude" height="28"/></a>
  <a href="#openai"><img src="https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white" alt="OpenAI" height="28"/></a>
  <a href="#gemini-cli"><img src="https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white" alt="Gemini" height="28"/></a>
  <a href="#mistral-lechat"><img src="https://img.shields.io/badge/LeChat-FF6B6B?logo=mistralai&logoColor=white" alt="LeChat" height="28"/></a>
  <a href="#vscode"><img src="https://img.shields.io/badge/VSCode-007ACC?logo=visualstudiocode&logoColor=white" alt="VSCode" height="28"/></a>
</div>

---

## What Can This Server Do?

<details open>
<summary><strong>Example Use Cases</strong> (click command to copy)</summary>

| **Analyze Experiments** | **Debug Traces** | **Create Reports** | **Get Help** |
|:---|:---|:---|:---|
| Show me the top 5 runs by eval/accuracy in wandb-smle/hiring-agent-demo-public? | How did the latency of my hiring agent predict traces evolve over the last months? | Generate a wandb report comparing the decisions made by the hiring agent last month | How do I create a leaderboard in Weave - ask SupportBot? |

New tools for auto-clustering coming soon:<br>
*"Go through the last 100 traces of my last training run in grpo-cuda/axolotl-grpo and tell me why rollout traces of my RL experiment were bad sometimes?"*
</details>

<details>
<summary><strong>Available Tools</strong> (6 powerful tools)</summary>

| Tool | Description | Example Query |
|------|-------------|---------------|
| **query_wandb_tool** | Query W&B runs, metrics, and experiments | *"Show me runs with loss < 0.1"* |
| **query_weave_traces_tool** | Analyze LLM traces and evaluations | *"What's the average latency?"* |
| **count_weave_traces_tool** | Count traces and get storage metrics | *"How many traces failed?"* |
| **create_wandb_report_tool** | Create W&B reports programmatically | *"Create a performance report"* |
| **query_wandb_entity_projects** | List projects for an entity | *"What projects exist?"* |
| **query_wandb_support_bot** | Get help from W&B documentation | *"How do I use sweeps?"* |

</details>

<details>
<summary><strong>Usage Tips</strong> (best practices)</summary>

**→ Provide your W&B project and entity name**  
LLMs are not mind readers, ensure you specify the W&B Entity and W&B Project to the LLM.

**→ Avoid asking overly broad questions**  
Questions such as "what is my best evaluation?" are probably overly broad and you'll get to an answer faster by refining your question to be more specific such as: "what eval had the highest f1 score?"

**→ Ensure all data was retrieved**  
When asking broad, general questions such as "what are my best performing runs/evaluations?" it's always a good idea to ask the LLM to check that it retrieved all the available runs. The MCP tools are designed to fetch the correct amount of data, but sometimes there can be a tendency from the LLMs to only retrieve the latest runs or the last N runs.

</details>

---

## Quick Start

We recommend using our **hosted server** at `https://mcp.withwandb.com` - no installation required! <br>

> 🔑 Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize) <br>

> 🌐 To connect to a **W&B Dedicated / On-Prem Instance** currently only the **local** MCP configuration can be used with an additional `WANDB_BASE_URL` env variable (the default is `api.wandb.ai`)

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

> **Note**: OpenAI's MCP is server-side, so localhost URLs won't work. For local servers, see [Option 2](#general-installation-guide) with ngrok.
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
  Mistral is currently the best supported Chat assistant based on API-key based authentication. 
  Simply navigate to "[Connectors](https://mistral.ai/news/le-chat-mcp-connectors-memories)" and 1) paste in the URL `https://mcp.withwandb.com/mcp` and 2) select API Key Authentication and paste WANDB API key. 
</details>

### Claude Desktop
<details>
<summary>Configuration setup</summary>

Add to your Claude config file. Claude desktop currently doesn't support remote MCPs to be added so we're adding the local MCP. Be careful to add the full path to `uv` for the command because Claude Desktop potentially doesn't find your `uv` installation otherwise. 

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
     "command": "/Users/niware_wb/.local/bin/uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
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

We're working on adding OAuth support so that we can integrate with ChatGPT. 

---

## General Installation Guide

<details>
<summary><strong>Option 1: Hosted Server (Recommended)</strong></summary>

The hosted server provides a zero-configuration experience with enterprise-grade reliability. This server is maintained by the W&B team, automatically updated with new features, and scales to handle any workload. Perfect for teams and production use cases where you want to focus on your ML work rather than infrastructure.

### Using the Public Server

The easiest way is using our hosted server at `https://mcp.withwandb.com`.

**Benefits:**
- ✅ Zero installation
- ✅ Always up-to-date
- ✅ Automatic scaling
- ✅ No maintenance

Simply use the configurations shown in [Quick Start](#quick-start).
</details>

<details>
<summary><strong>Option 2: Local Development (STDIO)</strong></summary>

Run the MCP server locally for development, testing, or when you need full control over your data. The local server runs directly on your machine with STDIO transport for desktop clients or HTTP transport for web-based clients. Ideal for developers who want to customize the server or work in air-gapped environments. **See below for client specific installation**.

### Running the Server Locally

**Quick Start:**
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the server
uv pip install git+https://github.com/wandb/wandb-mcp-server

# Run with STDIO transport (for desktop clients)
export WANDB_API_KEY="your-api-key"
uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
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
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "YOUR_API_KEY",
        "WANDB_BASE_URL": "YOUR_BASE_URL", #optional for dedicated or on-prem installations
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
      "git+https://github.com/wandb/wandb-mcp-server",
      "wandb_mcp_server"
    ],
    "env": {
      "WANDB_API_KEY": "YOUR-API_KEY",
      "WANDB_BASE_URL": "https://your-wandb-instance.example.com", # optional
    }
}
```


### Codex
```bash
codex mcp add wandb \
    --env WANDB_API_KEY=your_api_key_here \
    --env WANDB_BASE_URL=https://your-wandb-instance.example.com \
    -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```

### Claude Code
Add `--scope user` for global config.
```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -e WANDB_BASE_URL=your-base-url -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
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
     "command": "/Users/niware_wb/.local/bin/uvx",
      "args": [
        "--from",
        "git+https://github.com/wandb/wandb-mcp-server",
        "wandb_mcp_server"
      ],
      "env": {
        "WANDB_API_KEY": "<your-api-key>",
        "WANDB_BASE_URL": "https://your-wandb-instance.example.com", # optional
      }
    }
  }
}
```

Restart Claude Desktop to activate.

### Testing with ngrok (for server-side clients)

For clients like OpenAI and LeChat that require public URLs:

```bash
# 1. Start HTTP server
uvx wandb-mcp-server --transport http --port 8080

# 2. Expose with ngrok
ngrok http 8080

# 3. Use the ngrok URL in your client configuration
```

</details>

<details>
<summary><strong>Option 3: Self-Hosted HTTP Server (Advanced)</strong></summary>

This public repository focuses on the STDIO transport. If you need a fully managed HTTP deployment (Docker, Cloud Run, Hugging Face, etc.), start from this codebase and add your own HTTP entrypoint in a separate repo. The production-grade hosted server maintained by W&B now lives in a private repository built on top of this one.

### Running HTTP Server Locally

For lightweight experimentation and testing, you can run the FastMCP HTTP transport directly:

```bash
# Basic HTTP server
uvx wandb_mcp_server --transport http --host 0.0.0.0 --port 8080

# With Weave tracing enabled
uvx wandb_mcp_server \
  --transport http \
  --host 0.0.0.0 \
  --port 8080 \
  --weave_entity your-entity \
  --weave_project mcp-server-logs
```

> 📖 For all available command line options, see the [Command Line Reference](#command-line-reference) in the More Information section.

**Note**: Clients must continue to provide their own W&B API key via Bearer token per the MCP spec.
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
| `--wandb_api_key` | string | None | Weights & Biases API key for authentication |
| `--weave_entity` | string | None | The W&B entity to log traced MCP server calls to |
| `--weave_project` | string | `weave-mcp-server` | The W&B project to log traced MCP server calls to |

#### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | Your W&B API key (alternative to `--wandb_api_key` flag) | Yes |
| `WANDB_BASE_URL` | Custom W&B instance URL (for dedicated/on-prem instances) | No |
| `MCP_SERVER_LOG_LEVEL` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | No |
| `WANDB_SILENT` | Set to `"False"` to suppress W&B output | No |
| `WEAVE_SILENT` | Set to `"False"` to suppress Weave output | No |
| `WANDB_DEBUG` | Set to `"true"` to enable detailed W&B logging | No |
| `MCP_AUTH_DISABLED` | Disable HTTP authentication (development only) | No |

#### Usage Examples

**STDIO Transport (default for desktop clients):**
```bash
# Basic usage with environment variable
export WANDB_API_KEY="your-api-key"
uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server

# Or with API key as argument
uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server --wandb_api_key your-api-key
```

**HTTP Transport (for testing and development):**
```bash
# Basic HTTP server on localhost:8080
uvx wandb_mcp_server --transport http --host 127.0.0.1 --port 8080

# Bind to all interfaces with custom port
uvx wandb_mcp_server --transport http --host 0.0.0.0 --port 9090
```

**With Weave Tracing (log MCP calls to W&B):**
```bash
uvx wandb_mcp_server \
  --transport http \
  --port 8080 \
  --weave_entity my-team \
  --weave_project mcp-monitoring
```

**View all options:**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server --help
```

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
            "require_approval": "never",
        },
    ],
    input="How many traces are in wandb-smle/hiring-agent-demo-public?",
)

print(resp.output_text)
```
</details>

### Support

- [GitHub Issues](https://github.com/wandb/wandb-mcp-server/issues)
- Email support@wandb.com
