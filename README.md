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

We recommend using our **hosted server** at `https://mcp.withwandb.com` - no installation required!

> 🔑 Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### Cursor
<details>
<summary>One-click installation</summary>

1. Open Cursor Settings (`⌘,` or `Ctrl,`)
2. Navigate to **Features** → **Model Context Protocol**
3. Click **"Install from Registry"** or **"Add MCP Server"**
4. Search for "wandb" or enter:
   - **Name**: `wandb`
   - **URL**: `https://mcp.withwandb.com/mcp`
   - **API Key**: Your W&B API key

For local installation, see [Option 2](#option-2-local-development-stdio) below.
</details>

### Claude Desktop
<details>
<summary>Configuration setup</summary>

Add to your Claude config file:

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
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "YOUR_WANDB_API_KEY"
    }
  }
}
```

Restart Claude Desktop to activate.

For local installation, see [Option 2](#option-2-local-development-stdio) below.
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
    }],
    input="How many traces are in my project?"
)
print(resp.output_text)
```

> **Note**: OpenAI's MCP is server-side, so localhost URLs won't work. For local servers, see [Option 2](#option-2-local-development-stdio) with ngrok.
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

For local installation, see [Option 2](#option-2-local-development-stdio) below.
</details>

### Mistral LeChat
<details>
<summary>Configuration setup</summary>

In LeChat settings, add an MCP server:
- **URL**: `https://mcp.withwandb.com/mcp`
- **API Key**: Your W&B API key

For local installation, see [Option 2](#option-2-local-development-stdio) below.
</details>

### VSCode
<details>
<summary>Settings configuration</summary>

```bash
# Open settings
code ~/.config/Code/User/settings.json
```

```json
{
  "mcp.servers": {
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY"
      }
    }
  }
}
```

For local installation, see [Option 2](#option-2-local-development-stdio) below.
</details>

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

Run the MCP server locally for development, testing, or when you need full control over your data. The local server runs directly on your machine with STDIO transport for desktop clients or HTTP transport for web-based clients. Ideal for developers who want to customize the server or work in air-gapped environments.

### Manual Configuration
Add to your MCP client config:

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
        "WANDB_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Using uv (recommended)
uv pip install wandb-mcp-server

# Or from GitHub
pip install git+https://github.com/wandb/wandb-mcp-server
```

### Client-Specific Installation Commands

#### Cursor (Project-only)
Enable the server for a specific project:
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client --config_path .cursor/mcp.json && uvx wandb login
```

#### Cursor (Global)
Enable the server for all Cursor projects:
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client --config_path ~/.cursor/mcp.json && uvx wandb login
```

#### Windsurf
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client --config_path ~/.codeium/windsurf/mcp_config.json && uvx wandb login
```

#### Claude Code
```bash
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server && uvx wandb login
```

With API key:
```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```

#### Claude Desktop
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server add_to_client --config_path "~/Library/Application Support/Claude/claude_desktop_config.json" && uvx wandb login
```

### Testing with ngrok (for server-side clients)

For clients like OpenAI and LeChat that require public URLs:

```bash
# 1. Start HTTP server
uvx wandb-mcp-server --transport http --port 8080

# 2. Expose with ngrok
ngrok http 8080

# 3. Use the ngrok URL in your client configuration
```

> **Note**: These utilities are inspired by the OpenMCP Server Registry [add-to-client pattern](https://www.open-mcp.org/servers).
</details>

<details>
<summary><strong>Option 3: Self-Hosted HTTP Server</strong></summary>

Deploy your own W&B MCP server for team-wide access or custom infrastructure requirements. This option gives you complete control over deployment, security, and scaling while maintaining compatibility with all MCP clients. Perfect for organizations that need on-premises deployment or want to integrate with existing infrastructure.

### Using Docker

```bash
# Client-authenticated mode (recommended - no server API key needed)
docker run -p 7860:7860 ghcr.io/wandb/wandb-mcp-server
```

**Note**: HTTP deployment uses **client-authenticated mode**. Clients must provide their W&B API key via Bearer token. See [CLIENT_AUTH_MODE.md](CLIENT_AUTH_MODE.md) for details.

### From Source

```bash
# Clone repository
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server

# Install and run (no WANDB_API_KEY needed)
uv pip install -r requirements.txt
uv run app.py
```

### Deploy to HuggingFace Spaces

1. Fork [wandb-mcp-server](https://github.com/wandb/wandb-mcp-server)
2. Create new Space on [Hugging Face](https://huggingface.co/spaces)
3. Choose "Docker" SDK
4. Connect your fork
5. Deploy (no WANDB_API_KEY secret needed - clients provide their own)

Server URL: `https://YOUR-SPACE.hf.space/mcp`

**Authentication**: All clients must include `Authorization: Bearer <WANDB_API_KEY>` header in requests.

### Deploy to Google Cloud Run

For production deployments with enhanced security (HMAC-SHA256 session management):

```bash
# Use the automated deployment script
./deploy.sh
```

The script handles:
- ✅ Service account configuration
- ✅ HMAC-SHA256 API key hashing
- ✅ Multi-tenant session isolation
- ✅ Automatic deployment logging
- ✅ Health check verification

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed documentation.
</details>

---

## More Information

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
