---
title: Weights & Biases MCP Server
emoji: 🪄🐝
colorFrom: yellow
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>

# Weights & Biases MCP Server

A Model Context Protocol (MCP) server that provides seamless access to [Weights & Biases](https://www.wandb.ai/) for ML experiments and agent applications.

## Quick Install Buttons

### IDEs & Editors
[![Install in Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/en/install-mcp?name=wandb&config=eyJ0cmFuc3BvcnQiOiJodHRwIiwidXJsIjoiaHR0cHM6Ly9tY3Aud2l0aHdhbmRiLmNvbS9tY3AiLCJoZWFkZXJzIjp7IkF1dGhvcml6YXRpb24iOiJCZWFyZXIge3tXQU5EQl9BUElfS0VZfX0iLCJBY2NlcHQiOiJhcHBsaWNhdGlvbi9qc29uLCB0ZXh0L2V2ZW50LXN0cmVhbSJ9fQ%3D%3D)
[![Install in VSCode](https://img.shields.io/badge/Install%20in-VSCode-blue?style=for-the-badge&logo=visualstudiocode)](#vscode-hosted-server)
[![Install in Windsurf](https://img.shields.io/badge/Install%20in-Windsurf-green?style=for-the-badge&logo=windsurf)](#windsurf-ide-hosted-server)

### AI Coding Agents
[![Install in Claude Code](https://img.shields.io/badge/Install%20in-Claude%20Code-orange?style=for-the-badge&logo=anthropic)](#claude-code-hosted)
[![Install in Gemini CLI](https://img.shields.io/badge/Install%20in-Gemini%20CLI-purple?style=for-the-badge&logo=google)](#gemini-hosted-server)
[![Setup GitHub Copilot](https://img.shields.io/badge/Setup-GitHub%20Copilot-black?style=for-the-badge&logo=github)](#github-codex)

### AI Chat Clients
[![Install in ChatGPT](https://img.shields.io/badge/Install%20in-ChatGPT-teal?style=for-the-badge&logo=openai)](#chatgpt-hosted-server)
[![Install in LeChat](https://img.shields.io/badge/Install%20in-LeChat-red?style=for-the-badge&logo=mistral)](#mistral-lechat-hosted-server)
[![Install in Claude Desktop](https://img.shields.io/badge/Install%20in-Claude%20Desktop-orange?style=for-the-badge&logo=anthropic)](#claude-desktop-hosted-server)
[![Other Web Clients](https://img.shields.io/badge/Other-Web%20Clients-gray?style=for-the-badge&logo=web)](#other-web-clients)

> **Quick Setup:** Click the button for your client above. For Cursor, it auto-installs with one click. For others, you'll be taken to the setup instructions. Just replace `YOUR_WANDB_API_KEY` with your actual API key from [wandb.ai/authorize](https://wandb.ai/authorize).


## Example Use Cases

<details>
<summary><b>📋 Available MCP Tools & Descriptions</b></summary>

### W&B Models Tools

**`query_wandb_tool`** - Execute GraphQL queries against W&B experiment tracking data (runs, sweeps, artifacts)
- Query experiment runs, metrics, and performance comparisons
- Access artifact management and model registry data
- Analyze hyperparameter optimization and sweeps
- Retrieve project dashboards and reports data
- Supports pagination with `max_items` and `items_per_page` parameters
- Accepts custom GraphQL queries with variables

### Weave Tools (LLM/GenAI)

**`query_weave_traces_tool`** - Query LLM traces and evaluations with advanced filtering and pagination
- Retrieve execution traces and paths of LLM operations
- Access LLM inputs, outputs, and intermediate results
- Filter by display name, operation name, trace ID, status, time range, latency
- Sort by various fields (started_at, latency, cost, etc.)
- Support for metadata-only queries to avoid context window overflow
- Includes cost calculations and token usage analysis
- Configurable data truncation and column selection

**`count_weave_traces_tool`** - Efficiently count traces without returning full data
- Get total trace counts and root trace counts
- Apply same filtering options as query tool
- Useful for understanding project scope before detailed queries
- Returns storage size information in bytes
- Much faster than full trace queries when you only need counts

### Support & Knowledge

**`query_wandb_support_bot`** - Get help from [wandbot](https://github.com/wandb/wandbot)
- RAG-powered technical support agent for W&B/Weave questions
- Provides code examples and debugging assistance
- Covers experiment tracking, Weave tracing, model management
- Explains W&B features, best practices, and troubleshooting
- Works out-of-the-box with no configuration needed

### Reporting & Documentation

**`create_wandb_report_tool`** - Create shareable W&B Reports with markdown and visualizations
- Generate reports with markdown text and HTML-rendered charts
- Support for multiple chart sections with proper organization
- Interactive visualizations with hover effects and SVG elements
- Permanent, shareable documentation for analysis findings
- Accepts both single HTML strings and dictionaries of multiple charts

### Discovery & Navigation

**`query_wandb_entity_projects`** - List available entities and projects
- Discover accessible W&B entities (teams/usernames) and their projects
- Get project metadata including descriptions, visibility, tags
- Essential for understanding available data sources
- Helps with proper entity/project specification in queries
- Returns creation/update timestamps and project details

</details>

### 1. 🔍 Analyze ML Experiments
```
"Show me the top 5 runs with the highest accuracy from my wandb-smle/hiring-agent-demo-public project and create a report comparing their hyperparameters"
```
The MCP server queries W&B runs, compares metrics, and generates a shareable report with visualizations.

### 2. 🐛 Debug LLM Applications  
```
"Find all failed OpenAI chat traces in my weave project from the last 24 hours and analyze their error patterns"
```
The server retrieves Weave traces, filters by status, and provides detailed error analysis for debugging.

### 3. 📊 Evaluate Model Performance
```
"Compare the F1 scores across all evaluations in my RAG pipeline and identify which prompts performed best"
```
The server queries Weave evaluations, aggregates scores, and highlights top-performing configurations.

### 4. 🤖 Get Expert Help with W&B/Weave
```
"How do I implement custom metrics in Weave evaluations? Show me an example with async scorers"
```
The integrated [wandbot](https://github.com/wandb/wandbot) support agent provides detailed answers, code examples, and debugging assistance for any W&B or Weave-related questions.

## Installation & Deployment

This MCP server can be deployed in three ways. **We recommend starting with the hosted server** for the easiest setup experience.

### 🌐 Option 1: Hosted Server (Recommended - No Installation Required)

Use our publicly hosted server on Hugging Face Spaces - **zero installation needed!**

**Server URL:** `https://mcp.withwandb.com/mcp`

> **ℹ️ Quick Setup:** Click the button for your client above, then use the configuration examples in the sections below. Just replace `YOUR_WANDB_API_KEY` with your actual API key from [wandb.ai/authorize](https://wandb.ai/authorize).

### 💻 Option 2: Local Development (STDIO)

Run the server locally with direct stdio communication - best for development and testing.

### 🔌 Option 3: Self-Hosted HTTP Server

Deploy your own HTTP server with API key authentication - great for team deployments or custom infrastructure.

---

## Hosted Server Setup (Recommended)

**No installation required!** Just configure your MCP client to connect to our hosted server.

### Get Your W&B API Key

Get your Weights & Biases API key at: [https://wandb.ai/authorize](https://wandb.ai/authorize)

### Configuration by Client Type

Choose your MCP client below for easy hosted server setup. All configurations use the same hosted server URL: `https://mcp.withwandb.com/mcp`

#### IDEs & Code Editors

<details>
<summary><b>Cursor IDE (Hosted Server)</b></summary>

**Quick Setup:**
1. Open Cursor settings → MCP
2. Add the configuration below
3. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)
4. Restart Cursor

**Configuration for `.cursor/mcp.json` or `~/.cursor/mcp.json`:**

```json
{
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

✅ **That's it!** No installation, no dependencies, just configuration.
</details>

<details>
<summary><b id="windsurf-ide-hosted-server">Windsurf IDE (Hosted Server)</b></summary>

**Quick Setup:**
1. Open Windsurf settings → MCP
2. Add the configuration below
3. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)
4. Restart Windsurf

**Configuration for `mcp_config.json`:**

```json
{
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

✅ **That's it!** No installation required.
</details>

<details>
<summary><b id="vscode-hosted-server">VSCode (Hosted Server)</b></summary>

**Quick Setup:**
1. Create a `.vscode/mcp.json` file in your project root
2. Add the configuration below
3. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)
4. Restart VSCode or reload the window

**Configuration for `.vscode/mcp.json`:**

```json
{
  "servers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

✅ **That's it!** No installation required.
</details>

#### AI Coding Agents

<details>
<summary><b id="claude-code-hosted">Claude Code (Hosted Server)</b></summary>

**Quick Setup:**
1. Install Claude Code if you haven't already
2. Configure the MCP server with HTTP transport:
   ```bash
   claude mcp add wandb \
     --transport http \
     --url https://mcp.withwandb.com/mcp \
     --header "Authorization: Bearer YOUR_WANDB_API_KEY" \
     --header "Accept: application/json, text/event-stream"
   ```
3. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)

**Alternative: Manual Configuration**

Edit your Claude Code MCP config file:
```json
{
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

✅ **That's it!** No local installation required.
</details>

<details>
<summary><b id="github-codex">GitHub Copilot/Codex (Hosted Server)</b></summary>

**Quick Setup:**

GitHub Copilot doesn't directly support MCP servers, but you can use the W&B API through code comments:

1. Install the W&B Python SDK in your project:
   ```bash
   pip install wandb
   ```

2. Use Copilot to generate W&B code by adding comments like:
   ```python
   # Log metrics to wandb project my-project
   # Query the last 10 runs from wandb
   ```

**Note:** For direct MCP integration, consider using Cursor or VSCode with MCP extensions.
</details>

<details>
<summary><b id="gemini-hosted-server">Gemini CLI (Hosted Server)</b></summary>

**Quick Setup:**
1. Create a `gemini-extension.json` file in your project:

```json
{
  "name": "wandb-mcp-server",
  "version": "0.1.0",
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

2. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)

3. Install the extension:
   ```bash
   gemini extensions install --path .
   ```

✅ **That's it!** No installation required.
</details>

#### AI Chat Clients

<details>
<summary><b id="chatgpt-hosted-server">ChatGPT (Actions)</b></summary>

**Quick Setup:**

To use the W&B MCP Server with ChatGPT, create a Custom GPT with Actions:

1. Go to [ChatGPT](https://chat.openai.com) → Explore GPTs → Create
2. In the "Actions" section, click "Create new action"
3. Configure Authentication:
   - **Authentication Type**: API Key
   - **Auth Type**: Bearer
   - **API Key**: `YOUR_WANDB_API_KEY`

3. Add the OpenAPI schema:

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "W&B MCP Server",
    "version": "1.0.0",
    "description": "Access W&B experiment tracking and Weave traces"
  },
  "servers": [
    {
      "url": "https://mcp.withwandb.com"
    }
  ],
  "paths": {
    "/mcp": {
      "post": {
        "operationId": "callTool",
        "summary": "Execute W&B MCP tools",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": ["tool", "params"],
                "properties": {
                  "tool": {
                    "type": "string",
                    "description": "The MCP tool to call"
                  },
                  "params": {
                    "type": "object",
                    "description": "Parameters for the tool"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

4. Test the action and publish your Custom GPT

✅ **That's it!** ChatGPT can now access W&B data through Actions.
</details>

<details>
<summary><b id="mistral-lechat-hosted-server">Mistral LeChat (Hosted Server)</b></summary>

**Quick Setup:**
1. Go to LeChat Settings → Custom MCP Connectors
2. Click "Add MCP Connector"
3. Configure with:
   - **Server URL**: `https://mcp.withwandb.com/mcp`
   - **Authentication**: Choose "API Key Authentication"
   - **Token**: Enter your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize)

✅ **That's it!** No installation required.
</details>

<details>
<summary><b id="claude-desktop-hosted-server">Claude Desktop (Hosted Server)</b></summary>

**Quick Setup:**
1. [Download Claude Desktop](https://claude.ai/download) if you haven't already
2. Open Claude Desktop
3. Go to Settings → Features → Model Context Protocol
4. Add the configuration below
5. Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)
6. Restart Claude Desktop

**Configuration for `claude_desktop_config.json`:**

```json
{
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://mcp.withwandb.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

✅ **That's it!** No installation required.
</details>

<details>
<summary><b id="other-web-clients">Other Web Clients</b></summary>

**Quick Setup:**
1. Use our hosted public version: [HF Spaces](https://wandb-wandb-mcp-server.hf.space)
2. Configure your `WANDB_API_KEY` directly in the interface
3. Follow the instructions in the space to add it to your preferred client

This version allows you to access your own projects with your API key or work with all public projects otherwise.

✅ **That's it!** No installation required.
</details>

---

## 💻 Local Installation (Advanced Users)

If you prefer to run the MCP server locally or need custom configurations, follow these instructions.

### Prerequisites

#### 1. Install UV Package Manager

UV is required to run the MCP server. Install it using one of these methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**macOS (Homebrew):**
```bash
brew install uv
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Get Your W&B API Key

You'll need a Weights & Biases API key. Get yours at: [https://wandb.ai/authorize](https://wandb.ai/authorize)

Configure your API key using one of these methods (first one recommended to have the other default parameters too):

1. **`.env` file** in your project (copy from `env.example`):
   ```bash
   cp env.example .env
   # Edit .env and add your API key
   ```

2. **`.netrc` file**:
   ```bash
   uvx wandb login
   ```

3. **Environment variable** (recommended):
   ```bash
   export WANDB_API_KEY=your-api-key
   ```

4. **Command-line argument**:
   ```bash
   wandb_mcp_server --wandb-api-key your-api-key
   ```

#### 3. Environment Configuration (Optional)

The server includes [wandbot](https://github.com/wandb/wandbot) support for answering W&B/Weave questions. **wandbot works out-of-the-box without any configuration!** It uses the default public endpoint automatically.

See `env.example` for optional configuration like custom wandbot instances or other advanced settings.

### MCP Client Setup for Local Server

Choose your MCP client from the options below for local server setup:

<details>
<summary><b>Cursor IDE</b></summary>

**Quick Install (Project-specific):**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path .cursor/mcp.json && uvx wandb login
```

**Quick Install (Global):**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.cursor/mcp.json && uvx wandb login
```

<details>
<summary>Manual Configuration</summary>

Add to `.cursor/mcp.json` or `~/.cursor/mcp.json`:

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
        "WANDB_API_KEY": "your-api-key"
      }
    }
  }
}
```
</details>
</details>

<details>
<summary><b>Windsurf IDE</b></summary>

**Quick Install:**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.codeium/windsurf/mcp_config.json && uvx wandb login
```

<details>
<summary>Manual Configuration</summary>

Add to `~/.codeium/windsurf/mcp_config.json`:

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
        "WANDB_API_KEY": "your-api-key"
      }
    }
  }
}
```
</details>
</details>

<details>
<summary><b>Gemini</b></summary>
**Quick Install:**

1. Make sure to have your API key exported:

```bash
# Option 1: Export API key directly
export WANDB_API_KEY=your-api-key

# Option 2: Use wandb login (opens browser)
uvx wandb login
```

2. Then add the extension using the following command (based on the `gemini-extension.json` file)

```bash
gemini extensions install https://github.com/wandb/wandb-mcp-server
```

<details>
<summary>Manual Configuration</summary>
Create `gemini-extension.json` in your project root (use `--path=path/to/folder-with-gemini-extension.json` to add local folder):

```json
{
    "name": "wandb-mcp-server",
    "version": "0.1.0",
    "mcpServers": {
      "wandb": {
        "httpUrl": "https://mcp.withwandb.com/mcp",
        "trust": true,
        "headers": {
            "Authorization": "Bearer $WANDB_API_KEY",
            "Accept": "application/json, text/event-stream"
        }
      }
    }
  }
```

<details>
<summary><b>🤖 Claude Desktop</b></summary>

**Quick Install:**
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path "~/Library/Application Support/Claude/claude_desktop_config.json" && uvx wandb login
```

<details>
<summary>Manual Configuration</summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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
        "WANDB_API_KEY": "your-api-key"
      }
    }
  }
}
```
</details>
</details>

<details>
<summary><b id="claude-code">💻 Claude Code</b></summary>

**Quick Install:**
```bash
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server && uvx wandb login
```

**With API Key:**
```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```
</details>



## Usage Tips

### Be Specific About Projects
Always specify the W&B entity and project name in your queries:

✅ **Good:** "Show traces from wandb-team/my-project"  
❌ **Bad:** "Show my traces"

### Avoid Overly Broad Questions
Be specific to get better results:

✅ **Good:** "What eval had the highest F1 score in the last week?"  
❌ **Bad:** "What's my best evaluation?"

### Verify Complete Data Retrieval
When analyzing performance across multiple runs, ask the LLM to confirm it retrieved all available data to ensure comprehensive analysis.

## Self-Hosting Guide

### Deploy to Hugging Face Spaces

Deploy your own instance of the W&B MCP Server on Hugging Face Spaces:

1. **Fork this repository** or clone it locally
2. **Create a new Space on Hugging Face:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Set visibility as needed

3. **Push the code to your Space:**
   ```bash
   git remote add hf-space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf-space main
   ```

4. **Your server will be available at:**
   ```
   https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/mcp
   ```

See [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md) for detailed deployment instructions.

### Run Local HTTP Server

Run the server locally with HTTP transport for development or testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with authentication (recommended)
python app.py

# Or run without authentication (development only)
MCP_AUTH_DISABLED=true python app.py
```

The server will be available at `http://localhost:7860/mcp`

**Authentication:** See [AUTH_README.md](AUTH_README.md) for details on Bearer token authentication.

### File Structure for Deployment

```
wandb-mcp-server/
├── app.py                    # HF Spaces/HTTP server entry point
├── Dockerfile                # Container configuration for HF Spaces
├── requirements.txt          # Python dependencies for HTTP deployment
├── index.html               # Landing page for web interface
├── AUTH_README.md           # Authentication documentation
├── HUGGINGFACE_DEPLOYMENT.md # HF Spaces deployment guide
├── src/
│   └── wandb_mcp_server/
│       ├── server.py        # Core MCP server (STDIO & HTTP)
│       ├── auth.py          # Bearer token authentication
│       └── mcp_tools/       # Tool implementations
└── pyproject.toml           # Package configuration for local/pip install
```

## Advanced Configuration

### Enabling Weave Tracing for MCP Operations

Track all MCP tool calls using [Weave's MCP integration](https://weave-docs.wandb.ai/guides/integrations/mcp):

```bash
# Enable Weave tracing for MCP operations
export WEAVE_DISABLED=false
export MCP_LOGS_WANDB_ENTITY=your-entity
export MCP_LOGS_WANDB_PROJECT=mcp-logs

# Optional: trace list operations
export MCP_TRACE_LIST_OPERATIONS=true
```

This provides detailed observability into tool calls, resource access, and prompt generation across your MCP system.

### Logging Configuration

Control server logging with environment variables:

```bash
# Server log level
export MCP_SERVER_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# W&B/Weave output control
export WANDB_SILENT=False  # Show W&B output
export WEAVE_SILENT=False  # Show Weave output

# Debug mode
export WANDB_DEBUG=true    # Verbose W&B logging
```

### Transport Options

#### STDIO Transport (Default for Local Development)
For local development where the MCP client and server run on the same machine:
```bash
wandb_mcp_server --transport stdio
# Or with UV:
uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```
- Requires W&B API key in environment
- Direct communication via stdin/stdout
- Best for local IDE integrations (Cursor, Windsurf, etc.)

#### HTTP Transport (For Remote Access)
For remote access, web applications, or hosted deployments:
```bash
# Using the FastAPI app (recommended)
python app.py  # Runs on port 7860 by default

# Or using the CLI
wandb_mcp_server --transport http --host 0.0.0.0 --port 8080
```
- Clients provide W&B API key as Bearer token
- Supports authentication middleware
- Uses Server-Sent Events (SSE) for streaming
- Ideal for hosted deployments and web clients

### Running from Source

```bash
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server
wandb login
uv run src/wandb_mcp_server/server.py
```

## Troubleshooting

### Error: spawn uv ENOENT

If `uv` cannot be found:

1. Reinstall UV:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a system-wide symlink:
   ```bash
   sudo ln -s ~/.local/bin/uv /usr/local/bin/uv
   ```

3. Restart your application/IDE

### Authentication Issues

Verify W&B authentication:
```bash
uvx wandb login
```

Or check if your API key is set:
```bash
echo $WANDB_API_KEY
```

## Testing

Run integration tests with LLM providers:

```bash
# Set API key in .env
echo "ANTHROPIC_API_KEY=your-key" >> .env

# Run specific test file
uv run pytest -s -n 10 tests/test_query_wandb_gql.py

# Debug single test
pytest -s -n 1 "tests/test_query_weave_traces.py::test_query_weave_trace[sample_name]" -v --log-cli-level=DEBUG
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [W&B Documentation](https://docs.wandb.ai)
- [Weave Documentation](https://weave-docs.wandb.ai)  
- [GitHub Issues](https://github.com/wandb/wandb-mcp-server/issues)
- [W&B Community Forum](https://community.wandb.ai)