<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/wandb/wandb/main/assets/logo-light.svg" width="600" alt="Weights & Biases">
  </picture>
</p>

# Weights & Biases MCP Server

A Model Context Protocol (MCP) server that provides seamless access to [Weights & Biases](https://www.wandb.ai/) for ML experiments and agent applications.

## Example Use Cases

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

## Installation

These are instructions to run the local MCP server - we're working on a hosted version that is coming out soon too!

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

The server includes [wandbot](https://github.com/wandb/wandbot) support for answering W&B/Weave questions. See `env.example` for all available configuration options including custom endpoints.

### MCP Client Setup

Choose your MCP client from the options below:

<details>
<summary><b>🖱️ Cursor IDE</b></summary>

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
<summary><b>🌊 Windsurf IDE</b></summary>

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
<summary><b>💬 Gemini</b></summary>
**Quick Install:**
Uses the `.gemini-extension.json` in this repo's root:

```bash
gemini extensions install https://github.com/wandb/wandb-mcp-server
```
<details>
<summary>Manual Configuration</summary>
Create `gemini-extension.json` in your project root (use `--path=path/to/gemini-extension.json` to add local folder):

```json
{
    "name": "Weights and Biases MCP Server",
    "version": "0.1.0",
    "mcpServers": {
        "wandb": {
            "command": "uv",
            "args": [
                "run", 
                "--directory",
                "/path/to/wandb-mcp-server",
                "wandb_mcp_server",
                "--transport",
                "stdio"
            ],
            "env": {
                "WANDB_API_KEY": "$WANDB_API_KEY"
            }
        }
    }
}
```
</details>

Note: Replace `/path/to/wandb-mcp-server` with your installation path.
</details>

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
<summary><b>💻 Claude Code</b></summary>

**Quick Install:**
```bash
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server && uvx wandb login
```

**With API Key:**
```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```
</details>

<details>
<summary><b>🌐 ChatGPT, LeChat, Claude</b></summary>
Try our hosted public version: [HF Spaces](https://huggingface.co/spaces/NiWaRe/wandb-mcp-server)

This version allows you to configure your WANDB_API_KEY directly in the interface to access your own projects or to work with all publich projects otherwise. Follow the instructions in the space to add it to LeChat, ChatGPT, or Claude. We'll have an official hosted version soon.
</details>

## Available Tools

The server provides the following MCP tools:

### W&B Models Tools
- **`query_wandb_tool`** - Execute GraphQL queries against W&B experiment tracking data (runs, sweeps, artifacts)

### Weave Tools  
- **`query_weave_traces_tool`** - Query LLM traces and evaluations with filtering and pagination
- **`count_weave_traces_tool`** - Efficiently count traces without returning data

### Support & Reporting
- **`query_wandb_support_bot`** - Get help from [wandbot](https://github.com/wandb/wandbot), our RAG-powered technical support agent that can answer any W&B/Weave questions, help debug issues, and provide code examples
- **`create_wandb_report_tool`** - Create W&B Reports with markdown and visualizations
- **`query_wandb_entity_projects`** - List available entities and projects

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

#### Stdio Transport (Default)
For local and server being on the same machine:
```bash
wandb_mcp_server --transport stdio
```

#### HTTP Transport with SSE
For remote access or web applications:
```bash
wandb_mcp_server --transport http --host 0.0.0.0 --port 8080
```

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