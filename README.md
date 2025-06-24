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
- run python code in isolated E2B or Pyodide sandboxes for data analysis
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
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path .cursor/mcp.json --add_deno_path && uvx wandb login
```

### Cursor global
Enable the server for all Cursor projects, doesn't matter where this is run:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.cursor/mcp.json --add_deno_path && uvx wandb login
```

### Windsurf 

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path ~/.codeium/windsurf/mcp_config.json --add_deno_path && uvx wandb login
```

### Claude Code

```bash
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server --add_deno_path && uvx wandb login
```

Passing an environment variable to Claude Code, e.g. api key:

```bash
claude mcp add wandb -e WANDB_API_KEY=your-api-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
```

### Claude Desktop
First ensure `uv` is installed, you might have to use `homebrew` to install depite `uv` being available in your terminal. Then run the below:

```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client --config_path "~/Library/Application Support/Claude/claude_desktop_config.json" --add_deno_path && uvx wandb login
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
- **`query_weave_traces_tool`** Queries Weave traces with powerful filtering, sorting, and pagination options.
  Returns either complete trace data or just metadata to avoid overwhelming the LLM context window.

- **`count_weave_traces_tool`** Efficiently counts Weave traces matching given filters without returning the trace data.
  Returns both total trace count and root traces count to understand project scope before querying.

### 3. W&B Support agent
- **`query_wandb_support_bot`** Connect your client to [wandbot](https://github.com/wandb/wandbot), our RAG-powered support agent for general help on how to use Weigths & Biases products and features.

### 4. Python code sandbox
- **`execute_sandbox_code_tool`** Execute Python code in secure, isolated sandbox environments, either a hosted E2B sandbox or a local Pyodide sandbox, WebAssembly-based execution that uses Deno to isolate execution from the host system (inspired by [Pydantic AI's Run Python MCP](https://ai.pydantic.dev/mcp/run-python/)). See sandbox setup instructions above.

  **Sandbox Behavior:**
  - **E2B**: Maintains a single persistent sandbox instance during the MCP server session. Files written in one execution are available in subsequent executions. The sandbox is automatically terminated when the server stops. Default E2B sandbox lifetime is 15 minutes of inactivity (configurable via `E2B_SANDBOX_TIMEOUT_SECONDS`), but is kept alive by code executions.
  - **Pyodide**: Maintains a persistent Pyodide environment for the lifetime of the MCP server. Files written in one execution are available in subsequent executions. The Pyodide process is initialized when the server starts and terminates when the server stops.
  
  **File Operations:**
  - Both sandboxes support standard Python file I/O operations
  - Query results from `query_wandb_tool` and `query_weave_traces_tool` can be automatically saved as json files in the sandbox if the LLM passes a filename to `save_filename` to the tool call
  - Use the `save_filename` parameter to save results: `save_filename="my_data.json"`
  - Files are saved to `/tmp/` directory in the sandbox

### 5. Saving Analysis
- **`create_wandb_report_tool`** Creates a new W&B Report with markdown text and HTML-rendered visualizations.
  Provides a permanent, shareable document for saving analysis findings and generated charts.

### 6. General W&B helpers
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

### Code sandbox setup (optional)

The wandb MCP server exposes a secure, isolated python code sandbox tool to the client to let it send code (e.g. pandas) for additional data analysis to be run on queried W&B data. 

**Option 1: Local Pyodide sandbox - Install Deno**

The local Pyodide sandbox uses Deno to run Python in a WebAssembly environment, providing secure isolation from the host system. This option is automatically used if Deno is installed and no E2B API key is found.

```bash
# One-line install for macOS/Linux:
curl -fsSL https://deno.land/install.sh | sh -s -- -y

# Add Deno to your PATH (if not done automatically):
echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc  # for bash
echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.zshrc   # for zsh
source ~/.bashrc  # or ~/.zshrc

# Or on Windows (PowerShell):
irm https://deno.land/install.ps1 | iex
```

After installation, verify Deno is available:
```bash
# Restart your terminal or source your shell config
source ~/.bashrc  # or ~/.zshrc

# Verify installation
deno --version
```

If `deno --version` doesn't work, you may need to manually add Deno to your PATH:
```bash
echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc  # for bash
echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.zshrc   # for zsh
source ~/.bashrc  # or ~/.zshrc
```

Note, first execution may take longer as Pyodide downloads required packages

**Option 2: Hosted E2B sandbox - Set E2B api key**

The sandbox tool will use E2B if an E2B API key is detected. E2B provides persistent cloud VMs with full Python environment:

1. Sign up to E2B at [e2b.dev](https://e2b.dev)
2. Get your API key from the E2B dashboard
3. Set the `E2B_API_KEY` environment variable in the client settings.json

- To explicitly disable the sandbox tool completely, set `DISABLE_CODE_SANDBOX=1` environment variable


### Writing environment variables to the config file

The `add_to_client` function accepts a number of flags to enable writing optional environment variables to the server's config file. Below is an example of using the built-in convenience flag, `--e2b_api_key`, as well as setting other env variables that don't have dedicated flags.

```bash
# Write the server config file with additional env vars
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client \
  --config_path ~/.codeium/windsurf/mcp_config.json \
  --e2b_api_key 12345abcde \
  --add_deno_path \
  --write_env_vars MCP_LOGS_WANDB_ENTITY=my_wandb_entity E2B_PACKAGE_ALLOWLIST=numpy,pandas

# Then login to W&B
uvx wandb login
```

Arguments passed to `--write_env_vars` must be space separated and the key and value of each env variable must be separated only by a `=`.

### Running from Source

Run the server from source by running the below in the root dir:

```bash
wandb login && uv run src/wandb_mcp_server/server.py
```

### Environment Variables

The full list of environment variables used to control the server's settings can be found in the `.env.example` file.


## Sandbox Configuration (Optional)

You can configure sandbox behavior using environment variables:

#### Disable Sandbox
- `DISABLE_CODE_SANDBOX`: Set to any value to completely disable the code sandbox tool (e.g., `DISABLE_CODE_SANDBOX=1`)

#### Deno path
Use the `--add_deno_path` when using the `add_to_client` helper to automatically add Deno to your MCP configuration's PATH if Deno is installed but not in your system PATH. The flag automatically detects Deno installations from common locations including:
- **macOS/Linux**: `~/.deno/bin`, Homebrew (`/opt/homebrew/bin`, `/usr/local/bin`), MacPorts, system packages, Snap, Flatpak, asdf, vfox, Nix, Cargo, npm global
- **Windows**: `~/.deno/bin`, Scoop, Chocolatey, Winget, npm global, system locations

#### Package Installation Security
Control which packages can be installed in E2B sandboxes:
- `E2B_PACKAGE_ALLOWLIST`: Comma-separated list of allowed packages (e.g., `numpy,pandas,matplotlib`)
- `E2B_PACKAGE_DENYLIST`: Comma-separated list of denied packages (default includes potentially dangerous packages)

#### Cache Settings
- `E2B_CACHE_TTL_SECONDS`: Execution cache TTL in seconds (default: 900 = 15 minutes)

#### E2B Sandbox Lifetime
- `E2B_SANDBOX_TIMEOUT_SECONDS`: Sandbox lifetime in seconds (default: 900 = 15 minutes)
  - The sandbox will automatically shut down after this timeout if no code is executed
  - Each code execution resets the timeout
  - Example: `E2B_SANDBOX_TIMEOUT_SECONDS=600` for 10-minute timeout

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

### Code Sandbox Issues

If the code execution tool is not available or failing, here's how to diagnose and fix common issues:

<details>
<summary><strong>Deno/Pyodide Sandbox Issues</strong></summary>

**Problem:** Deno is installed but not detected by the MCP server

**Symptoms:**
- You can run `deno --version` in your terminal
- But MCP server logs show "Pyodide not available (Deno not installed)"

**Root Cause:** Deno is not properly installed or not in the system PATH

**Solutions:**

1. **Install Deno correctly:**
   ```bash
   curl -fsSL https://deno.land/install.sh | sh -s -- -y
   ```

2. **Add to your shell configuration:**
   ```bash
   echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc  # for bash
   echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.zshrc   # for zsh
   source ~/.bashrc  # or ~/.zshrc
   ```

3. **Restart your MCP client** (Claude Desktop, Cursor, etc.) to pick up the new PATH

4. **Verify your setup:**
   ```bash
   # Check if Deno is installed
   ls -la ~/.deno/bin/deno
   
   # Test Deno works
   deno --version
   
   # Test in a fresh shell
   zsh -c "deno --version"  # or bash -c "deno --version"
   ```

**Quick Fix:** If you have Deno installed but it's not being detected, try using the automatic PATH fix with the `add_deno_path` flag:
```bash
uvx --from git+https://github.com/wandb/wandb-mcp-server -- add_to_client \
  --config_path /path/to/your/mcp_config.json \
  --add_deno_path
```

This automatically searches for Deno in common installation locations across different package managers and installation methods.

**Alternative:** If PATH issues persist, you can manually specify the environment in your MCP client config:
```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"],
      "env": {
        "WANDB_API_KEY": "your-key",
        "PATH": "/Users/yourusername/.deno/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

3. **Test the detection directly:**
   ```bash
   # Test if Python can detect Deno the same way the MCP server does
   python3 -c "
   import subprocess
   try:
       result = subprocess.run(['deno', '--version'], capture_output=True, text=True, timeout=5)
       print(f'Success: {result.returncode == 0}')
       print(f'Output: {result.stdout}')
   except FileNotFoundError:
       print('Deno not found in PATH')
   except Exception as e:
       print(f'Error: {e}')
   "
   ```

**Problem:** Deno installation fails or PATH issues persist

**Solutions:**
- Reinstall Deno with automatic PATH setup:
  ```bash
  curl -fsSL https://deno.land/install.sh | sh -s -- -y
  source ~/.bashrc  # or ~/.zshrc
  ```
 - Verify installation location:
   ```bash
   ls -la ~/.deno/bin/deno
   ```

</details>

<details>
<summary><strong>E2B Sandbox Issues</strong></summary>

**Problem:** E2B sandbox not available

**Check:** Verify your API key is set:
```bash
echo $E2B_API_KEY
```

**Solutions:**
1. Get an API key from [e2b.dev](https://e2b.dev)
 2. Set it in your environment or MCP configuration
 3. Restart your MCP server

</details>

<details>
<summary><strong>General Debugging Steps</strong></summary>

1. **Check MCP server startup logs** for sandbox availability messages
2. **Verify environment variables** are properly set in your MCP configuration
3. **Test sandbox detection** using the Python snippet above
 4. **Restart your MCP client** (Claude Desktop, Cursor, etc.) after making configuration changes
 5. **Check for conflicting installations** - ensure you don't have multiple Deno installations

</details>

<details>
<summary><strong>Still Having Issues?</strong></summary>

If sandbox detection still fails:
1. Check the exact error message in your MCP server logs
2. Verify your shell configuration files (`.zshrc`, `.bashrc`) are properly formatted
 3. Try running the MCP server directly from terminal to see if PATH issues persist
 4. Consider using E2B as an alternative if Deno setup is problematic

</details>


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

#### Sandbox tests

Run sandbox-specific tests:

```bash
# Unit tests (with mocking, no real sandboxes needed)
uv run pytest tests/test_sandbox_execution.py -v

# Integration tests (requires E2B_API_KEY or Deno)
uv run pytest tests/test_sandbox_integration.py -v

# Run all sandbox tests
uv run pytest tests/test_sandbox*.py -v
```

For E2B tests, ensure `E2B_API_KEY` is set in your environment or `.env` file.
For Pyodide tests, ensure Deno is installed and available in your PATH.
