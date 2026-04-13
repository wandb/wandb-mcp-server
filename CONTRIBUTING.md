# Contributing to the W&B MCP Server

## Development Setup

```bash
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server

# Create virtual environment
uv venv .venv --python 3.12
uv pip install -e ".[test,http]"

# Install pre-commit hooks (ruff check + format on staged files)
pre-commit install
```

## Running Tests

All unit tests are mock-based -- no API keys or network calls required.

```bash
# Run all tests
uv run pytest tests/ -v --tb=short

# Run a specific test file
uv run pytest tests/test_weave_api.py -v

# Run with coverage
uv run pytest tests/ --cov=src/wandb_mcp_server
```

## Linting

Ruff config is in `pyproject.toml`. Do not pass `--select` or `--ignore` on the CLI.

```bash
# Check
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Auto-fix
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

Pre-commit hooks run these automatically on staged files.

## Running Locally

### STDIO mode (for desktop clients)

```bash
export WANDB_API_KEY=your-key
uv run wandb_mcp_server
```

### HTTP mode (for testing)

```bash
export WANDB_API_KEY=your-key
uv run wandb_mcp_server --transport http --port 8080
```

Then test with curl:

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

## Branch Naming

- `feat/mcp{N}-{short-name}` for features linked to Jira tickets
- `fix/{description}` for bug fixes
- `chore/{description}` for maintenance
- Always branch from `main`

## PR Process

1. Create a branch from `main`
2. Make changes, ensure tests pass locally
3. Push and create a PR targeting `main`
4. CI runs automatically (ruff + pytest on Python 3.11 + 3.12)
5. Get review from `@NiWaRe` or another team member
6. Merge via squash merge

### PR Title Format

```
feat(MCP-{N}): short description
fix(MCP-{N}): short description
chore(MCP-{N}): short description
```

### PR Body

Include: summary, changes, Jira link, test plan.

## Architecture

```
src/wandb_mcp_server/
├── server.py              # FastMCP app, tool registration, CLI
├── auth.py                # Bearer token validation, session binding
├── api_client.py          # WandBApiManager (per-request API keys)
├── session_manager.py     # Multi-tenant session management
├── analytics.py           # Cloud Logging + Segment events
├── config.py              # WANDB_BASE_URL, token budgets
├── mcp_tools/             # 14 MCP tool implementations
│   ├── query_weave.py     # Trace querying with token budget
│   ├── query_wandb_gql.py # GraphQL with auto-pagination
│   ├── create_report.py   # W&B Reports with panels
│   ├── query_artifacts.py # Artifact list/get/compare
│   ├── query_registry.py  # Registry + collection listing
│   └── ...
└── weave_api/             # HTTP client for Weave trace server
    ├── service.py         # TraceService orchestrator
    ├── processors.py      # Token estimation + truncation
    ├── query_builder.py   # Filter -> trace server query
    └── client.py          # HTTP client (Basic auth)
```

## Adding a New Tool

1. Create `src/wandb_mcp_server/mcp_tools/your_tool.py`
2. Define the tool function with a descriptive docstring
3. Create a `TOOL_DESCRIPTION` constant with proactive guidance
4. Register it in `server.py` via `register_tools()`
5. Add unit tests in `tests/test_your_tool.py`
6. Update the tool count in `README.md`

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `WANDB_API_KEY` | W&B authentication (STDIO mode) | From netrc |
| `WANDB_BASE_URL` | W&B API endpoint | `https://api.wandb.ai` |
| `WF_TRACE_SERVER_URL` | Weave trace server | `https://trace.wandb.ai` |
| `MAX_RESPONSE_TOKENS` | Token budget for truncation | `30000` |
| `MCP_SERVER_LOG_LEVEL` | Log level | `WARNING` |

## Deployment

This repo contains only the MCP server logic. Deployment (Docker, Cloud Run, Helm) lives in the private `wandb/wandb-mcp-server-test` repo. See [RELEASING.md](RELEASING.md) for the release process.
