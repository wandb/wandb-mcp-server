# Weights & Biases MCP Server

## Repository Overview

This is a **Weights & Biases MCP Server** that provides LLM clients with access to W&B's ecosystem through standardized Model Context Protocol tools. It bridges two main W&B products:

1. **W&B Models** - ML experiment tracking, hyperparameter optimization, artifacts
2. **W&B Weave** - LLM/GenAI observability, tracing, and evaluation

## Tool Categories and Functionality

### Weave Tools (LLM/GenAI Focus)
- `query_weave_traces_tool` - Query LLM execution traces with filtering, sorting, pagination
- `count_weave_traces_tool` - Efficiently count traces without returning full data

### W&B Models Tools (ML Experiment Tracking)
- `query_wandb_gql_tool` - Execute GraphQL queries against experiment tracking data

### Support & Documentation
- `query_wandb_support_bot` - RAG-powered support via wandbot for W&B questions

### Output & Reporting
- `create_wandb_report_tool` - Create shareable reports with markdown and HTML visualizations

### Discovery & Navigation
- `query_wandb_entity_projects` - List available entities and projects

## Architecture Patterns

### Tool Registration Pattern
Tools are registered in `server.py` using FastMCP decorators:
```python
@mcp.tool(description=TOOL_DESCRIPTION)
async def tool_function(params...):
    # Tool implementation
    return result
```

### Directory Structure
```
src/wandb_mcp_server/
├── server.py                    # Main MCP server with tool registrations
├── mcp_tools/                   # Individual tool implementations
│   ├── query_weave.py          # Weave trace querying
│   ├── count_traces.py         # Trace counting
│   ├── query_wandb_gql.py      # GraphQL querying
│   ├── query_wandbot.py        # Support bot integration
│   └── tools_utils.py          # Shared utilities
├── weave_api/                   # Structured Weave API layer
│   ├── client.py               # HTTP client
│   ├── service.py              # Business logic layer
│   ├── models.py               # Pydantic data models
│   ├── processors.py           # Data processing
│   └── query_builder.py        # Query construction
├── tool_prompts.py             # Tool descriptions/prompts
├── report.py                   # Report creation logic
└── utils.py                    # Common utilities
```

### Key Architecture Patterns
1. **Layered Architecture** - Clear separation between API client, service layer, and tools
2. **Pydantic Models** - Strong typing with `QueryResult`, `WeaveTrace`, `TraceMetadata`
3. **Async/Await Support** - Modern async patterns for concurrent operations
4. **Retry Logic** - Built-in retry mechanisms for network resilience
5. **Configuration Management** - Environment-based config with `.env` support
6. **Comprehensive Logging** - Rich logging with configurable levels

### Tool Implementation Pattern
Each tool follows this structure:
```python
# 1. Tool description constant
TOOL_DESCRIPTION = """Detailed description..."""

# 2. Core implementation function
def core_function(params):
    # Business logic
    return result

# 3. MCP tool wrapper (in server.py)
@mcp.tool(description=TOOL_DESCRIPTION)
async def mcp_tool_wrapper(params):
    return core_function(params)
```

## Dependencies & Tech Stack

### Core Dependencies
- `mcp[cli]>=1.3.0` - Model Context Protocol framework
- `weave>=0.51.47` - W&B Weave SDK
- `wandb>=0.19.8` - W&B core SDK
- `httpx>=0.28.1` - Async HTTP client
- `pydantic` - Data validation and modeling

### Supporting Libraries
- `python-dotenv` - Environment variable management
- `simple-parsing` - CLI argument parsing
- `tiktoken` - Token counting
- `networkx` - Graph operations
- `requests` - HTTP requests with retry logic

### Testing Stack
- `pytest>=8.3.1` - Testing framework
- `pytest-xdist` - Parallel test execution
- `anthropic>=0.50.0` - LLM integration testing
- `litellm>=1.67.2` - Multi-LLM testing support

## Adding New Tools - Recommended Approach

1. **Create tool implementation** in `mcp_tools/new_tool.py`
2. **Define tool description** in `tool_prompts.py` 
3. **Add Pydantic models** if needed in `weave_api/models.py`
4. **Register tool** in `server.py` using `@mcp.tool` decorator
5. **Add tests** following existing patterns in `tests/`
6. **Use existing utilities** from `tools_utils.py` for common functionality

## Running and Testing

### Local Development
```bash
# Install dependencies
uv sync

# Run server
uv run wandb-mcp-server

# Run tests
uv run pytest
```

### Testing Commands
- `uv run pytest` - Run all tests
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run pytest -xvs` - Run with verbose output and stop on first failure


## Environment Configuration

Required environment variables:
- `WANDB_API_KEY` - W&B API key for accessing data
- `WEAVE_PROJECT_ID` - Weave project identifier

Optional environment variables:
- `MCP_SERVER_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

The codebase demonstrates excellent separation of concerns, comprehensive error handling, and scalable patterns for adding new W&B integrations.