# Hugging Face Spaces Deployment Guide

This repository is configured for deployment on Hugging Face Spaces as a Model Context Protocol (MCP) server for Weights & Biases.

## Architecture

The application runs as a FastAPI server on port 7860 (HF Spaces default) with:
- **Main landing page**: `/` - Serves the index.html with setup instructions
- **Health check**: `/health` - Returns server status and W&B configuration
- **MCP endpoint**: `/mcp` - Streamable HTTP transport endpoint for MCP
  - Server can intelligently decide to return plan plan JSON or a SSE stream (the client always requests in the same way, see below)
  - Requires `Accept: application/json, text/event-stream` header
  - Supports initialize, tools/list, tools/call methods

More information on the details of [streamable http](https://modelcontextprotocol.io/specification/draft/basic/transports#streamable-http) are in the official docs and [this PR](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206).

## Key Changes for HF Spaces

### 1. app.py
- Creates a FastAPI application that serves the landing page
- Mounts FastMCP server using `mcp.streamable_http_app()` pattern (following [example from Mistral here](https://huggingface.co/spaces/Jofthomas/Multiple_mcp_fastapi_template))
- Uses lifespan context manager for session management
- Configured to run on `0.0.0.0:7860` (HF Spaces requirement)
- Sets W&B cache directories to `/tmp` to avoid permission issues

### 2. server.py
- Exports necessary functions for HF Spaces initialization
- Support for being imported as a module
- Maintains backward compatibility with CLI usage

### 3. Dependencies
- FastAPI and uvicorn as main dependencies
- All dependencies listed in requirements.txt for HF Spaces

### 4. Lazy Loading Fix
- `TraceService` initialization in `query_weave.py` to use lazy loading
- This allows the server to start even without a W&B API key (when first adding in LeChat for example without connecting)
- The service is only initialized when first needed

## Environment Variables

No environment variables are required! The server works without any configuration.

**Note**: Users provide their own W&B API keys as Bearer tokens. No server configuration needed (see AUTH_README.md).

## Deployment Steps

1. **Create a new Space on Hugging Face**
   - Choose "Docker" as the SDK
   - Set visibility as needed

2. **Configure Secrets**
   - Go to Settings → Variables and secrets
   - Add `MCP_SERVER_URL` as a variable for the URL to be correctly

3. **Push the Code**
   ```bash
   git add .
   git commit -m "Configure for HF Spaces deployment"
   git push
   ```

4. **Connect to the MCP Server**
   - Use the endpoint: `https://[your-username]-[space-name].hf.space/mcp`
   - Configure your MCP client with this URL and "streamable-http" transport

## File Structure

```
.
├── app.py              # HF Spaces entry point
├── index.html          # Landing page
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Package configuration
└── src/
    └── wandb_mcp_server/
        ├── server.py   # MCP server implementation
        └── ...         # Tool implementations
```

## Testing Locally

To test the HF Spaces configuration locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export WANDB_API_KEY=your_key_here

# Run the server
python app.py
```

The server will start on http://localhost:7860

## MCP Architecture & Key Learnings

### Understanding MCP and FastMCP

The Model Context Protocol (MCP) is a protocol for communication between AI assistants and external tools/services. Through our experimentation, we discovered several important aspects:

#### 1. FastMCP Framework
- **FastMCP** is a Python framework that simplifies MCP server implementation
- It provides decorators (`@mcp.tool()`) for easy tool registration
- Internally uses Starlette for HTTP handling
- Supports multiple transports: stdio, SSE, and streamable HTTP

#### 2. Streamable HTTP Transport
The streamable HTTP transport (introduced in [MCP PR #206](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206)) is the modern approach for remote MCP:

- **Single endpoint** (`/mcp`) handles all communication
- **Dual mode operation**:
  - Regular POST requests for stateless operations
  - SSE (Server-Sent Events) upgrade for streaming responses
- **Key advantages**:
  - Stateless servers possible (no persistent connections required)
  - Better infrastructure compatibility ("just HTTP")
  - Supports both request-response and streaming patterns

#### 3. Implementation Patterns

##### The HuggingFace Pattern
Based on the [reference implementation](https://huggingface.co/spaces/Jofthomas/Multiple_mcp_fastapi_template), the correct pattern is:

```python
# Create MCP server
mcp = FastMCP("server-name")

# Register tools
@mcp.tool()
def my_tool(): ...

# Get streamable HTTP app (returns Starlette app)
mcp_app = mcp.streamable_http_app()

# Mount in FastAPI
app.mount("/", mcp_app)  # Note: mount at root, not at /mcp
```

##### Why Mount at Root?
- `streamable_http_app()` creates internal routes at `/mcp`
- Mounting at `/mcp` would create `/mcp/mcp` (double path)
- Mounting at root gives us the clean `/mcp` endpoint

#### 4. Session Management
- FastMCP includes a `session_manager` for handling stateful operations
- Use lifespan context manager to properly initialize/cleanup:
  ```python
  async with mcp.session_manager.run():
      yield
  ```

#### 5. Response Format
- MCP uses **Server-Sent Events (SSE)** for responses
- Responses are prefixed with `event: message` and `data: `
- JSON-RPC format for the actual message content
- Example response:
  ```
  event: message
  data: {"jsonrpc":"2.0","id":1,"result":{...}}
  ```

### Critical Implementation Details

#### 1. Required Headers
Clients MUST send:
- `Content-Type: application/json`
- `Accept: application/json, text/event-stream`

Without the correct Accept header, the server returns a "Not Acceptable" error.

#### 2. Lazy Loading Pattern
To avoid initialization issues (e.g., API keys required at import time):
```python
# Instead of this:
_service = Service()  # Fails if no API key

# Use lazy loading:
_service = None
def get_service():
    global _service
    if _service is None:
        _service = Service()
    return _service
```

#### 3. Environment Setup for HF Spaces
Critical for avoiding permission errors:
```python
os.environ["WANDB_CACHE_DIR"] = "/tmp/.wandb_cache"
os.environ["HOME"] = "/tmp"
```

### Common Pitfalls & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Double path (`/mcp/mcp`) | 404 errors on `/mcp` | Mount streamable_http_app() at root (`/`) |
| Missing Accept header | "Not Acceptable" error | Include `Accept: application/json, text/event-stream` |
| Import-time API key errors | Server fails to start | Use lazy loading pattern |
| Permission errors in HF Spaces | `mkdir /.cache: permission denied` | Set cache dirs to `/tmp` |
| Can't access MCP methods | Methods not exposed | Use FastMCP's built-in decorators and methods |