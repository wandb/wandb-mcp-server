# Hugging Face Spaces Deployment Guide

This repository is configured for deployment on Hugging Face Spaces as a Model Context Protocol (MCP) server for Weights & Biases.

## Architecture

The application runs as a FastAPI server on port 7860 (HF Spaces default) with:
- **Main landing page**: `/` - Serves the index.html with setup instructions
- **Health check**: `/health` - Returns server status and W&B configuration
- **MCP endpoint**: `/mcp` - Streamable HTTP transport endpoint for MCP
  - Uses Server-Sent Events (SSE) for responses
  - Requires `Accept: application/json, text/event-stream` header
  - Supports initialize, tools/list, tools/call methods

## Key Changes for HF Spaces

### 1. app.py
- Creates a FastAPI application that serves the landing page
- Mounts FastMCP server using `mcp.streamable_http_app()` pattern (following HuggingFace example)
- Uses lifespan context manager for session management
- Configured to run on `0.0.0.0:7860` (HF Spaces requirement)
- Sets W&B cache directories to `/tmp` to avoid permission issues

### 2. server.py
- Exports necessary functions for HF Spaces initialization
- Support for being imported as a module
- Maintains backward compatibility with CLI usage

### 3. Dependencies
- FastAPI and uvicorn moved to main dependencies (not optional)
- All dependencies listed in requirements.txt for HF Spaces

### 4. Lazy Loading Fix
- Fixed `TraceService` initialization in `query_weave.py` to use lazy loading
- This allows the server to start even without a W&B API key
- The service is only initialized when first needed

## Environment Variables

No environment variables are required! The server works without any configuration.

**Note**: Users provide their own W&B API keys as Bearer tokens. No server configuration needed.

## Deployment Steps

1. **Create a new Space on Hugging Face**
   - Choose "Docker" as the SDK
   - Set visibility as needed

2. **Configure Secrets**
   - Go to Settings → Variables and secrets
   - Add `WANDB_API_KEY` as a secret

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

## MCP Client Configuration

### Important Notes

The MCP server uses the Streamable HTTP transport which:
- Returns responses in Server-Sent Events (SSE) format
- Requires the client to send `Accept: application/json, text/event-stream` header
- Uses session management for stateful operations

### Testing with curl

```bash
# Initialize the server
curl -X POST https://[your-username]-[space-name].hf.space/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "protocolVersion": "0.1.0",
      "capabilities": {},
      "clientInfo": {"name": "test-client", "version": "1.0"}
    },
    "id": 1
  }'

# List available tools
curl -X POST https://[your-username]-[space-name].hf.space/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}'
```

### MCP Client Configuration

For MCP clients that support streamable HTTP:

```json
{
  "mcpServers": {
    "wandb": {
      "url": "https://[your-username]-[space-name].hf.space/mcp",
      "transport": "streamable-http"
    }
  }
}
```

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

### Testing Strategy

1. **Local Testing**: Always test with correct headers
2. **Check Routes**: Verify mounting creates `/mcp` endpoint
3. **Test Initialize First**: This method doesn't require session state
4. **SSE Response Parsing**: Remember responses are SSE formatted, not plain JSON

### Evolution of Our Implementation

Our journey to the correct implementation went through several iterations:

#### Attempt 1: Direct Protocol Implementation
- **Approach**: Implement MCP protocol directly in FastAPI
- **Issue**: Reinventing the wheel, not using FastMCP's built-in capabilities
- **Learning**: FastMCP already handles the protocol complexity

#### Attempt 2: Trying to Extract FastMCP's Internal App
- **Approach**: Access FastMCP's internal FastAPI app via attributes
- **Issue**: FastMCP doesn't expose its app in an accessible way
- **Learning**: Need to use FastMCP's intended methods

#### Attempt 3: Using http_app() Method
- **Approach**: Try various methods like `http_app()`, `asgi_app()`, etc.
- **Issue**: These methods either don't exist or don't work as expected
- **Learning**: Documentation and examples are crucial

#### Attempt 4: The Correct Pattern
- **Approach**: Use `streamable_http_app()` following HuggingFace example
- **Success**: Works perfectly when mounted at root
- **Key Insight**: The example pattern exists for a reason - follow it!

### Key Takeaways

1. **Follow Existing Examples**: The HuggingFace example was the key to success
2. **Understand the Protocol**: MCP uses SSE for good reasons (streaming, stateless option)
3. **Lazy Loading is Critical**: Avoid initialization-time dependencies
4. **Environment Matters**: HF Spaces has specific constraints (ports, permissions)
5. **Test Incrementally**: Start with basic endpoints before complex operations

## Differences from Standard Deployment

| Feature | Standard | HF Spaces |
|---------|----------|-----------|
| Transport | stdio/http | streamable-http only |
| Port | Configurable | Fixed at 7860 |
| Host | Configurable | Fixed at 0.0.0.0 |
| Entry Point | CLI (server.py) | FastAPI (app.py) |
| Static Files | Optional directories | Embedded in app |

## Troubleshooting

1. **Server not starting**: Check WANDB_API_KEY is set in Space secrets
2. **MCP connection fails**: Ensure using `/mcp` endpoint with correct transport ("streamable-http")
3. **Tools not working**: Verify W&B API key has necessary permissions
4. **Landing page not loading**: Check index.html is included in deployment