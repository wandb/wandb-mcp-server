# Hugging Face Spaces Deployment Guide

This repository is configured for deployment on Hugging Face Spaces as a Model Context Protocol (MCP) server for Weights & Biases.

## Architecture

The application runs as a FastAPI server on port 7860 (HF Spaces default) with:
- **Main landing page**: `/` - Serves the index.html with setup instructions
- **Health check**: `/health` - Returns server status and W&B configuration
- **MCP endpoint**: `/mcp` - Streamable HTTP transport endpoint for MCP (supports both regular HTTP and SSE upgrade)

## Key Changes for HF Spaces

### 1. app.py
- Creates a FastAPI application that serves the landing page
- Mounts the MCP server as a sub-application at `/mcp`
- Configured to run on `0.0.0.0:7860` (HF Spaces requirement)
- No need for static/templates directories - index.html is served directly

### 2. server.py
- Exports necessary functions for HF Spaces initialization
- Support for being imported as a module
- Maintains backward compatibility with CLI usage

### 3. Dependencies
- FastAPI and uvicorn moved to main dependencies (not optional)
- All dependencies listed in requirements.txt for HF Spaces

## Environment Variables

Required in HF Spaces settings:
```
WANDB_API_KEY=your_api_key_here
```

Optional:
```
WANDB_ENTITY=your_wandb_entity
MCP_LOGS_WANDB_PROJECT=wandb-mcp-logs
WEAVE_DISABLED=false  # Set to enable Weave tracing
```

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

Example configuration for Claude Desktop or other MCP clients:

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