# PR: Add API Key Authentication & HuggingFace Spaces Deployment

**Branch:** `nico/hf_api_auth` → `nico/remote_support`  
**Type:** Feature/Infrastructure

## Summary
Implements Bearer token authentication for HTTP transport and enables HuggingFace Spaces deployment for the W&B MCP Server, allowing secure multi-user access without exposing server credentials.

## Key Changes

### 🔐 Authentication (`src/wandb_mcp_server/auth.py`)
- **Bearer token authentication** middleware for FastAPI/MCP
- Each client provides their W&B API key as Bearer token 
- Server uses client's token for all W&B operations (perfect isolation)
- Validates token format (20-100 chars, alphanumeric)
- Sets `WANDB_API_KEY` env var per-request for SDK usage
- Optional `MCP_AUTH_DISABLED` flag for development

### 🚀 HuggingFace Spaces Deployment
- **Dockerfile** for containerized deployment with UV package manager
- **app.py** - FastAPI app with MCP mounted at `/mcp` route
- Landing page (`index.html`) with API documentation
- Proper CORS and SSE headers for MCP protocol
- Port 7860 configuration for HF Spaces

### 📝 Documentation
- **AUTH_README.md** - Complete auth flow documentation
- **HUGGINGFACE_DEPLOYMENT.md** - Deployment guide
- Client configuration examples for Cursor, Claude, Mistral
- Updated README with hosted server instructions

### 🐛 Fixes
- Fixed `query_wandb_entity_projects` - handles missing `updated_at` attribute
- wandbot now works with default URL (no env var required)
- Improved error handling in Weave API calls

## Security Considerations
- ✅ No server-side API key exposure
- ✅ Per-client credential isolation  
- ✅ Bearer tokens follow MCP spec
- ✅ Auth can be disabled for local dev only

## Testing
```bash
# Test auth middleware
curl -H "Authorization: Bearer YOUR_API_KEY" https://your-server/mcp

# Test without auth (should fail)
curl https://your-server/mcp
```

## Files Changed
- **New:** `auth.py`, `app.py`, `Dockerfile`, `index.html`, auth docs
- **Modified:** Server initialization, MCP tools for auth awareness
- **21 files changed:** 1,378 insertions(+), 41 deletions(-)

## Deployment Status
✅ Successfully deployed at: https://niware-wandb-mcp-server.hf.space

---
*Ready for review. Key areas to focus on: auth.py middleware, Bearer token validation, and HF Spaces configuration.*
