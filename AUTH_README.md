# W&B MCP Server Authentication

The W&B MCP Server uses **Bearer token authentication** with W&B API keys for secure access to your Weights & Biases data.

## How It Works

### API Key Authentication

The server uses standard HTTP Bearer token authentication where your W&B API key serves as the Bearer token:

```http
Authorization: Bearer YOUR_WANDB_API_KEY
```

**Key Features:**
- Each client provides their own W&B API key
- Server uses the client's key for all W&B operations  
- Perfect isolation between users
- No server-side API key needed for HTTP transport

### Getting Your W&B API Key

1. Go to [https://wandb.ai/authorize](https://wandb.ai/authorize)
2. Log in with your W&B account
3. Copy your API key (exactly 40 characters, no spaces)
4. Use it as the Bearer token in your MCP client

**Important**: W&B API keys must be exactly 40 alphanumeric characters. Make sure you copy the entire key without any extra spaces or line breaks.

## Client Configuration

### Mistral LeChat

In Mistral LeChat, add a Custom MCP Connector:

1. **Server URL**: `https://niware-wandb-mcp-server.hf.space/mcp`
2. **Authentication**: Choose "API Key Authentication"
3. **Token**: Enter your W&B API key

### Claude Desktop / Cursor

Configure in your MCP settings:

```json
{
  "mcpServers": {
    "wandb": {
      "transport": "http",
      "url": "https://niware-wandb-mcp-server.hf.space/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

**Important Headers:**
- `Authorization`: Your W&B API key as Bearer token
- `Accept`: Must include both `application/json` and `text/event-stream` for MCP Streamable HTTP

### Python Client Example

```python
import requests

# Initialize MCP session
response = requests.post(
    "https://niware-wandb-mcp-server.hf.space/mcp",
    headers={
        "Authorization": "Bearer YOUR_WANDB_API_KEY",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    },
    json={
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {},
        "id": 1
    }
)
```

## OAuth: WIP

### Current Solution

We currently use W&B API keys directly as Bearer tokens. This approach:
- ✅ Works with all W&B functionality
- ✅ Compatible with MCP specification
- ✅ Simple and reliable
- ✅ Follows industry patterns (GitHub, OpenAI)
- ❌ Requires manual key management
- ❌ No automatic token refresh

### Tests
We attempted to implement a OAuth-assisted authentication flow (clients like ChatGPT would forward to login in the beginning):

1. **OAuth Discovery Endpoints**: `/.well-known/oauth-authorization-server`
2. **Authorization Flow**: Redirect to W&B's Auth0 login
3. **Token Exchange**: Accept W&B API keys as "access tokens"
4. **Device Flow**: Guide users to get their API key

### Still WIP
We're running into some issues with the OAuth-assisted approach we tried out (forward clients to wandb.ai/protect and return OAuth style metdata) - with some issus:

1. **W&B Doesn't Provide OAuth for Third Parties**
   - W&B uses Auth0 internally but doesn't allow third-party OAuth client registration
   - No way to register our MCP server as an OAuth application
   - Can't receive authorization codes or callbacks from W&B

2. **API Keys Are Not OAuth Tokens**
   - W&B provides permanent API keys, not temporary OAuth tokens
   - No refresh mechanism, no expiration, no scopes
   - Keys are managed through W&B's web interface, not OAuth flows

3. **Cross-Domain Issues**
   - OAuth requires the authorization server and resource server to cooperate
   - W&B's Auth0 instance (`wandb.auth0.com`) doesn't know about our server
   - Can't validate tokens or handle callbacks

For proper OAuth 2.0 support, W&B would need to:

1. **Allow OAuth Client Registration**
   - Let developers register OAuth applications
   - Provide client ID and secret
   - Support redirect URIs for callbacks

2. **Issue Real OAuth Tokens**
   - Temporary access tokens (e.g., 1-hour expiry)
   - Refresh tokens for obtaining new access tokens
   - Scoped permissions (read-only, write, admin)

3. **Provide Token Validation**
   - Introspection endpoint for validating tokens
   - Revocation endpoint for invalidating tokens
   - JWKS endpoint for JWT validation

## Troubleshooting

### Common Issues

#### "Authorization required" (401)
- Ensure Bearer token is included in header
- Check API key is valid at [wandb.ai/settings](https://wandb.ai/settings)

#### "Not Acceptable" (406)
- Include `Accept: application/json, text/event-stream` header
- Required for MCP Streamable HTTP transport

#### "Missing session ID" (400)
- Call `initialize` method first
- Include session ID from response in subsequent requests

#### "Invalid API key format"
- W&B API keys are ~40 alphanumeric characters
- Get a valid key from [wandb.ai/authorize](https://wandb.ai/authorize)

### Testing Authentication

```bash
# Test with curl
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer YOUR_WANDB_API_KEY" \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

## Development Mode

For local development only, you can disable authentication:

```bash
export MCP_AUTH_DISABLED=true
```

⚠️ **Warning**: Never use this in production or on public servers!