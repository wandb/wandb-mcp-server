---
title: Weights & Biases MCP Server
emoji: 🪄🐝
colorFrom: yellow
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# Weights & Biases MCP Server on HuggingFace Spaces

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for querying [Weights & Biases](https://www.wandb.ai/) data, hosted on HuggingFace Spaces.

This server allows MCP clients to:
- 📊 Query W&B Models runs and sweeps
- 🔍 Query W&B Weave traces, evaluations and datasets  
- 🤖 Query [wandbot](https://github.com/wandb/wandbot), the W&B support agent
- 📝 Write text and charts to W&B Reports

## 🚀 Quick Start - Use This Server Directly!

### No Setup Required! 🎉

You can use this server immediately with your own W&B API key:

1. **Get your W&B API key** from [wandb.ai/authorize](https://wandb.ai/authorize)
2. **Configure your MCP client** with:
   - **Server URL**: `https://niware-wandb-mcp-server.hf.space/mcp`
   - **Authentication**: Your W&B API key as Bearer token
3. **Start querying** your W&B data!

That's it! No server configuration, no duplication needed. Each user provides their own API key, ensuring secure access to their own W&B projects.

## 🖥️ Using with MCP Clients

### Mistral LeChat
1. Go to Settings → Custom MCP Connectors
2. Add a new connector:
   - **Server URL**: `https://niware-wandb-mcp-server.hf.space/mcp`
   - **Authentication**: Choose "API Key Authentication"
   - **API Key**: Your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### Cursor
Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):
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

### Claude Desktop / Claude Code
Configure in your MCP settings with the server URL and Bearer token authentication.

## 🔧 Available MCP Tools

### W&B Models
- **`query_wandb_tool`**: Execute GraphQL queries against W&B experiment tracking data

### W&B Weave  
- **`query_weave_traces_tool`**: Query Weave evaluations and traces with filtering and pagination
- **`count_weave_traces_tool`**: Count traces matching filters without returning data

### Support & Reports
- **`query_wandb_support_bot`**: Get help from wandbot, the W&B support agent
- **`create_wandb_report_tool`**: Create W&B Reports with markdown and visualizations
- **`query_wandb_entity_projects`**: List available W&B entities and projects

## 📝 Example Queries

```
How many openai.chat traces are in my wandb-team/my-project weave project?
```

```
Show me the latest 10 runs from my experiment tracking project and create a report with the results.
```

```
What's the best performing model in my latest sweep? Plot the results.
```

## 💻 Run Locally (Optional)

Want to run your own instance or develop locally? Check out the main repository:

```bash
# Install and run from source
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server
uv run src/wandb_mcp_server/server.py --transport http
```

See the [GitHub repository](https://github.com/wandb/wandb-mcp-server) for installation instructions for various MCP clients.

## 🔍 Troubleshooting

### Connection Issues?
1. **Verify your API key**: Ensure it's correctly copied from [wandb.ai/authorize](https://wandb.ai/authorize)
2. **Check the endpoint**: Must include `/mcp` suffix: `https://niware-wandb-mcp-server.hf.space/mcp`
3. **Headers required**: Include both `Authorization` and `Accept` headers as shown above

### Query Tips
- Always specify your W&B entity and project name in queries
- Be specific rather than overly broad in your questions
- Verify you have access to the projects you're querying

## 🔐 Security

- **Your API key is never stored** - It's only used transiently for your requests
- **Each user is isolated** - Your key only accesses your W&B data
- **No server configuration needed** - The server doesn't have its own W&B access
- **Industry-standard Bearer tokens** - Same pattern as GitHub, OpenAI, etc.

## 📚 Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [Source Code Repository](https://github.com/wandb/wandb-mcp-server)
- [Get Your W&B API Key](https://wandb.ai/authorize)

## 📄 License

This project is licensed under the MIT License. See the [GitHub repository](https://github.com/wandb/wandb-mcp-server) for details.

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference