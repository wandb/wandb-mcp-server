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

## 🚀 Quick Start - Use This Space

### Option 1: Duplicate This Space (Recommended)

1. **Duplicate this Space** by clicking the three dots menu (⋮) in the top right → "Duplicate this Space"
2. **Set your W&B API Key** in your duplicated Space:
   - Go to Settings → "Variables and secrets"
   - Add a new secret: `WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. **Use your personal MCP server endpoint**:
   ```
   https://huggingface.co/spaces/[your-username]/[your-space-name]/mcp
   ```

By duplicating the Space, you get your own private instance that can access your W&B projects securely!

### Option 2: Use the Public Space

If the Space owner has configured a public API key, you can use the public endpoint directly:
```
https://huggingface.co/spaces/[original-space]/mcp
```

⚠️ **Note**: The public Space will only have access to projects accessible by the configured API key.

## 🖥️ Using with MCP Clients

### Mistral le Chat
1. Go to your chat interface
2. Click on MCP settings
3. Add server with your Space URL: `https://huggingface.co/spaces/[your-username]/[your-space-name]/mcp`
4. Start querying your W&B data!

### Other MCP Clients
Use the endpoint with any MCP-compatible client that supports HTTP transport with Server-Sent Events (SSE).

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

## ⚙️ Configuration

### Required Environment Variables
- `WANDB_API_KEY`: Your Weights & Biases API key (set as Secret in Space settings)

### Optional Environment Variables
- `MCP_SERVER_LOG_LEVEL`: Set to `DEBUG` for verbose logging (default: `WARNING`)
- `PORT`: Server port (automatically set by HuggingFace Spaces)

## 🔍 Troubleshooting

### Space Not Working?
1. **Check your API key**: Ensure `WANDB_API_KEY` is set correctly in Space settings → "Variables and secrets"
2. **Verify the endpoint**: Make sure you're using `/mcp` suffix: `https://huggingface.co/spaces/[username]/[space]/mcp`
3. **Check Space status**: Ensure the Space is running (not crashed or building)

### Query Tips
- Always specify your W&B entity and project name in queries
- Be specific rather than overly broad in your questions
- Verify you have access to the projects you're querying

## 🏗️ Development & Advanced Usage

This HuggingFace Space is built from the open-source [wandb-mcp-server](https://github.com/wandb/wandb-mcp-server) repository.

## 📚 Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [Source Code & Desktop Installation](https://github.com/wandb/wandb-mcp-server)

## 📄 License

This project is licensed under the MIT License. See the [GitHub repository](https://github.com/wandb/wandb-mcp-server) for details.

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference