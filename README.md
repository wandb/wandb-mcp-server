---
title: Weights & Biases MCP Server
emoji: 🏋️‍♂️
colorFrom: yellow
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

# Weights & Biases MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for querying [Weights & Biases](https://www.wandb.ai/) data, hosted on HuggingFace Spaces.

This server allows MCP clients to:
- 📊 Query W&B Models runs and sweeps
- 🔍 Query W&B Weave traces, evaluations and datasets  
- 🤖 Query [wandbot](https://github.com/wandb/wandbot), the W&B support agent
- 📝 Write text and charts to W&B Reports

## 🚀 Quick Start

### 1. Get Your W&B API Key
Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### 2. Configure Environment Variables
⚠️ **Important**: You must set your `WANDB_API_KEY` in the Space settings under "Variables and secrets" for this to work.

### 3. Use the MCP Server
The server runs on HTTP transport with Server-Sent Events (SSE) at:
```
https://huggingface.co/spaces/[your-username]/[space-name]/mcp
```

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

## 🖥️ Using with MCP Clients

### Mistral le Chat
1. Go to your chat interface
2. Add MCP server with URL: `https://huggingface.co/spaces/[your-username]/[space-name]/mcp`
3. Start querying your W&B data!

### Other MCP Clients
Use the endpoint with any MCP-compatible client that supports HTTP transport.

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

### Authentication Issues
- Ensure your `WANDB_API_KEY` is set correctly in the Space environment variables
- Verify your API key is valid at [wandb.ai/authorize](https://wandb.ai/authorize)

### Connection Issues
- Make sure you're using the correct endpoint with `/mcp` suffix
- Check that the Space is running and not in a crashed state

### Query Issues
- Always specify your W&B entity and project name in queries
- Be specific rather than overly broad in your questions
- Verify you have access to the projects you're querying

## 📚 Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [Source Code](https://github.com/wandb/wandb-mcp-server)

## 📄 License

This project is licensed under the Apache License 2.0.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
