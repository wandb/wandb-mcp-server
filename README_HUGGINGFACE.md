# Weights & Biases MCP Server - HuggingFace Space

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for querying [Weights & Biases](https://www.wandb.ai/) data, hosted on HuggingFace Spaces.

This server allows MCP clients to:
- Query W&B Models runs and sweeps
- Query W&B Weave traces, evaluations and datasets  
- Query [wandbot](https://github.com/wandb/wandbot), the W&B support agent
- Write text and charts to W&B Reports

## 🚀 Quick Start

### 1. Get Your W&B API Key
Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

### 2. Configure Environment Variables
In your HuggingFace Space settings, add:
- `WANDB_API_KEY`: Your Weights & Biases API key

### 3. Use the MCP Server
The server runs on HTTP transport with Server-Sent Events (SSE) at:
```
https://your-space-name.hf.space/mcp
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
2. Add MCP server with URL: `https://your-space-name.hf.space/mcp`
3. Start querying your W&B data!

### Other MCP Clients
Use the endpoint `https://your-space-name.hf.space/mcp` with any MCP-compatible client that supports HTTP transport.

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

## 🔍 Troubleshooting

### Authentication Issues
- Ensure your `WANDB_API_KEY` is set correctly in the Space environment variables
- Verify your API key is valid at [wandb.ai/authorize](https://wandb.ai/authorize)

### Connection Issues
- Make sure you're using the correct endpoint: `https://your-space-name.hf.space/mcp`
- Check that the Space is running and not in a crashed state

### Query Issues
- Always specify your W&B entity and project name in queries
- Be specific rather than overly broad in your questions
- Verify you have access to the projects you're querying

## 🏗️ Development

This Space is built from the [wandb-mcp-server](https://github.com/wandb/wandb-mcp-server) repository.

### Local Development
```bash
# Clone the repository
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### Docker Build
```bash
docker build -t wandb-mcp-server .
docker run -p 8080:8080 -e WANDB_API_KEY=your-key wandb-mcp-server
```

## 📚 Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [Source Code](https://github.com/wandb/wandb-mcp-server)

## 📄 License

This project is licensed under the Apache License 2.0.
