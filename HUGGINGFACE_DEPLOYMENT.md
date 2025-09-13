# HuggingFace Spaces Deployment Instructions

This guide will help you deploy the Weights & Biases MCP Server to HuggingFace Spaces.

## 📋 Prerequisites

1. **HuggingFace Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **W&B API Key**: Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

## 🚀 Step-by-Step Deployment

### 1. Create a New Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Space name**: `wandb-mcp-server` (or your preferred name)
   - **License**: `apache-2.0`
   - **Select the SDK**: `Docker`
   - **Hardware**: `CPU basic` (free tier is sufficient)
   - **Visibility**: `Public` or `Private` (your choice)

### 2. Upload Files

Upload the following files to your Space repository:

#### Required Files:
- `Dockerfile` - Container configuration
- `app.py` - Entry point for the Space
- `requirements.txt` - Python dependencies
- `README_HUGGINGFACE.md` - Rename this to `README.md` in the Space
- `src/` - Copy the entire source directory
- `pyproject.toml` - Project configuration

#### File Structure in Your Space:
```
your-space/
├── Dockerfile
├── app.py
├── requirements.txt
├── README.md (renamed from README_HUGGINGFACE.md)
├── pyproject.toml
└── src/
    └── wandb_mcp_server/
        ├── __init__.py
        ├── server.py
        ├── utils.py
        ├── trace_utils.py
        ├── mcp_tools/
        └── weave_api/
```

### 3. Configure Environment Variables

1. In your Space settings, go to the "Variables and secrets" section
2. Add the following environment variable:
   - **Name**: `WANDB_API_KEY`
   - **Value**: Your Weights & Biases API key from step 1
   - **Type**: Secret (recommended for security)

### 4. Deploy and Test

1. Commit your changes to trigger the build
2. Wait for the Space to build and start (this may take a few minutes)
3. Once running, your MCP server will be available at:
   ```
   https://your-username-wandb-mcp-server.hf.space/mcp
   ```

### 5. Verify Deployment

1. Check the Space logs to ensure it started successfully
2. Look for messages like:
   ```
   Starting Weights & Biases MCP Server on HuggingFace Spaces
   WANDB_API_KEY configured: Yes
   Starting HTTP server on port 8080
   MCP endpoint will be available at: /mcp
   ```

## 🔧 Using Your Deployed Server

### With Mistral le Chat
1. Open [chat.mistral.ai](https://chat.mistral.ai)
2. Go to MCP settings
3. Add your server URL: `https://your-username-wandb-mcp-server.hf.space/mcp`

### With Other MCP Clients
Use the endpoint URL in any MCP client that supports HTTP transport.

### Example Usage
Once connected, you can ask questions like:
```
How many traces are in my wandb-team/my-project weave project?
```

```
Show me the latest runs from my experiment and create a report.
```

## 🛠️ Customization Options

### Hardware Upgrades
- For better performance with large datasets, consider upgrading to `CPU upgrade` or `GPU` hardware
- This can be changed in Space settings under "Hardware"

### Environment Variables
You can add additional environment variables in Space settings:
- `MCP_SERVER_LOG_LEVEL`: Set to `DEBUG` for verbose logging
- `WANDB_SILENT`: Set to `False` if you want W&B logging
- `WEAVE_SILENT`: Set to `False` if you want Weave logging

### Custom Configuration
Modify `app.py` to customize:
- Port configuration (though HuggingFace handles this automatically)
- Logging levels
- Additional startup logic

## 🔍 Troubleshooting

### Build Failures
- Check the build logs for specific error messages
- Ensure all required files are uploaded
- Verify `requirements.txt` has all necessary dependencies

### Runtime Issues
- Check the Space logs for error messages
- Verify `WANDB_API_KEY` is set correctly
- Ensure your W&B API key is valid

### Connection Problems
- Verify the Space is running (not crashed)
- Check that you're using the correct endpoint URL
- Ensure your MCP client supports HTTP transport

### Performance Issues
- Consider upgrading to better hardware
- Monitor Space resource usage
- Optimize query parameters to reduce data transfer

## 🔄 Updates and Maintenance

### Updating the Server
1. Update files in your Space repository
2. Commit changes to trigger a rebuild
3. Monitor the build process and test functionality

### Monitoring
- Check Space logs regularly for errors
- Monitor resource usage in Space settings
- Set up notifications for Space status changes

## 💡 Tips for Success

1. **Start Small**: Test with simple queries first
2. **Monitor Resources**: Keep an eye on CPU/memory usage
3. **Secure Secrets**: Always use "Secret" type for API keys
4. **Documentation**: Keep your Space README updated with usage examples
5. **Version Control**: Consider using Git integration for easier updates

## 📚 Additional Resources

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker on HuggingFace Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [W&B MCP Server Repository](https://github.com/wandb/wandb-mcp-server)
