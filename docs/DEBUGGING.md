Useful curl request to get a raw trace payload:

```
curl -L "https://trace.wandb.ai/calls/stream_query" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Basic <base64_wandb_api_key>" \
  -d '{
    "project_id": "wandb-applied-ai-team/mcp-tests",
    "filter": {
      "call_ids": ["01958ab9-3c68-7c23-8ccd-c135c7037769"]
    },
    "limit": 10
  }'
```