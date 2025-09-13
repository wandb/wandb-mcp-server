# Raw Weave API Client

This extension to the wandb-mcp-server package provides a raw HTTP-based implementation of the Weave API client. It allows you to interact with the Weave server using direct HTTP requests, without requiring the Weave client to be initialized.

## Features

- Drop-in replacement for the original Weave client-based implementation
- Same function signatures and behavior as the original implementation
- Uses raw HTTP requests to interact with the Weave server
- No dependency on the Weave client
- Supports all the same filtering and sorting options as the original implementation

## Usage

### Basic Usage

```python
from wandb_mcp_server import query_traces

# Query traces
traces = query_traces(
    entity_name="your_entity",
    project_name="your_project",
    filters={
        "op_name": "weave:///your_op_name",
        "trace_roots_only": True
    },
    limit=5,
    api_key="your_api_key"
)

# Print the results
for trace in traces:
    print(f"ID: {trace['id']}")
    print(f"Op Name: {trace['op_name']}")
    print(f"Display Name: {trace['display_name']}")
    print(f"Started At: {trace['started_at']}")
    print(f"Ended At: {trace['ended_at']}")
    print("---")
```

### Paginated Query

```python
import asyncio
from wandb_mcp_server import paginated_query_traces

async def main():
    # Query traces with pagination
    result = await paginated_query_traces(
        entity_name="your_entity",
        project_name="your_project",
        chunk_size=20,
        filters={
            "op_name": "weave:///your_op_name",
            "trace_roots_only": True
        },
        target_limit=100,
        api_key="your_api_key"
    )
    
    # Print the results
    print(f"Total Traces: {result['metadata']['total_traces']}")
    for trace in result['traces']:
        print(f"ID: {trace['id']}")
        print(f"Op Name: {trace['op_name']}")
        print(f"Display Name: {trace['display_name']}")
        print("---")

# Run the async function
asyncio.run(main())
```

### Using the Original Implementation

If you need to use the original Weave client-based implementation, you can still do so:

```python
from wandb_mcp_server import query_traces_client, paginated_query_traces_client

# Use the original implementation
traces = query_traces_client(
    entity_name="your_entity",
    project_name="your_project",
    filters={
        "op_name": "weave:///your_op_name",
        "trace_roots_only": True
    },
    limit=5
)
```

## Direct HTTP Requests with curl

The raw HTTP implementation uses the following endpoints:

### Query Traces

```bash
curl -X POST "https://trace.wandb.ai/calls/stream_query" \
  -u "your_api_key:" \
  -H "Content-Type: application/json" \
  -H "Accept: application/jsonl" \
  -d '{
    "project_id": "your_entity/your_project",
    "filter": {
      "trace_roots_only": true,
      "op_names": ["weave:///your_op_name"]
    },
    "limit": 5,
    "sort_by": [{"field": "started_at", "direction": "desc"}],
    "include_costs": true,
    "include_feedback": true
  }'
```

### Get Call

```bash
curl -X POST "https://trace.wandb.ai/calls/get" \
  -u "your_api_key:" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your_entity/your_project",
    "call_id": "your_call_id",
    "include_costs": true,
    "include_storage_size": true,
    "include_total_storage_size": true
  }'
```

### Get Call Stats

```bash
curl -X POST "https://trace.wandb.ai/calls/stats" \
  -u "your_api_key:" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your_entity/your_project",
    "filter": {
      "trace_roots_only": true,
      "op_names": ["weave:///your_op_name"]
    },
    "include_total_storage_size": true
  }'
```

## Example Script

An example script is provided in the `examples` directory:

```bash
# Run the example script
python examples/raw_weave_api_example.py \
  --entity your_entity \
  --project your_project \
  --api-key your_api_key \
  --limit 5
```

## Implementation Details

The raw HTTP implementation is provided in the `query_weave_raw.py` file. It uses the `requests` library to make HTTP requests to the Weave server. The implementation is designed to be a drop-in replacement for the original Weave client-based implementation, with the same function signatures and behavior.

The main functions provided are:

- `query_traces`: Query Weave traces with flexible filtering and sorting options
- `paginated_query_traces`: Query traces with pagination

The implementation also includes a compatibility layer to support the original Weave client-based implementation, so that existing code can continue to work without changes.

## License

MIT