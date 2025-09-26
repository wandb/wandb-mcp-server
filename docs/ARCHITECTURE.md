# W&B MCP Server - Architecture & Scalability Guide

## Table of Contents
1. [Architecture Decision](#architecture-decision)
2. [Stateless HTTP Design](#stateless-http-design)
3. [Performance & Scalability](#performance--scalability)
4. [Load Test Results](#load-test-results)
5. [Deployment Recommendations](#deployment-recommendations)

---

## Architecture Decision

### Decision: Pure Stateless HTTP Mode

**The W&B MCP Server uses pure stateless HTTP mode (`stateless_http=True`).**

This fundamental architecture decision enables:
- ✅ **Universal client compatibility** (OpenAI, Cursor, LeChat, Claude)
- ✅ **Horizontal scaling** capabilities
- ✅ **Simpler operations** and maintenance
- ✅ **Cloud-native** deployment patterns

### Why Stateless?

The Model Context Protocol traditionally used stateful sessions, but this created issues:

| Client | Behavior | Problem with Stateful |
|--------|----------|----------------------|
| **OpenAI** | Deletes session after listing tools, then reuses ID | Session not found errors |
| **Cursor** | Sends Bearer token with every request | Expects stateless behavior |
| **Claude** | Can work with either model | No issues |

### The Solution

```python
# Pure stateless operation - no session persistence
mcp = FastMCP("wandb-mcp-server", stateless_http=True)
```

With this approach:
- **Session IDs are correlation IDs only** - they match requests to responses
- **No state persists between requests** - each request is independent
- **Authentication required per request** - Bearer token must be included
- **Any worker can handle any request** - enables horizontal scaling

---

## Stateless HTTP Design

### Architecture Overview

```
┌─────────────────────────────────────┐
│    MCP Clients (OpenAI/Cursor/etc)  │
│     Bearer Token with Each Request   │
└─────────────┬───────────────────────┘
              │ HTTPS
┌─────────────▼───────────────────────┐
│         Load Balancer (Optional)     │
│      Round-Robin Distribution        │
└──┬──────────┬──────────┬────────────┘
   │          │          │
┌──▼───┐  ┌──▼───┐  ┌──▼───┐
│ W1   │  │ W2   │  │ W3   │  (Multiple Workers Possible)
│      │  │      │  │      │
│ ASGI │  │ ASGI │  │ ASGI │  Uvicorn/Gunicorn
└──┬───┘  └──┬───┘  └──┬───┘
   │          │          │
┌──▼──────────▼──────────▼────────────┐
│         FastAPI Application         │
│  ┌────────────────────────────┐     │
│  │  Stateless Auth Middleware  │     │
│  │  (Bearer Token Validation)  │     │
│  └────────────────────────────┘     │
│  ┌────────────────────────────┐     │
│  │    MCP Stateless Handler    │     │
│  │  (No Session Storage)       │     │
│  └────────────────────────────┘     │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│         W&B API Integration         │
└─────────────────────────────────────┘
```

### Request Flow

1. **Client sends request** with Bearer token and session ID
2. **Middleware validates** Bearer token
3. **MCP processes** request (session ID used for correlation only)
4. **Response sent** with matching session ID
5. **No state persisted** - request complete

### Key Implementation Details

```python
async def thread_safe_auth_middleware(request: Request, call_next):
    """Stateless authentication middleware."""
    
    # Session IDs are correlation IDs only
    session_id = request.headers.get("Mcp-Session-Id")
    if session_id:
        logger.debug(f"Correlation ID: {session_id[:8]}...")
    
    # Every request must have Bearer token
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
        # Use API key for this request only
        # No session storage or retrieval
```

---

## Performance & Scalability

### Single Worker Performance

Based on testing with stateless mode:

| Metric | Local Server | Remote (HF Spaces) |
|--------|--------------|-------------------|
| **Max Concurrent** | 1000 clients | 500+ clients |
| **Throughput** | ~50-60 req/s | ~35 req/s |
| **Latency (p50)** | <500ms | <2s |
| **Memory Usage** | 200-500MB | 300-600MB |

### Horizontal Scaling Potential

With stateless mode, the server supports true horizontal scaling:

| Workers | Max Concurrent | Total Throughput | Notes |
|---------|----------------|------------------|-------|
| 1 | 1000 | ~50 req/s | Current deployment |
| 2 | 2000 | ~100 req/s | Linear scaling |
| 4 | 4000 | ~200 req/s | Near-linear |
| 8 | 8000 | ~400 req/s | Some overhead |

**Key Advantage**: No session affinity required - any worker can handle any request!

---

## Load Test Results

### Latest Test Results (2025-09-25)

#### Local Server (MacOS, Single Worker)

| Concurrent Clients | Success Rate | Throughput | Mean Response |
|--------------------|-------------|------------|---------------|
| 10 | 100% | 47 req/s | 89ms |
| 100 | 100% | 47 req/s | 1.2s |
| 500 | 100% | 56 req/s | 4.4s |
| **1000** | **100%** | **48 req/s** | **9.3s** |
| 1500 | 80% | 51 req/s | 15.4s |
| 2000 | 70% | 53 req/s | 20.8s |

**Breaking Point**: ~1500 concurrent connections

#### Remote Server (mcp.withwandb.com)

| Concurrent Clients | Success Rate | Throughput | Mean Response |
|--------------------|-------------|------------|---------------|
| 10 | 100% | 10 req/s | 0.8s |
| 50 | 100% | 29 req/s | 1.2s |
| 100 | 100% | 33 req/s | 1.9s |
| 200 | 100% | 34 req/s | 3.3s |
| **500** | **100%** | **35 req/s** | **7.5s** |

**Key Finding**: Remote server handles 500+ concurrent connections reliably!

### Performance Sweet Spots

1. **Low Latency** (<1s response): Use ≤50 concurrent connections
2. **Balanced** (good throughput & latency): Use 100-200 concurrent connections  
3. **Maximum Throughput**: Use 200-300 concurrent connections
4. **Maximum Capacity**: Up to 500 concurrent (remote) or 1000 (local)

---

## Deployment Recommendations

### Current Deployment (HuggingFace Spaces)

```yaml
Configuration:
  - Single worker (can be increased)
  - Stateless HTTP mode
  - 2 vCPU, 16GB RAM
  - Port 7860

Performance:
  - 500+ concurrent connections
  - ~35 req/s throughput
  - 100% reliability up to 500 concurrent
```

### Scaling Options

#### Option 1: Vertical Scaling
- Increase CPU/RAM on HuggingFace Spaces
- Can improve single-worker throughput

#### Option 2: Horizontal Scaling (Recommended)
```python
# app.py - Enable multiple workers
uvicorn.run(app, host="0.0.0.0", port=PORT, workers=4)
```

#### Option 3: Multi-Region Deployment
- Deploy to multiple regions
- Use global load balancer
- Reduce latency for users worldwide

### Production Checklist

✅ **Stateless mode enabled** (`stateless_http=True`)  
✅ **Bearer authentication** on every request  
✅ **Health check endpoint** (`/health`)  
✅ **Monitoring** for response times and errors  
✅ **Rate limiting** (recommended: 100 req/s per client)  
✅ **Connection limits** (recommended: 500 concurrent)  

### Configuration Example

```python
# Production configuration
mcp = FastMCP("wandb-mcp-server", stateless_http=True)

# Uvicorn with multiple workers (if needed)
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        workers=1,  # Increase for horizontal scaling
        limit_concurrency=1000,  # Connection limit
        timeout_keep_alive=30,  # Keepalive timeout
    )
```

### Security Considerations

1. **API Key Validation**: Every request validates Bearer token
2. **No Session Storage**: No risk of session hijacking
3. **Rate Limiting**: Protect against abuse
4. **HTTPS Only**: Always use TLS in production
5. **Token Rotation**: Encourage regular API key rotation

---

## Summary

The W&B MCP Server's stateless architecture provides:

- **Universal Compatibility**: Works with all MCP clients
- **Excellent Performance**: 500+ concurrent connections, ~35 req/s
- **Horizontal Scalability**: Add workers to increase capacity
- **Simple Operations**: No session management complexity
- **Production Ready**: Deployed and tested at scale

The stateless design is not a compromise - it's the optimal architecture for MCP servers in production environments.
