# W&B MCP Server - Testing Guide

This directory contains all tests and testing utilities for the W&B MCP Server. Tests are organized by functionality and include both unit tests and integration tests.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Load Testing](#load-testing)
4. [Authentication Testing](#authentication-testing)
5. [Server Health Checks](#server-health-checks)
6. [Test Configuration](#test-configuration)

---

## Test Structure

### Unit Tests
- `test_query_weave_traces.py` - Tests for Weave trace querying functionality
- `test_count_traces.py` - Tests for trace counting operations
- `test_query_wandb_gql.py` - Tests for GraphQL querying
- `test_query_wandbot.py` - Tests for support bot integration
- `test_weave_api.py` - Tests for the Weave API layer
- `test_wandb_gql_examples.py` - Example GraphQL queries and tests

### Integration Tests
- `test_auth_flow.py` - Complete authentication flow testing
- `test_simple_auth.py` - Basic authentication tests
- `test_simplified_auth.py` - Simplified auth scenarios

### Performance & Load Tests
- `load_test.py` - Comprehensive load testing script
- `check_server.py` - Server health and connectivity checks

### Test Utilities
- `conftest.py` - Pytest configuration and fixtures
- `anthropic_test_utils.py` - Utilities for testing with Anthropic models
- `weave_test_aggregator.py` - Aggregated test utilities for Weave
- `tests_descriptions.md` - Detailed test descriptions and documentation

---

## Running Tests

### Prerequisites

Ensure you have the required environment variables set:

```bash
export WANDB_API_KEY="your-wandb-api-key"
export WEAVE_PROJECT_ID="your-project-id"  # Optional for some tests
```

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_query_weave_traces.py

# Run tests in parallel
uv run pytest -n auto

# Run tests with coverage
uv run pytest --cov=src/wandb_mcp_server

# Stop on first failure
uv run pytest -x
```

### Test Categories

```bash
# Run only unit tests
uv run pytest tests/test_*.py -k "not auth and not load"

# Run only integration tests
uv run pytest tests/test_*auth*.py

# Run only API tests
uv run pytest tests/test_*api*.py
```

---

## Load Testing

The `load_test.py` script provides comprehensive performance testing capabilities.

### Quick Load Test

```bash
# Run standard load tests (recommended)
uv run python tests/load_test.py --mode standard

# Test against local server
uv run python tests/load_test.py --url http://localhost:7860 --mode standard

# Test against remote server
uv run python tests/load_test.py --url https://mcp.withwandb.com --mode standard
```

### Custom Load Testing

```bash
# Custom load test with specific parameters
uv run python tests/load_test.py --mode custom --clients 50 --requests 20 --delay 0.1

# Stress test to find breaking point
uv run python tests/load_test.py --mode stress
```

### Load Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--url` | Server URL to test | `http://localhost:7860` |
| `--api-key` | W&B API key (optional) | Test key |
| `--mode` | Test mode: standard/stress/custom | `standard` |
| `--clients` | Number of concurrent clients | `10` |
| `--requests` | Requests per client | `10` |
| `--delay` | Delay between requests (seconds) | `0.1` |

### Expected Performance

Based on our testing:

| Server Type | Max Concurrent | Throughput | Success Rate |
|-------------|----------------|------------|--------------|
| Local | 1000+ clients | ~50 req/s | 100% up to 1000 |
| Remote (HF) | 500+ clients | ~35 req/s | 100% up to 500 |

---

## Authentication Testing

### Authentication Tests

```bash
# Run authentication unit tests
uv run pytest tests/test_authentication.py -v

# Run only unit tests (mocked, no server required)
uv run pytest tests/test_authentication.py -m unit

# Run integration tests (requires running server)
uv run pytest tests/test_authentication.py -m integration
```

### Manual Auth Testing

```bash
# Test authentication with curl
curl -X POST http://localhost:7860/mcp \
  -H "Authorization: Bearer YOUR_WANDB_API_KEY" \
  -H "Accept: application/json, text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}'
```

### Auth Test Scenarios

1. **Valid API Key**: Tests successful authentication
2. **Invalid API Key**: Tests proper error handling
3. **Missing Authorization**: Tests missing header handling
4. **Malformed Token**: Tests invalid token format handling
5. **Session Management**: Tests session creation and reuse

---

## Server Health Checks

Use the `check_server.py` script to verify server health and connectivity.

```bash
# Check local server
uv run python tests/check_server.py

# Check remote server
uv run python tests/check_server.py --url https://mcp.withwandb.com

# Check with specific API key
uv run python tests/check_server.py --api-key YOUR_WANDB_API_KEY
```

### Health Check Features

- ✅ Server connectivity
- ✅ MCP protocol compliance
- ✅ Authentication validation
- ✅ Tool availability
- ✅ Response time measurement
- ✅ Error rate monitoring

---

## Test Configuration

### Environment Variables

```bash
# Required for most tests
export WANDB_API_KEY="your-40-char-api-key"

# Optional - for specific project tests
export WEAVE_PROJECT_ID="entity/project"

# Optional - for debugging
export MCP_SERVER_LOG_LEVEL="DEBUG"

# Optional - disable auth for local dev only
export MCP_AUTH_DISABLED="true"  # ⚠️ Development only!
```

### Test Data

Some tests require access to specific W&B projects:

- **Public Test Project**: `wandb-smle/hiring-agent-demo-public`
- **Weave Test Project**: Your own Weave project with traces
- **GraphQL Test Data**: Any W&B project with runs and metrics

### CI/CD Configuration

For continuous integration, ensure:

1. **API Key**: Set `WANDB_API_KEY` as a secret
2. **Test Isolation**: Use separate test projects
3. **Rate Limiting**: Respect W&B API limits
4. **Timeout Settings**: Set appropriate test timeouts
5. **Parallel Execution**: Use `pytest-xdist` for faster tests

### Debugging Failed Tests

```bash
# Run with maximum verbosity
uv run pytest -vvv --tb=long

# Run with pdb on failure
uv run pytest --pdb

# Run specific failing test
uv run pytest tests/test_specific.py::test_function_name -v

# Capture print statements
uv run pytest -s
```

---

## Contributing Tests

When adding new tests:

1. **Follow naming convention**: `test_*.py` for test files
2. **Use fixtures**: Leverage `conftest.py` fixtures
3. **Mock external calls**: Use mocks for W&B API calls when possible
4. **Test edge cases**: Include error scenarios and edge cases
5. **Document test purpose**: Add docstrings explaining test objectives
6. **Update this README**: Document new test categories or utilities

### Test Template

```python
import pytest
from unittest.mock import Mock, patch
from src.wandb_mcp_server.mcp_tools.your_tool import your_function

class TestYourFunction:
    """Tests for your_function functionality."""
    
    def test_successful_case(self):
        """Test successful execution of your_function."""
        # Arrange
        input_data = {"key": "value"}
        
        # Act
        result = your_function(input_data)
        
        # Assert
        assert result is not None
        assert "expected_key" in result
    
    def test_error_handling(self):
        """Test error handling in your_function."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
    
    @patch('src.wandb_mcp_server.api_client.make_request')
    def test_with_mock(self, mock_request):
        """Test with mocked external dependencies."""
        mock_request.return_value = {"mocked": "response"}
        result = your_function({"test": "data"})
        assert result["mocked"] == "response"
```

For more detailed test descriptions, see [`tests_descriptions.md`](tests_descriptions.md).
