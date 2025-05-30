"""
Integration tests for sandbox code execution functionality.
These tests actually execute code in real sandboxes (when available).
"""

import asyncio
import os
import pytest
from dotenv import load_dotenv

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    PyodideSandbox,
)

load_dotenv()


class TestSandboxIntegration:
    """Integration tests for all sandbox types."""
    
    @pytest.mark.asyncio
    async def test_basic_execution_all_sandboxes(self):
        """Test basic code execution across all available sandboxes."""
        code = """
import math
result = math.sqrt(16)
print(f"The square root of 16 is: {result}")
print("Python execution successful!")
"""
        
        # Test with auto-selection
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        assert "4.0" in result["output"]
        assert "Python execution successful!" in result["output"]
        assert result["sandbox_used"] in ["e2b", "pyodide", "none"]
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_specific_features(self):
        """Test E2B-specific features like package installation."""
        # Test package installation
        code = """
import numpy as np
import pandas as pd

# Create a simple dataset
data = np.random.randn(5, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("DataFrame created successfully!")
print(df.describe())
"""
        
        result = await execute_sandbox_code(
            code,
            sandbox_type="e2b",
            install_packages=["numpy", "pandas"]
        )
        
        assert result["success"] is True
        assert "DataFrame created successfully!" in result["output"]
        assert result["sandbox_used"] == "e2b"
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_dangerous_code_allowed(self):
        """Test that E2B allows 'dangerous' operations since it's properly sandboxed."""
        # This code demonstrates E2B's secure isolation
        code = """
import os
import subprocess

# These operations are safe in E2B's isolated environment
print(f"Current directory: {os.getcwd()}")
print(f"Environment USER: {os.environ.get('USER', 'not set')}")

# Try to run a command (safe in sandbox)
try:
    result = subprocess.run(['echo', 'Hello from subprocess'], capture_output=True, text=True)
    print(f"Subprocess output: {result.stdout.strip()}")
except Exception as e:
    print(f"Subprocess failed (expected in some environments): {e}")

# Try file operations (safe in sandbox)
try:
    with open('/tmp/test.txt', 'w') as f:
        f.write('Test file content')
    print("File written successfully")
    
    with open('/tmp/test.txt', 'r') as f:
        content = f.read()
    print(f"File content: {content}")
except Exception as e:
    print(f"File operation failed: {e}")
"""
        
        result = await execute_sandbox_code(code, sandbox_type="e2b")
        
        assert result["success"] is True
        assert result["sandbox_used"] == "e2b"
        # The code should execute without security errors
        assert "Security validation failed" not in result.get("error", "")
        
    @pytest.mark.asyncio
    async def test_pyodide_sandbox(self):
        """Test Pyodide sandbox functionality."""
        # Check if Node.js is available
        pyodide = PyodideSandbox()
        if not pyodide.available:
            pytest.skip("Node.js not available for Pyodide testing")
        
        code = """
# Test basic Python functionality
import json
import datetime

data = {
    'timestamp': str(datetime.datetime.now()),
    'values': [1, 2, 3, 4, 5],
    'message': 'Pyodide execution successful'
}

print(json.dumps(data, indent=2))

# Test mathematical operations
import math
print(f"Pi value: {math.pi}")
print(f"E value: {math.e}")
"""
        
        result = await execute_sandbox_code(code, sandbox_type="pyodide")
        
        assert result["success"] is True
        assert "Pyodide execution successful" in result["output"]
        assert result["sandbox_used"] == "pyodide"
        
    @pytest.mark.asyncio
    async def test_pyodide_dangerous_code_allowed(self):
        """Test that Pyodide allows 'dangerous' operations since it's WebAssembly sandboxed."""
        pyodide = PyodideSandbox()
        if not pyodide.available:
            pytest.skip("Node.js not available for Pyodide testing")
        
        code = """
# These operations are safe in Pyodide's WebAssembly sandbox
try:
    # __import__ is safe in WebAssembly sandbox
    sys = __import__('sys')
    print(f"Python version: {sys.version}")
    
    # eval is safe in WebAssembly sandbox
    result = eval('2 + 2')
    print(f"Eval result: {result}")
    
    # globals() is safe in WebAssembly sandbox
    g = globals()
    print(f"Number of global variables: {len(g)}")
    
except Exception as e:
    print(f"Error: {e}")
"""
        
        result = await execute_sandbox_code(code, sandbox_type="pyodide")
        
        assert result["success"] is True
        assert result["sandbox_used"] == "pyodide"
        # The code should execute without security errors
        assert "Security validation failed" not in result.get("error", "")
        
            
    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """Test that long-running code is properly timed out."""
        code = """
import time
print("Starting long operation...")
time.sleep(10)  # This should timeout
print("This should not be printed")
"""
        
        result = await execute_sandbox_code(code, timeout=2)
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower() or "timed out" in result["error"].lower()
        
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in code execution."""
        code = """
print("Before error")
raise ValueError("This is a test error")
print("After error - should not print")
"""
        
        result = await execute_sandbox_code(code)
        
        assert result["success"] is False
        assert "ValueError" in result["error"]
        assert "This is a test error" in result["error"]
        assert "Before error" in result["output"]
        assert "After error" not in result["output"]
        
    @pytest.mark.asyncio
    async def test_execution_caching(self):
        """Test that identical code executions are cached."""
        code = """
import random
# Using a deterministic operation to test caching
print("Result: 42")
"""
        
        # First execution
        result1 = await execute_sandbox_code(code)
        
        # Second execution (should be cached)
        result2 = await execute_sandbox_code(code)
        execution_time2 = result2.get("execution_time_ms", 0)
        
        assert result1["output"] == result2["output"]
        assert result1["sandbox_used"] == result2["sandbox_used"]
        # Cached execution should be much faster (0ms)
        assert execution_time2 == 0
        
    @pytest.mark.asyncio
    async def test_concurrent_executions(self):
        """Test that multiple concurrent executions work correctly."""
        codes = [
            "print(f'Task 1: {1 + 1}')",
            "print(f'Task 2: {2 + 2}')",
            "print(f'Task 3: {3 + 3}')",
            "print(f'Task 4: {4 + 4}')",
            "print(f'Task 5: {5 + 5}')",
        ]
        
        # Execute all codes concurrently
        tasks = [execute_sandbox_code(code) for code in codes]
        results = await asyncio.gather(*tasks)
        
        # Verify all executions succeeded
        for i, result in enumerate(results):
            assert result["success"] is True
            expected_output = f"Task {i+1}: {(i+1) * 2}"
            assert expected_output in result["output"]
            
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works correctly."""
        # This test would need to be adjusted based on rate limit settings
        # For now, we'll just verify the mechanism exists
        code = "print('Rate limit test')"
        
        # Execute within normal limits
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        
        # Note: To properly test rate limiting, we'd need to:
        # 1. Reduce the rate limit for testing
        # 2. Execute many requests rapidly
        # 3. Verify that some are rejected
        
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_pool_functionality(self):
        """Test E2B sandbox pooling functionality."""
        # Execute multiple times to test pool
        codes = [
            "print('Pool test 1')",
            "print('Pool test 2')",
            "print('Pool test 3')",
        ]
        
        results = []
        for code in codes:
            result = await execute_sandbox_code(code, sandbox_type="e2b")
            results.append(result)
            assert result["success"] is True
            assert result["sandbox_used"] == "e2b"
        
        # All should succeed with pool
        assert all(r["success"] for r in results)
        
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        code = """
# Test unicode support
print("Hello, 世界! 🌍")
print("Special chars: €£¥")
print('Quotes: "double" and \\'single\\'')
print(\"\"\"Triple quotes: '''test'''\"\"\")

# Test with unicode in variables
emoji = "🚀"
print(f"Rocket emoji: {emoji}")
"""
        
        result = await execute_sandbox_code(code)
        
        assert result["success"] is True
        assert "世界" in result["output"]
        assert "🌍" in result["output"]
        assert "€£¥" in result["output"]
        
    @pytest.mark.asyncio
    async def test_large_output_handling(self):
        """Test handling of large outputs."""
        code = """
# Generate large output
for i in range(100):
    print(f"Line {i}: " + "x" * 100)
    
print("Final line after large output")
"""
        
        result = await execute_sandbox_code(code)
        
        assert result["success"] is True
        assert "Line 0:" in result["output"]
        assert "Line 99:" in result["output"]
        assert "Final line after large output" in result["output"]
        
    @pytest.mark.asyncio
    async def test_data_science_workflow(self):
        """Test a realistic data science workflow."""
        code = """
import json

# Simulate data analysis
data = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
    {"name": "Diana", "score": 95},
    {"name": "Eve", "score": 88}
]

# Calculate statistics
scores = [d["score"] for d in data]
avg_score = sum(scores) / len(scores)
max_score = max(scores)
min_score = min(scores)

# Find top performer
top_performer = max(data, key=lambda x: x["score"])

# Print results
print(f"Average score: {avg_score}")
print(f"Max score: {max_score}")
print(f"Min score: {min_score}")
print(f"Top performer: {top_performer['name']} with score {top_performer['score']}")

# Create summary
summary = {
    "total_students": len(data),
    "average_score": avg_score,
    "score_range": {"min": min_score, "max": max_score},
    "top_performer": top_performer["name"]
}

print(f"\\nSummary: {json.dumps(summary, indent=2)}")
"""
        
        result = await execute_sandbox_code(code)
        
        assert result["success"] is True
        assert "Average score: 87.6" in result["output"]
        assert "Top performer: Diana" in result["output"]
        assert '"total_students": 5' in result["output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])