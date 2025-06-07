"""
Tests for sandbox file persistence and session management.
These tests verify that files persist across sandbox sessions and that
sandbox instances are properly reused.
"""

import asyncio
import json
import os
import time
import pytest
from dotenv import load_dotenv

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    check_sandbox_availability,
    E2BSandbox,
    PyodideSandbox,
)
from wandb_mcp_server.mcp_tools.code_sandbox.sandbox_file_utils import (
    write_json_to_sandbox,
)

load_dotenv()


class TestSandboxPersistence:
    """Test file persistence across sandbox sessions."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_file_persistence(self):
        """Test that files persist across E2B sandbox sessions."""
        # Step 1: Write a file in the first session
        write_code = """
import json
import time

data = {
    "session": "first",
    "timestamp": time.time(),
    "message": "This file should persist across sessions",
    "items": [1, 2, 3, 4, 5]
}

with open('/tmp/persistence_test.json', 'w') as f:
    json.dump(data, f, indent=2)
    
print("File written successfully")
print(f"Data: {json.dumps(data, indent=2)}")
"""
        
        result1 = await execute_sandbox_code(write_code, sandbox_type="e2b")
        assert result1["success"] is True
        assert "File written successfully" in result1["output"]
        
        # Step 2: Read and modify the file in a second session
        modify_code = """
import json

# Read the existing file
with open('/tmp/persistence_test.json', 'r') as f:
    data = json.load(f)
    
print(f"Original session: {data['session']}")
print(f"Original timestamp: {data['timestamp']}")

# Modify the data
data['session'] = 'second'
data['modified'] = True
data['items'].extend([6, 7, 8])

# Write it back
with open('/tmp/persistence_test_modified.json', 'w') as f:
    json.dump(data, f, indent=2)
    
print("Modified file written successfully")
"""
        
        result2 = await execute_sandbox_code(modify_code, sandbox_type="e2b")
        assert result2["success"] is True
        assert "Original session: first" in result2["output"]
        assert "Modified file written successfully" in result2["output"]
        
        # Step 3: Verify both files exist in a third session
        verify_code = """
import json
import os

files = os.listdir('/tmp')
json_files = [f for f in files if f.endswith('.json')]
print(f"JSON files in /tmp: {json_files}")

# Read both files
for filename in ['persistence_test.json', 'persistence_test_modified.json']:
    filepath = f'/tmp/{filename}'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"\\n{filename}:")
        print(f"  Session: {data.get('session')}")
        print(f"  Items count: {len(data.get('items', []))}")
        print(f"  Modified: {data.get('modified', False)}")
    else:
        print(f"\\n{filename}: NOT FOUND")
"""
        
        result3 = await execute_sandbox_code(verify_code, sandbox_type="e2b")
        assert result3["success"] is True
        assert "persistence_test.json" in result3["output"]
        assert "persistence_test_modified.json" in result3["output"]
        assert "Session: first" in result3["output"]
        assert "Session: second" in result3["output"]
        assert "Items count: 5" in result3["output"]
        assert "Items count: 8" in result3["output"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_same_instance(self):
        """Test that E2B uses the same sandbox instance across calls."""
        # Create a unique instance identifier
        instance_id_code = """
import os
import hashlib

# Create a unique identifier for this sandbox instance
instance_id_file = '/tmp/sandbox_instance_id.txt'

if os.path.exists(instance_id_file):
    with open(instance_id_file, 'r') as f:
        instance_id = f.read().strip()
    print(f"Existing sandbox instance: {instance_id}")
else:
    # First time in this sandbox
    instance_id = hashlib.md5(str(os.getpid()).encode()).hexdigest()[:8]
    with open(instance_id_file, 'w') as f:
        f.write(instance_id)
    print(f"New sandbox instance: {instance_id}")
    
# Also write a counter file
counter_file = '/tmp/execution_counter.txt'
if os.path.exists(counter_file):
    with open(counter_file, 'r') as f:
        counter = int(f.read().strip())
else:
    counter = 0

counter += 1
with open(counter_file, 'w') as f:
    f.write(str(counter))
    
print(f"Execution count: {counter}")
"""
        
        # Run multiple times
        instance_ids = []
        for i in range(3):
            result = await execute_sandbox_code(
                instance_id_code, 
                sandbox_type="e2b"
            )
            assert result["success"] is True
            
            # Extract instance ID from output
            output = result["output"]
            if "Existing sandbox instance:" in output:
                # Not the first run
                assert i > 0, "Should be new instance on first run"
                line = [l for l in output.split('\n') if "Existing sandbox instance:" in l][0]
                instance_id = line.split(": ")[1]
            else:
                # First run
                assert i == 0, "Should be existing instance after first run"
                line = [l for l in output.split('\n') if "New sandbox instance:" in l][0]
                instance_id = line.split(": ")[1]
            
            instance_ids.append(instance_id)
            
            # Check execution count
            assert f"Execution count: {i + 1}" in output
        
        # All instance IDs should be the same
        assert len(set(instance_ids)) == 1, f"Expected same instance, got {instance_ids}"
    
    @pytest.mark.asyncio
    async def test_pyodide_no_persistence(self):
        """Test that Pyodide does NOT persist files across sessions (current behavior)."""
        # Check if Pyodide is available
        available, types, _ = check_sandbox_availability()
        if "pyodide" not in types:
            pytest.skip("Pyodide not available (Deno not installed)")
        
        # Write a file
        write_code = """
import json

data = {"test": "pyodide persistence", "value": 42}
with open('/tmp/pyodide_test.json', 'w') as f:
    json.dump(data, f)
print("File written in Pyodide")

# List files to confirm
import os
files = os.listdir('/tmp')
print(f"Files after write: {files}")
"""
        
        result1 = await execute_sandbox_code(write_code, sandbox_type="pyodide")
        assert result1["success"] is True
        assert "File written in Pyodide" in result1["output"]
        
        # Try to read in a new session
        read_code = """
import os
import json

# List files
files = os.listdir('/tmp')
print(f"Files in new session: {files}")

# Try to read the file
try:
    with open('/tmp/pyodide_test.json', 'r') as f:
        data = json.load(f)
    print(f"File found! Data: {data}")
except FileNotFoundError:
    print("File NOT FOUND - Pyodide doesn't persist files across sessions")
"""
        
        result2 = await execute_sandbox_code(read_code, sandbox_type="pyodide")
        assert result2["success"] is True
        # Currently, Pyodide creates a new instance each time, so files don't persist
        assert "File NOT FOUND" in result2["output"]
    
    @pytest.mark.asyncio
    async def test_sandbox_file_utils(self):
        """Test the sandbox file utilities."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        # Test data
        test_data = {
            "utility_test": True,
            "timestamp": time.time(),
            "nested": {
                "key1": "value1",
                "key2": [1, 2, 3]
            }
        }
        
        # Write using utility function
        await write_json_to_sandbox(
            json_data=test_data,
            filename="utility_test.json",
            path_prefix="/tmp/"
        )
        
        # Verify the file was written by reading it back
        verify_code = """
import json
import os

# Check if file exists
filepath = '/tmp/utility_test.json'
if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print("File found!")
    print(f"utility_test: {data.get('utility_test')}")
    print(f"Has timestamp: {'timestamp' in data}")
    print(f"Nested keys: {list(data.get('nested', {}).keys())}")
else:
    print("File NOT FOUND")
"""
        
        result = await execute_sandbox_code(verify_code)
        assert result["success"] is True
        assert "File found!" in result["output"]
        assert "utility_test: True" in result["output"]
        assert "Nested keys: ['key1', 'key2']" in result["output"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_native_file_operations(self):
        """Test E2B native file operations."""
        # Create an E2B sandbox instance
        sandbox = E2BSandbox(os.getenv("E2B_API_KEY"))
        await sandbox.create_sandbox()
        
        try:
            # Write a file using native operations
            test_content = json.dumps({
                "native_write": True,
                "timestamp": time.time(),
                "data": [1, 2, 3, 4, 5]
            }, indent=2)
            
            await sandbox.writeFile("/tmp/native_test.json", test_content)
            
            # Verify by executing code that reads the file
            read_code = """
import json

with open('/tmp/native_test.json', 'r') as f:
    data = json.load(f)
    
print(f"Native write successful: {data.get('native_write')}")
print(f"Data length: {len(data.get('data', []))}")
"""
            
            result = await sandbox.execute_code(read_code)
            assert result["success"] is True
            assert "Native write successful: True" in result["output"]
            assert "Data length: 5" in result["output"]
            
        finally:
            await sandbox.close_sandbox()
    
    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self):
        """Test concurrent file operations in sandboxes."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        # Create multiple files concurrently
        async def write_file(index: int):
            code = f"""
import json

data = {{"file_index": {index}, "squared": {index**2}}}
with open('/tmp/concurrent_{index}.json', 'w') as f:
    json.dump(data, f)
print(f"File {index} written")
"""
            return await execute_sandbox_code(code)
        
        # Write files concurrently
        tasks = [write_file(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["success"] for r in results)
        
        # Verify all files exist (in E2B they should persist)
        verify_code = """
import os
import json

files = sorted([f for f in os.listdir('/tmp') if f.startswith('concurrent_')])
print(f"Concurrent files found: {len(files)}")

for f in files:
    with open(f'/tmp/{f}', 'r') as file:
        data = json.load(file)
        print(f"{f}: index={data['file_index']}, squared={data['squared']}")
"""
        
        result = await execute_sandbox_code(verify_code)
        assert result["success"] is True
        
        # E2B should find all 5 files, Pyodide might not
        if "e2b" in types and result.get("sandbox_used") == "e2b":
            assert "Concurrent files found: 5" in result["output"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_cleanup(self):
        """Test E2B sandbox cleanup functionality."""
        # First, ensure we have a sandbox instance
        code = "print('Creating sandbox for cleanup test')"
        result = await execute_sandbox_code(code, sandbox_type="e2b")
        assert result["success"] is True
        
        # Now cleanup
        await E2BSandbox.cleanup_shared_sandbox()
        
        # Next execution should create a new instance
        instance_code = """
import os

# This should be a fresh sandbox
marker_file = '/tmp/cleanup_test_marker.txt'
if os.path.exists(marker_file):
    print("ERROR: Marker file exists - sandbox was not cleaned up!")
else:
    print("SUCCESS: Clean sandbox - no marker file")
    with open(marker_file, 'w') as f:
        f.write("marker")
"""
        
        result = await execute_sandbox_code(instance_code, sandbox_type="e2b")
        assert result["success"] is True
        assert "SUCCESS: Clean sandbox" in result["output"]
        assert "ERROR: Marker file exists" not in result["output"]


class TestSandboxWandbIntegration:
    """Test sandbox integration with W&B query results."""
    
    @pytest.mark.asyncio
    async def test_wandb_result_file_writing(self):
        """Test that W&B query results are written to sandbox files."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        # Simulate a W&B query result
        mock_wandb_result = {
            "project": "test-project",
            "runs": [
                {"id": "run1", "name": "experiment-1", "state": "finished"},
                {"id": "run2", "name": "experiment-2", "state": "finished"}
            ],
            "total": 2
        }
        
        # Write to sandbox
        await write_json_to_sandbox(
            json_data=mock_wandb_result,
            filename="wandb_query_result.json"
        )
        
        # Process the result in sandbox
        analysis_code = """
import json

# Read the W&B query result
with open('/tmp/wandb_query_result.json', 'r') as f:
    data = json.load(f)

print(f"Project: {data['project']}")
print(f"Total runs: {data['total']}")

# Analyze the runs
finished_runs = [r for r in data['runs'] if r['state'] == 'finished']
print(f"Finished runs: {len(finished_runs)}")

# Create a summary
summary = {
    "project": data['project'],
    "total_runs": data['total'],
    "finished_runs": len(finished_runs),
    "run_names": [r['name'] for r in data['runs']]
}

# Write summary
with open('/tmp/wandb_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
    
print("Summary created successfully")
"""
        
        result = await execute_sandbox_code(analysis_code)
        assert result["success"] is True
        assert "Project: test-project" in result["output"]
        assert "Total runs: 2" in result["output"]
        assert "Summary created successfully" in result["output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])