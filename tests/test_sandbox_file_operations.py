"""
Tests for sandbox file operations and native file system access.
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    check_sandbox_availability,
    E2BSandbox,
    PyodideSandbox,
)

load_dotenv()


class TestNativeFileOperations:
    """Test native file operations in sandboxes."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_direct_file_write_read(self):
        """Test E2B's native file.write() and reading back."""
        sandbox = E2BSandbox(os.getenv("E2B_API_KEY"))
        await sandbox.create_sandbox()
        
        try:
            # Test 1: Write a simple text file
            await sandbox.writeFile("/tmp/test.txt", "Hello from E2B!")
            
            # Read it back using code execution
            result = await sandbox.execute_code("""
with open('/tmp/test.txt', 'r') as f:
    content = f.read()
print(f"Content: '{content}'")
""")
            assert result["success"] is True
            assert "Content: 'Hello from E2B!'" in result["output"]
            
            # Test 2: Write JSON data
            json_data = {"test": True, "values": [1, 2, 3]}
            await sandbox.writeFile("/tmp/data.json", json.dumps(json_data))
            
            result = await sandbox.execute_code("""
import json
with open('/tmp/data.json', 'r') as f:
    data = json.load(f)
print(f"JSON data: {data}")
print(f"Values sum: {sum(data['values'])}")
""")
            assert result["success"] is True
            assert "Values sum: 6" in result["output"]
            
            # Test 3: Write to nested directories
            await sandbox.writeFile("/tmp/nested/dir/file.txt", "Nested file content")
            
            result = await sandbox.execute_code("""
import os
print(f"Nested dir exists: {os.path.exists('/tmp/nested/dir')}")
with open('/tmp/nested/dir/file.txt', 'r') as f:
    print(f"Nested file: '{f.read()}'")
""")
            assert result["success"] is True
            assert "Nested file: 'Nested file content'" in result["output"]
            
        finally:
            await sandbox.close_sandbox()
    
    @pytest.mark.asyncio
    async def test_pyodide_file_operations(self):
        """Test Pyodide file operations."""
        available, types, _ = check_sandbox_availability()
        if "pyodide" not in types:
            pytest.skip("Pyodide not available")
        
        sandbox = PyodideSandbox()
        
        # Test writing a file through code execution
        await sandbox.writeFile("/tmp/pyodide_test.txt", "Pyodide file content")
        
        # Note: In current implementation, this executes code to write the file
        # Each execution creates a new Pyodide instance, so we can't read it back
        # This test mainly verifies no errors occur
    
    @pytest.mark.asyncio
    async def test_binary_file_handling(self):
        """Test handling of binary files."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        # Create binary data using base64
        code = """
import base64
import json

# Create some binary data
binary_data = bytes([0, 1, 2, 3, 255, 254, 253, 252])
encoded = base64.b64encode(binary_data).decode('utf-8')

# Write as base64
with open('/tmp/binary_data.b64', 'w') as f:
    f.write(encoded)

# Also write the length for verification
with open('/tmp/binary_info.json', 'w') as f:
    json.dump({
        "length": len(binary_data),
        "first_byte": binary_data[0],
        "last_byte": binary_data[-1]
    }, f)

print(f"Binary data length: {len(binary_data)}")
print(f"Base64 length: {len(encoded)}")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        assert "Binary data length: 8" in result["output"]
    
    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        # Create a large file (1MB of data)
        code = """
import json

# Create 1MB of JSON data
large_data = {
    "items": [
        {"id": i, "value": f"item_{i}" * 10} 
        for i in range(10000)
    ]
}

# Write it
with open('/tmp/large_file.json', 'w') as f:
    json.dump(large_data, f)

# Check file size
import os
size = os.path.getsize('/tmp/large_file.json')
print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
print(f"Number of items: {len(large_data['items'])}")
"""
        
        result = await execute_sandbox_code(code, timeout=60)
        assert result["success"] is True
        assert "Number of items: 10000" in result["output"]
    
    @pytest.mark.asyncio
    async def test_file_permissions_and_errors(self):
        """Test file permission handling and error cases."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        code = """
import os

# Test 1: Try to read non-existent file
try:
    with open('/tmp/does_not_exist.txt', 'r') as f:
        content = f.read()
except FileNotFoundError as e:
    print("✓ FileNotFoundError caught correctly")

# Test 2: Try to write to a directory as if it were a file
try:
    os.makedirs('/tmp/test_dir', exist_ok=True)
    with open('/tmp/test_dir', 'w') as f:
        f.write("This should fail")
except (IsADirectoryError, OSError) as e:
    print("✓ Directory write error caught correctly")

# Test 3: Create and check file permissions
test_file = '/tmp/permission_test.txt'
with open(test_file, 'w') as f:
    f.write("Permission test")

# Check if file exists and is readable
print(f"✓ File exists: {os.path.exists(test_file)}")
print(f"✓ File readable: {os.access(test_file, os.R_OK)}")
print(f"✓ File writable: {os.access(test_file, os.W_OK)}")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        assert "FileNotFoundError caught correctly" in result["output"]
        assert "File exists: True" in result["output"]
    
    @pytest.mark.asyncio
    async def test_directory_operations(self):
        """Test directory creation and listing."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        code = """
import os
import json

# Create nested directory structure
dirs = [
    '/tmp/test_project',
    '/tmp/test_project/src',
    '/tmp/test_project/tests',
    '/tmp/test_project/data'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")

# Create files in each directory
files = {
    '/tmp/test_project/README.md': '# Test Project',
    '/tmp/test_project/src/main.py': 'print("Hello")',
    '/tmp/test_project/tests/test_main.py': 'assert True',
    '/tmp/test_project/data/config.json': json.dumps({"debug": True})
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)

# List directory structure
for root, dirs, files in os.walk('/tmp/test_project'):
    level = root.replace('/tmp/test_project', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        assert "Created: /tmp/test_project" in result["output"]
        assert "README.md" in result["output"]
        assert "main.py" in result["output"]
    
    @pytest.mark.asyncio
    async def test_csv_data_processing(self):
        """Test CSV file handling for data processing."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        code = """
import csv
import json

# Create CSV data
csv_data = [
    ['name', 'score', 'grade'],
    ['Alice', '95', 'A'],
    ['Bob', '87', 'B'],
    ['Charlie', '92', 'A'],
    ['Diana', '78', 'C'],
    ['Eve', '91', 'A']
]

# Write CSV file
with open('/tmp/grades.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print("CSV file created")

# Read and process CSV
with open('/tmp/grades.csv', 'r') as f:
    reader = csv.DictReader(f)
    students = list(reader)

# Calculate statistics
scores = [int(s['score']) for s in students]
avg_score = sum(scores) / len(scores)
a_grades = len([s for s in students if s['grade'] == 'A'])

print(f"Average score: {avg_score:.1f}")
print(f"Number of A grades: {a_grades}")

# Save summary as JSON
summary = {
    'total_students': len(students),
    'average_score': avg_score,
    'grade_distribution': {
        grade: len([s for s in students if s['grade'] == grade])
        for grade in set(s['grade'] for s in students)
    }
}

with open('/tmp/grade_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to JSON")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        assert "CSV file created" in result["output"]
        assert "Average score:" in result["output"]
        assert "Number of A grades: 3" in result["output"]


class TestSandboxSecurity:
    """Test sandbox security and isolation."""
    
    @pytest.mark.asyncio
    async def test_filesystem_isolation(self):
        """Test that sandboxes have proper filesystem isolation."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        code = """
import os
import sys

# Test 1: Check current directory
print(f"Current directory: {os.getcwd()}")

# Test 2: Try to access parent directories
try:
    # This should work in sandboxes
    parent_files = os.listdir('/')
    print(f"Root directory accessible (OK in sandbox): {len(parent_files)} items")
except Exception as e:
    print(f"Cannot access root: {e}")

# Test 3: Check Python path
print(f"Python paths: {len(sys.path)} entries")

# Test 4: Check environment
env_vars = dict(os.environ)
# Remove sensitive vars for display
safe_env = {k: v for k, v in env_vars.items() 
            if 'KEY' not in k and 'TOKEN' not in k and 'SECRET' not in k}
print(f"Environment variables: {len(env_vars)} total, {len(safe_env)} safe to display")

# Test 5: Check temp directory
temp_files = os.listdir('/tmp')
print(f"Temp directory: {len(temp_files)} files")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        # Should execute without security errors
        assert "Current directory:" in result["output"]
    
    @pytest.mark.asyncio
    async def test_network_isolation(self):
        """Test network isolation in sandboxes."""
        available, types, _ = check_sandbox_availability()
        if not available:
            pytest.skip("No sandboxes available")
        
        code = """
# Test network access (should fail in Pyodide, might work in E2B)
import socket

try:
    # Try to resolve a domain
    ip = socket.gethostbyname('example.com')
    print(f"Network access available: resolved example.com to {ip}")
except Exception as e:
    print(f"Network access restricted: {type(e).__name__}")

# Test if we can create sockets
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()
    print("Socket creation allowed")
except Exception as e:
    print(f"Socket creation restricted: {type(e).__name__}")
"""
        
        result = await execute_sandbox_code(code)
        assert result["success"] is True
        # Different sandboxes have different network policies
        # Just verify it executes without crashing


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])