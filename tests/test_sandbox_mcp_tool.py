"""
Tests for the MCP sandbox tool integration.
"""

import json
import pytest
from unittest.mock import patch
from dotenv import load_dotenv

from tests.anthropic_test_utils import (
    call_anthropic,
    extract_anthropic_text,
    extract_anthropic_tool_use,
    get_anthropic_tool_result_message,
)
from wandb_mcp_server.mcp_tools.tools_utils import generate_anthropic_tool_schema
from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION

load_dotenv()


class TestSandboxMCPTool:
    """Test the MCP sandbox tool integration."""

    def test_anthropic_tool_schema_generation(self):
        """Test that the tool schema is properly generated for Anthropic."""
        schema = generate_anthropic_tool_schema(
            "execute_sandbox_code_tool",
            EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION,
            {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                "sandbox_type": {"type": "string", "description": "Sandbox type (e2b or pyodide)", "enum": ["e2b", "pyodide"]},
                "install_packages": {"type": "array", "items": {"type": "string"}, "description": "Packages to install (E2B only)"}
            }
        )
        
        assert schema["name"] == "execute_sandbox_code_tool"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "code" in schema["input_schema"]["properties"]
        assert "timeout" in schema["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution."""
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Hello, World!\n",
                "error": None,
                "logs": [],
                "sandbox_used": "pyodide"
            }
            
            # Import here to avoid circular imports during testing
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            result = await execute_sandbox_code_tool(
                code="print('Hello, World!')",
                timeout=30
            )
            
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert result_dict["output"] == "Hello, World!\n"
            assert result_dict["sandbox_used"] == "pyodide"

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution with error."""
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.return_value = {
                "success": False,
                "output": "",
                "error": "NameError: name 'undefined_variable' is not defined",
                "logs": [],
                "sandbox_used": "pyodide"
            }
            
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            result = await execute_sandbox_code_tool(
                code="print(undefined_variable)",
                timeout=30
            )
            
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "NameError" in result_dict["error"]
            assert result_dict["sandbox_used"] == "pyodide"

    @pytest.mark.asyncio
    async def test_tool_exception_handling(self):
        """Test tool exception handling."""
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.side_effect = Exception("Unexpected error")
            
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            result = await execute_sandbox_code_tool(
                code="print('test')",
                timeout=30
            )
            
            result_dict = json.loads(result)
            assert result_dict["success"] is False
            assert "Tool execution failed" in result_dict["error"]
            assert result_dict["sandbox_used"] == "none"

    @pytest.mark.asyncio
    async def test_tool_with_parameters(self):
        """Test tool execution with various parameters."""
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Success with E2B\n",
                "error": None,
                "logs": ["Starting E2B sandbox", "Code executed successfully"],
                "sandbox_used": "e2b"
            }
            
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            result = await execute_sandbox_code_tool(
                code="print('Success with E2B')",
                timeout=60,
                sandbox_type="e2b",
                install_packages=["numpy", "pandas"]
            )
            
            # Verify the mock was called with correct parameters
            mock_execute.assert_called_once_with(
                code="print('Success with E2B')",
                timeout=60,
                sandbox_type="e2b",
                install_packages=["numpy", "pandas"]
            )
            
            result_dict = json.loads(result)
            assert result_dict["success"] is True
            assert result_dict["sandbox_used"] == "e2b"
            assert len(result_dict["logs"]) == 2


@pytest.mark.integration
@pytest.mark.skipif(
    not any(key in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY'] for key in __import__('os').environ),
    reason="No AI API key available for integration testing"
)
class TestSandboxAnthropicIntegration:
    """Integration tests with Anthropic API."""

    @pytest.mark.asyncio
    async def test_anthropic_tool_usage(self):
        """Test that Anthropic can successfully use the sandbox tool."""
        tool_schema = generate_anthropic_tool_schema(
            "execute_sandbox_code_tool",
            EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION,
            {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                "sandbox_type": {"type": "string", "description": "Sandbox type (e2b or pyodide)", "enum": ["e2b", "pyodide"]},
                "install_packages": {"type": "array", "items": {"type": "string"}, "description": "Packages to install (E2B only)"}
            }
        )
        
        messages = [
            {
                "role": "user",
                "content": "Please calculate the square root of 144 using Python code in a sandbox."
            }
        ]
        
        response = await call_anthropic(messages, tools=[tool_schema])
        
        # Extract tool use
        tool_use = extract_anthropic_tool_use(response)
        assert tool_use is not None
        assert tool_use["name"] == "execute_sandbox_code_tool"
        assert "code" in tool_use["input"]
        
        # Mock the tool execution
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "The square root of 144 is: 12.0\n",
                "error": None,
                "logs": [],
                "sandbox_used": "pyodide"
            }
            
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            tool_result = await execute_sandbox_code_tool(**tool_use["input"])
            
            # Send tool result back to Anthropic
            tool_result_message = get_anthropic_tool_result_message(
                tool_use["id"], tool_result
            )
            
            messages.append(response)
            messages.append(tool_result_message)
            
            final_response = await call_anthropic(messages, tools=[tool_schema])
            final_text = extract_anthropic_text(final_response)
            
            assert "12" in final_text  # Should mention the result

    @pytest.mark.asyncio
    async def test_anthropic_error_handling(self):
        """Test Anthropic handling of sandbox errors."""
        tool_schema = generate_anthropic_tool_schema(
            "execute_sandbox_code_tool",
            EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION,
            {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
            }
        )
        
        messages = [
            {
                "role": "user",
                "content": "Please run this Python code: print(undefined_variable)"
            }
        ]
        
        response = await call_anthropic(messages, tools=[tool_schema])
        tool_use = extract_anthropic_tool_use(response)
        
        # Mock a failed execution
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.execute_sandbox_code') as mock_execute:
            mock_execute.return_value = {
                "success": False,
                "output": "",
                "error": "NameError: name 'undefined_variable' is not defined",
                "logs": [],
                "sandbox_used": "pyodide"
            }
            
            from wandb_mcp_server.server import execute_sandbox_code_tool
            
            tool_result = await execute_sandbox_code_tool(**tool_use["input"])
            
            tool_result_message = get_anthropic_tool_result_message(
                tool_use["id"], tool_result
            )
            
            messages.append(response)
            messages.append(tool_result_message)
            
            final_response = await call_anthropic(messages, tools=[tool_schema])
            final_text = extract_anthropic_text(final_response)
            
            # Anthropic should acknowledge the error
            assert any(word in final_text.lower() for word in ["error", "failed", "undefined"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])