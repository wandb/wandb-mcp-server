"""Base classes for agent runners.

AgentRunner is the abstract interface for running a coding agent against
the MCP server. AgentResult is the standardized output format that scorers
consume.

The stream-json parser extracts tool calls from Claude Code's structured
output format (each line is a JSON object with a "type" field).
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResult:
    """Standardized result from an agent run.

    Attributes:
        response_text: The agent's final text response.
        tools_called: Unique MCP tool names invoked during the run.
        workflow_steps: Inferred workflow steps from the tool call sequence.
        total_tool_calls: Total number of tool invocations.
        duration_ms: Wall-clock time for the run in milliseconds.
        exit_code: Process exit code (0 = success).
        raw_output: Full agent output for debugging.
        error: Error message if the run failed.
    """

    response_text: str = ""
    tools_called: list[str] = field(default_factory=list)
    workflow_steps: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    duration_ms: int = 0
    exit_code: int = 0
    raw_output: str = ""
    error: Optional[str] = None


# Mapping from MCP tool names to workflow step labels
TOOL_TO_STEP = {
    "count_weave_traces_tool": "count",
    "query_weave_traces_tool": "query_traces",
    "query_wandb_tool": "query_runs",
    "query_wandb_entity_projects": "identify_project",
    "create_wandb_report_tool": "create_report",
}


def parse_claude_stream_json(raw: str) -> AgentResult:
    """Parse Claude Code's stream-json output into an AgentResult.

    Claude's --output-format stream-json emits one JSON object per line.
    Relevant event types:
    - {"type": "assistant", "message": {...}} -- contains text/tool_use blocks
    - {"type": "result", "result": "..."} -- final result text

    Args:
        raw: Raw stdout from claude -p --output-format stream-json.

    Returns:
        Parsed AgentResult with tool calls extracted.
    """
    tools_called = []
    tool_names_seen: set[str] = set()
    response_parts: list[str] = []
    total_tool_calls = 0

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "result":
            result_text = event.get("result", "")
            if result_text:
                response_parts.append(result_text)

        elif event_type == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "text":
                    response_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    total_tool_calls += 1
                    if tool_name and tool_name not in tool_names_seen:
                        tool_names_seen.add(tool_name)
                        tools_called.append(tool_name)

    mcp_tools = [t for t in tools_called if t.startswith("mcp__wandb__")]
    clean_tools = [t.replace("mcp__wandb__", "") for t in mcp_tools]

    workflow_steps = []
    for tool in clean_tools:
        step = TOOL_TO_STEP.get(tool, tool)
        if step not in workflow_steps:
            workflow_steps.append(step)

    return AgentResult(
        response_text="\n".join(response_parts),
        tools_called=clean_tools,
        workflow_steps=workflow_steps,
        total_tool_calls=total_tool_calls,
    )


def parse_codex_output(raw: str) -> AgentResult:
    """Parse Codex CLI output into an AgentResult.

    Codex exec outputs a mix of text and tool call indicators.
    We extract MCP tool calls by looking for patterns like:
    - "Calling tool: tool_name" or similar structured output
    - The final text output as the response

    Args:
        raw: Raw stdout from codex exec.

    Returns:
        Parsed AgentResult with tool calls extracted.
    """
    tools_called = []
    tool_names_seen: set[str] = set()
    total_tool_calls = 0

    tool_pattern = re.compile(r"(?:Calling|Using|Tool).*?:\s*([\w_]+)")
    mcp_pattern = re.compile(r"mcp__wandb__([\w_]+)")

    for line in raw.strip().splitlines():
        for match in mcp_pattern.finditer(line):
            tool_name = match.group(1)
            total_tool_calls += 1
            if tool_name not in tool_names_seen:
                tool_names_seen.add(tool_name)
                tools_called.append(tool_name)

        for match in tool_pattern.finditer(line):
            tool_name = match.group(1)
            if tool_name.startswith("mcp__wandb__"):
                clean = tool_name.replace("mcp__wandb__", "")
                total_tool_calls += 1
                if clean not in tool_names_seen:
                    tool_names_seen.add(clean)
                    tools_called.append(clean)

    workflow_steps = []
    for tool in tools_called:
        step = TOOL_TO_STEP.get(tool, tool)
        if step not in workflow_steps:
            workflow_steps.append(step)

    return AgentResult(
        response_text=raw,
        tools_called=tools_called,
        workflow_steps=workflow_steps,
        total_tool_calls=max(total_tool_calls, len(tools_called)),
    )


class AgentRunner(ABC):
    """Abstract base class for agent runners.

    Subclasses implement run() to invoke a specific coding agent CLI
    against the W&B MCP server and return a standardized AgentResult.
    """

    name: str = "base"

    @abstractmethod
    def run(self, prompt: str, skill_name: str, timeout: int = 120) -> AgentResult:
        """Run a prompt against the MCP server via the agent CLI.

        Args:
            prompt: The user prompt to send to the agent.
            skill_name: Name of the skill being evaluated (for context).
            timeout: Maximum seconds to wait for the agent.

        Returns:
            AgentResult with the agent's response and extracted tool calls.
        """
        ...
