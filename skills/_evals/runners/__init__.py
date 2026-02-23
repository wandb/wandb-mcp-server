"""Agent runners for MCP skill evaluations.

Provides CLI-based runners that invoke real coding agents (Claude Code, Codex)
against the W&B MCP server and parse their structured output.
"""

from skills._evals.runners.base import AgentResult, AgentRunner

__all__ = ["AgentRunner", "AgentResult"]
