"""Claude Code CLI runner for MCP skill evaluations.

Invokes `claude -p` (print mode) with the W&B MCP server configured via
--mcp-config, and parses the stream-json output to extract tool calls.

Requires:
    - `claude` CLI installed and authenticated (Claude Code >= 2.0)
    - Run `claude login` or `claude setup-token` first
    - WANDB_API_KEY environment variable set (for MCP server)

Known issues:
    - `claude -p` may hang in non-TTY environments (subprocess, CI).
      If this happens, try running from a terminal directly or use
      `claude setup-token` for non-interactive auth.
    - The Claude CLI needs its own auth separate from ANTHROPIC_API_KEY.

Example:
    runner = ClaudeRunner()
    result = runner.run("Add tracing to my OpenAI app", skill_name="quickstart")
    print(result.tools_called)  # ['query_weave_traces_tool', ...]
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

from skills._evals.runners.base import AgentResult, AgentRunner, parse_claude_stream_json


class ClaudeRunner(AgentRunner):
    """Runs prompts via the Claude Code CLI with W&B MCP server attached."""

    name: str = "claude"

    def __init__(
        self,
        model: str = "sonnet",
        max_budget_usd: float = 0.50,
        mcp_server_command: str | None = None,
    ):
        """Initialize the Claude runner.

        Args:
            model: Claude model alias (e.g., "sonnet", "opus", "haiku").
            max_budget_usd: Maximum dollar spend per eval run.
            mcp_server_command: Override for the MCP server command.
                Defaults to running wandb-mcp-server from the repo.
        """
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.mcp_server_command = mcp_server_command
        self._validate_cli()

    def _validate_cli(self):
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude Code CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _build_mcp_config(self) -> str:
        """Build MCP server config JSON and write to a temp file.

        Returns the path to the temp config file. Claude CLI's --mcp-config
        accepts a file path or inline JSON.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        config = {
            "mcpServers": {
                "wandb": {
                    "command": self.mcp_server_command or "uv",
                    "args": self.mcp_server_command and [] or [
                        "run",
                        "--directory", repo_root,
                        "wandb-mcp-server",
                    ],
                    "env": {
                        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    },
                }
            }
        }

        fd, path = tempfile.mkstemp(suffix=".json", prefix="mcp_config_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)
        return path

    def run(self, prompt: str, skill_name: str, timeout: int = 120) -> AgentResult:
        """Run a prompt via claude -p with the MCP server.

        Args:
            prompt: The user prompt to evaluate.
            skill_name: Skill being tested (included in system prompt context).
            timeout: Max seconds to wait for the agent.

        Returns:
            AgentResult with parsed tool calls and response.
        """
        config_path = self._build_mcp_config()

        system_context = (
            f"You are testing the '{skill_name}' skill for the W&B MCP server. "
            f"Use the available MCP tools (prefixed mcp__wandb__) to accomplish the task. "
            f"Be concise and focus on using the right tools."
        )

        cmd = [
            "claude", "-p", prompt,
            "--mcp-config", config_path,
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--model", self.model,
            "--max-budget-usd", str(self.max_budget_usd),
            "--append-system-prompt", system_context,
            "--no-session-persistence",
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
            )

            duration_ms = int((time.monotonic() - start) * 1000)

            result = parse_claude_stream_json(proc.stdout)
            result.duration_ms = duration_ms
            result.exit_code = proc.returncode
            result.raw_output = proc.stdout

            if proc.returncode != 0 and not result.response_text:
                result.error = proc.stderr or f"Exit code {proc.returncode}"

            return result

        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return AgentResult(
                duration_ms=duration_ms,
                exit_code=-1,
                error=f"Timed out after {timeout}s",
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return AgentResult(
                duration_ms=duration_ms,
                exit_code=-1,
                error=str(e),
            )
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
