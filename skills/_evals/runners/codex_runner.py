"""OpenAI Codex CLI runner for MCP skill evaluations.

Invokes `codex exec` (non-interactive mode) with the W&B MCP server
configured and parses the output to extract tool calls.

Requires:
    - `npx @openai/codex` available (Codex CLI >= 0.100)
    - OPENAI_API_KEY environment variable set
    - WANDB_API_KEY environment variable set

Example:
    runner = CodexRunner()
    result = runner.run("Add tracing to my OpenAI app", skill_name="quickstart")
    print(result.tools_called)
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

from skills._evals.runners.base import AgentResult, AgentRunner, parse_codex_output


class CodexRunner(AgentRunner):
    """Runs prompts via the Codex CLI with W&B MCP server attached."""

    name: str = "codex"

    def __init__(
        self,
        model: str = "o3-mini",
        mcp_server_command: str | None = None,
    ):
        """Initialize the Codex runner.

        Args:
            model: OpenAI model to use (e.g., "o3-mini", "o3", "gpt-4.1").
            mcp_server_command: Override for the MCP server command.
        """
        self.model = model
        self.mcp_server_command = mcp_server_command
        self._validate_cli()

    def _validate_cli(self):
        result = subprocess.run(
            ["npx", "@openai/codex", "--version"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Codex CLI not found. Install via: npm install -g @openai/codex"
            )

    def _write_mcp_config(self) -> str:
        """Write a temporary Codex MCP config file.

        Codex reads MCP config from a JSON file passed via
        `codex mcp add-from-file` or from ~/.codex/config.toml.
        For eval isolation, we use a temp config.

        Returns:
            Path to the temp config file.
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

        fd, path = tempfile.mkstemp(suffix=".json", prefix="codex_mcp_config_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)
        return path

    def run(self, prompt: str, skill_name: str, timeout: int = 180) -> AgentResult:
        """Run a prompt via codex exec with the MCP server.

        Args:
            prompt: The user prompt to evaluate.
            skill_name: Skill being tested.
            timeout: Max seconds to wait.

        Returns:
            AgentResult with parsed tool calls and response.
        """
        config_path = self._write_mcp_config()

        full_prompt = (
            f"You are testing the '{skill_name}' skill for the W&B MCP server. "
            f"Use the available MCP tools to accomplish the following task:\n\n"
            f"{prompt}"
        )

        cmd = [
            "npx", "@openai/codex", "exec",
            full_prompt,
            "--full-auto",
            "-c", f'model="{self.model}"',
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ},
            )

            duration_ms = int((time.monotonic() - start) * 1000)

            result = parse_codex_output(proc.stdout)
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
