#!/usr/bin/env bash
# Convenience script for running MCP skill evaluations.
#
# Usage:
#   ./skills/_evals/run.sh                                    # quickstart + mock + TUI (default profile)
#   ./skills/_evals/run.sh quickstart claude                  # quickstart + Claude Code
#   ./skills/_evals/run.sh all claude --profile hackathon     # hackathon profile
#   ./skills/_evals/run.sh quickstart mock --no-tui           # table output only
#   ./skills/_evals/run.sh all mock --profile hackathon --seed # seed hackathon data first
#
# Environment:
#   WANDB_API_KEY          -- Required for real agent runners and seed
#   OPENAI_API_KEY         -- Required for Codex runner
#   MCP_EVAL_SEED_ENTITY   -- W&B entity for seed project (default: a-sh0ts)
#   MCP_EVAL_SEED_PROJECT  -- W&B project for seed data (default: mcp-skill-eval-seed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKILL="${1:-quickstart}"
RUNNER="${2:-mock}"
shift 2 2>/dev/null || true

USE_TUI=true
EXTRA_FLAGS=()

for arg in "$@"; do
    case "$arg" in
        --no-tui)
            USE_TUI=false
            ;;
        *)
            EXTRA_FLAGS+=("$arg")
            ;;
    esac
done

cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    uv venv .venv --python 3.12
fi

source .venv/bin/activate

if ! python -c "import textual" 2>/dev/null; then
    echo "Installing eval dependencies..."
    uv pip install "textual>=3.0.0" "rich>=14.0.0"
fi

CMD=(python -m skills._evals.run_evals --skill "$SKILL" --runner "$RUNNER")

if [ "$USE_TUI" = true ]; then
    CMD+=(--tui)
fi

CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
