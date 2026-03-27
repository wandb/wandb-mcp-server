#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# test_local_changes.sh — Validate MCP tool changes against live W&B APIs
#
# This script installs the local wandb-mcp-server branch into the test repo's
# venv, then runs validate_tools_live.py which exercises every tool against the
# real W&B/Weave backend. Use this before deploying to staging or opening a PR.
#
# Prerequisites:
#   - WANDB_API_KEY set in environment or in WandBAgentFactory/.env
#   - The wandb-mcp-server-test repo checked out as a sibling directory
#
# Usage:
#   cd wandb-mcp-server
#   ./scripts/test_local_changes.sh
#
# What it does:
#   1. Locates the test repo (../wandb-mcp-server-test)
#   2. Installs the current local branch into the test repo's venv
#   3. Runs validate_tools_live.py with the test repo's Python
#   4. Reports pass/fail
#
# For WBAF evals (full agent loop), deploy to staging first:
#   cd ../wandb-mcp-server-test
#   ./deploy.sh staging feat/api-efficiency-audit
# Then run evals from WandBAgentFactory as normal.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_REPO="${SERVER_ROOT}/../wandb-mcp-server-test"
WBAF_REPO="${SERVER_ROOT}/../WandBAgentFactory"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

# ── Locate repos ─────────────────────────────────────────────────────────────

if [ ! -d "$TEST_REPO" ]; then
    fail "Cannot find wandb-mcp-server-test at $TEST_REPO"
fi
info "Server repo: $SERVER_ROOT"
info "Test repo:   $TEST_REPO"

# ── Resolve WANDB_API_KEY ────────────────────────────────────────────────────

if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -f "$WBAF_REPO/.env" ]; then
        WANDB_API_KEY=$(awk -F= '/^WANDB_API_KEY=/{print $2}' "$WBAF_REPO/.env")
        export WANDB_API_KEY
        info "Loaded WANDB_API_KEY from WandBAgentFactory/.env"
    elif [ -f "$SERVER_ROOT/.env" ]; then
        WANDB_API_KEY=$(awk -F= '/^WANDB_API_KEY=/{print $2}' "$SERVER_ROOT/.env")
        export WANDB_API_KEY
        info "Loaded WANDB_API_KEY from wandb-mcp-server/.env"
    fi
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    fail "WANDB_API_KEY is not set. Export it or add to WandBAgentFactory/.env"
fi
info "WANDB_API_KEY loaded (length=${#WANDB_API_KEY})"

# ── Install local branch into test venv ──────────────────────────────────────

info "Installing local wandb-mcp-server into test repo venv..."
cd "$TEST_REPO"
uv pip install -e "${SERVER_ROOT}[test]" --quiet 2>&1 | tail -5
info "Install complete"

# ── Verify the install picked up our changes ─────────────────────────────────

TIMEOUT=$(.venv/bin/python -c "from wandb_mcp_server.weave_api.client import WeaveApiClient; print(WeaveApiClient.DEFAULT_TIMEOUT)" 2>/dev/null)
if [ "$TIMEOUT" != "30" ]; then
    fail "Install verification failed: DEFAULT_TIMEOUT=$TIMEOUT (expected 30)"
fi
info "Install verified: DEFAULT_TIMEOUT=$TIMEOUT"

# ── Run the live validation script ───────────────────────────────────────────

info "Running live tool validation..."
echo ""
.venv/bin/python "$SERVER_ROOT/scripts/validate_tools_live.py"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    info "All validations passed!"
    echo ""
    info "Next steps:"
    info "  1. Deploy to staging:  cd $TEST_REPO && ./deploy.sh staging feat/api-efficiency-audit"
    info "  2. Run WBAF evals:     cd $WBAF_REPO && uv run python -m core.run_eval evals/mcp-ci.yaml --agent codex-mcp-v3-lazy"
else
    fail "Some validations failed. See output above."
fi
