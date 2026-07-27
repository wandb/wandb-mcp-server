# Contributing to the W&B MCP Server

Thank you for improving the W&B MCP server. This repository contains the public
tool implementations, protocol boundary, configuration, and unit tests.

## Development setup

Prerequisites:

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

```bash
git clone https://github.com/wandb/wandb-mcp-server
cd wandb-mcp-server
uv sync --frozen --extra test --extra http --python 3.12
uv run pre-commit install
```

Do not add API keys to the repository or test configuration. The
non-integration suite is mock-based and does not require network access.

## Validation

Run the same core checks as CI:

```bash
uv lock --check
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/ -m "not integration" -x -v --tb=short -q
uv run bandit -q -r src -ll
uv build
```

CI runs the non-integration suite and a fresh-wheel smoke test on Python 3.11
and 3.12. It also tests compatibility with the latest W&B SDK and scans the
lockfile and source with Grype, Bandit, and Socket Security.

Integration tests may require external services or credentials. Run them only
when the test documents its prerequisites; never use customer data for routine
development.

## Running locally

STDIO, for desktop and CLI clients:

```bash
export WANDB_API_KEY=your-key
uv run wandb_mcp_server
```

HTTP, for local transport testing:

```bash
export WANDB_API_KEY=your-key
uv run wandb_mcp_server --transport http --host 127.0.0.1 --port 8080
curl --fail http://127.0.0.1:8080/health
```

Do not expose the development HTTP server directly to the internet.

## Branches and pull requests

Normal contributions branch from the current `main`:

```bash
git fetch origin
git switch -c fix/short-description origin/main
```

Release integration branches such as `staging/0.4.0` are maintainer-owned.
Only target one when the release owner has explicitly assigned the change to
that release.

A pull request should include:

- The user-visible problem and the chosen behavior.
- A concise description of the implementation.
- Compatibility, security, and privacy implications.
- The exact validation performed.
- A migration note for public tool or configuration changes.

Use the merge method required by the target branch. Component PRs entering a
release branch use merge commits so their reviewed history remains visible.
Never bypass required reviews or checks to assemble a release.

## Architecture

```text
src/wandb_mcp_server/
├── server.py                 # FastMCP construction and tool registration
├── instrumented_server.py    # One telemetry/admission boundary per public call
├── admission.py              # Weighted actor and process concurrency limits
├── auth.py                   # HTTP bearer-token validation
├── session_manager.py        # MCP session metadata
├── analytics*.py             # Bounded product and operational telemetry
├── api_client.py             # Actor-isolated public W&B SDK clients
├── config.py                 # Public/internal URLs and workload profiles
├── wandb_selective_reads.py  # Bounded W&B field projections
├── wandb_urls.py             # Public-link construction and rewriting
├── mcp_tools/
│   ├── query_wandb.py        # Default structured W&B SDK query surface
│   ├── query_wandb_gql.py    # Optional, query-only raw GraphQL escape hatch
│   ├── run_history.py        # Bounded history reads
│   ├── query_weave.py        # Weave trace queries
│   └── ...
└── weave_api/                # Weave trace-server HTTP client
```

Backend W&B requests must use the resolved internal API URL when configured.
User-visible links must use the public W&B URL. Telemetry must not perform
extra W&B lookups solely to enrich an event.

## Changing or adding tools

Prefer improving an existing public tool when it already owns the capability.
Adding a tool expands the public MCP surface and requires an explicit product
decision.

For any tool change:

1. Implement the function under `src/wandb_mcp_server/mcp_tools/`.
2. Keep its description specific about when to use it, limits, and coverage.
3. Register it through `register_tools()` in `server.py`.
4. Classify its cost in `admission.py`; unknown tools intentionally default to
   the heaviest class.
5. Add only low-cardinality, allowlisted telemetry dimensions. Never emit raw
   prompts, queries, filters, API keys, or customer resource identifiers.
6. Respect read-only registration and feature gates.
7. Use bounded iteration, response budgets, and the current tool deadline.
8. Add unit, registration, description, security, and error-path tests.
9. Update the README when the public interface or operator configuration
   changes.

Raw GraphQL is an opt-in compatibility escape hatch, not the default
implementation path. Application-owned documents must remain query-only.

## Documentation

- Keep examples executable and JSON examples valid JSON.
- Avoid embedding temporary PR numbers, staging revisions, customer names, or
  current test counts in durable documentation.
- Put release-specific customer changes in `docs/releases/`.
- Put maintainer release mechanics in [RELEASING.md](RELEASING.md).
- Remove documentation for deleted APIs instead of preserving misleading
  compatibility instructions.

## Security

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
Do not include secrets, customer data, or undisclosed vulnerabilities in public
issues or pull requests.
