"""Shared fixtures for skills evaluation."""

import os

import pytest

EVAL_ENTITY = os.environ.get("MCP_LOGS_WANDB_ENTITY", "a-sh0ts")
EVAL_PROJECT = os.environ.get("MCP_EVAL_PROJECT", "mcp-skill-evals")


@pytest.fixture(scope="session", autouse=True)
def weave_session():
    """Initialize Weave once for the entire eval session."""
    try:
        import weave

        weave.init(f"{EVAL_ENTITY}/{EVAL_PROJECT}")
        yield
        weave.finish()
    except Exception:
        yield


@pytest.fixture()
def sample_entity():
    return EVAL_ENTITY


@pytest.fixture()
def sample_project():
    return os.environ.get("MCP_EVAL_TARGET_PROJECT", "mcp-logs")


EXPERIMENT_SCENARIOS = [
    {
        "id": "compare-top-runs",
        "user_request": "Compare the top 5 runs by eval_loss in my project",
        "expected_tools": ["query_wandb_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze"],
    },
    {
        "id": "best-run",
        "user_request": "Which run had the best accuracy?",
        "expected_tools": ["query_wandb_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze"],
    },
    {
        "id": "create-report",
        "user_request": "Create a report comparing my latest training runs",
        "expected_tools": ["query_wandb_tool", "create_wandb_report_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze", "create_report"],
    },
]

TRACE_SCENARIOS = [
    {
        "id": "trace-overview",
        "user_request": "Give me an overview of my traces",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "summarize"],
    },
    {
        "id": "error-investigation",
        "user_request": "Why are my traces failing?",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_errors", "analyze"],
    },
    {
        "id": "eval-summary",
        "user_request": "Summarize the results of my latest evaluation",
        "expected_tools": ["query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_eval", "summarize"],
    },
]

QUICKSTART_SCENARIOS = [
    {
        "id": "openai-app",
        "user_request": "Add tracing to my OpenAI chatbot",
        "framework": "openai",
        "expected_output_contains": ["weave.init", "import weave"],
    },
    {
        "id": "langchain-app",
        "user_request": "I want to trace my LangChain RAG pipeline",
        "framework": "langchain",
        "expected_output_contains": ["weave.init", "import weave"],
    },
    {
        "id": "custom-app",
        "user_request": "Instrument my custom LLM wrapper",
        "framework": "custom",
        "expected_output_contains": ["weave.init", "@weave.op()"],
    },
]

FAILURE_SCENARIOS = [
    {
        "id": "rate-limit-cluster",
        "user_request": "My pipeline keeps hitting rate limits, analyze the failures",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "sample_errors", "cluster", "report"],
    },
    {
        "id": "taxonomy-generation",
        "user_request": "Generate a failure taxonomy for my traces and create a scorer",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "sample_errors", "cluster", "generate_taxonomy", "create_scorer"],
    },
]
