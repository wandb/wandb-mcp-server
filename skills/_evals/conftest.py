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
        "rubric": [
            {"id": "queried_wandb", "text": "Used query_wandb_tool to fetch runs from the project"},
            {"id": "compared_metrics", "text": "Compared eval_loss across the top runs"},
            {"id": "presented_table", "text": "Presented results in a clear comparison format"},
        ],
        "regex_checks": [
            {"id": "has_metric_name", "pattern": r"eval.loss|eval_loss", "description": "Mentions the eval_loss metric"},
        ],
    },
    {
        "id": "best-run",
        "user_request": "Which run had the best accuracy?",
        "expected_tools": ["query_wandb_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze"],
        "rubric": [
            {"id": "queried_wandb", "text": "Used query_wandb_tool to fetch runs sorted by accuracy"},
            {"id": "identified_best", "text": "Identified a specific run as having the best accuracy"},
        ],
        "regex_checks": [
            {"id": "has_accuracy", "pattern": r"accuracy|acc", "description": "Mentions accuracy metric"},
        ],
    },
    {
        "id": "create-report",
        "user_request": "Create a report comparing my latest training runs",
        "expected_tools": ["query_wandb_tool", "create_wandb_report_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze", "create_report"],
        "rubric": [
            {"id": "queried_runs", "text": "Fetched runs data before creating the report"},
            {"id": "created_report", "text": "Used create_wandb_report_tool to generate a report"},
            {"id": "returned_url", "text": "Returned the report URL to the user"},
        ],
        "regex_checks": [
            {"id": "has_url", "pattern": r"https?://", "description": "Response contains a URL"},
        ],
    },
]

TRACE_SCENARIOS = [
    {
        "id": "trace-overview",
        "user_request": "Give me an overview of my traces",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "summarize"],
        "rubric": [
            {"id": "counted_first", "text": "Used count_weave_traces_tool before querying traces"},
            {"id": "used_metadata", "text": "Used metadata_only=True for initial overview"},
            {"id": "summarized", "text": "Provided a structured summary of trace behavior"},
        ],
        "regex_checks": [
            {"id": "has_count", "pattern": r"\d+\s*(traces|calls|total)", "description": "Reports a trace count"},
        ],
    },
    {
        "id": "error-investigation",
        "user_request": "Why are my traces failing?",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_errors", "analyze"],
        "rubric": [
            {"id": "counted_first", "text": "Counted traces before detailed query"},
            {"id": "filtered_errors", "text": "Filtered for error status traces"},
            {"id": "identified_patterns", "text": "Identified error patterns or categories"},
        ],
        "regex_checks": [
            {"id": "has_error_word", "pattern": r"error|fail|exception", "description": "Mentions errors"},
        ],
    },
    {
        "id": "eval-summary",
        "user_request": "Summarize the results of my latest evaluation",
        "expected_tools": ["query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_eval", "summarize"],
        "rubric": [
            {"id": "found_eval", "text": "Located evaluation traces via op_name filter"},
            {"id": "drilled_children", "text": "Queried child traces using parent_id"},
            {"id": "summarized_results", "text": "Provided pass/fail rates or score summary"},
        ],
        "regex_checks": [
            {"id": "has_rate", "pattern": r"\d+%|pass|success", "description": "Reports a rate or pass/fail"},
        ],
    },
]

QUICKSTART_SCENARIOS = [
    {
        "id": "openai-app",
        "user_request": "Add tracing to my OpenAI chatbot",
        "framework": "openai",
        "expected_output_contains": ["weave.init", "import weave"],
        "regex_checks": [
            {"id": "has_init", "pattern": r"weave\.init\(", "description": "Contains weave.init() call"},
            {"id": "has_import", "pattern": r"import weave", "description": "Contains import weave"},
        ],
    },
    {
        "id": "langchain-app",
        "user_request": "I want to trace my LangChain RAG pipeline",
        "framework": "langchain",
        "expected_output_contains": ["weave.init", "import weave"],
        "regex_checks": [
            {"id": "has_init", "pattern": r"weave\.init\(", "description": "Contains weave.init() call"},
        ],
    },
    {
        "id": "custom-app",
        "user_request": "Instrument my custom LLM wrapper",
        "framework": "custom",
        "expected_output_contains": ["weave.init", "@weave.op()"],
        "regex_checks": [
            {"id": "has_decorator", "pattern": r"@weave\.op\(\)", "description": "Contains @weave.op() decorator"},
        ],
    },
]

FAILURE_SCENARIOS = [
    {
        "id": "rate-limit-cluster",
        "user_request": "My pipeline keeps hitting rate limits, analyze the failures",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "sample_errors", "cluster", "report"],
        "rubric": [
            {"id": "quantified", "text": "Quantified the error rate before diving into details"},
            {"id": "sampled_errors", "text": "Sampled error traces with relevant columns"},
            {"id": "clustered", "text": "Grouped errors into meaningful categories"},
            {"id": "identified_rate_limit", "text": "Identified rate limit errors as a category"},
        ],
        "regex_checks": [
            {"id": "has_rate_limit", "pattern": r"rate.?limit|429|RateLimitError", "description": "Mentions rate limits"},
        ],
    },
    {
        "id": "taxonomy-generation",
        "user_request": "Generate a failure taxonomy for my traces and create a scorer",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "sample_errors", "cluster", "generate_taxonomy", "create_scorer"],
        "rubric": [
            {"id": "sampled_errors", "text": "Sampled failing traces to build intuition"},
            {"id": "generated_taxonomy", "text": "Produced a taxonomy with named categories"},
            {"id": "created_scorer", "text": "Created or described a Scorer class for classification"},
        ],
        "regex_checks": [
            {"id": "has_scorer", "pattern": r"Scorer|scorer|taxonomy", "description": "Mentions scorer or taxonomy"},
        ],
    },
]
