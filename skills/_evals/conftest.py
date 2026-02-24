"""Shared fixtures, scenarios, and profiles for MCP skill evaluations.

Scenarios are used in two modes:
1. Unit tests (pytest): _simulate_*_skill() returns mock outputs, scored locally.
2. Live evals (run_evals.py): Real agent CLIs run prompts, scored against real MCP data.

Profiles:
    "default"   -- Generic scenarios for all skills (always used by pytest).
    "hackathon" -- Mistral Worldwide Hackathon scenarios matching judging criteria.

Environment variables:
    MCP_LOGS_WANDB_ENTITY    -- W&B entity for eval logging (default: a-sh0ts)
    MCP_EVAL_PROJECT         -- Weave project for eval results (default: mcp-skill-evals)
    MCP_EVAL_SEED_ENTITY     -- W&B entity for seed data (default: a-sh0ts)
    MCP_EVAL_SEED_PROJECT    -- W&B project for seed data (default: mcp-skill-eval-seed)
    MCP_EVAL_PROFILE         -- Active profile (default: default)
"""

import os

import pytest

from skills._evals.scorers import (
    MetricComparisonScorer,
    TaxonomyCoverageScorer,
    TraceCountAccuracyScorer,
    ValidPythonScorer,
)

EVAL_ENTITY = os.environ.get("MCP_LOGS_WANDB_ENTITY", "a-sh0ts")
EVAL_PROJECT = os.environ.get("MCP_EVAL_PROJECT", "mcp-skill-evals")
EVAL_SEED_ENTITY = os.environ.get("MCP_EVAL_SEED_ENTITY", EVAL_ENTITY)
EVAL_SEED_PROJECT = os.environ.get("MCP_EVAL_SEED_PROJECT", "mcp-skill-eval-seed")


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
        "custom_scorers": [MetricComparisonScorer()],
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
        "custom_scorers": [TraceCountAccuracyScorer()],
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
        "custom_scorers": [ValidPythonScorer()],
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
        "custom_scorers": [ValidPythonScorer()],
    },
    {
        "id": "verify-traces-live",
        "user_request": (
            f"I already instrumented my app with Weave. "
            f"Verify that traces are appearing in entity={EVAL_SEED_ENTITY} "
            f"project={EVAL_SEED_PROJECT}. Tell me how many traces exist and "
            f"give me the Weave UI link."
        ),
        "framework": "any",
        "expected_tools": ["count_weave_traces_tool"],
        "expected_workflow": ["count", "verify_traces"],
        "expected_output_contains": ["wandb.ai"],
        "rubric": [
            {"id": "used_count_tool", "text": "Used count_weave_traces_tool to check for traces"},
            {"id": "reported_count", "text": "Reported a specific number of traces found"},
            {"id": "provided_link", "text": "Gave the user a Weave UI link to view traces"},
        ],
        "regex_checks": [
            {"id": "has_count", "pattern": r"\d+\s*(traces|calls|total)", "description": "Reports a trace count"},
            {"id": "has_link", "pattern": r"wandb\.ai", "description": "Contains W&B link"},
        ],
    },
    {
        "id": "instrument-and-verify-live",
        "user_request": (
            f"I have a Python app that uses OpenAI. Show me how to add Weave tracing, "
            f"then verify traces exist in entity={EVAL_SEED_ENTITY} "
            f"project={EVAL_SEED_PROJECT} using the MCP tools."
        ),
        "framework": "openai",
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["detect_framework", "add_init", "verify_traces"],
        "expected_output_contains": ["weave.init", "import weave"],
        "rubric": [
            {"id": "provided_code", "text": "Provided weave.init() and import weave instrumentation code"},
            {"id": "verified_traces", "text": "Used MCP tools to verify traces exist in the project"},
            {"id": "showed_results", "text": "Showed the user trace count or summary from the project"},
        ],
        "regex_checks": [
            {"id": "has_init", "pattern": r"weave\.init\(", "description": "Contains weave.init() call"},
            {"id": "has_entity", "pattern": EVAL_SEED_ENTITY, "description": "References the eval entity"},
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
        "custom_scorers": [TaxonomyCoverageScorer(min_categories=2)],
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
        "custom_scorers": [TaxonomyCoverageScorer(min_categories=3), ValidPythonScorer()],
    },
]


# ---------------------------------------------------------------------------
# Hackathon profile: Mistral Worldwide Hackathon (Feb 28 - Mar 1, 2026)
# Scenarios derived from judging criteria:
#   - Technical Quality (fine-tuning, hyperparams)
#   - E2E Points (Models + Weave together)
#   - Experiment Tracking and Artifacts (loss plots, LoRA artifacts)
#   - Tracing and Evaluation (agent traces, benchmarks)
#   - W&B Report (summary report creation)
#   - Self-Improvement Mini-Challenge (automated eval-improve loop)
# ---------------------------------------------------------------------------

HACKATHON_QUICKSTART_SCENARIOS = [
    {
        "id": "hackathon-mistral-tracing",
        "user_request": (
            "I'm building a Mistral-powered agent for the hackathon. "
            "Show me how to add Weave tracing so all my Mistral API calls are logged."
        ),
        "framework": "mistral",
        "expected_output_contains": ["weave.init", "import weave"],
        "regex_checks": [
            {"id": "has_init", "pattern": r"weave\.init\(", "description": "Contains weave.init() call"},
            {"id": "has_mistral", "pattern": r"mistral|Mistral", "description": "References Mistral"},
        ],
        "custom_scorers": [ValidPythonScorer()],
    },
    {
        "id": "hackathon-wandb-finetuning",
        "user_request": (
            "I'm fine-tuning Mistral Small with Unsloth for the hackathon. "
            "How do I log my training metrics and save my LoRA adapter as a W&B Artifact?"
        ),
        "framework": "wandb",
        "expected_output_contains": ["wandb.init", "wandb.log"],
        "regex_checks": [
            {"id": "has_wandb_init", "pattern": r"wandb\.init\(", "description": "Contains wandb.init()"},
            {"id": "has_wandb_log", "pattern": r"wandb\.log\(", "description": "Contains wandb.log()"},
            {"id": "has_artifact", "pattern": r"[Aa]rtifact", "description": "Mentions Artifact"},
        ],
        "custom_scorers": [ValidPythonScorer()],
    },
    {
        "id": "hackathon-mcp-setup",
        "user_request": (
            "How do I set up the W&B MCP server in Cursor for the hackathon? "
            "I want my agent to query my runs and traces automatically."
        ),
        "framework": "mcp",
        "expected_output_contains": ["mcp.withwandb.com"],
        "regex_checks": [
            {"id": "has_mcp_url", "pattern": r"mcp\.withwandb\.com", "description": "Contains MCP server URL"},
            {"id": "has_auth", "pattern": r"Authorization|Bearer|api.key", "description": "Mentions auth setup"},
        ],
    },
]

HACKATHON_EXPERIMENT_SCENARIOS = [
    {
        "id": "hackathon-compare-finetuning",
        "user_request": (
            f"Compare my Mistral fine-tuning runs by eval_accuracy in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}. "
            f"Which run had the best performance?"
        ),
        "expected_tools": ["query_wandb_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze"],
        "rubric": [
            {"id": "queried_runs", "text": "Used query_wandb_tool to fetch fine-tuning runs"},
            {"id": "compared_accuracy", "text": "Compared eval_accuracy across runs"},
            {"id": "identified_best", "text": "Identified the best-performing run"},
        ],
        "regex_checks": [
            {"id": "has_accuracy", "pattern": r"accuracy|eval_accuracy", "description": "Mentions accuracy metric"},
            {"id": "has_mistral", "pattern": r"[Mm]istral", "description": "References Mistral models"},
        ],
        "custom_scorers": [MetricComparisonScorer()],
    },
    {
        "id": "hackathon-create-report",
        "user_request": (
            f"Create a W&B Report summarizing my fine-tuning experiments in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}. "
            f"Include training curves and a comparison of my LoRA adapters."
        ),
        "expected_tools": ["query_wandb_tool", "create_wandb_report_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze", "create_report"],
        "rubric": [
            {"id": "fetched_data", "text": "Fetched run data before creating the report"},
            {"id": "created_report", "text": "Used create_wandb_report_tool to generate a report"},
            {"id": "included_metrics", "text": "Report includes training metrics or curves"},
            {"id": "returned_url", "text": "Returned the report URL to the user"},
        ],
        "regex_checks": [
            {"id": "has_url", "pattern": r"https?://", "description": "Response contains a URL"},
            {"id": "has_loss", "pattern": r"loss|training", "description": "Mentions training metrics"},
        ],
    },
    {
        "id": "hackathon-lora-comparison",
        "user_request": (
            f"Which of my LoRA fine-tuning runs in entity={EVAL_SEED_ENTITY} "
            f"project={EVAL_SEED_PROJECT} achieved the lowest loss? "
            f"Show me the configs and final metrics."
        ),
        "expected_tools": ["query_wandb_tool"],
        "expected_workflow": ["identify_project", "query_runs", "analyze"],
        "rubric": [
            {"id": "found_runs", "text": "Found and compared LoRA fine-tuning runs"},
            {"id": "showed_configs", "text": "Displayed run configs (learning rate, LoRA rank, etc.)"},
            {"id": "identified_best_loss", "text": "Identified the run with lowest loss"},
        ],
        "regex_checks": [
            {"id": "has_loss", "pattern": r"loss|eval_loss", "description": "Mentions loss metric"},
            {"id": "has_config", "pattern": r"lora|learning.rate|rank", "description": "Mentions config params"},
        ],
        "custom_scorers": [MetricComparisonScorer()],
    },
]

HACKATHON_TRACE_SCENARIOS = [
    {
        "id": "hackathon-agent-traces",
        "user_request": (
            f"Show me an overview of my Mistral agent traces in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}. "
            f"How many calls were made and what's the error rate?"
        ),
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "summarize"],
        "rubric": [
            {"id": "counted", "text": "Used count_weave_traces_tool to get total traces"},
            {"id": "summarized_status", "text": "Reported success/error breakdown"},
            {"id": "mentioned_mistral", "text": "Referenced Mistral agent operations"},
        ],
        "regex_checks": [
            {"id": "has_count", "pattern": r"\d+\s*(traces|calls|total)", "description": "Reports trace count"},
            {"id": "has_error_rate", "pattern": r"\d+%|error", "description": "Reports error information"},
        ],
        "custom_scorers": [TraceCountAccuracyScorer()],
    },
    {
        "id": "hackathon-eval-results",
        "user_request": (
            f"Summarize the results of my latest Weave evaluation in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}. "
            f"What's the pass rate and what are the common failure patterns?"
        ),
        "expected_tools": ["query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_eval", "summarize"],
        "rubric": [
            {"id": "found_eval", "text": "Located evaluation traces"},
            {"id": "reported_pass_rate", "text": "Reported pass/fail rate or score summary"},
            {"id": "identified_failures", "text": "Described common failure patterns"},
        ],
        "regex_checks": [
            {"id": "has_rate", "pattern": r"\d+%|pass|fail|success", "description": "Reports evaluation rate"},
        ],
    },
    {
        "id": "hackathon-latency-analysis",
        "user_request": (
            f"What's the latency breakdown for my Mistral agent in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}? "
            f"Which operations are slowest?"
        ),
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "query_traces", "summarize"],
        "rubric": [
            {"id": "queried_latency", "text": "Queried traces with latency information"},
            {"id": "identified_slow_ops", "text": "Identified the slowest operations"},
            {"id": "provided_breakdown", "text": "Provided a latency breakdown or distribution"},
        ],
        "regex_checks": [
            {"id": "has_latency", "pattern": r"latency|ms|seconds|slow", "description": "Mentions latency"},
        ],
    },
]

HACKATHON_FAILURE_SCENARIOS = [
    {
        "id": "hackathon-debug-timeouts",
        "user_request": (
            f"My Mistral agent keeps timing out during the hackathon. "
            f"Debug the failures in entity={EVAL_SEED_ENTITY} "
            f"project={EVAL_SEED_PROJECT}. What's going wrong?"
        ),
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "sample_errors", "cluster", "report"],
        "rubric": [
            {"id": "found_errors", "text": "Found and sampled error traces"},
            {"id": "identified_timeouts", "text": "Identified timeout errors as a pattern"},
            {"id": "suggested_fix", "text": "Suggested a fix or mitigation strategy"},
        ],
        "regex_checks": [
            {"id": "has_timeout", "pattern": r"timeout|timed?.out|TimeoutError", "description": "Mentions timeouts"},
        ],
        "custom_scorers": [TaxonomyCoverageScorer(min_categories=1)],
    },
    {
        "id": "hackathon-self-improvement",
        "user_request": (
            f"I want to build a self-improvement loop for my Mistral agent. "
            f"Use the W&B MCP tools to analyze my evaluation results in "
            f"entity={EVAL_SEED_ENTITY} project={EVAL_SEED_PROJECT}, "
            f"identify the weakest areas, and suggest specific improvements "
            f"I should make to my agent."
        ),
        "expected_tools": ["count_weave_traces_tool", "query_weave_traces_tool"],
        "expected_workflow": ["count", "metadata", "sample_errors", "cluster", "report"],
        "rubric": [
            {"id": "analyzed_results", "text": "Analyzed evaluation or trace results"},
            {"id": "identified_weaknesses", "text": "Identified specific weakness areas"},
            {"id": "suggested_improvements", "text": "Provided actionable improvement suggestions"},
            {"id": "mentioned_iteration", "text": "Mentioned re-evaluation or iteration after changes"},
        ],
        "regex_checks": [
            {"id": "has_improve", "pattern": r"improv|optimi|better|fix|change", "description": "Mentions improvement"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Profile registry: maps profile name -> skill -> scenario list
# ---------------------------------------------------------------------------

PROFILES = {
    "default": {
        "quickstart": QUICKSTART_SCENARIOS,
        "trace": TRACE_SCENARIOS,
        "experiment": EXPERIMENT_SCENARIOS,
        "failure": FAILURE_SCENARIOS,
    },
    "hackathon": {
        "quickstart": HACKATHON_QUICKSTART_SCENARIOS,
        "trace": HACKATHON_TRACE_SCENARIOS,
        "experiment": HACKATHON_EXPERIMENT_SCENARIOS,
        "failure": HACKATHON_FAILURE_SCENARIOS,
    },
}
