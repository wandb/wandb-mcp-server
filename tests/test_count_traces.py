import os

import pytest
from wandb_mcp_server.utils import get_rich_logger

from tests.anthropic_test_utils import call_anthropic, extract_anthropic_tool_use
from wandb_mcp_server.mcp_tools.count_traces import (
    COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    count_traces,
)
from wandb_mcp_server.mcp_tools.tools_utils import generate_anthropic_tool_schema

logger = get_rich_logger(__name__)

os.environ["WANDB_SILENT"] = "true"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set; skipping live Anthropic tests.",
        allow_module_level=True,
    )

TEST_WANDB_ENTITY = "wandb-applied-ai-team"  # "c-metrics"
TEST_WANDB_PROJECT = "wandb-mcp-tests"
model_name = "claude-3-7-sonnet-20250219"

available_tools = {
    "count_traces": {
        "function": count_traces,
        "schema": generate_anthropic_tool_schema(
            func=count_traces, description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
        ),
    }
}

tools = [available_tools["count_traces"]["schema"]]


test_queries = [
    {
        "question": "Please count the total number of traces recorded in the `{project_name}` project under the `{entity_name}` entity.",
        "expected_output": 21639,
    },
    {
        "question": "How many Weave call logs exist for the `{project_name}` project in my `{entity_name}` entity?",  # (Uses "call logs" instead of "traces")
        "expected_output": 21639,
    },
    {
        "question": "What's the volume of traces for `{project_name}` in the `{entity_name}` entity?",  # (Assumes default entity or requires clarification, implies counting)
        "expected_output": 21639,
    },
    {
        "question": "Count the calls that resulted in an error within the `{entity_name}/{project_name}` project.",  # (Requires filtering by status='error')
        "expected_output": 136,
    },
    {
        "question": "How many times has the `generate_joke` operation been invoked in the `{project_name}` project for the `{entity_name}`?",  # (Requires filtering by op_name)
        "expected_output": 4,
    },
    {
        "question": "The date is March 12th, 2025. Give me the parent trace count for `{entity_name}/{project_name}` last month.",  # (Requires calculating and applying a time filter)
        "expected_output": 262,
    },
    {
        "question": "Can you count the parent traces in `{entity_name}/{project_name}`?",  # (Requires, root traces)
        "expected_output": 475,
    },
    {
        "question": "`{entity_name}/{project_name}` trace tally?",  # (Requires inferring the need for counting and likely asking for the entity)
        "expected_output": 21639,
    },
    {
        "question": "How many traces in `{entity_name}/{project_name}` took more than 10 minutes to run?",  # (Requires an attribute filter)
        "expected_output": 155,
    },
    {
        "question": "How many traces in `{entity_name}/{project_name}` took less than 2 seconds to run?",  # (Requires an attribute filter)
        "expected_output": 12357,
    },
    {
        "question": "THe date is April 20th, 2025. Count failed traces for the `openai.chat.completions` op within the `{entity_name}/{project_name}` project since the 27th of February 2025 up to March 1st..",  #  (Requires combining status='success', trace_roots_only=True, op_name, and a time filter)
        "expected_output": 15,
    },
]

# -----------------------
# Pytest integration
# -----------------------


@pytest.mark.parametrize(
    "sample", test_queries, ids=[f"sample_{i}" for i, _ in enumerate(test_queries)]
)
def test_count_traces(sample):
    """Run each natural-language query end-to-end through the Anthropic model and
    verify that the invoked tool returns the expected value."""

    query_text = sample["question"].format(
        entity_name=TEST_WANDB_ENTITY,
        project_name=TEST_WANDB_PROJECT,
    )
    expected_output = sample["expected_output"]

    logger.info("==============================")
    logger.info(f"QUERY: {query_text}")

    messages = [{"role": "user", "content": query_text}]

    response = call_anthropic(model_name, messages, tools)
    _, tool_name, tool_input, _ = extract_anthropic_tool_use(response)

    logger.info(f"Tool emitted by model: {tool_name}")
    logger.debug(f"Tool input: {tool_input}")

    assert tool_name is not None, "Model did not emit a tool call"

    # Execute the real tool — no mocking.
    tool_result = available_tools[tool_name]["function"](**tool_input)

    logger.info(f"Tool result: {tool_result} (expected {expected_output})")

    assert tool_result == expected_output, (
        f"Unexpected result for query `{query_text}`: {tool_result} (expected {expected_output})"
    )
