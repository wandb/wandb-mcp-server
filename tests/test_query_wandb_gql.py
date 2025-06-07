# noqa: D100
"""Integration tests that verify Anthropic selects `query_wandb_tool`.

These tests send natural-language questions about the W&B *Models* data for the
`wandb-applied-ai-team/mcp-tests` project.  The Anthropic model should respond
with a `tool_use` invoking `query_wandb_tool`, which we then execute and
validate.
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytest

from tests.anthropic_test_utils import (
    call_anthropic,
    check_correctness_tool,
    extract_anthropic_tool_use,
)
from wandb_mcp_server.mcp_tools.query_wandb_gql import (
    QUERY_WANDB_GQL_TOOL_DESCRIPTION,
    query_paginated_wandb_gql,
)
from wandb_mcp_server.mcp_tools.tools_utils import generate_anthropic_tool_schema
from wandb_mcp_server.utils import get_git_commit, get_rich_logger

# Root logging configuration
logger = get_rich_logger(__name__)

# weave.init("wandb-applied-ai-team/wandb-mcp-server-test-outputs")
# os.environ["WANDB_SILENT"] = "true"


# -----------------------------------------------------------------------------
# Custom JSON encoder for datetime objects (similar to test_query_weave_traces.py)
# -----------------------------------------------------------------------------
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that can handle datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# -----------------------------------------------------------------------------
# Environment guards
# -----------------------------------------------------------------------------

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not WANDB_API_KEY:
    pytest.skip(
        "WANDB_API_KEY environment variable not set; skipping live GraphQL tests.",
        allow_module_level=True,
    )
if not ANTHROPIC_API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set; skipping Anthropic tests.",
        allow_module_level=True,
    )

# -----------------------------------------------------------------------------
# Static test context
# -----------------------------------------------------------------------------

TEST_WANDB_ENTITY = "wandb-applied-ai-team"
TEST_WANDB_PROJECT = "mcp-tests"
# MODEL_NAME = "claude-3-7-sonnet-20250219"
# MODEL_NAME = "claude-4-sonnet-20250514"
MODEL_NAME = "claude-4-opus-20250514"
CORRECTNESS_MODEL_NAME = "claude-3-5-haiku-20241022"

# -----------------------------------------------------------------------------
# Build tool schema for Anthropic
# -----------------------------------------------------------------------------

available_tools: Dict[str, Dict[str, Any]] = {
    "query_paginated_wandb_gql": {
        "function": query_paginated_wandb_gql,
        "schema": generate_anthropic_tool_schema(
            func=query_paginated_wandb_gql,
            description=QUERY_WANDB_GQL_TOOL_DESCRIPTION,
        ),
    }
}

tools: List[Dict[str, Any]] = [available_tools["query_paginated_wandb_gql"]["schema"]]

# -----------------------------------------------------------------------------
# Compute baseline runCount once so that tests have a stable expected value
# -----------------------------------------------------------------------------

BASELINE_QUERY = """
query ProjectRunCount($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runCount
  }
}
"""
BASELINE_VARIABLES = {"entity": TEST_WANDB_ENTITY, "project": TEST_WANDB_PROJECT}

# Compute baseline
logger.info(
    "Fetching baseline runCount for %s/%s", TEST_WANDB_ENTITY, TEST_WANDB_PROJECT
)
_baseline_result = query_paginated_wandb_gql(BASELINE_QUERY, BASELINE_VARIABLES)
BASELINE_RUN_COUNT: int = _baseline_result["project"]["runCount"]
logger.info("Baseline runCount = %s", BASELINE_RUN_COUNT)

# -----------------------------------------------------------------------------
# Natural-language queries to test
# -----------------------------------------------------------------------------

test_queries = [
    {
        "index": 0,
        "question": "How many runs are currently logged in the `{project_name}` project under the `{entity_name}` entity?",
        "expected_output": 37,
    },
    {
        "index": 1,
        "question": "What's the total experiment count for `{entity_name}/{project_name}`?",
        "expected_output": 37,
    },
    {
        "index": 2,
        "question": "In `{project_name}` in entity `{entity_name}` how many runs were run on April 29th 2025?",
        "expected_output": 37,
    },
    {
        "index": 3,
        "question": "Could you report the number of tracked runs in `{entity_name}/{project_name}` with lr 0.002?",
        "expected_output": 7,
    },
    {
        "index": 4,
        "question": "what was the run with the best eval loss in the `{project_name}` project belonging to `{entity_name}`.",
        "expected_output": "run_id: h0fm5qp5 OR run_name: transformer_7_bs-128_lr-0.008_5593616",
    },
    {
        "index": 5,
        "question": "How many steps in run gtng2y4l `{entity_name}/{project_name}` right now.",
        "expected_output": 750000,
    },
    {
        "index": 6,
        "question": "How many steps in run transformer_25_bs-33554432_lr-0.026000000000000002_2377215 `{entity_name}/{project_name}` right now.",
        "expected_output": 750000,
    },
    {
        "index": 7,
        "question": "What's the batch size of the run with best evaluation accuracy for `{project_name}` inside `{entity_name}`?",
        "expected_output": 16,
    },
    # {
    #     "index": 8, # Example if uncommented
    #     "question": "Count the runs in my `{entity_name}` entity for the `{project_name}` project.",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
    # {
    #     "index": 9, # Example if uncommented
    #     "question": "How big is the experiment set for `{entity_name}/{project_name}`?",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
    # {
    #     "index": 10, # Example if uncommented
    #     "question": "Tell me the number of runs tracked in `{project_name}` (entity `{entity_name}`).",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    test_queries,
    ids=[f"sample_{i}" for i, _ in enumerate(test_queries)],
)
def test_query_wandb_gql(sample, weave_results_dir):
    """End-to-end test: NL question → Anthropic → tool_use → result validation."""

    start_time = time.monotonic()
    current_git_commit = get_git_commit()
    git_commit_id = f"commit_{current_git_commit}"
    current_test_file_name = os.path.basename(__file__)

    # Find the index of the current sample for unique naming and metadata
    sample_index = -1
    for i, s in enumerate(test_queries):
        if s == sample:
            sample_index = i
            break

    test_case_name = f"gql_query_{sample_index}_{sample.get('question', 'unknown_question')[:20].replace(' ', '_')}"

    query_text = sample["question"].format(
        entity_name=TEST_WANDB_ENTITY,
        project_name=TEST_WANDB_PROJECT,
    )
    expected_output = sample["expected_output"]

    logger.info("\n==============================")
    logger.info("QUERY: %s", query_text)

    # --- Retry Logic Setup ---
    max_retries = 1
    last_reasoning = "No correctness check performed yet."
    last_is_correct = False
    first_call_assistant_response = None  # Store the response dict from the first model
    tool_result = None  # Store the result of executing the tool
    tool_name_used_in_test = None
    tool_input_used_in_test = None

    # Initialize log_data_for_file for the current test sample
    final_log_data_for_file = {
        "metadata": {
            "sample_name": test_case_name,
            "test_case_index": sample_index,
            "git_commit_id": git_commit_id,
            "source_test_file_name": current_test_file_name,
            "test_query_text": query_text,
            "expected_test_output": str(expected_output),
            "retry_attempt": 0,  # Will be updated in the loop
            "max_retries_configured": max_retries,
        },
        "inputs": {  # Inputs to the overall test/evaluation
            "test_query": query_text,
            "expected_value": str(expected_output),
        },
        "output": {},  # Will store tool output and correctness check details
        "score": False,  # Default to False, updated on success
        "scorer_name": "gql_correctness_assertion",  # Specific scorer for these tests
        "metrics": {},  # Will store execution_latency_seconds
    }

    try:
        # Initial messages for the first attempt
        messages_first_call = [{"role": "user", "content": query_text}]

        for attempt in range(max_retries + 1):
            logger.info(f"\n--- Attempt {attempt + 1} / {max_retries + 1} ---")
            final_log_data_for_file["metadata"]["retry_attempt"] = attempt + 1

            if attempt > 0:
                # We are retrying. Add the previous assistant response and a user message with feedback.
                if first_call_assistant_response:
                    messages_first_call.append(
                        first_call_assistant_response
                    )  # Add previous assistant message (contains tool use)
                else:
                    # Should not happen in retry logic, but defensively handle
                    logger.warning(
                        "Attempting retry, but no previous assistant response found."
                    )

                # Construct the user message asking for a retry
                retry_user_message_content = f"""
Executing the previous tool call resulted in:
```json
{json.dumps(tool_result, indent=2, cls=DateTimeEncoder)}
```
A separate check determined this result was incorrect for the original query.
The reasoning provided was: "{last_reasoning}".

Please re-analyze the original query ("{query_text}") and the result from your previous attempt, then try generating the 'query_paginated_wandb_gql' tool call again.
"""
                messages_first_call.append(
                    {"role": "user", "content": retry_user_message_content}
                )

            # --- First Call: Get the query_paginated_wandb_gql tool use ---
            response = call_anthropic(
                model_name=MODEL_NAME,
                messages=messages_first_call,
                tools=tools,  # Provide the GQL tool schema
            )
            first_call_assistant_response = (
                response  # Store this response for potential next retry
            )
            _, tool_name, tool_input, _ = extract_anthropic_tool_use(response)

            tool_name_used_in_test = tool_name
            tool_input_used_in_test = tool_input

            logger.info(f"Attempt {attempt + 1}: Tool emitted by model: {tool_name}")
            logger.info(
                f"Attempt {attempt + 1}: Tool input: {json.dumps(tool_input, indent=2)}"
            )

            assert tool_name == "query_paginated_wandb_gql", (
                f"Attempt {attempt + 1}: Expected 'query_paginated_wandb_gql', got '{tool_name}'"
            )

            # --- Execute the GQL tool ---
            try:
                tool_result = available_tools[tool_name]["function"](**tool_input)
                logger.info(
                    f"Attempt {attempt + 1}: Tool result: {json.dumps(tool_result, indent=2, cls=DateTimeEncoder)}"
                )  # Log full result
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1}: Error executing tool '{tool_name}' with input {tool_input}: {e}",
                    exc_info=True,
                )
                final_log_data_for_file["output"]["tool_execution_error_details"] = str(
                    e
                )
                # If tool execution fails, we might want to stop retrying for this sample or handle differently.
                # For now, it will proceed to correctness check which will likely fail or be skipped.
                # Depending on the error, we might want to `pytest.fail` or `raise` to stop the current attempt.
                # For this iteration, we'll let it go to the correctness check, which will likely fail it.
                last_is_correct = False
                last_reasoning = f"Tool execution failed: {e}"
                if attempt >= max_retries:  # If this was the last attempt
                    raise  # Re-raise the exception to fail the test
                continue  # Skip to next retry attempt

            # --- Second Call: Perform Correctness Check (Separate Task) ---
            logger.info(
                f"\n--- Starting Correctness Check for Attempt {attempt + 1} ---"
            )

            try:
                # Prepare the prompt for the check - provide all context clearly
                correctness_prompt = f"""
                Please evaluate if the provided 'Actual Tool Result' correctly addresses the 'Original User Query' and seems consistent with the 'Expected Output'. Use the 'check_correctness_tool' to provide your reasoning and conclusion.

                Original User Query:
                "{query_text}"

                Expected Output (for context, may not be directly comparable in structure):
                {json.dumps(expected_output, indent=2, cls=DateTimeEncoder)}

                Actual Tool Result from 'query_paginated_wandb_gql':
                {json.dumps(tool_result, indent=2, cls=DateTimeEncoder)}
                """

                messages_check_call = [{"role": "user", "content": correctness_prompt}]
                correctness_response = call_anthropic(
                    model_name=CORRECTNESS_MODEL_NAME,
                    messages=messages_check_call,
                    check_correctness_tool=check_correctness_tool,
                )

                logger.info(
                    f"Attempt {attempt + 1}: Correctness check response:\n{correctness_response}\n\n"
                )

                # --- Extract and Validate Correctness Tool Use ---
                _, check_tool_name, check_tool_input, _ = extract_anthropic_tool_use(
                    correctness_response
                )

                assert check_tool_name == "check_correctness_tool", (
                    f"Attempt {attempt + 1}: Expected correctness tool, got {check_tool_name}"
                )
                assert "reasoning" in check_tool_input, (
                    f"Attempt {attempt + 1}: Correctness tool missing 'reasoning'"
                )
                assert "is_correct" in check_tool_input, (
                    f"Attempt {attempt + 1}: Correctness tool missing 'is_correct'"
                )

                # 2. Extract the data from the input dictionary
                try:
                    reasoning_text = check_tool_input["reasoning"]
                    is_correct_flag = check_tool_input["is_correct"]

                    # Store the latest results
                    last_reasoning = reasoning_text
                    last_is_correct = is_correct_flag

                    logger.info(
                        f"Attempt {attempt + 1}: Correctness Reasoning: {reasoning_text}"
                    )
                    logger.info(
                        f"Attempt {attempt + 1}: Is Correct according to LLM: {is_correct_flag}"
                    )

                    if is_correct_flag:
                        logger.info(
                            f"--- Correctness check passed on attempt {attempt + 1}. ---"
                        )
                        final_log_data_for_file["score"] = True
                        break  # Exit the loop successfully

                    # If not correct, and this is the last attempt, the loop will end naturally.

                except KeyError as e:
                    logger.error(
                        f"Attempt {attempt + 1}: Missing expected key in correctness tool input: {e}"
                    )
                    logger.error(
                        f"Attempt {attempt + 1}: Full input received: {check_tool_input}"
                    )
                    last_is_correct = False
                    last_reasoning = f"Correctness tool response missing key: {e}"
                    final_log_data_for_file["output"]["assertion_error_details"] = (
                        f"Correctness tool response missing key: {e}"
                    )
                    if attempt >= max_retries:
                        pytest.fail(
                            f"Attempt {attempt + 1}: Correctness tool response was missing key: {e}"
                        )
                    continue  # To next retry
                except Exception as e:
                    logger.error(
                        f"Attempt {attempt + 1}: Error processing correctness tool input: {e}",
                        exc_info=True,
                    )
                    last_is_correct = False
                    last_reasoning = f"Failed to process correctness tool input: {e}"
                    final_log_data_for_file["output"]["assertion_error_details"] = (
                        f"Failed to process correctness tool input: {e}"
                    )
                    if attempt >= max_retries:
                        pytest.fail(
                            f"Attempt {attempt + 1}: Failed to process correctness tool input: {e}"
                        )
                    continue  # To next retry

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1}: Error during correctness check for query '{query_text}': {e}",
                    exc_info=True,
                )
                last_is_correct = False
                last_reasoning = f"Correctness check failed with exception: {e}"
                final_log_data_for_file["output"]["assertion_error_details"] = (
                    f"Correctness check failed with exception: {e}"
                )
                if attempt >= max_retries:
                    pytest.fail(
                        f"Attempt {attempt + 1}: Correctness check failed with exception: {e}"
                    )
                continue  # To next retry

        # After the loop, if not last_is_correct, it means all retries failed or it failed on the last attempt.
        if not last_is_correct and attempt >= max_retries:
            pytest.fail(
                f"LLM evaluation failed after {max_retries + 1} attempts for sample {sample_index}. "
                f"Final is_correct_flag is `{last_is_correct}`. "
                f"Final Reasoning: '{last_reasoning}'"
            )

    except Exception as test_exec_exception:
        # Catch any exception that might cause the test to fail before all retries are done
        # or even before the loop fully completes.
        logger.error(
            f"Test execution for sample {sample_index} failed globally: {test_exec_exception}",
            exc_info=True,
        )
        final_log_data_for_file["score"] = False
        final_log_data_for_file["output"]["test_exception"] = str(test_exec_exception)
        # We will write the JSON in `finally`, then re-raise or let pytest handle the failure.
        raise  # Re-raise the caught exception to ensure the test is marked as failed by pytest

    finally:
        end_time = time.monotonic()
        execution_latency_seconds = end_time - start_time
        final_log_data_for_file["metrics"]["execution_latency_seconds"] = (
            execution_latency_seconds
        )
        final_log_data_for_file["metadata"]["final_attempt_number_for_json"] = (
            final_log_data_for_file["metadata"]["retry_attempt"]
        )  # Should be updated inside loop

        # Populate output details from the last successful (or last attempted) tool call
        final_log_data_for_file["output"]["tool_name"] = tool_name_used_in_test
        final_log_data_for_file["output"]["tool_input"] = (
            json.dumps(tool_input_used_in_test, indent=2)
            if tool_input_used_in_test
            else None
        )
        final_log_data_for_file["output"]["tool_result"] = (
            json.dumps(tool_result, indent=2, cls=DateTimeEncoder)
            if tool_result
            else None
        )
        final_log_data_for_file["output"]["correctness_reasoning"] = last_reasoning
        final_log_data_for_file["score"] = last_is_correct  # Ensure final score is set

        # Generate a unique filename for the JSON output
        unique_file_id = str(uuid.uuid4())
        worker_id = os.environ.get(
            "PYTEST_XDIST_WORKER", "main_thread"
        )  # Default if not in xdist

        # Sanitize test_case_name for filename (take first 30 chars, replace spaces)
        safe_test_name_part = (
            test_case_name.replace(" ", "_").replace("/", "_").replace("\\", "_")[:30]
        )

        file_name = f"gql_test_idx_{sample_index}_{safe_test_name_part}_w_{worker_id}_attempt_{final_log_data_for_file['metadata']['final_attempt_number_for_json']}_{('pass' if final_log_data_for_file['score'] else 'fail')}_{unique_file_id}.json"
        file_path = weave_results_dir / file_name

        logger.critical(
            f"WRITING JSON for GQL Test: {test_case_name} (Index: {sample_index}, Last Attempt: {final_log_data_for_file['metadata']['final_attempt_number_for_json']}, Score: {final_log_data_for_file['score']}) to {file_path}"
        )
        try:
            with open(file_path, "w") as f:
                json.dump(final_log_data_for_file, f, indent=2, cls=DateTimeEncoder)
            logger.info(
                f"Result for GQL test {test_case_name} (Latency: {execution_latency_seconds:.2f}s) written to {file_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to write result JSON for GQL test {test_case_name} to {file_path}: {e}"
            )

    # If we reach here and no exception was raised by pytest.fail or re-raised from the try block,
    # it means the correctness check passed within the allowed attempts.
    if not last_is_correct:  # Final check if loop exited due to retries without success
        pytest.fail(
            f"LLM evaluation failed after {max_retries + 1} attempts for sample {sample_index}. "
            f"Final is_correct_flag is `{last_is_correct}`. "
            f"Final Reasoning: '{last_reasoning}'"
        )

    logger.info(
        f"--- Test for sample {sample_index} ({test_case_name}) completed. Score: {last_is_correct} ---"
    )
