from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest
from wandb_mcp_server.utils import get_rich_logger

from tests.anthropic_test_utils import (
    call_anthropic,
    check_correctness_tool,
    extract_anthropic_tool_use,
    get_anthropic_tool_result_message,
)

from wandb_mcp_server.mcp_tools.query_wandbot import (
    WANDBOT_TOOL_DESCRIPTION,
    query_wandbot_api,
)
from wandb_mcp_server.mcp_tools.tools_utils import generate_anthropic_tool_schema

# -----------------------------------------------------------------------------
# Logging & env guards
# -----------------------------------------------------------------------------
logger = get_rich_logger(__name__)

# -----------------------------------------------------------------------------
# Environment guards
# -----------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WANDBOT_BASE_URL = os.getenv("WANDBOT_TEST_URL", "https://morg--wandbot-api-wandbotapi-serve.modal.run")

if not ANTHROPIC_API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set; skipping Anthropic tests.",
        allow_module_level=True,
    )

# -----------------------------------------------------------------------------
# Static test context
# -----------------------------------------------------------------------------

MODEL_NAME = "claude-3-7-sonnet-20250219"
CORRECTNESS_MODEL_NAME = "claude-3-5-haiku-20241022"

# -----------------------------------------------------------------------------
# Build tool schema for Anthropic
# -----------------------------------------------------------------------------

available_tools: Dict[str, Dict[str, Any]] = {
    "query_wandbot_api": {
        "function": query_wandbot_api,
        "schema": generate_anthropic_tool_schema(
            func=query_wandbot_api,  # Pass the function itself
            description=WANDBOT_TOOL_DESCRIPTION,  # Use the imported description
        ),
    }
}


tools: List[Dict[str, Any]] = [available_tools["query_wandbot_api"]["schema"]]

# -----------------------------------------------------------------------------
# Natural-language queries to test
# -----------------------------------------------------------------------------

test_queries = [
    {
        "question": "What kinds of scorers does weave support?",
        "expected_output": "There are 2 types of scorers in weave, Function-based and Class-based.",
    },
    # Add more test cases here later
]

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    test_queries,
    ids=[f"sample_{i}" for i, _ in enumerate(test_queries)],
)
def test_query_wandbot(sample):
    """End-to-end test: NL question → Anthropic → tool_use → result validation with correctness check."""

    query_text = sample["question"]
    expected_output = sample[
        "expected_output"
    ]  # Get expected output for correctness check

    logger.info("\n==============================")
    logger.info("QUERY: %s", query_text)

    # --- Retry Logic Setup ---
    max_retries = 1
    last_reasoning = "No correctness check performed yet."
    last_is_correct = False
    first_call_assistant_response = None  # Store the response dict from the first model
    tool_result = None  # Store the result of executing the tool
    tool_use_id = None  # Initialize tool_use_id *before* the loop

    # Initial messages for the first attempt
    messages_first_call = [{"role": "user", "content": query_text}]

    for attempt in range(max_retries + 1):
        logger.info(f"\n--- Attempt {attempt + 1} / {max_retries + 1} ---")

        current_messages = messages_first_call  # Start with the base messages

        if attempt > 0:
            # Retry logic: Add previous assistant response, tool result, and user feedback
            retry_messages = []
            if first_call_assistant_response:
                # 1. Add previous assistant message (contains tool use)
                retry_messages.append(
                    {
                        "role": first_call_assistant_response.role,
                        "content": first_call_assistant_response.content,
                    }
                )
                # 2. Add the result from executing the tool in the previous attempt
                if tool_result is not None and tool_use_id is not None:
                    tool_result_message = get_anthropic_tool_result_message(
                        tool_result, tool_use_id
                    )
                    retry_messages.append(tool_result_message)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}: Cannot add tool result message, tool_result or tool_use_id missing."
                    )

                # 3. Add the user message asking for a retry
                retry_user_message_content = f"""
Executing the previous tool call resulted in:
```json
{json.dumps(tool_result, indent=2)}
```
A separate check determined this result was incorrect for the original query.
The reasoning provided was: "{last_reasoning}".

Please re-analyze the original query ("{query_text}") and the result from your previous attempt, then try generating the '{available_tools["query_wandbot_api"]["schema"]["name"]}' tool call again.
"""
                retry_messages.append(
                    {"role": "user", "content": retry_user_message_content}
                )
                current_messages = (
                    messages_first_call[:1] + retry_messages
                )  # Rebuild message list for retry
            else:
                logger.warning(
                    "Attempting retry, but no previous assistant response or tool_use_id found."
                )
                # If retry is needed but we lack context, we probably should just fail or stick with original messages
                # For now, let's proceed with original messages, though this might not be ideal.
                current_messages = messages_first_call

        # --- First Call: Get the query_wandbot_api tool use ---
        try:
            response = call_anthropic(
                model_name=MODEL_NAME,
                messages=current_messages,  # Use the potentially updated message list
                tools=tools,
            )
            first_call_assistant_response = response  # Store for potential *next* retry
        except Exception as e:
            pytest.fail(f"Attempt {attempt + 1}: Anthropic API call failed: {e}")

        try:
            # Extract tool_use_id here
            _, tool_name, tool_input, tool_use_id = extract_anthropic_tool_use(response)
            if tool_use_id is None:
                logger.warning(
                    f"Attempt {attempt + 1}: Model did not return a tool use block."
                )
                # Decide how to handle this - maybe fail, maybe retry without tool use?
                # For now, continue to execution, it might fail gracefully or correctness check will catch it.

        except ValueError as e:
            logger.error(
                f"Attempt {attempt + 1}: Failed to extract tool use from response: {response}"
            )
            pytest.fail(f"Attempt {attempt + 1}: Could not extract tool use: {e}")

        logger.info(f"Attempt {attempt + 1}: Tool emitted by model: {tool_name}")
        logger.info(
            f"Attempt {attempt + 1}: Tool input: {json.dumps(tool_input, indent=2)}"
        )

        assert tool_name == "query_wandbot_api", (
            f"Attempt {attempt + 1}: Expected 'query_wandbot_api', got '{tool_name}'"
        )
        assert "question" in tool_input, (
            f"Attempt {attempt + 1}: Tool input missing 'question'"
        )

        # --- Execute the WandBot tool ---
        try:
            # --- Ensure only expected args based on the *current* function signature are passed ---
            # Assuming the function now only takes 'question'
            if "question" not in tool_input:
                pytest.fail(
                    f"Attempt {attempt + 1}: Tool input missing required 'question' argument."
                )

            actual_args = {"question": tool_input["question"]}

            tool_result = available_tools[tool_name]["function"](**actual_args)
            logger.info(
                f"Attempt {attempt + 1}: Tool result: {json.dumps(tool_result, indent=2)}"
            )  # Log full result

            # Basic structure check before correctness check
            assert isinstance(tool_result, dict), "Tool result should be a dictionary"
            assert isinstance(tool_result.get("answer"), str), (
                "'answer' should be a string"
            )
            assert isinstance(tool_result.get("sources"), list), (
                "'sources' should be a list"
            )

        except Exception as e:
            logger.error(
                f"Attempt {attempt + 1}: Error executing or validating tool '{tool_name}' with input {actual_args}: {e}",
                exc_info=True,
            )
            pytest.fail(
                f"Attempt {attempt + 1}: Tool execution or basic validation failed: {e}"
            )

        # --- Second Call: Perform Correctness Check ---
        logger.info(f"\n--- Starting Correctness Check for Attempt {attempt + 1} ---")
        try:
            correctness_prompt = f"""
Please evaluate if the provided 'Actual Tool Result' provides a helpful and relevant answer to the 'Original User Query'. 
The 'Expected Output Hint' gives guidance on what a good answer should contain. 
Use the 'check_correctness_tool' to provide your reasoning and conclusion.

Original User Query:
"{query_text}"

Expected Output:
"{expected_output}"

Actual Tool Result from '{tool_name}':
```json
{json.dumps(tool_result, indent=2)}
```
            """
            messages_check_call = [{"role": "user", "content": correctness_prompt}]
            correctness_response = call_anthropic(
                model_name=CORRECTNESS_MODEL_NAME,
                messages=messages_check_call,
                check_correctness_tool=check_correctness_tool,  # Pass the imported tool schema
            )
            logger.info(
                f"Attempt {attempt + 1}: Correctness check response:\n{correctness_response}\n\n"
            )

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

            last_reasoning = check_tool_input["reasoning"]
            last_is_correct = check_tool_input["is_correct"]

            logger.info(
                f"Attempt {attempt + 1}: Correctness Reasoning: {last_reasoning}"
            )
            logger.info(
                f"Attempt {attempt + 1}: Is Correct according to LLM: {last_is_correct}"
            )

            if last_is_correct:
                logger.info(
                    f"--- Correctness check passed on attempt {attempt + 1}. ---"
                )
                break  # Exit the loop successfully

        except KeyError as e:
            logger.error(
                f"Attempt {attempt + 1}: Missing expected key in correctness tool input: {e}"
            )
            logger.error(
                f"Attempt {attempt + 1}: Full input received: {check_tool_input}"
            )
            last_is_correct = False
            last_reasoning = f"Correctness tool response missing key: {e}"
            # Continue loop if retries left, fail otherwise handled after loop

        except Exception as e:
            logger.error(
                f"Attempt {attempt + 1}: Error during correctness check for query '{query_text}': {e}",
                exc_info=True,
            )
            last_is_correct = False
            last_reasoning = f"Correctness check failed with exception: {e}"
            # Continue loop if retries left, fail otherwise handled after loop

    # --- After the loop, fail the test if the last attempt wasn't correct ---
    if not last_is_correct:
        pytest.fail(
            f"LLM evaluation failed after {max_retries + 1} attempts. "
            f"Final is_correct_flag is `{last_is_correct}`. "
            f"Final Reasoning: '{last_reasoning}'"
        )

    # If we reach here, it means the correctness check passed within the allowed attempts.
    logger.info("--- Test passed within allowed attempts. ---")
