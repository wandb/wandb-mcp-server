import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import weave

# Load .env file before anything else that might need environment variables
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from wandb_mcp_server.utils import get_rich_logger

load_dotenv()

logger = get_rich_logger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# -----------------------------------------------------------------------------
# Test tool response correctness
# -----------------------------------------------------------------------------


class CheckCorrectness(BaseModel):
    """Check if the tool response is correct."""

    reasoning: str = Field(
        ...,
        description="Reasoning about the correctness of the tool response given \
the user query, the expected value of the output and the response data from the W&B Api. The \
expected outout should be clear to see in the response data.",
    )
    is_correct: bool = Field(
        ...,
        description="Whether the tool response is correct given the \
expected output.",
    )


check_correctness_schema = CheckCorrectness.model_json_schema()

check_correctness_tool = {
    "name": "check_correctness_tool",
    "description": "Check if the assistant's response is correct given an expected output.",
    "input_schema": check_correctness_schema,
}

# -----------------------------------------------------------------------------
# Call Anthropic
# -----------------------------------------------------------------------------


@weave.op
def call_anthropic(
    model_name: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    check_correctness_tool: Optional[Dict[str, Any]] = None,
):
    """Send a chat completion request to the Anthropic client with the supplied tools."""
    if client is None:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable must be set for live Anthropic calls."
        )
    if tools:
        return client.messages.create(
            model=model_name, max_tokens=4000, tools=tools, messages=messages
        )
    elif check_correctness_tool:
        return client.messages.create(
            model=model_name,
            max_tokens=4000,
            tools=[check_correctness_tool],
            messages=messages,
            tool_choice={"type": "tool", "name": "check_correctness_tool"},
        )
    else:
        return client.messages.create(
            model=model_name, max_tokens=4000, messages=messages
        )


@weave.op
def extract_anthropic_tool_use(
    response,
) -> Tuple[Any, str | None, Dict[str, Any] | None, str | None]:
    """Grab the first tool_use block from an Anthropic response and return (tool_use, name, input, id)."""
    for idx, content in enumerate(response.content):
        logger.debug(f"LLM response content {idx}: {content}")
        if content.type == "tool_use":
            return content, content.name, content.input, content.id
    return None, None, None, None


@weave.op
def extract_anthropic_text(
    response,
) -> Tuple[Any, str | None, Dict[str, Any] | None, str | None]:
    """Grab the first text block from an Anthropic response and return (text, id)."""
    for idx, content in enumerate(response.content):
        logger.debug(f"LLM response content {idx}: {content}")
        if content.type == "text":
            return content.text
    return None, None


@weave.op
def get_anthropic_tool_result_message(tool_result: Any, tool_id: str) -> Dict[str, Any]:
    """Helper for feeding a tool result back to Anthropic in the required format."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(tool_result),
            }
        ],
    }


# Export symbols for ease of import
__all__ = [
    "call_anthropic",
    "extract_anthropic_tool_use",
    "get_anthropic_tool_result_message",
]
