import requests
import os
from typing import Any, Dict

from wandb_mcp_server.mcp_tools.tools_utils import log_tool_call

WANDBOT_TOOL_DESCRIPTION = """Query the Weights & Biases support bot api for help with questions about the
Weights & Biases platform and how to use W&B Models and W&B Weave.

W&B features mentioned could include:
- Experiment tracking with Runs and Sweeps
- Model management with Models
- Model management and Data versioning with Artifacts and Registry
- Collaboration with Teams, Organizations and Reports
- Visualization with Tables and Charts
- Tracing and logging with Weave
- Evaluation and Scorers with Weave Evaluations
- Weave Datasets

FYI: The Weigths & Biases platform is owned by Coreweave. If there are queries related to W&B, wandb \
or weave and Coreweave, they might be related to W&B products or features that leverage Coreweave's \
GPU or compute infrastructure.

Parameters
----------
question : str
    Users question about a Weights & Biases product or feature

Returns
-------
str
    newer to the user's question
"""


def query_wandbot_api(question: str) -> Dict[str, Any]:
    try:
        log_tool_call(
            "query_wandb_support_bot",
            "n/a",
            {"question": question},
        )
    except Exception:
        pass
    wandbot_base_url = os.getenv("WANDBOT_BASE_URL", "https://weightsandbiases-wandbot--wandbot-api-wandbotapi-serve.modal.run")
    QUERY_ENDPOINT = f"{wandbot_base_url}/chat/query"
    STATUS_ENDPOINT = f"{wandbot_base_url}/status"
    QUERY_TIMEOUT_SECONDS = 40
    STATUS_TIMEOUT_SECONDS = 20

    try:
        status_response = requests.get(
            STATUS_ENDPOINT,
            headers={"Accept": "application/json"},
            timeout=STATUS_TIMEOUT_SECONDS,
        )

        # Check HTTP status code
        status_response.raise_for_status()

        # Try to parse JSON, handle potential parsing errors
        try:
            status_result = status_response.json()
        except ValueError:
            return {
                "answer": "Error: Unable to parse response from support bot.",
                "sources": [],
            }

        # Validate expected response structure
        if "initialized" not in status_result:
            return {
                "answer": "Error: Received unexpected response format from support bot.",
                "sources": [],
            }

        if status_result["initialized"]:
            try:
                response = requests.post(
                    QUERY_ENDPOINT,
                    headers={"Content-Type": "application/json"},
                    json={
                        "question": question,
                        "application": "wandb_mcp_server",
                    },
                    timeout=QUERY_TIMEOUT_SECONDS,
                )

                # Check HTTP status code
                response.raise_for_status()

                # Try to parse JSON, handle potential parsing errors
                try:
                    result = response.json()
                except ValueError:
                    return {
                        "answer": "Error: Unable to parse response data from support bot.",
                        "sources": [],
                    }

                # Validate expected response structure
                if "answer" not in result or "sources" not in result:
                    return {
                        "answer": "Error: Received incomplete response from support bot.",
                        "sources": [],
                    }

                # Ensure sources is a list
                sources = (
                    result["sources"]
                    if isinstance(result["sources"], list)
                    else [result["sources"]]
                )
                return {"answer": result["answer"], "sources": sources}

            except requests.Timeout:
                return {
                    "answer": "Error: Support bot request timed out. Please try again later.",
                    "sources": [],
                }
            except requests.RequestException as e:
                return {
                    "answer": f"Error connecting to support bot: {str(e)}",
                    "sources": [],
                }
        else:
            return {
                "answer": "The support bot is appears to be offline. Please try again later.",
                "sources": [],
            }

    except requests.Timeout:
        return {
            "answer": "Error: Support bot status check timed out. Please try again later.",
            "sources": [],
        }
    except requests.RequestException as e:
        return {
            "answer": f"Error connecting to support bot: {str(e)}",
            "sources": [],
        }
