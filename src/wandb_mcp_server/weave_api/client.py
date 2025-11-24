"""
Weave API client.

This module provides a client for interacting with the Weights & Biases Weave API.
It handles authentication, request construction, and response parsing.
"""

import base64
import json
import os
from typing import Any, Dict, Iterator, Optional

import requests
from requests.exceptions import RetryError

from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.config import WF_TRACE_SERVER_URL

logger = get_rich_logger(__name__)


class WeaveApiClient:
    """Client for interacting with the Weights & Biases Weave API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        retries: int = 3,
        timeout: int = 10,
    ):
        """Initialize the WeaveApiClient.

        Args:
            api_key: API key for authentication. If None, try to get from environment.
            server_url: Weave API server URL. Defaults to 'https://trace.wandb.ai'.
            retries: Number of retries for failed requests.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Set up a session for connection pooling and better request handling
        self.session = requests.Session()

        # NO FALLBACKS! API key must be explicitly provided
        # For HTTP: Comes from auth middleware via TraceService
        # For STDIO: Set at server startup via TraceService
        
        # Validate API key
        if not api_key:
            raise ValueError(
                "API key not provided to WeaveApiClient. "
                "API key must be explicitly passed from TraceService."
            )

        self.api_key = api_key
        self.server_url = server_url or WF_TRACE_SERVER_URL
        self.retries = retries
        self.timeout = timeout

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for the Weave API.

        Returns:
            Dictionary of authentication headers.
        """
        auth_token = base64.b64encode(f":{self.api_key}".encode()).decode()
        return {
            "Content-Type": "application/json",
            "Accept": "application/jsonl",
            "Authorization": f"Basic {auth_token}",
        }

    def query_traces(self, query_params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Query traces from the Weave API.

        Args:
            query_params: Dictionary of query parameters.

        Returns:
            Iterator of trace dictionaries.

        Raises:
            Exception: If the request fails.
        """
        url = f"{self.server_url}/calls/stream_query"
        headers = self._get_auth_headers()

        logger.info(
            f"Sending request to Weave server:\n{json.dumps(query_params, indent=2)[:1000]}...\n"
        )
        logger.debug(f"Full query parameters:\n{json.dumps(query_params, indent=2)}\n")

        try:
            response = self.session.post(
                url,
                headers=headers,
                data=json.dumps(query_params),
                timeout=self.timeout,
                stream=True,
            )

            # Check for errors
            if response.status_code != 200:
                error_msg = f"Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

            logger.info(f"Response status: {response.status_code}")

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Parse the JSON line
                    trace_data = json.loads(line.decode("utf-8"))
                    logger.debug(f"Received trace data with ID: {trace_data.get('id')}")
                    yield trace_data

        except requests.RequestException as e:
            logger.error(
                f"Error executing HTTP request to Weave server: {e}. "
                f"Request body snippet: {str(query_params)[:1000]}"
            )
            if isinstance(e, RetryError):
                cause = e.__cause__
                if cause and hasattr(cause, "reason"):
                    logger.error(
                        f"Specific reason for retry exhaustion: {cause.reason}"
                    )
            raise Exception(f"Failed to query Weave traces due to network error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Weave server: {e}")
            raise Exception(f"Failed to parse Weave API response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during HTTP request to Weave server: {e}")
            raise
