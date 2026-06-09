"""GraphQL helpers compatible across W&B SDK GraphQL transports."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

import requests

from wandb_mcp_server.config import WANDB_BASE_URL


def execute_graphql(
    api: Any,
    query: str,
    variables: Mapping[str, Any] | None = None,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Execute a GraphQL document with the current or legacy W&B SDK transport."""
    variables_dict = dict(variables or {})
    if api_key:
        response = requests.post(
            f"{WANDB_BASE_URL.rstrip('/')}/graphql",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables_dict},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GraphQL response was not an object")
        return payload

    service_api = getattr(api, "__dict__", {}).get("_service_api")
    if service_api is not None and hasattr(service_api, "execute_graphql"):
        return service_api.execute_graphql(query, variables=variables_dict)

    gql = importlib.import_module("wandb_gql").gql
    return api.client.execute(gql(query), variable_values=variables_dict)
