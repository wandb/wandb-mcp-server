"""GraphQL helpers compatible across W&B SDK GraphQL transports."""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def execute_graphql(
    api: Any,
    query: str,
    variables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a GraphQL document with the current or legacy W&B SDK transport."""
    variables_dict = dict(variables or {})
    service_api = getattr(api, "__dict__", {}).get("_service_api")
    if service_api is not None and hasattr(service_api, "execute_graphql"):
        return service_api.execute_graphql(query, variables=variables_dict)

    gql = importlib.import_module("wandb_gql").gql
    return api.client.execute(gql(query), variable_values=variables_dict)
