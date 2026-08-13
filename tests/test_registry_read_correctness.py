"""Registry correctness tests using W&B 0.28 public paginator classes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from wandb.errors import AuthenticationError, CommError
from wandb.apis.public import Api
from wandb.proto import wandb_api_pb2
from wandb.sdk.lib.service.service_connection import WandbApiFailedError
from mcp.shared.memory import create_connected_server_and_client_session

from wandb_mcp_server.mcp_tools.query_artifacts import list_artifact_versions
from wandb_mcp_server.mcp_tools.query_registry import list_registries, list_registry_collections
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.registry_support import (
    RegistryMalformedResponse,
    registry_error_result,
    resolve_registry_organization,
)
from wandb_mcp_server.wandb_selective_reads import REGISTRY_ORGANIZATION_QUERY
from wandb_mcp_server.server import create_mcp_server
from wandb_mcp_server.wandb_graphql import validate_read_only_graphql


def _registry_node(name: str = "models") -> dict[str, Any]:
    return {
        "__typename": "Project",
        "id": f"registry-{name}",
        "internalId": f"registry-internal-{name}",
        "name": f"wandb-registry-{name}",
        "entity": {"name": "registry-entity", "organization": {"name": "Example Org"}},
        "description": None,
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": None,
        "access": "PRIVATE",
        "allowAllArtifactTypes": True,
        "artifactTypes": {"edges": []},
    }


def _collection_node(name: str = "my-model") -> dict[str, Any]:
    return {
        "__typename": "ArtifactPortfolio",
        "id": f"collection-{name}",
        "name": name,
        "description": None,
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": None,
        "project": {
            "id": "registry-models-project",
            "internalId": "registry-models-project-internal",
            "name": "wandb-registry-models",
            "entity": {"name": "registry-entity"},
        },
        "type": {"name": "model"},
        "tags": {"edges": []},
    }


class FakeRegistryService:
    """Small fake transport beneath the real W&B registry paginators."""

    def __init__(
        self,
        *,
        organization_response: dict[str, Any] | None = None,
        registries: list[dict[str, Any]] | None = None,
        collections: list[dict[str, Any]] | None = None,
        versions: list[dict[str, Any]] | None = None,
        default_entity: str | None = "team-entity",
        registry_pages: list[tuple[list[dict[str, Any]], str | None, bool]] | None = None,
        collection_pages: list[tuple[list[dict[str, Any]], str | None, bool]] | None = None,
    ) -> None:
        self.organization_response = organization_response or {
            "entity": {
                "organization": {
                    "name": "Example Org",
                    "orgEntity": {"name": "registry-entity"},
                },
                "user": None,
            }
        }
        self.registry_nodes = [_registry_node()] if registries is None else registries
        self.collection_nodes = [_collection_node()] if collections is None else collections
        self.version_nodes = (
            [
                {
                    "versionIndex": 0,
                    "aliases": [{"alias": "latest"}],
                    "artifactCollection": {"name": "my-model"},
                    "artifact": {
                        "id": "artifact-0",
                        "state": "COMMITTED",
                        "description": None,
                        "size": 12,
                        "fileCount": 1,
                        "createdAt": "2025-01-01T00:00:00Z",
                        "updatedAt": None,
                        "digest": "digest-0",
                        "tags": [{"name": "production"}],
                    },
                }
            ]
            if versions is None
            else versions
        )
        self.default_entity = default_entity
        self.registry_pages = registry_pages
        self.collection_pages = collection_pages
        self.page_indexes = {"registries": 0, "collections": 0}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def feature_enabled(self, feature: Any) -> bool:
        return True

    def execute_graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        parse: Any = None,
        **_: Any,
    ) -> Any:
        variables = dict(variables or {})
        if "MCPRegistryOrganization" in query:
            operation = "organization"
            if variables.get("hasEntity"):
                result = self.organization_response
            else:
                entity = self.organization_response.get("entity") or {}
                user = entity.get("user") or {}
                organization = entity.get("organization")
                organizations = user.get("organizations")
                if organizations is None and organization is not None:
                    organizations = [organization]
                result = {
                    "entity": None,
                    "viewer": {
                        "entity": self.default_entity,
                        "organizations": organizations or [],
                    },
                }
        elif "GetDefaultEntity" in query:
            operation = "default_entity"
            result = {"viewer": {"id": "viewer-id", "entity": self.default_entity}}
        elif "FetchRegistries" in query:
            operation = "registries"
            nodes, end_cursor, has_next = self._page(
                operation,
                self.registry_nodes,
                self.registry_pages,
            )
            result = {
                "organization": {
                    "orgEntity": {
                        "projects": {
                            "pageInfo": {
                                "__typename": "PageInfo",
                                "endCursor": end_cursor,
                                "hasNextPage": has_next,
                            },
                            "edges": [{"node": node} for node in nodes],
                        }
                    }
                }
            }
        elif "RegistryCollections" in query:
            operation = "collections"
            nodes, end_cursor, has_next = self._page(
                operation,
                self.collection_nodes,
                self.collection_pages,
            )
            result = {
                "organization": {
                    "orgEntity": {
                        "name": "registry-entity",
                        "artifactCollections": {
                            "totalCount": len(self.collection_nodes),
                            "pageInfo": {
                                "__typename": "PageInfo",
                                "endCursor": end_cursor,
                                "hasNextPage": has_next,
                            },
                            "edges": [{"node": node} for node in nodes],
                        },
                    }
                }
            }
        elif "MCPRegistryArtifactVersions" in query:
            operation = "versions"
            result = {
                "organization": {
                    "orgEntity": {
                        "artifactMemberships": {
                            "edges": [
                                {"cursor": f"version-{index}", "node": node}
                                for index, node in enumerate(self.version_nodes)
                            ],
                            "pageInfo": {"endCursor": None, "hasNextPage": False},
                        }
                    }
                }
            }
        else:  # pragma: no cover - makes an unexpected SDK query fail loudly
            raise AssertionError("unexpected registry GraphQL operation")
        self.calls.append((operation, variables))
        return parse(json.dumps(result)) if callable(parse) else result

    def _page(
        self,
        operation: str,
        default_nodes: list[dict[str, Any]],
        pages: list[tuple[list[dict[str, Any]], str | None, bool]] | None,
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        if pages is None:
            return default_nodes, None, False
        index = self.page_indexes[operation]
        self.page_indexes[operation] += 1
        if index >= len(pages):
            raise AssertionError(f"{operation} exceeded the configured fake pages")
        return pages[index]


def _api(service: FakeRegistryService, *, settings: dict[str, Any] | None = None) -> Api:
    api = object.__new__(Api)
    api._service_api = service
    api._default_entity = None
    api.settings = dict({"entity": "team-entity"} if settings is None else settings)
    return api


def _personal_orgs(*names: str) -> dict[str, Any]:
    return {
        "entity": {
            "organization": None,
            "user": {
                "organizations": [{"name": name, "orgEntity": {"name": f"{name.lower()}-entity"}} for name in names]
            },
        }
    }


def test_explicit_organization_wins_without_resolution_request() -> None:
    service = FakeRegistryService(organization_response={"entity": None})
    api = _api(service)

    assert resolve_registry_organization(api, "Explicit Org") == "Explicit Org"
    assert service.calls == []


def test_configured_organization_wins_without_resolution_request() -> None:
    service = FakeRegistryService(organization_response={"entity": None})
    api = _api(service, settings={"organization": "Configured Org", "entity": "team"})

    assert resolve_registry_organization(api) == "Configured Org"
    assert service.calls == []


def test_team_entity_resolution_uses_same_transport_and_is_cached() -> None:
    service = FakeRegistryService()
    api = _api(service)

    assert resolve_registry_organization(api) == "Example Org"
    assert resolve_registry_organization(api) == "Example Org"
    assert [operation for operation, _ in service.calls] == ["organization"]
    assert service.calls[0][1] == {"entity": "team-entity", "hasEntity": True}


def test_team_organization_and_entity_names_are_bounded() -> None:
    service = FakeRegistryService(
        organization_response={
            "entity": {
                "organization": {
                    "name": "x" * 513,
                    "orgEntity": {"name": "registry-entity"},
                },
                "user": None,
            }
        }
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "malformed_response"


def test_organization_cache_is_isolated_per_api_client() -> None:
    first_service = FakeRegistryService(organization_response=_personal_orgs("First Org"))
    second_service = FakeRegistryService(organization_response=_personal_orgs("Second Org"))

    assert resolve_registry_organization(_api(first_service)) == "First Org"
    assert resolve_registry_organization(_api(second_service)) == "Second Org"
    assert [operation for operation, _ in first_service.calls] == ["organization"]
    assert [operation for operation, _ in second_service.calls] == ["organization"]


def test_sdk_default_entity_and_single_personal_org_are_supported() -> None:
    service = FakeRegistryService(organization_response=_personal_orgs("Only Org"))
    api = _api(service, settings={})

    assert resolve_registry_organization(api) == "Only Org"
    assert [operation for operation, _ in service.calls] == ["organization"]
    assert service.calls[0][1] == {"entity": "", "hasEntity": False}


def test_unconfigured_default_org_is_resolved_in_one_fixed_request() -> None:
    service = FakeRegistryService(default_entity="registry-entity")
    api = _api(service, settings={})

    assert resolve_registry_organization(api) == "Example Org"
    assert [operation for operation, _ in service.calls] == ["organization"]


def test_multiple_personal_orgs_return_bounded_candidates() -> None:
    service = FakeRegistryService(organization_response=_personal_orgs("Org B", "Org A"))
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result == {
        "error": "organization_required",
        "message": "Specify organization because more than one organization is accessible.",
        "organization_candidates": ["Org A", "Org B"],
        "candidates_truncated": False,
    }
    assert [operation for operation, _ in service.calls] == ["organization"]


def test_many_personal_orgs_return_bounded_candidates() -> None:
    service = FakeRegistryService(organization_response=_personal_orgs(*(f"Org {index:02d}" for index in range(25))))
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "organization_required"
    assert len(result["organization_candidates"]) == 20
    assert result["candidates_truncated"] is True


@pytest.mark.parametrize(
    "response",
    [
        {"entity": None},
        {"entity": {"organization": None, "user": None}},
        {"entity": {"organization": None, "user": {"organizations": []}}},
    ],
)
def test_null_or_missing_organization_is_stable(response: dict[str, Any]) -> None:
    service = FakeRegistryService(organization_response=response)
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "organization_resolution_failed"
    assert "TypeError" not in json.dumps(result)


def test_malformed_organization_connection_is_stable() -> None:
    service = FakeRegistryService(
        organization_response={"entity": {"organization": None, "user": {"organizations": {"bad": "shape"}}}}
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "malformed_response"


def test_registry_and_collection_tools_use_real_sdk_paginators_without_alias_reads() -> None:
    service = FakeRegistryService()
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        registries = json.loads(list_registries())
        collections = json.loads(list_registry_collections("models"))

    assert registries["returned_count"] == 1
    assert registries["items"][0]["description"] is None
    assert registries["items"][0]["updated_at"] is None
    assert collections["returned_count"] == 1
    assert collections["items"][0]["description"] is None
    assert collections["items"][0]["updated_at"] is None
    assert collections["items"][0]["aliases"] is None
    assert collections["items"][0]["aliases_loaded"] is False
    operations = [operation for operation, _ in service.calls]
    assert operations == ["organization", "registries", "registries", "collections"]
    assert "ArtifactCollectionAliases" not in " ".join(operations)


def test_registry_version_workflow_reuses_resolved_org_and_public_registry_search() -> None:
    service = FakeRegistryService()
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_artifacts.WandBApiManager.get_api", return_value=api):
        versions = json.loads(
            list_artifact_versions(
                "my-model",
                registry_name="models",
                source="registry",
            )
        )

    assert versions["returned_count"] == 1
    assert versions["items"][0]["aliases"] == ["latest"]
    assert versions["items"][0]["description"] is None
    assert versions["items"][0]["updated_at"] is None
    assert [operation for operation, _ in service.calls] == ["organization", "registries", "versions"]


def test_fifty_collections_use_bounded_pages_and_zero_alias_requests() -> None:
    service = FakeRegistryService(
        collections=[_collection_node(f"model-{index}") for index in range(50)],
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registry_collections("models", max_items=50))

    assert result["returned_count"] == 50
    assert result["total_count"] == 50
    assert [operation for operation, _ in service.calls] == ["organization", "registries", "collections"]


def test_registry_paginator_crosses_one_empty_page_under_a_request_cap() -> None:
    service = FakeRegistryService(
        registry_pages=[
            ([], "registry-page-1", True),
            ([_registry_node()], None, False),
        ]
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["returned_count"] == 1
    assert [operation for operation, _ in service.calls] == ["organization", "registries", "registries"]


def test_registry_paginator_rejects_repeated_cursor_without_request_amplification() -> None:
    service = FakeRegistryService(
        registry_pages=[
            ([], "repeated", True),
            ([], "repeated", True),
        ]
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "malformed_response"
    assert [operation for operation, _ in service.calls] == ["organization", "registries", "registries"]


def test_registry_paginator_enforces_a_hard_empty_page_request_ceiling() -> None:
    service = FakeRegistryService(registry_pages=[([], f"page-{index}", True) for index in range(10)])
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert result["error"] == "malformed_response"
    assert [operation for operation, _ in service.calls].count("registries") == 3


def test_collection_paginator_crosses_one_empty_page_under_a_request_cap() -> None:
    service = FakeRegistryService(
        collection_pages=[
            ([], "collection-page-1", True),
            ([_collection_node()], None, False),
        ]
    )
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registry_collections("models"))

    assert result["returned_count"] == 1
    assert [operation for operation, _ in service.calls] == [
        "organization",
        "registries",
        "collections",
        "collections",
    ]


def test_nonexistent_registry_returns_not_found_before_collection_query() -> None:
    service = FakeRegistryService(registries=[])
    api = _api(service)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registry_collections("missing"))

    assert result["error"] == "resource_not_found"
    assert [operation for operation, _ in service.calls] == ["organization", "registries"]


def test_filter_limits_fail_before_api_construction() -> None:
    too_deep: dict[str, Any] = {}
    cursor = too_deep
    for _ in range(14):
        nested: dict[str, Any] = {}
        cursor["next"] = nested
        cursor = nested

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api") as get_api:
        result = json.loads(list_registries(filter=too_deep))

    assert result["error"] == "invalid_input"
    get_api.assert_not_called()


def test_oversized_filter_fails_before_api_construction() -> None:
    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api") as get_api:
        result = json.loads(list_registries(filter={"name": "x" * (64 * 1024)}))

    assert result["error"] == "invalid_input"
    get_api.assert_not_called()


@pytest.mark.parametrize("bad_filter", [{"metric": math.nan}, {"metric": object()}])
def test_non_json_filters_fail_before_api_construction(bad_filter: dict[str, Any]) -> None:
    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api") as get_api:
        result = json.loads(list_registries(filter=bad_filter))

    assert result["error"] == "invalid_input"
    get_api.assert_not_called()


def test_registry_response_obeys_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FakeRegistryService()
    service.registry_nodes[0]["description"] = "large-value-" * 20_000
    api = _api(service)
    monkeypatch.setattr("wandb_mcp_server.registry_support.MAX_RESPONSE_TOKENS", 512)

    with patch("wandb_mcp_server.mcp_tools.query_registry.WandBApiManager.get_api", return_value=api):
        result = json.loads(list_registries())

    assert "large-value" not in json.dumps(result)
    assert result.get("truncated") is True or result["error"] == "response_too_large"


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (401, "secret canary", "authentication_failed"),
        (403, "secret canary", "permission_denied"),
        (404, "secret canary", "resource_not_found"),
        (429, "rate limited secret canary", "server_busy"),
        (503, "service overloaded secret canary", "server_busy"),
    ],
)
def test_stable_http_error_mapping_does_not_echo_upstream(
    status: int,
    message: str,
    expected: str,
) -> None:
    class UpstreamError(RuntimeError):
        status_code = status

    result = registry_error_result(UpstreamError(message))

    assert result["error"] == expected
    assert "canary" not in json.dumps(result)


def test_timeout_and_malformed_errors_are_stable() -> None:
    class RequestTimeout(RuntimeError):
        pass

    timeout = registry_error_result(RequestTimeout("secret at internal.svc.cluster.local"))
    malformed = registry_error_result(RegistryMalformedResponse("secret"))

    assert timeout["error"] == "upstream_timeout"
    assert malformed["error"] == "malformed_response"
    assert "secret" not in json.dumps([timeout, malformed])


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (401, "authentication rejected", "authentication_failed"),
        (403, "permission rejected", "permission_denied"),
        (404, "resource absent", "resource_not_found"),
        (429, "too many requests", "server_busy"),
        (503, "service unavailable: capacity exhausted", "server_busy"),
    ],
)
def test_real_wandb_service_error_shapes_map_to_stable_errors(
    status: int,
    message: str,
    expected: str,
) -> None:
    response = wandb_api_pb2.ApiErrorResponse(http_status=status, message=message)
    result = registry_error_result(WandbApiFailedError("W&B API request failed", response))

    assert result["error"] == expected


def test_real_wandb_authentication_and_timeout_shapes_are_stable() -> None:
    authentication = registry_error_result(AuthenticationError("API key rejected"))
    timeout = registry_error_result(
        CommError(
            "The W&B service process is busy and did not respond in time.",
            exc=TimeoutError("internal.svc secret canary"),
        )
    )

    assert authentication["error"] == "authentication_failed"
    assert timeout["error"] == "upstream_timeout"
    assert "canary" not in json.dumps([authentication, timeout])


def test_organization_projection_is_fixed_and_read_only() -> None:
    assert REGISTRY_ORGANIZATION_QUERY.lstrip().startswith("query MCPRegistryOrganization")
    document = validate_read_only_graphql(REGISTRY_ORGANIZATION_QUERY)
    assert len(document.definitions) == 1


def test_registry_tools_do_not_use_implicit_sdk_registry_resolution() -> None:
    package_root = Path(__file__).parents[1] / "src" / "wandb_mcp_server" / "mcp_tools"
    for filename in ("query_registry.py", "query_artifacts.py"):
        source = (package_root / filename).read_text()
        assert ".registry(" not in source


@pytest.mark.asyncio
async def test_complete_registry_workflow_through_official_mcp_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = FakeRegistryService()
    api = _api(service)
    artifact = type("ArtifactFixture", (), {})()
    artifact.id = "artifact-0"
    artifact.name = "my-model:v0"
    artifact.type = "model"
    artifact.version = "v0"
    artifact.state = "COMMITTED"
    artifact.description = None
    artifact.size = 12
    artifact.file_count = 1
    artifact.tags = ["production"]
    artifact.aliases = ["latest"]
    artifact.metadata = {}
    artifact.created_at = "2025-01-01T00:00:00Z"
    artifact.digest = "digest-0"
    artifact.commit_hash = None
    artifact.source_qualified_name = None
    artifact.linked_artifacts = []
    artifact.source_artifact = artifact
    artifact.logged_by = lambda: None
    artifact.used_by = lambda: []
    artifact.files = lambda: iter(())
    api.artifact = lambda *args, **kwargs: artifact  # type: ignore[method-assign]
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")

    with patch.object(WandBApiManager, "get_api", return_value=api):
        server = create_mcp_server("stdio")
        async with create_connected_server_and_client_session(server) as session:
            registries = _tool_json(await session.call_tool("list_registries_tool", {}))
            collections = _tool_json(
                await session.call_tool(
                    "list_registry_collections_tool",
                    {"registry_name": registries["items"][0]["name"]},
                )
            )
            versions = _tool_json(
                await session.call_tool(
                    "list_artifact_versions_tool",
                    {
                        "collection_name": collections["items"][0]["name"],
                        "registry_name": registries["items"][0]["name"],
                        "source": "registry",
                    },
                )
            )
            details = _tool_json(
                await session.call_tool(
                    "get_artifact_details_tool",
                    {"artifact_name": "registry-entity/wandb-registry-models/my-model:v0"},
                )
            )

    assert registries["returned_count"] == 1
    assert collections["returned_count"] == 1
    assert versions["returned_count"] == 1
    assert details["artifact"]["version"] == "v0"
    assert [operation for operation, _ in service.calls] == [
        "organization",
        "registries",
        "registries",
        "collections",
        "registries",
        "versions",
    ]


def _tool_json(result: Any) -> dict[str, Any]:
    assert result.isError is False
    assert result.content
    text = getattr(result.content[0], "text", None)
    assert isinstance(text, str)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed
