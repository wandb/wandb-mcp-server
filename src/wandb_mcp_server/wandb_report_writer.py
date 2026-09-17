"""Bounded persistence for W&B Workspaces reports.

``wandb_workspaces.Report.save`` currently enumerates every project in an
entity before creating the target project.  That existence check is
unbounded, and can exhaust the MCP tool deadline for large entities.  A
public ``Api.project`` point lookup can establish whether creation is needed
without requiring project-update permission for an existing project.

This module intentionally owns the only application-side GraphQL mutation.
Callers cannot provide a document or alter the selected operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import requests
import wandb_workspaces.reports.v2.interface as wr_interface

from wandb_mcp_server.api_client import WandBWriteOutcomeUnknown


class WandBReportWriteError(RuntimeError):
    """Raised when W&B does not acknowledge a report upsert."""


def save_report_bounded(report: Any, api: Any) -> Any:
    """Persist one Workspaces report with bounded point operations.

    The supplied ``api`` is the request-scoped, centrally configured
    ``wandb.Api``.  Its base URL, API key, and timeout therefore match every
    other MCP W&B call. Existing projects take one lookup and one report
    upsert; missing projects add one project-create call. No global API state
    is changed.
    """
    model = report._to_model()

    _ensure_project_exists(
        api,
        entity_name=model.project.entity_name,
        project_name=model.project.name,
    )

    variables = {
        "id": model.id or None,
        "name": model.name or wr_interface.internal._generate_name(),
        "entityName": model.project.entity_name,
        "projectName": model.project.name,
        "description": model.description,
        "displayName": model.display_name,
        "type": "runs",
        "spec": model.spec.model_dump_json(by_alias=True, exclude_none=True),
    }
    result = _execute_fixed_report_upsert(api, variables)
    report_id = _extract_report_id(result)
    report.id = report_id
    return report


def _ensure_project_exists(
    api: Any,
    *,
    entity_name: str,
    project_name: str,
) -> None:
    """Use a bounded public point lookup, creating only on genuine absence."""
    project = api.project(project_name, entity=entity_name)
    try:
        # Api.project is lazy. Accessing id performs one GET_PROJECT_GQL query.
        project.id
    except ValueError as exc:
        if str(exc) != f"Project {project_name} not found":
            # Permission, authentication, and transport failures must not be
            # mistaken for absence. In particular, an unconditional
            # create_project call can require ProjectUpdatePermission.
            raise
        api.create_project(project_name, entity_name)


def _execute_fixed_report_upsert(api: Any, variables: Mapping[str, Any]) -> object:
    """Run the fixed mutation through Workspaces' supported SDK adapter."""
    try:
        return wr_interface.execute_graphql(
            api,
            wr_interface.gql.upsert_view,
            dict(variables),
        )
    except Exception as exc:
        if _is_transport_outcome_uncertain(exc):
            raise WandBWriteOutcomeUnknown() from exc
        raise


def _is_transport_outcome_uncertain(exc: BaseException) -> bool:
    """Recognize transport loss after the fixed write was dispatched."""
    uncertain_types = (
        TimeoutError,
        ConnectionError,
        httpx.TimeoutException,
        httpx.NetworkError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, uncertain_types):
            return True
        for nested in (
            getattr(current, "exc", None),
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _extract_report_id(result: object) -> str:
    """Validate only the stable response field needed by ``Report.url``."""
    if isinstance(result, Mapping):
        upsert = result.get("upsertView")
        if isinstance(upsert, Mapping):
            view = upsert.get("view")
            if isinstance(view, Mapping):
                report_id = view.get("id")
                if isinstance(report_id, str) and report_id:
                    return report_id
    # Do not include the upstream payload: it may contain customer report
    # content or infrastructure details.
    raise WandBReportWriteError("W&B returned an invalid report upsert response.")


__all__ = ["WandBReportWriteError", "save_report_bounded"]
