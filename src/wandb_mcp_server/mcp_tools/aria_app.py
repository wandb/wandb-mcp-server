"""Interactive compact ARIA conversation card MCP App (`ui://wandb/aria-agent.html`)."""

from __future__ import annotations

from importlib.resources import files
import os
from typing import Any, Dict, Literal

from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.wandb_urls import public_forge_aria_url

ARIA_RESOURCE_URI = "ui://wandb/aria-agent.html"
ARIA_UI_META: Dict[str, Any] = {
    "ui": {"resourceUri": ARIA_RESOURCE_URI},
    "ui/resourceUri": ARIA_RESOURCE_URI,
}


def register_aria_agent_resource(mcp: FastMCP) -> None:
    """Register the offline single-file ARIA conversation card MCP App resource."""
    if not hasattr(mcp, "resource"):
        return
    ui: Dict[str, Any] = {
        "prefersBorder": True,
        "csp": {"connectDomains": [], "resourceDomains": ["data:"]},
    }
    if domain := os.environ.get("WANDB_MCP_APP_DOMAIN"):
        ui["domain"] = domain

    @mcp.resource(ARIA_RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def aria_agent_html() -> str:
        return files("wandb_mcp_server").joinpath("data/aria-agent.html").read_text(encoding="utf-8")


def enrich_aria_result_for_ui(
    result: Dict[str, Any],
    *,
    access_mode: Literal["read-only", "read-write"],
) -> Dict[str, Any]:
    """Attach server access mode and Forge ARIA deep links on successful ARIA tool payloads."""
    if not isinstance(result, dict) or result.get("ok") is False:
        return result
    enriched = dict(result)
    enriched.setdefault("access_mode", access_mode)
    enriched.setdefault("forge_home_url", public_forge_aria_url())
    thread_id = enriched.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        enriched.setdefault("forge_url", public_forge_aria_url(thread_id))
    return enriched


__all__ = [
    "ARIA_RESOURCE_URI",
    "ARIA_UI_META",
    "enrich_aria_result_for_ui",
    "register_aria_agent_resource",
]
