from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
from importlib.resources import files
import json
import re

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult
import pytest

from wandb_mcp_server.admission import current_tool_deadline, tool_cost
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.mcp_tools import line_plot
from wandb_mcp_server.runtime_contract import resolve_runtime_selection
from wandb_mcp_server.server import register_tools
from wandb_mcp_server.trace_utils import count_tokens_conservative


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-only")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-only")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    monkeypatch.delenv("WANDB_MCP_TOOL_EXTENSIONS", raising=False)
    monkeypatch.delenv("WANDB_MCP_APP_DOMAIN", raising=False)


@pytest.fixture
def history(monkeypatch):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "run_name": "gentle-sweep",
                "sampled": True,
                "truncated": False,
                "rows": [
                    {"_step": 2, "loss": 0.3, "accuracy": 0.8},
                    {"_step": 0, "loss": 0.9},
                    {"_step": 1, "loss": 0.5, "accuracy": 0.6},
                ],
            }
        )

    monkeypatch.setattr(line_plot, "get_run_history", fetch)
    return calls


def render(**kwargs):
    return line_plot.render_line_plot(
        **{
            "entity_name": "team",
            "project_name": "project",
            "run_ids": ["a"],
            "keys": ["loss"],
            **kwargs,
        }
    )


def test_happy_path_compact_series_and_text_fallback(history):
    result = render(run_ids=["a", "b"], keys=["loss", "accuracy"], title="Learning curves")
    assert isinstance(result, CallToolResult)
    assert not result.isError
    data = result.structuredContent
    assert data["title"] == "Learning curves"
    assert data["url"].endswith("/team/project")
    assert [call["run_id"] for call in history] == ["a", "b"]
    assert data["runs"][0]["name"] == "gentle-sweep"
    assert data["runs"][0]["url"].endswith("/team/project/runs/a")
    assert data["runs"][0]["series"] == {"loss": [[0, 0.9], [1, 0.5], [2, 0.3]], "accuracy": [[1, 0.6], [2, 0.8]]}
    text = result.content[0].text
    assert "loss: 3 points; first=0.9, last=0.3, min=0.3, max=0.9" in text
    assert "accuracy: 2 points" in text
    assert "Open in W&B:" in text
    assert "[[" not in text
    assert len(line_plot.LINE_PLOT_DESCRIPTION) < 4000


@pytest.mark.parametrize(
    "arguments",
    [
        {"run_ids": []},
        {"run_ids": ["a"] * 11},
        {"run_ids": ["a", "a"]},
        {"run_ids": None},
        {"keys": []},
        {"keys": [str(i) for i in range(6)]},
        {"keys": ["loss", "loss"]},
        {"keys": [None]},
        {"keys": [" "]},
        {"entity_name": ""},
        {"project_name": None},
        {"x_axis": ""},
        {"samples": 0},
        {"samples": True},
        {"samples": float("nan")},
        {"samples": None},
        {"min_step": -1},
        {"max_step": True},
        {"max_step": float("inf")},
        {"min_step": 10, "max_step": 2},
        {"max_step": line_plot.MAX_SAFE_HISTORY_STEP + 1},
        {"min_step": 0, "max_step": line_plot.MCP_MAX_HISTORY_RANGE_STEPS},
        {"title": "x" * 201},
        {"keys": ["x" * 513]},
        {"entity_name": "\ud800"},
    ],
)
def test_invalid_input_does_not_read_history(history, arguments):
    result = render(**arguments)
    assert result.isError
    assert result.structuredContent["error"] == "invalid_input"
    assert history == []


def test_range_profile_validation_and_sample_clamp(history, monkeypatch):
    monkeypatch.setattr(line_plot, "MCP_WORKLOAD_PROFILE", "shared")
    monkeypatch.setattr(line_plot, "MCP_MAX_HISTORY_SAMPLES", 25)
    assert render(min_step=1).isError
    assert history == []
    result = render(samples=10000, min_step=5, max_step=10)
    assert not result.isError
    assert history[0]["samples"] == 25
    assert (history[0]["min_step"], history[0]["max_step"]) == (5, 10)
    assert result.structuredContent["arguments"]["samples"] == 25


def test_partial_failure_preserves_success_and_hides_upstream_details(history, monkeypatch):
    fetch = line_plot.get_run_history

    def sometimes(**kwargs):
        if kwargs["run_id"] == "bad":
            raise PermissionError("Bearer secret-private-content")
        return fetch(**kwargs)

    monkeypatch.setattr(line_plot, "get_run_history", sometimes)
    result = render(run_ids=["bad", "a"])
    assert not result.isError
    assert result.structuredContent["runs"][0]["error"] == "history_fetch_failed"
    assert result.structuredContent["runs"][1]["series"]["loss"]
    assert "secret-private-content" not in result.model_dump_json()


@pytest.mark.parametrize("response", ["null", "[]", "not json", '{"error":"auth_required"}', '{"rows":null}'])
def test_all_run_failures_are_tool_errors(monkeypatch, response):
    monkeypatch.setattr(line_plot, "get_run_history", lambda **_: response)
    result = render()
    assert result.isError
    assert "history_fetch_failed" in result.content[0].text


def test_deadline_prevents_backend_work(history):
    token = current_tool_deadline.set(0)
    try:
        result = render(run_ids=["a", "b"])
    finally:
        current_tool_deadline.reset(token)
    assert history == []
    assert result.isError


@pytest.mark.parametrize("axis", ["_timestamp", "epoch"])
def test_custom_axis_empty_and_non_finite_values(monkeypatch, axis):
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **_: json.dumps(
            {
                "rows": [
                    {axis: 2, "loss": 0.3},
                    {axis: 1, "loss": float("nan")},
                    {axis: float("inf"), "loss": 1},
                    {axis: 3, "loss": True},
                    {axis: 4, "loss": "text"},
                    {"_step": 5, "loss": 1},
                ]
            }
        ),
    )
    result = render(x_axis=axis, keys=["loss", "missing"])
    assert not result.isError
    assert result.structuredContent["runs"][0]["series"] == {"loss": [[2, 0.3]], "missing": []}
    assert "missing: 0 points; no finite numeric points" in result.content[0].text
    json.dumps(result.model_dump(), allow_nan=False)


def test_response_budget_and_source_truncation(monkeypatch):
    monkeypatch.setattr(line_plot, "MAX_RESPONSE_TOKENS", 1800)
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **_: json.dumps(
            {
                "source_truncated": True,
                "rows": [{"_step": i, "loss": i / 9} for i in range(500)],
            }
        ),
    )
    result = render(run_ids=["a", "b"])
    data = result.structuredContent
    assert data["truncated"]
    assert data["runs"][0]["truncated"]
    assert 0 < len(data["runs"][0]["series"]["loss"]) < 500
    assert (
        count_tokens_conservative(
            json.dumps({"content": result.content[0].text, "structuredContent": data}, separators=(",", ":"))
        )
        <= 1800
    )


def test_tiny_budget_fails_closed(history, monkeypatch):
    monkeypatch.setattr(line_plot, "MAX_RESPONSE_TOKENS", 1)
    result = render()
    assert result.isError
    assert result.structuredContent == {"error": "response_too_large"}


def test_resource_is_only_registered_with_selected_tool(monkeypatch):
    selection = resolve_runtime_selection()
    for selected in (True, False):
        server = FastMCP("resource-test")
        chosen = selection if selected else replace(selection, tools=selection.tools - {line_plot.TOOL_NAME})
        register_tools(server, chosen)
        resources = asyncio.run(server.list_resources())
        assert len(resources) == int(selected)
        if selected:
            assert str(resources[0].uri) == line_plot.RESOURCE_URI
            assert resources[0].mimeType == "text/html;profile=mcp-app"


@pytest.mark.asyncio
async def test_official_client_preserves_metadata_content_and_structured_result(history, monkeypatch):
    monkeypatch.setenv("WANDB_MCP_APP_DOMAIN", "configured.claudemcpcontent.com")
    server = InstrumentedFastMCP("line-plot-test", stateless_http=True, json_response=True)
    register_tools(server)
    app = server.streamable_http_app()
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        tools = await client.list_tools()
                        tool = next(t for t in tools.tools if t.name == line_plot.TOOL_NAME)
                        assert tool.meta == {
                            "ui": {"resourceUri": line_plot.RESOURCE_URI},
                            "ui/resourceUri": line_plot.RESOURCE_URI,
                        }
                        resource = await client.read_resource(line_plot.RESOURCE_URI)
                        content = resource.contents[0]
                        assert content.mimeType == "text/html;profile=mcp-app"
                        assert content.meta["ui"] == {
                            "domain": "configured.claudemcpcontent.com",
                            "csp": {"connectDomains": [], "resourceDomains": []},
                        }
                        assert "new uPlot" in content.text
                        result = await client.call_tool(
                            line_plot.TOOL_NAME,
                            {"entity_name": "team", "project_name": "project", "run_ids": ["a"], "keys": ["loss"]},
                        )
                        assert not result.isError
                        assert result.structuredContent["runs"][0]["series"]["loss"] == [[0, 0.9], [1, 0.5], [2, 0.3]]
                        assert result.content[0].text.startswith("loss\nOpen in W&B:")
                        assert "[[" not in result.content[0].text
                        invalid = await client.call_tool(
                            line_plot.TOOL_NAME,
                            {"entity_name": "team", "project_name": "project", "run_ids": [], "keys": ["loss"]},
                        )
                        assert invalid.isError
    finally:
        server._sync_executor.shutdown(wait=True)


def test_app_is_offline_and_vendor_hashes_match():
    html = files("wandb_mcp_server").joinpath("data/line-plot.html").read_text()

    class Loads(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src", "href"} and value:
                    assert not re.match(r"(?:https?:)?//", value)

    Loads().feed(html)
    assert "fetch(" not in html
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    js = re.search(r"<script>(.*?)</script>", html, re.S)[1]
    css = re.search(r"<style>(.*?)</style>", html, re.S)[1]
    assert hashlib.sha256(js.encode()).hexdigest() in html
    assert hashlib.sha256(css.encode()).hexdigest() in html
    assert "MIT License" in files("wandb_mcp_server").joinpath("data/UPLOT_LICENSE.txt").read_text()
    assert tool_cost(line_plot.TOOL_NAME) == ("heavy", 4)
