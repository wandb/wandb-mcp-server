"""Unit and stateless MCP protocol tests for the Agent Lens & Weave Conversation MCP Apps."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
from importlib.resources import files
import json
import re
from unittest.mock import MagicMock, patch

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
import pytest

from tests.agent_lens_fixtures import BASE_URL, ENTITY, PROJECT, WINDOW, _FakeResponse, _mocked, _ok
from wandb_mcp_server.admission import tool_cost
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.mcp_tools import agent_conversation_app, agent_lens_insights_app
from wandb_mcp_server.runtime_contract import resolve_runtime_selection
from wandb_mcp_server.server import register_tools
from wandb_mcp_server.wandb_urls import public_forge_agent_lens_url


def test_public_forge_agent_lens_url_encodes_segments_and_enforces_https(monkeypatch: pytest.MonkeyPatch):
    assert (
        public_forge_agent_lens_url("acme", "support/bot v1")
        == "https://forge.coreweave.com/acme/support%2Fbot%20v1/agent-lens"
    )
    monkeypatch.setenv("FORGE_APP_BASE_URL", "http://insecure.internal.local")
    assert public_forge_agent_lens_url("acme", "proj") == "https://forge.coreweave.com/acme/proj/agent-lens"
    monkeypatch.setenv("FORGE_APP_BASE_URL", "https://forge-staging.coreweave.com/")
    assert public_forge_agent_lens_url("acme", "proj") == "https://forge-staging.coreweave.com/acme/proj/agent-lens"


def test_render_agent_lens_insights_gathers_parallel_sections_and_merges_clusters():
    responses = [
        _ok({"first_week": "2026-10-05", "latest_week": "2026-11-16", "classified_turns": 470, "total_turns": 500}),
        _ok([{"signature_type": "failure", "completed_at": "2026-11-23T02:00:00Z"}]),
        _ok(
            {
                "turn_count": 20,
                "categories": [
                    {
                        "category": "tool_failure",
                        "count": 15,
                        "severity": "critical",
                        "cluster_breakdowns": [{"id": "f1", "count": 12}],
                    }
                ],
            }
        ),
        _ok(
            {
                "turn_count": 450,
                "categories": [
                    {"category": "billing_question", "count": 300, "cluster_breakdowns": [{"id": "i1", "count": 200}]}
                ],
            }
        ),
        _ok([{"id": "f1", "title": "Stripe 422 Missing Currency", "share": 0.6, "attribution": "tool_error"}]),
        _ok([{"id": "i1", "title": "Invoice Lookup", "share": 0.44}]),
        _ok([{"trace_id": "trace-1", "root_cause": "tool_error", "summary": "Missing currency argument"}]),
    ]
    with _mocked(responses) as session:
        result = asyncio.run(
            agent_lens_insights_app.render_agent_lens_insights(
                ENTITY,
                PROJECT,
                window_days=7,
                agent_name="billing-router",
                trace_ids=["trace-1"],
                _access_mode="read-write",
            )
        )

    assert not result.isError
    assert len(session.calls) == 7
    sc = result.structuredContent
    assert sc["schema"] == "wandb.agent_lens_insights/v1"
    assert sc["forgeUrl"] == f"https://forge.coreweave.com/{ENTITY}/{PROJECT}/agent-lens"
    assert sc["window"]["windowDays"] == 7
    assert sc["window"]["agentName"] == "billing-router"
    assert sc["coverage"]["coveragePct"] == 94.0
    assert sc["failure"]["turnCount"] == 20
    assert sc["failure"]["categories"][0]["categoryId"] == "tool_failure"
    assert sc["failure"]["categories"][0]["share"] == 0.75
    assert sc["failure"]["clusters"][0]["id"] == "f1"
    assert sc["failure"]["clusters"][0]["title"] == "Stripe 422 Missing Currency"
    assert sc["failure"]["clusters"][0]["count"] == 12
    assert sc["failure"]["clusters"][0]["categoryId"] == "tool_failure"
    assert sc["failureAttributions"][0]["rootCause"] == "tool_error"
    assert sc["warnings"] == []

    text_summary = json.loads(result.content[0].text)
    assert text_summary["schema"] == "wandb.agent_lens_insights/v1"
    assert text_summary["failure"]["turn_count"] == 20


def test_render_agent_lens_insights_partial_cluster_404_degrades_gracefully():
    responses = [
        _ok({"first_week": "2026-11-16", "latest_week": "2026-11-16"}),
        _FakeResponse({"detail": "not clustered yet"}, status_code=404),
        _ok({"turn_count": 4, "categories": [{"category": "timeout", "count": 4, "cluster_breakdowns": []}]}),
        _ok({"turn_count": 10, "categories": []}),
        _FakeResponse({"detail": "not clustered yet"}, status_code=404),
        _FakeResponse({"detail": "not clustered yet"}, status_code=404),
    ]
    with _mocked(responses):
        result = asyncio.run(
            agent_lens_insights_app.render_agent_lens_insights(
                ENTITY,
                PROJECT,
                start_at=WINDOW["start_at"],
                end_at=WINDOW["end_at"],
            )
        )

    assert not result.isError
    sc = result.structuredContent
    assert sc["failure"]["turnCount"] == 4
    assert sc["coverage"]["coveragePct"] is None
    assert len(sc["warnings"]) == 3


def test_render_agent_lens_insights_rejects_invalid_arguments_and_propagates_forbidden():
    with _mocked(_ok({})) as session:
        bad_days = asyncio.run(agent_lens_insights_app.render_agent_lens_insights(ENTITY, PROJECT, window_days=5))
        half_window = asyncio.run(
            agent_lens_insights_app.render_agent_lens_insights(ENTITY, PROJECT, start_at=WINDOW["start_at"])
        )
    assert session.calls == []
    assert bad_days.isError
    assert bad_days.structuredContent["error"] == "agent_lens_invalid_request"
    assert half_window.isError

    with _mocked(_FakeResponse({"detail": "forbidden"}, status_code=403)):
        forbidden = asyncio.run(agent_lens_insights_app.render_agent_lens_insights(ENTITY, PROJECT))
    assert forbidden.isError
    assert forbidden.structuredContent["error"] == "agent_lens_forbidden"


def test_render_agent_conversation_and_trace_preserve_exact_text_and_populate_structured_content():
    canned_conv = {
        "conversation_id": "conv-99",
        "agent_name": "support-router",
        "agent_version": "v2",
        "total_turns": 2,
        "has_more": False,
        "total_cost_usd": 0.0084,
        "turns": [
            {
                "trace_id": "tr-1",
                "status_code": "OK",
                "total_duration_ms": 310,
                "input_tokens": 120,
                "output_tokens": 45,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Where is my refund?"}]},
                    {"role": "assistant", "content": "Checking order #55."},
                ],
                "feedback": [{"id": "fb-1", "feedback_type": "wandb.agent_user_feedback", "scorer_tags": ["👍"]}],
            },
            {
                "trace_id": "tr-2",
                "status_code": "ERROR",
                "total_duration_ms": 890,
                "input_tokens": 200,
                "output_tokens": 60,
                "messages": [
                    {
                        "role": "tool",
                        "tool_name": "refund_order",
                        "arguments": {"order_id": "55"},
                        "status_code": "ERROR",
                        "error": "Refund window expired",
                    }
                ],
            },
        ],
    }

    fake_resp = MagicMock(status_code=200, headers={})
    fake_resp.json.return_value = canned_conv
    fake_session = MagicMock()
    fake_session.post.return_value = fake_resp

    with (
        patch.object(WandBApiManager, "get_api_key", return_value="test-key"),
        patch.object(WandBApiManager, "get_api", return_value=MagicMock(viewer=MagicMock())),
        patch("wandb_mcp_server.mcp_tools.agents.get_no_retry_session", return_value=fake_session),
    ):
        conv_res = agent_conversation_app.render_agent_conversation(
            "acme",
            "support-bot",
            conversation_id="conv-99",
            include_feedback=True,
            _access_mode="read-only",
            _agent_lens_available=True,
        )

    assert not conv_res.isError
    assert json.loads(conv_res.content[0].text) == canned_conv
    sc = conv_res.structuredContent
    assert sc["schema"] == "wandb.agent_conversation/v1"
    assert sc["mode"] == "conversation"
    assert sc["accessMode"] == "read-only"
    assert sc["agentLensAvailable"] is True
    assert sc["forgeUrl"] == "https://forge.coreweave.com/acme/support-bot/agent-lens"
    assert sc["summary"]["turnCount"] == 2
    assert sc["summary"]["errorCount"] == 1
    assert sc["summary"]["totalInputTokens"] == 320
    assert sc["turns"][0]["messages"][0]["content"] == "Where is my refund?"
    assert sc["turns"][1]["isError"] is True
    assert sc["turns"][1]["errorMessage"] == "Refund window expired"
    assert sc["feedback"][0]["emoji"] == "👍"


def test_mcp_app_resources_follow_selected_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-agents")
    monkeypatch.setenv("WANDB_MCP_TOOL_EXTENSIONS", "agent-lens")
    monkeypatch.setenv("AGENT_LENS_BASE_URL", BASE_URL)
    selection = resolve_runtime_selection()

    full_server = FastMCP("mcp-apps-selected")
    register_tools(full_server, selection)
    uris = {str(r.uri) for r in asyncio.run(full_server.list_resources())}
    assert agent_conversation_app.CONVERSATION_RESOURCE_URI in uris
    assert agent_lens_insights_app.INSIGHTS_RESOURCE_URI in uris

    stripped = replace(
        selection,
        tools=selection.tools
        - {
            "get_weave_agent_conversation_tool",
            "get_weave_agent_trace_tool",
            agent_lens_insights_app.INSIGHTS_TOOL_NAME,
        },
    )
    stripped_server = FastMCP("mcp-apps-stripped")
    register_tools(stripped_server, stripped)
    stripped_uris = {str(r.uri) for r in asyncio.run(stripped_server.list_resources())}
    assert agent_conversation_app.CONVERSATION_RESOURCE_URI not in stripped_uris
    assert agent_lens_insights_app.INSIGHTS_RESOURCE_URI not in stripped_uris


@pytest.mark.asyncio
async def test_stateless_streamable_http_serves_both_new_mcp_apps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave-agents")
    monkeypatch.setenv("WANDB_MCP_TOOL_EXTENSIONS", "agent-lens")
    monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", "read-only")
    monkeypatch.setenv("AGENT_LENS_BASE_URL", BASE_URL)

    server = InstrumentedFastMCP("agent-mcp-apps-test", stateless_http=True, json_response=True)
    register_tools(server)
    app = server.streamable_http_app()

    canned_trace = {
        "trace_id": "trace-abc",
        "conversation_id": "conv-abc",
        "agent_name": "triage-bot",
        "status_code": "OK",
        "total_duration_ms": 210,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_resp = MagicMock(status_code=200, headers={})
    fake_resp.json.return_value = canned_trace
    fake_session = MagicMock()
    fake_session.post.return_value = fake_resp

    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                async with streamable_http_client("http://localhost:8000/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        tools = {t.name: t for t in (await client.list_tools()).tools}

                        for conv_tool_name in ("get_weave_agent_conversation_tool", "get_weave_agent_trace_tool"):
                            assert tools[conv_tool_name].meta == {
                                "ui": {"resourceUri": agent_conversation_app.CONVERSATION_RESOURCE_URI},
                                "ui/resourceUri": agent_conversation_app.CONVERSATION_RESOURCE_URI,
                            }
                        assert tools[agent_lens_insights_app.INSIGHTS_TOOL_NAME].meta == {
                            "ui": {"resourceUri": agent_lens_insights_app.INSIGHTS_RESOURCE_URI},
                            "ui/resourceUri": agent_lens_insights_app.INSIGHTS_RESOURCE_URI,
                        }

                        for uri, expected_marker in (
                            (agent_conversation_app.CONVERSATION_RESOURCE_URI, "wandb-agent-conversation"),
                            (agent_lens_insights_app.INSIGHTS_RESOURCE_URI, "wandb-agent-lens-insights"),
                        ):
                            res = await client.read_resource(uri)
                            block = res.contents[0]
                            assert block.mimeType == "text/html;profile=mcp-app"
                            assert block.meta["ui"]["csp"] == {
                                "connectDomains": [],
                                "resourceDomains": ["data:"],
                            }
                            assert expected_marker in block.text

                        with (
                            patch.object(WandBApiManager, "get_api_key", return_value="test-key"),
                            patch.object(WandBApiManager, "get_api", return_value=MagicMock(viewer=MagicMock())),
                            patch("wandb_mcp_server.mcp_tools.agents.get_no_retry_session", return_value=fake_session),
                        ):
                            trace_result = await client.call_tool(
                                "get_weave_agent_trace_tool",
                                {
                                    "entity_name": "acme",
                                    "project_name": "support",
                                    "trace_id": "trace-abc",
                                },
                            )
                        assert not trace_result.isError
                        assert trace_result.structuredContent["schema"] == "wandb.agent_conversation/v1"
                        assert trace_result.structuredContent["accessMode"] == "read-only"
                        assert trace_result.structuredContent["agentLensAvailable"] is True
                        assert json.loads(trace_result.content[0].text) == canned_trace
    finally:
        server._sync_executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "bundle_name",
    ["line-plot.html", "agent-conversation.html", "agent-lens-insights.html"],
)
def test_all_mcp_app_bundles_are_offline_and_integrity_stamped(bundle_name: str):
    html = files("wandb_mcp_server").joinpath(f"data/{bundle_name}").read_text(encoding="utf-8")

    class ExternalAttrScanner(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            del tag
            for key, value in attrs:
                if key in {"src", "href"} and value:
                    assert not re.match(r"(?:https?:)?//", value), f"External attr in {bundle_name}: {value}"

    ExternalAttrScanner().feed(html)
    assert "fetch(" not in html
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    js = re.search(r"<script[^>]*>(.*?)</script>", html, re.S)[1]
    css = re.search(r"<style[^>]*>(.*?)</style>", html, re.S)[1]
    assert hashlib.sha256(js.encode("utf-8")).hexdigest() in html
    assert hashlib.sha256(css.encode("utf-8")).hexdigest() in html

    notices = files("wandb_mcp_server").joinpath("data/THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    assert bundle_name in notices
    assert "@modelcontextprotocol/ext-apps@" in notices
    assert "uplot@" in notices
    assert "zod@" in notices
    assert tool_cost(agent_lens_insights_app.INSIGHTS_TOOL_NAME) == ("heavy", 4)


def test_tools_list_serialized_payload_stays_under_128_kib_across_all_22_profiles(
    monkeypatch: pytest.MonkeyPatch,
):
    import asyncio
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ListToolsResult
    from wandb_mcp_server.runtime_contract import (
        load_runtime_contract,
        profile_combinations,
    )
    from wandb_mcp_server.server import register_tools

    contract = load_runtime_contract()
    combinations = profile_combinations(contract)
    assert len(combinations) == 11

    max_bytes = 0
    checked_profiles = 0
    for combination in combinations:
        for access_mode in ("read-write", "read-only"):
            monkeypatch.setenv("MCP_WORKLOAD_PROFILE", "local")
            monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", combination.base_profile)
            monkeypatch.setenv("WANDB_MCP_TOOL_EXTENSIONS", ",".join(combination.extensions))
            monkeypatch.setenv("WANDB_MCP_ACCESS_MODE", access_mode)
            monkeypatch.setenv("WF_TRACE_SERVER_URL", "https://trace.wandb.ai")
            monkeypatch.setenv("WB_AGENT_BASE_URL", "https://wb-agent.wandb.ai")
            monkeypatch.setenv("AGENT_LENS_BASE_URL", "https://agent-pulse.wandb.ai")
            mcp = FastMCP("test")
            register_tools(mcp)
            tools = asyncio.run(mcp.list_tools())
            wire_json = ListToolsResult(tools=tools).model_dump_json(by_alias=True, exclude_none=True)
            wire_bytes = len(wire_json.encode("utf-8"))
            assert wire_bytes < 131_072, (
                f"Profile {combination.name} ({access_mode}) exceeded 128 KiB: {wire_bytes} bytes"
            )
            max_bytes = max(max_bytes, wire_bytes)
            checked_profiles += 1

    assert checked_profiles == 22
    assert max_bytes < 126_000
