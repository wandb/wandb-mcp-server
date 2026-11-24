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

from tests.agent_lens_fixtures import (
    BASE_URL,
    ENTITY,
    PROJECT,
    WINDOW,
    _FakeResponse,
    _mocked,
    _ok,
    load_contract_fixture,
)
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
        == "https://forge.coreweave.com/agent-lens/acme/support%2Fbot%20v1"
    )
    assert (
        public_forge_agent_lens_url("acme", "proj", "insights", "failure")
        == "https://forge.coreweave.com/agent-lens/acme/proj/insights/failure"
    )
    monkeypatch.setenv("FORGE_APP_BASE_URL", "http://insecure.internal.local")
    assert public_forge_agent_lens_url("acme", "proj") == "https://forge.coreweave.com/agent-lens/acme/proj"
    monkeypatch.setenv("FORGE_APP_BASE_URL", "https://forge-staging.coreweave.com/")
    assert public_forge_agent_lens_url("acme", "proj") == "https://forge-staging.coreweave.com/agent-lens/acme/proj"


def test_render_agent_lens_insights_contract_fixture_parity():
    contract = load_contract_fixture()
    responses = [
        _ok(contract["latest_week"]["data"]),
        _ok(contract["clustering_status"]["data"]),
        _ok(contract["failure_breakdowns"]["data"]),
        _ok(contract["intent_breakdowns"]["data"]),
        _ok(contract["failure_clusters"]["data"]),
        _ok(contract["intent_clusters"]["data"]),
        _ok(contract["failure_attributions"]["data"]),
    ]
    with _mocked(responses) as session:
        result = asyncio.run(
            agent_lens_insights_app.render_agent_lens_insights(
                ENTITY,
                PROJECT,
                window_days=7,
                agent_name="billing-router",
                trace_ids=["trace-stripe-8841"],
                _access_mode="read-write",
            )
        )

    assert not result.isError
    assert len(session.calls) == 7
    sc = result.structuredContent
    assert sc["schema"] == "wandb.agent_lens_insights/v1"
    assert sc["forgeUrl"] == f"https://forge.coreweave.com/agent-lens/{ENTITY}/{PROJECT}"
    assert sc["window"]["windowDays"] == 7
    assert sc["window"]["agentName"] == "billing-router"

    # 1. Coverage has firstWeek / latestWeek only (no unreachable coveragePct)
    assert sc["coverage"] == {
        "firstWeek": "2026-10-05",
        "latestWeek": "2026-11-16",
    }

    # 2. no_failure is filtered out; failingTurnCount = turn_count - no_failure.count
    assert sc["failure"]["turnCount"] == 12840
    assert sc["failure"]["failingTurnCount"] == 612
    category_ids = [c["categoryId"] for c in sc["failure"]["categories"]]
    assert "no_failure" not in category_ids
    assert category_ids == ["tool_misuse", "tool_failure", "wrong_output"]

    # 3. Product category labels + impact metrics + no category-level severity
    top_cat = sc["failure"]["categories"][0]
    assert top_cat["label"] == "Tool misuse"
    assert sc["failure"]["categories"][1]["label"] == "Tool failure"
    assert sc["failure"]["categories"][2]["label"] == "Incorrect result"
    assert sc["intent"]["categories"][0]["label"] == "Task request"
    assert "severity" not in top_cat
    assert top_cat["conversationCount"] == 248
    assert top_cat["frustratedCount"] == 74
    assert top_cat["averageLatencyMs"] == 1420.5
    assert top_cat["averageCostPerTurnUsd"] == 0.0046

    # 4. Cluster severity (info | minor | major) and description preserved
    top_cluster = top_cat["clusterBreakdowns"][0]
    assert top_cluster["id"] == "01a1fc1d-f6a4-7e82-9d62-fbdeb71f5d12"
    assert top_cluster["title"] == "lookup_invoice omits required currency"
    assert top_cluster["severity"] == "major"
    assert "422 Unprocessable Entity" in top_cluster["description"]
    assert top_cluster["conversationCount"] == 196

    # 5. Real TurnFailureAttribution fields mapped
    attr = sc["failureAttributions"][0]
    assert attr["traceId"] == "trace-stripe-8841"
    assert attr["failureId"] == "77869ce8-2917-5d33-bc74-ab63603e278e"
    assert attr["failureCategory"] == "tool_misuse"
    assert attr["categoryLabel"] == "Tool misuse"
    assert attr["failureSeverity"] == "major"
    assert attr["failureEvidenceSpanIds"] == ["span-tool-8841"]
    assert attr["failureCluster"] == {
        "id": "01a1fc1d-f6a4-7e82-9d62-fbdeb71f5d12",
        "title": "lookup_invoice omits required currency",
    }
    assert sc["warnings"] == []

    text_summary = json.loads(result.content[0].text)
    assert text_summary["schema"] == "wandb.agent_lens_insights/v1"
    assert text_summary["failure"]["turn_count"] == 12840
    assert text_summary["failure"]["failing_turn_count"] == 612


def test_severity_and_unknown_category_fallback_strictness():
    assert agent_lens_insights_app._normalize_severity("major") == "major"
    assert agent_lens_insights_app._normalize_severity("minor") == "minor"
    assert agent_lens_insights_app._normalize_severity("info") == "info"
    assert agent_lens_insights_app._normalize_severity("critical") is None
    assert agent_lens_insights_app._normalize_severity("high") is None
    assert agent_lens_insights_app.label_for_category("failure", "custom_domain_slip") == "Custom Domain Slip"


def test_render_agent_lens_insights_partial_cluster_404_degrades_gracefully():
    responses = [
        _ok({"first_week": "2026-11-16", "latest_week": "2026-11-16"}),
        _FakeResponse({"detail": "not clustered yet"}, status_code=404),
        _ok(
            {
                "turn_count": 10,
                "categories": [
                    {"category": "no_failure", "count": 6, "cluster_breakdowns": []},
                    {"category": "tool_misuse", "count": 4, "cluster_breakdowns": []},
                ],
            }
        ),
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
    assert sc["failure"]["turnCount"] == 10
    assert sc["failure"]["failingTurnCount"] == 4
    assert len(sc["failure"]["categories"]) == 1
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


def test_render_agent_conversation_and_trace_preserve_exact_text_and_selected_failure():
    contract = load_contract_fixture()
    canned_conv = contract["weave_conversation"]

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
            conversation_id="conv-stripe-8841",
            include_feedback=True,
            selected_failure={
                "trace_id": "trace-stripe-8841",
                "failure_id": "77869ce8-2917-5d33-bc74-ab63603e278e",
            },
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
    assert sc["forgeUrl"] == "https://forge.coreweave.com/agent-lens/acme/support-bot"
    assert sc["selectedFailure"] == {
        "traceId": "trace-stripe-8841",
        "failureId": "77869ce8-2917-5d33-bc74-ab63603e278e",
    }
    assert sc["summary"]["turnCount"] == 2
    assert sc["summary"]["errorCount"] == 1
    assert sc["turns"][1]["isError"] is True


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
                                    "selected_failure": {
                                        "trace_id": "trace-abc",
                                        "failure_id": "fail-1",
                                    },
                                },
                            )
                        assert not trace_result.isError
                        assert trace_result.structuredContent["schema"] == "wandb.agent_conversation/v1"
                        assert trace_result.structuredContent["accessMode"] == "read-only"
                        assert trace_result.structuredContent["agentLensAvailable"] is True
                        assert trace_result.structuredContent["selectedFailure"] == {
                            "traceId": "trace-abc",
                            "failureId": "fail-1",
                        }
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
    from mcp.types import ListToolsResult
    from wandb_mcp_server.runtime_contract import (
        load_runtime_contract,
        profile_combinations,
    )

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


def test_agent_conversation_unpacks_weave_tagged_union_messages_and_weave_ref_feedback():
    turn_uuid = "019a4f10-8841-7000-8000-abcdef123456"
    turn_hex = turn_uuid.replace("-", "")
    raw_payload = {
        "conversation_id": "conv-live-01",
        "total_turns": 1,
        "has_more": False,
        "turns": [
            {
                "trace_id": turn_uuid,
                "root_span_name": "studio_turn",
                "agent_name": "composer-v2",
                "agent_version": "2.4.0",
                "status_code": "OK",
                "provider": "anthropic",
                "started_at": "2026-11-23T18:10:00Z",
                "ended_at": "2026-11-23T18:10:03.620Z",
                "total_duration_ms": 3620,
                "total_cost_usd": None,
                "messages": [
                    {
                        "type": "agent_start",
                        "span_id": "span-start-1",
                        "agent_name": "composer-v2",
                        "agent_version": "2.4.0",
                        "status_code": "UNSET",
                        "error_type": None,
                        "status_message": None,
                        "started_at": "2026-11-23T18:10:00Z",
                        "user_message": None,
                        "assistant_message": None,
                        "tool_call": None,
                        "agent_start": {
                            "model": "claude-sonnet-4.5",
                            "system_instructions": "Preserve focused_track_id.",
                            "tool_definitions": "[]",
                            "status": "OK",
                        },
                        "agent_handoff": None,
                        "context_compacted": None,
                        "feedback": None,
                    },
                    {
                        "type": "user_message",
                        "span_id": "span-user-1",
                        "agent_name": "composer-v2",
                        "agent_version": "2.4.0",
                        "status_code": "OK",
                        "error_type": None,
                        "status_message": None,
                        "started_at": "2026-11-23T18:10:00.050Z",
                        "user_message": {
                            "text": "Punch in 4 bars on chorus_synth.",
                            "content_refs": [],
                        },
                        "assistant_message": None,
                        "tool_call": None,
                        "agent_start": None,
                        "agent_handoff": None,
                        "context_compacted": None,
                        "feedback": None,
                    },
                    {
                        "type": "tool_call",
                        "span_id": "span-tool-1",
                        "agent_name": "composer-v2",
                        "agent_version": "2.4.0",
                        "status_code": "OK",
                        "error_type": None,
                        "status_message": None,
                        "started_at": "2026-11-23T18:10:01.100Z",
                        "user_message": None,
                        "assistant_message": None,
                        "tool_call": {
                            "tool_name": "edit_track_region",
                            "tool_arguments": '{"stem":"lead_vox","bars":[9,12]}',
                            "tool_result": '{"ok":true,"mutated_stem":"lead_vox"}',
                            "duration_ms": 410,
                            "status": "OK",
                            "content_refs": [],
                        },
                        "agent_start": None,
                        "agent_handoff": None,
                        "context_compacted": None,
                        "feedback": None,
                    },
                    {
                        "type": "assistant_message",
                        "span_id": "span-asst-1",
                        "agent_name": "composer-v2",
                        "agent_version": "2.4.0",
                        "status_code": "OK",
                        "error_type": None,
                        "status_message": None,
                        "started_at": "2026-11-23T18:10:01.600Z",
                        "user_message": None,
                        "assistant_message": {
                            "text": "Updated bars 9-12 on the chorus.",
                            "model": "claude-sonnet-4.5",
                            "reasoning_content": None,
                            "reasoning_tokens": 0,
                            "input_tokens": 1240,
                            "output_tokens": 185,
                            "input_cost_usd": 0.00372,
                            "output_cost_usd": 0.002775,
                            "total_cost_usd": 0.006495,
                            "duration_ms": 1920,
                            "status": "OK",
                            "content_refs": [],
                        },
                        "tool_call": None,
                        "agent_start": None,
                        "agent_handoff": None,
                        "context_compacted": None,
                        "feedback": None,
                    },
                ],
                "feedback": [
                    {
                        "id": "fb-note-1",
                        "feedback_type": "wandb.note.1",
                        "weave_ref": f"weave:///acme/composer-prod/agent_turn/{turn_hex}",
                        "payload": {"note": "Mutated lead_vox while chorus_synth was focused"},
                        "creator": "pm-alex",
                        "created_at": "2026-11-23T18:15:00Z",
                        "wb_user_id": "user-1",
                        "scorer_tags": [],
                    }
                ],
            }
        ],
    }

    res = agent_conversation_app.build_agent_conversation_result(
        json.dumps(raw_payload),
        mode="conversation",
        entity_name="acme",
        project_name="composer-prod",
        conversation_id="conv-live-01",
    )
    assert not res.isError
    sc = res.structuredContent
    assert sc["summary"]["totalInputTokens"] == 1240
    assert sc["summary"]["totalOutputTokens"] == 185
    assert sc["summary"]["totalTokens"] == 1425
    assert sc["summary"]["totalCostUsd"] == pytest.approx(0.006495)

    turn = sc["turns"][0]
    assert [m["role"] for m in turn["messages"]] == ["system", "user", "tool", "assistant"]
    assert turn["messages"][1]["content"] == "Punch in 4 bars on chorus_synth."
    assert turn["messages"][2]["toolName"] == "edit_track_region"
    assert '"stem":"lead_vox"' in turn["messages"][2]["arguments"]
    assert '"mutated_stem":"lead_vox"' in turn["messages"][2]["toolResult"]
    assert turn["messages"][3]["content"] == "Updated bars 9-12 on the chorus."
    assert turn["model"] == "claude-sonnet-4.5"
    assert sc["feedback"][0]["traceId"] == turn_uuid
    assert sc["feedback"][0]["note"] == "Mutated lead_vox while chorus_synth was focused"
