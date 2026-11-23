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
                "rows": [{"_step": 0, "loss": 1}, {"_step": 1, "loss": 3}, {"_step": 2, "loss": 5}],
            }
        )

    monkeypatch.setattr(line_plot, "get_run_history", fetch)
    return calls


def render(config=None, **kwargs):
    return line_plot.render_line_plot(
        **{
            "entity_name": "team",
            "project_name": "project",
            "run_ids": ["a"],
            "config": {"metrics": ["loss"], **(config or {})},
            **kwargs,
        }
    )


def test_happy_path_shape_and_text(history):
    result = render({"chartTitle": "Loss"}, run_ids=["a", "b"])
    data = result.structuredContent
    assert not result.isError
    assert data["schema"] == "wandb.graph/v1"
    assert [c["run_id"] for c in history] == ["a", "b"]
    assert data["series"][0]["historySeries"] == [{"x": 0, "y": 1}, {"x": 1, "y": 3}, {"x": 2, "y": 5}]
    s = data["series"][0]
    assert data["seriesMetadataById"][s["metadataId"]]["legendHref"].endswith("/runs/a")
    assert data["seriesStylesById"][s["seriesId"]]["color"] == line_plot.PALETTE[0]
    assert data["effectiveConfig"]["groupAgg"] == "mean"
    assert "first=1, last=5, min=1, max=5" in result.content[0].text
    assert "[{" not in result.content[0].text
    assert len(line_plot.LINE_PLOT_DESCRIPTION) <= 4000


@pytest.mark.parametrize(
    "cfg",
    [
        {"metrics": []},
        {"metrics": ["x"] * 6},
        {"metrics": ["x", "x"]},
        {"metrics": None},
        {"metrics": [True]},
        {"xAxis": ""},
        {"surprise": 1},
        {"x_axis": "_step"},
        {"xAxisMin": float("nan")},
        {"yAxisMax": True},
        {"xAxisMin": 2, "xAxisMax": 1},
        {"yLogScale": True, "yAxisMin": 0},
        {"showLegend": "yes"},
        {"limit": 11},
        {"limit": True},
        {"smoothingWeight": -1},
        {"smoothingWeight": float("inf")},
        {"smoothingType": "bad"},
        {"smoothingType": "gaussian", "smoothingWeight": 101},
        {"overrideMarks": {"a:loss": "bad"}},
        {"overrideLineWidths": {"a:loss": -1}},
        {"overrideColors": {"a:loss": "red"}},
        {"chartTitle": "x" * 201},
    ],
)
def test_invalid_config(history, cfg):
    result = render(cfg)
    assert result.isError
    assert result.structuredContent["error"] == "invalid_input"
    assert not history


@pytest.mark.parametrize(
    "args",
    [
        {"run_ids": []},
        {"run_ids": ["a", "a"]},
        {"run_ids": ["a"] * 11},
        {"samples": True},
        {"samples": 0},
        {"min_step": -1},
        {"max_step": True},
        {"max_step": line_plot.MAX_SAFE_HISTORY_STEP + 1},
        {"min_step": 10, "max_step": 2},
        {"entity_name": "\ud800"},
        {"config": None},
        {"config": []},
    ],
)
def test_invalid_execution(history, args):
    result = line_plot.render_line_plot(
        **{"entity_name": "team", "project_name": "project", "run_ids": ["a"], "config": {"metrics": ["loss"]}, **args}
    )
    assert result.isError
    assert not history


@pytest.mark.parametrize(
    "field,value",
    [
        ("expressions", []),
        ("metricRegex", "loss"),
        ("useMetricRegex", False),
        ("xExpression", ""),
        ("fullFidelityMode", "standard"),
        ("groupedFullFidelityMode", "standard"),
        ("pointVisualizationMethod", "bucketing-gorilla"),
        ("plotType", "bar"),
        ("groupAgg", "last"),
        ("groupAgg", "samples"),
        ("groupArea", "samples"),
        ("groupBy", "state"),
        ("xAxis", "_absolute_runtime"),
    ],
)
def test_unsupported(history, field, value):
    result = render({field: value})
    assert result.structuredContent["error"] == "unsupported_config"
    assert result.structuredContent["fields"] == [field]
    assert not history


def test_limits_ignored_and_domains(history, monkeypatch):
    monkeypatch.setattr(line_plot, "MCP_MAX_HISTORY_SAMPLES", 25)
    monkeypatch.setattr(line_plot, "MCP_WORKLOAD_PROFILE", "shared")
    assert render(min_step=1).isError
    result = render(
        {
            "fontSize": "large",
            "tooltipNumberOfRuns": "all_runs",
            "legendPosition": "west",
            "pointVisualizationMethod": "sampling",
            "xAxisMin": 1,
            "xAxisMax": 9,
            "yAxisMin": None,
            "limit": 1,
        },
        samples=5000,
        min_step=0,
        max_step=20,
        run_ids=["a", "b"],
    )
    p = result.structuredContent
    assert history[0]["samples"] == 25
    assert p["arguments"]["config"]["fontSize"] == "large"
    assert p["coverage"]["ignoredConfig"] == ["fontSize", "tooltipNumberOfRuns"]
    assert p["coverage"]["truncated"]
    assert len(p["series"]) == 1
    assert p["xDomain"] == [1, 9]
    assert p["legend"]["position"] == "west"
    assert render(min_step=0, max_step=line_plot.MCP_MAX_HISTORY_RANGE_STEPS).isError


@pytest.mark.parametrize("response", ["null", "[]", "bad json", '{"error":"auth_required"}', '{"rows":null}'])
def test_all_failures(history, monkeypatch, response):
    monkeypatch.setattr(line_plot, "get_run_history", lambda **kw: response)
    result = render()
    assert result.isError
    assert result.structuredContent["coverage"]["perRunErrors"][0]["error"] == "history_fetch_failed"


def test_partial_errors_and_deadline(history, monkeypatch):
    fetch = line_plot.get_run_history

    def some(**kw):
        if kw["run_id"] == "bad":
            raise PermissionError("secret-upstream-canary")
        return fetch(**kw)

    monkeypatch.setattr(line_plot, "get_run_history", some)
    result = render(run_ids=["bad", "a"])
    assert not result.isError
    assert len(result.structuredContent["series"]) == 1
    assert "secret-upstream-canary" not in result.model_dump_json()
    token = current_tool_deadline.set(0)
    try:
        before = len(history)
        assert render().isError
        assert len(history) == before
    finally:
        current_tool_deadline.reset(token)


@pytest.mark.parametrize("axis,multiplier", [("_timestamp", 1000), ("_runtime", 1), ("epoch", 1)])
def test_axes_and_empty(history, monkeypatch, axis, multiplier):
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps(
            {"rows": [{axis: 2, "loss": 3}, {axis: 3, "loss": float("nan")}, {axis: 4, "loss": True}]}
        ),
    )
    p = render({"xAxis": axis, "metrics": ["loss", "missing"]}).structuredContent
    assert p["series"][0]["historySeries"] == [{"x": 2 * multiplier, "y": 3}]
    assert p["series"][1]["historySeries"] == []
    assert p["axis"]["xIsDatetime"] == (axis == "_timestamp")


@pytest.mark.parametrize(
    "kind,weight,expected",
    [
        ("none", 0.25, [1, 3, 5]),
        ("exponential", 0.25, [1, 7 / 3, 27 / 7]),
        ("average", 2, [1, 2, 4]),
    ],
)
def test_smoothing_numeric(history, kind, weight, expected):
    p = render({"smoothingType": kind, "smoothingWeight": weight, "showOriginalAfterSmoothing": True}).structuredContent
    assert [v["y"] for v in p["series"][0]["historySeries"]] == pytest.approx(expected)
    assert len(p["series"]) == (1 if kind == "none" else 2)
    if kind != "none":
        assert p["series"][1]["role"] == "original"
        assert [v["y"] for v in p["series"][1]["historySeries"]] == [1, 3, 5]


def test_time_weighted_smoothing(history, monkeypatch):
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps(
            {"rows": [{"_step": 0, "loss": 1}, {"_step": 1, "loss": 3}, {"_step": 1500, "loss": 5}]}
        ),
    )
    alpha = (0.25 * 0.999) ** 0.5
    a = alpha ** (1 / 1500 * 1500)
    b = alpha ** (1499 / 1500 * 1500)
    expected = [1, (a + 3) / (a + 1), ((a + 3) * b + 5) / ((a + 1) * b + 1)]
    p = render({"smoothingType": "exponentialTimeWeighted", "smoothingWeight": 0.25}).structuredContent
    assert [v["y"] for v in p["series"][0]["historySeries"]] == pytest.approx(expected)


def test_gaussian_reflect_numeric(history):
    import math

    sigma = 0.25
    # radius ceil(4*sigma+.5)=2; reflected [3,1,1,3,5] at the first sample.
    a, b = math.exp(-1 / (2 * sigma * sigma)), math.exp(-4 / (2 * sigma * sigma))
    first = (b * 3 + a * 1 + 1 + a * 3 + b * 5) / (1 + 2 * a + 2 * b)
    p = render({"smoothingType": "gaussian", "smoothingWeight": sigma}).structuredContent
    assert [v["y"] for v in p["series"][0]["historySeries"]] == pytest.approx([first, 3, 6 - first])


@pytest.mark.parametrize("kind", ["none", "exponential", "exponentialTimeWeighted", "gaussian", "average"])
def test_smoothing_empty_and_singleton(kind):
    from wandb_mcp_server.mcp_tools.line_plot_smoothing import smooth

    assert smooth([], 0.25, kind) == []
    assert smooth([{"x": 0, "y": 4}], 0.25, kind) == [{"x": 0, "y": 4}]


def test_invalid_json_config(history):
    cycle = []
    cycle.append(cycle)
    assert render({"legendFields": cycle}).structuredContent["error"] == "invalid_input"
    assert render({"legendTemplate": "x" * 33000}).isError


def test_overflow_does_not_emit_nonfinite_json(grouping, monkeypatch):
    monkeypatch.setattr(line_plot, "get_run_history", lambda **kw: json.dumps({"rows": [{"_step": 0, "loss": 1e308}]}))
    p = render({"aggregate": True, "groupAgg": "sum"}, run_ids=["a", "b"]).structuredContent
    assert p["coverage"]["truncated"]
    assert p["series"][0]["historySeries"] == []
    json.dumps(p, allow_nan=False)


@pytest.fixture
def grouping(history, monkeypatch):
    calls = []
    monkeypatch.setattr(line_plot.WandBApiManager, "get_api", staticmethod(lambda: object()))

    def snapshot(api, **kw):
        calls.append(kw)
        return {"group": "train", "config": {k: "train" for k in kw["config_keys"]}}

    monkeypatch.setattr(line_plot, "fetch_projected_run", snapshot)
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps(
            {
                "rows": [{"_step": 0, "loss": 1}, {"_step": 1, "loss": 3}]
                if kw["run_id"] == "a"
                else [{"_step": 0, "loss": 5}, {"_step": 2, "loss": 7}]
            }
        ),
    )
    return calls


@pytest.mark.parametrize("agg,value", [("mean", 3), ("median", 3), ("min", 1), ("max", 5), ("sum", 6)])
@pytest.mark.parametrize("area", ["minmax", "stddev", "stderr", "none"])
def test_grouping(grouping, agg, value, area):
    p = render({"groupBy": "optimizer.value", "groupAgg": agg, "groupArea": area}, run_ids=["a", "b"]).structuredContent
    assert grouping[0]["config_keys"] == ["optimizer"]
    assert p["coverage"]["groupingAlignment"] == "exact-match-union"
    assert p["series"][0]["historySeries"] == [{"x": 0, "y": value}, {"x": 1, "y": 3}, {"x": 2, "y": 7}]
    if area == "none":
        assert len(p["series"]) == 1
    else:
        band = p["series"][1]
        delta = {"minmax": 2, "stddev": 8**0.5, "stderr": 2}[area]
        assert band["kind"] == "area"
        assert band["historySeries"][0] == pytest.approx({"x": 0, "y": 3 + delta, "y0": 3 - delta})
        assert band["historySeries"][1] == {"x": 1, "y": 3, "y0": 3}


@pytest.mark.parametrize("key,expected", [("group", []), ("config.optimizer", ["optimizer"]), ("a.value.b", ["a.b"])])
def test_group_paths(grouping, key, expected):
    assert not render({"groupBy": key}).isError
    assert grouping[0]["config_keys"] == expected


def test_overrides(history):
    p = render(
        {
            "overrideColors": {"a:loss": {"color": "#123456", "transparentColor": "#12345633"}},
            "overrideSeriesTitles": {"a:loss": "Custom"},
            "overrideLineWidths": {"a:loss": 3},
            "overrideMarks": {"a:loss": "dashed"},
            "yLogScale": True,
            "xAxisTitle": "Steps",
            "showLegend": False,
        }
    ).structuredContent
    s = p["series"][0]
    assert p["seriesStylesById"][s["seriesId"]] == {
        "color": "#123456",
        "colorKey": "a:loss",
        "lineWidth": 3,
        "mark": "dashed",
    }
    assert p["seriesMetadataById"][s["metadataId"]]["title"] == "Custom"
    assert p["axis"]["yScale"] == "log"
    assert not p["legend"]["show"]


def test_response_budget(history, monkeypatch):
    monkeypatch.setattr(line_plot, "MAX_RESPONSE_TOKENS", 2500)
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps({"source_truncated": True, "rows": [{"_step": i, "loss": i / 9} for i in range(500)]}),
    )
    result = render()
    assert result.structuredContent["coverage"]["truncated"]
    assert len(result.structuredContent["series"][0]["historySeries"]) < 500
    assert (
        count_tokens_conservative(
            json.dumps(
                {"content": result.content[0].text, "structuredContent": result.structuredContent},
                separators=(",", ":"),
            )
        )
        <= 2500
    )
    monkeypatch.setattr(line_plot, "MAX_RESPONSE_TOKENS", 1)
    assert render().structuredContent["error"] == "response_too_large"


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
                            "prefersBorder": True,
                            "domain": "configured.claudemcpcontent.com",
                            "csp": {"connectDomains": [], "resourceDomains": ["data:"]},
                        }
                        assert "wandb-line-plot" in content.text
                        assert line_plot.TOOL_NAME in content.text
                        result = await client.call_tool(
                            line_plot.TOOL_NAME,
                            {
                                "entity_name": "team",
                                "project_name": "project",
                                "run_ids": ["a"],
                                "config": {"metrics": ["loss"]},
                            },
                        )
                        assert not result.isError
                        assert result.structuredContent["series"][0]["historySeries"] == [
                            {"x": 0, "y": 1},
                            {"x": 1, "y": 3},
                            {"x": 2, "y": 5},
                        ]
                        assert result.content[0].text.startswith("loss\nOpen in W&B:")
                        assert "[[" not in result.content[0].text
                        invalid = await client.call_tool(
                            line_plot.TOOL_NAME,
                            {
                                "entity_name": "team",
                                "project_name": "project",
                                "run_ids": [],
                                "config": {"metrics": ["loss"]},
                            },
                        )
                        assert invalid.isError
    finally:
        server._sync_executor.shutdown(wait=True)


def test_app_is_offline_and_vendor_hashes_match():
    html = files("wandb_mcp_server").joinpath("data/line-plot.html").read_text(encoding="utf-8")

    class Loads(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src", "href"} and value:
                    assert not re.match(r"(?:https?:)?//", value)

    Loads().feed(html)
    assert "fetch(" not in html
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    js = re.search(r"<script[^>]*>(.*?)</script>", html, re.S)[1]
    css = re.search(r"<style[^>]*>(.*?)</style>", html, re.S)[1]
    assert hashlib.sha256(js.encode("utf-8")).hexdigest() in html
    assert hashlib.sha256(css.encode("utf-8")).hexdigest() in html
    notices = files("wandb_mcp_server").joinpath("data/THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    assert "@modelcontextprotocol/ext-apps@" in notices
    assert "uplot@" in notices
    assert "zod@" in notices
    assert "MIT License" in files("wandb_mcp_server").joinpath("data/UPLOT_LICENSE.txt").read_text(encoding="utf-8")
    assert tool_cost(line_plot.TOOL_NAME) == ("heavy", 4)


def test_one_call_shorthand_auto_discovers_entity_and_runs(monkeypatch):
    from wandb_mcp_server.wandb_selective_reads import ProjectedRunPage

    monkeypatch.setattr(line_plot.WandBApiManager, "get_api", staticmethod(lambda: object()))
    monkeypatch.setattr(line_plot, "fetch_viewer_default_entity", lambda api: "viewer-team")

    discovery_calls = []

    def fake_projected_runs(api, **kwargs):
        discovery_calls.append(kwargs)
        return ProjectedRunPage(
            items=[
                {"id": "live-1", "display_name": "Live Run", "state": "running", "group": None, "config": {}},
                {"id": "done-2", "display_name": "Done Run", "state": "finished", "group": None, "config": {}},
            ],
            total_count=2,
            has_more=False,
            next_cursor=None,
            requests=1,
        )

    monkeypatch.setattr(line_plot, "fetch_projected_runs", fake_projected_runs)
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps(
            {
                "run_name": "Live Run" if kw["run_id"] == "live-1" else "Done Run",
                "run_state": "running" if kw["run_id"] == "live-1" else "finished",
                "sampled": True,
                "rows": [{"_step": 0, "train/loss": 2.0}, {"_step": 1, "train/loss": 1.0}],
            }
        ),
    )

    result = line_plot.render_line_plot(project_name="llm-proj", metric="train/loss", limit=2)
    assert not result.isError
    data = result.structuredContent
    assert discovery_calls[0]["entity"] == "viewer-team"
    assert discovery_calls[0]["filters"] == {"$or": [{"keys_info.keys.train/loss": {"$ne": None}}]}
    assert discovery_calls[0]["limit"] == 2
    assert data["arguments"]["entity_name"] == "viewer-team"
    assert data["arguments"]["run_ids"] == ["live-1", "done-2"]
    assert data["coverage"]["autoDiscoveredRuns"] is True
    assert data["hasActiveRuns"] is True
    assert data["runsById"]["live-1"]["active"] is True
    assert data["runsById"]["done-2"]["active"] is False
    assert data["workspaceUrl"].endswith("/viewer-team/llm-proj/workspace")
    assert "active_runs=1" in result.content[0].text


def test_entity_and_discovery_edge_cases(history, monkeypatch):
    from wandb_mcp_server.wandb_selective_reads import ProjectedRunPage, SelectiveReadUnavailable

    monkeypatch.setattr(line_plot.WandBApiManager, "get_api", staticmethod(lambda: object()))
    monkeypatch.setattr(line_plot, "fetch_viewer_default_entity", lambda api: None)

    no_entity = line_plot.render_line_plot(project_name="proj", metric="loss")
    assert no_entity.isError
    assert no_entity.structuredContent["error"] == "no_default_entity"

    conflict_metric = line_plot.render_line_plot(
        entity_name="team",
        project_name="proj",
        run_ids=["a"],
        metric="loss",
        config={"metrics": ["acc"]},
    )
    assert conflict_metric.isError
    assert conflict_metric.structuredContent["error"] == "invalid_input"

    conflict_limit = line_plot.render_line_plot(
        entity_name="team",
        project_name="proj",
        run_ids=["a"],
        metric="loss",
        limit=2,
        config={"metrics": ["loss"], "limit": 5},
    )
    assert conflict_limit.isError
    assert conflict_limit.structuredContent["error"] == "invalid_input"

    # Fallback when keys_info filter raises SelectiveReadUnavailable, returning empty project cleanly
    attempts = []

    def fallback_runs(api, **kwargs):
        attempts.append(kwargs["filters"])
        if len(attempts) == 1:
            raise SelectiveReadUnavailable("keys_info unsupported on legacy server")
        return ProjectedRunPage(items=[], total_count=0, has_more=False, next_cursor=None, requests=1)

    monkeypatch.setattr(line_plot, "fetch_viewer_default_entity", lambda api: "team")
    monkeypatch.setattr(line_plot, "fetch_projected_runs", fallback_runs)
    empty_res = line_plot.render_line_plot(project_name="empty-proj", metric="loss")
    assert not empty_res.isError
    assert len(attempts) == 2
    assert empty_res.structuredContent["series"] == []
    assert "No runs found logging loss in team/empty-proj." in empty_res.content[0].text


def test_raw_history_series_preserved_when_smoothed_and_downsampled(history, monkeypatch):
    smoothed_res = render({"smoothingType": "exponential", "smoothingWeight": 0.25})
    entry = smoothed_res.structuredContent["series"][0]
    assert entry["rawHistorySeries"] == [{"x": 0, "y": 1}, {"x": 1, "y": 3}, {"x": 2, "y": 5}]
    assert [p["y"] for p in entry["historySeries"]] == pytest.approx([1, 7 / 3, 27 / 7])

    monkeypatch.setattr(line_plot, "MAX_RESPONSE_TOKENS", 2500)
    monkeypatch.setattr(
        line_plot,
        "get_run_history",
        lambda **kw: json.dumps({"rows": [{"_step": i, "loss": float(i)} for i in range(400)]}),
    )
    budget_res = render({"smoothingType": "exponential", "smoothingWeight": 0.4})
    s0 = budget_res.structuredContent["series"][0]
    assert len(s0["historySeries"]) == len(s0["rawHistorySeries"])
    assert len(s0["historySeries"]) < 400
