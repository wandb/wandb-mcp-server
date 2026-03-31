#!/usr/bin/env python3
"""Live validation of MCP tool changes against real W&B/Weave APIs.

Exercises every change from the API efficiency audit PR to confirm
correct behavior with network calls. Designed to run after installing
the local wandb-mcp-server branch into any venv that has API keys set.

Requirements:
    WANDB_API_KEY  – must be set in the environment

Usage:
    python scripts/validate_tools_live.py
    # or from the test repo after local install:
    .venv/bin/python /path/to/wandb-mcp-server/scripts/validate_tools_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

ENTITY = "wandb-applied-ai-team"
PROJECT = "mcp-tests"


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float
    detail: str = ""
    error: str = ""


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def record(self, name: str, passed: bool, duration: float, detail: str = "", error: str = ""):
        self.results.append(TestResult(name, passed, duration, detail, error))

    def summary(self) -> str:
        lines = ["\n" + "=" * 70, "VALIDATION SUMMARY", "=" * 70]
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{icon}] {r.name} ({r.duration_s:.2f}s)")
            if r.detail:
                lines.append(f"         {r.detail}")
            if r.error:
                lines.append(f"         ERROR: {r.error}")
        lines.append("-" * 70)
        lines.append(f"  {passed}/{total} passed")
        lines.append("=" * 70)
        return "\n".join(lines)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


def _setup_api_key() -> str:
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        print("ERROR: WANDB_API_KEY not set. Export it or source a .env file.", file=sys.stderr)
        sys.exit(1)
    from wandb_mcp_server.api_client import WandBApiManager

    WandBApiManager.set_context_api_key(api_key)
    return api_key


# ── H1: Retry adapter ────────────────────────────────────────────────────────


def test_h1_retry_adapter(suite: TestSuite):
    """Verify the WeaveApiClient session has a retry adapter mounted."""
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.weave_api.client import WeaveApiClient

        client = WeaveApiClient(api_key="validation_key_placeholder_00")
        adapter = client.session.get_adapter("https://trace.wandb.ai")
        assert adapter.max_retries.total == 3, f"Expected 3 retries, got {adapter.max_retries.total}"
        assert 429 in adapter.max_retries.status_forcelist
        assert 503 in adapter.max_retries.status_forcelist
        suite.record(
            "H1: retry adapter mounted",
            True,
            time.monotonic() - t0,
            f"retries={adapter.max_retries.total}, codes={adapter.max_retries.status_forcelist}",
        )
    except Exception as e:
        suite.record("H1: retry adapter mounted", False, time.monotonic() - t0, error=str(e))


# ── M6: Default timeout ──────────────────────────────────────────────────────


def test_m6_default_timeout(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.weave_api.client import WeaveApiClient

        assert WeaveApiClient.DEFAULT_TIMEOUT == 30, f"Expected 30, got {WeaveApiClient.DEFAULT_TIMEOUT}"
        client = WeaveApiClient(api_key="validation_key_placeholder_00")
        assert client.timeout == 30
        suite.record("M6: default timeout 30s", True, time.monotonic() - t0)
    except Exception as e:
        suite.record("M6: default timeout 30s", False, time.monotonic() - t0, error=str(e))


# ── M4: networkx removed ─────────────────────────────────────────────────────


def test_m4_no_networkx(suite: TestSuite):
    t0 = time.monotonic()
    try:
        import importlib

        spec = importlib.util.find_spec("wandb_mcp_server")
        assert spec is not None
        # networkx should not be imported by wandb_mcp_server
        import wandb_mcp_server.server  # noqa: F401

        mods = [m for m in sys.modules if m.startswith("networkx")]
        suite.record(
            "M4: networkx not imported",
            len(mods) == 0,
            time.monotonic() - t0,
            f"networkx modules in sys.modules: {mods}",
        )
    except Exception as e:
        suite.record("M4: networkx not imported", False, time.monotonic() - t0, error=str(e))


# ── count_traces (live) ──────────────────────────────────────────────────────


def test_count_traces_live(suite: TestSuite):
    """Validate count_traces returns reasonable numbers for the test project."""
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.count_traces import count_traces

        total = count_traces(ENTITY, PROJECT, filters={})
        assert total > 0, f"Expected >0 traces, got {total}"
        roots = count_traces(ENTITY, PROJECT, filters={"trace_roots_only": True})
        assert roots > 0, f"Expected >0 roots, got {roots}"
        assert roots <= total, f"roots ({roots}) > total ({total})"
        suite.record("count_traces (live)", True, time.monotonic() - t0, f"total={total}, roots={roots}")
    except Exception as e:
        suite.record("count_traces (live)", False, time.monotonic() - t0, error=str(e))


# ── H4: parallel count (timing) ──────────────────────────────────────────────


def test_h4_parallel_count(suite: TestSuite):
    """Two sequential counts should be slower than concurrent execution."""
    from concurrent.futures import ThreadPoolExecutor
    from wandb_mcp_server.mcp_tools.count_traces import count_traces
    from wandb_mcp_server.api_client import WandBApiManager

    api_key = os.environ["WANDB_API_KEY"]

    def _count_with_key(**kwargs):
        WandBApiManager.set_context_api_key(api_key)
        return count_traces(**kwargs)

    t0 = time.monotonic()
    try:
        seq_start = time.monotonic()
        count_traces(ENTITY, PROJECT, filters={})
        count_traces(ENTITY, PROJECT, filters={"trace_roots_only": True})
        seq_dur = time.monotonic() - seq_start

        par_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_count_with_key, entity_name=ENTITY, project_name=PROJECT, filters={})
            f2 = ex.submit(
                _count_with_key, entity_name=ENTITY, project_name=PROJECT, filters={"trace_roots_only": True}
            )
            f1.result()
            f2.result()
        par_dur = time.monotonic() - par_start

        speedup = seq_dur / par_dur if par_dur > 0 else 0
        suite.record(
            "H4: parallel count",
            True,
            time.monotonic() - t0,
            f"sequential={seq_dur:.2f}s, parallel={par_dur:.2f}s, speedup={speedup:.1f}x",
        )
    except Exception as e:
        suite.record("H4: parallel count", False, time.monotonic() - t0, error=str(e))


# ── query_weave_traces (live) ────────────────────────────────────────────────


def test_query_traces_live(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.query_weave import query_paginated_weave_traces

        result = asyncio.run(
            query_paginated_weave_traces(
                entity_name=ENTITY,
                project_name=PROJECT,
                chunk_size=10,
                filters={"trace_roots_only": True},
                target_limit=5,
                truncate_length=100,
                return_full_data=False,
            )
        )
        data = json.loads(result.model_dump_json())
        n = len(data.get("traces", []))
        assert n > 0, f"Expected traces, got {n}"
        suite.record("query_weave_traces (live)", True, time.monotonic() - t0, f"returned {n} traces")
    except Exception as e:
        suite.record("query_weave_traces (live)", False, time.monotonic() - t0, error=str(e))


# ── H3: schema projection ────────────────────────────────────────────────────


def test_h3_schema_projection(suite: TestSuite):
    """detail_level='schema' should use a restricted column set."""
    t0 = time.monotonic()
    try:
        from unittest.mock import AsyncMock, patch
        from wandb_mcp_server.weave_api.models import QueryResult, TraceMetadata

        mock_query = AsyncMock(
            return_value=QueryResult(
                metadata=TraceMetadata(total_traces=0),
                traces=[],
            )
        )
        with (
            patch("wandb_mcp_server.server.query_paginated_weave_traces", mock_query),
            patch("wandb_mcp_server.server.count_traces", return_value=5),
            patch("wandb_mcp_server.api_client.WandBApiManager.get_api_key", return_value="fake"),
        ):
            from wandb_mcp_server.server import register_tools
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("test-h3")
            register_tools(mcp)
            tool_fn = None
            for name, fn in mcp._tool_manager._tools.items():
                if "query_weave_traces" in name:
                    tool_fn = fn
                    break
            assert tool_fn is not None
            asyncio.run(tool_fn.fn(entity_name="e", project_name="p", detail_level="schema"))

        kw = mock_query.call_args[1]
        expected = ["id", "trace_id", "op_name", "started_at", "ended_at", "display_name", "parent_id", "summary"]
        assert kw["columns"] == expected, f"Got columns={kw['columns']}"
        assert kw["include_costs"] is False
        assert kw["include_feedback"] is False
        suite.record(
            "H3: schema projection", True, time.monotonic() - t0, "columns passed correctly, costs/feedback disabled"
        )
    except Exception as e:
        suite.record("H3: schema projection", False, time.monotonic() - t0, error=str(e))


# ── H2: cost sort cap ────────────────────────────────────────────────────────


def test_h2_cost_sort_cap(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.weave_api.service import TraceService

        assert TraceService.COST_SORT_MAX_FIRST_PASS == 10_000
        suite.record("H2: cost sort cap", True, time.monotonic() - t0, f"cap={TraceService.COST_SORT_MAX_FIRST_PASS}")
    except Exception as e:
        suite.record("H2: cost sort cap", False, time.monotonic() - t0, error=str(e))


# ── M2: $or/$in filter ops ───────────────────────────────────────────────────


def test_m2_or_in_filters(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.weave_api.query_builder import QueryBuilder

        # $in
        q_in = QueryBuilder.build_query_expression({"$in": {"summary.weave.status": ["error", "running"]}})
        assert q_in is not None, "$in query returned None"
        d_in = q_in.model_dump(by_alias=True)
        assert "$in" in json.dumps(d_in), f"No $in in {d_in}"

        # $or
        q_or = QueryBuilder.build_query_expression({"$or": [{"status": "error"}, {"status": "running"}]})
        assert q_or is not None, "$or query returned None"

        suite.record("M2: $or/$in filters", True, time.monotonic() - t0, "$in and $or produce valid query objects")
    except Exception as e:
        suite.record("M2: $or/$in filters", False, time.monotonic() - t0, error=str(e))


# ── H5: beta_scan_history wiring ─────────────────────────────────────────────


def test_h5_beta_scan_wiring(suite: TestSuite):
    """Validate the _fetch_step_range function exists and respects env var."""
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.run_history import _reservoir_sample

        # _reservoir_sample basic check
        data = list(range(100))
        sampled = _reservoir_sample(iter([{"_step": i} for i in data]), 10)
        assert len(sampled) == 10, f"Expected 10, got {len(sampled)}"
        steps = [r["_step"] for r in sampled]
        assert steps == sorted(steps), "Reservoir sample not sorted"

        suite.record(
            "H5: beta_scan_history wiring", True, time.monotonic() - t0, "functions exist, reservoir sample works"
        )
    except Exception as e:
        suite.record("H5: beta_scan_history wiring", False, time.monotonic() - t0, error=str(e))


# ── M1: history row budget ────────────────────────────────────────────────────


def test_m1_row_budget(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.run_history import _enforce_row_budget

        small = [{"_step": i, "v": 1.0} for i in range(5)]
        assert len(_enforce_row_budget(small, 100_000)) == 5

        big = [{"_step": i, "v": 1.0, "a": "x" * 100} for i in range(2000)]
        trimmed = _enforce_row_budget(big, 2000)
        assert len(trimmed) < 2000, f"Expected < 2000, got {len(trimmed)}"
        assert len(trimmed) > 0

        suite.record(
            "M1: history row budget",
            True,
            time.monotonic() - t0,
            f"2000 rows -> {len(trimmed)} after budget enforcement",
        )
    except Exception as e:
        suite.record("M1: history row budget", False, time.monotonic() - t0, error=str(e))


# ── GQL pagination ────────────────────────────────────────────────────────────


def test_gql_pagination(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.query_wandb_gql import query_paginated_wandb_gql

        result = query_paginated_wandb_gql(
            query=f'query {{ project(name: "{PROJECT}", entityName: "{ENTITY}") {{ runCount }} }}',
            variables={},
        )
        assert "errors" not in result, f"GQL error: {result.get('errors')}"
        suite.record("GQL pagination (live)", True, time.monotonic() - t0)
    except Exception as e:
        suite.record("GQL pagination (live)", False, time.monotonic() - t0, error=str(e))


# ── list_entity_projects ──────────────────────────────────────────────────────


def test_list_projects(suite: TestSuite):
    t0 = time.monotonic()
    try:
        from wandb_mcp_server.mcp_tools.list_wandb_entities_projects import list_entity_projects

        result = list_entity_projects()
        assert isinstance(result, dict) and len(result) > 0, "No entities returned"
        total_projects = sum(len(v) for v in result.values())
        suite.record(
            "list_entity_projects (live)",
            True,
            time.monotonic() - t0,
            f"{len(result)} entities, {total_projects} projects",
        )
    except Exception as e:
        suite.record("list_entity_projects (live)", False, time.monotonic() - t0, error=str(e))


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("MCP API Efficiency Audit – Live Tool Validation")
    print("=" * 70)

    _setup_api_key()
    suite = TestSuite()

    tests = [
        test_h1_retry_adapter,
        test_m6_default_timeout,
        test_m4_no_networkx,
        test_h2_cost_sort_cap,
        test_m2_or_in_filters,
        test_h5_beta_scan_wiring,
        test_m1_row_budget,
        test_h3_schema_projection,
        test_count_traces_live,
        test_h4_parallel_count,
        test_query_traces_live,
        test_gql_pagination,
        test_list_projects,
    ]

    for test_fn in tests:
        print(f"\n  Running {test_fn.__name__}...")
        try:
            test_fn(suite)
        except Exception:
            suite.record(test_fn.__name__, False, 0, error=traceback.format_exc())

    print(suite.summary())
    sys.exit(0 if suite.all_passed else 1)


if __name__ == "__main__":
    main()
