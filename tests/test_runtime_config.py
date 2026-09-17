"""Runtime safety configuration must fail closed during process startup."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

_LOAD_CONFIG = "import runpy; runpy.run_path('src/wandb_mcp_server/config.py')"
_RESOLVE_RUNTIME = (
    "from wandb_mcp_server.runtime_contract import resolve_runtime_selection; resolve_runtime_selection()"
)


@pytest.mark.parametrize(
    "variable",
    [
        "MAX_RESPONSE_TOKENS",
        "MAX_ACCUMULATED_BYTES",
        "MCP_TOOL_TIMEOUT_SECONDS",
        "MCP_WANDB_REQUEST_TIMEOUT_SECONDS",
        "MCP_ADMISSION_ACTOR_CAPACITY",
        "MCP_ADMISSION_PROCESS_CAPACITY",
        "MCP_MAX_QUERY_LIMIT",
        "MCP_MAX_FULL_TRACE_LIMIT",
        "MCP_MAX_HISTORY_SAMPLES",
        "MCP_MAX_HISTORY_KEYS",
        "MCP_MAX_HISTORY_RANGE_STEPS",
        "MCP_MAX_WANDB_QUERY_ITEMS",
        "MCP_MAX_FULL_DETAIL_ITEMS",
        "MCP_MAX_PROJECT_FIELDS",
        "MCP_MAX_PROBE_RUNS",
        "MCP_MAX_EVALUATION_ROWS",
        "MCP_MAX_SCHEMA_SAMPLE_ROWS",
        "MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE",
        "MCP_MAX_GQL_ITEMS",
        "MCP_MAX_GQL_ITEMS_PER_PAGE",
        "MCP_SYNC_TOOL_WORKERS",
        "MCP_COUNT_TOOL_WORKERS",
        "MCP_ANALYTICS_QUEUE_CAPACITY",
        "MCP_ANALYTICS_TEST_BUFFER_CAPACITY",
        "SESSION_TTL_SECONDS",
        "MAX_SESSIONS_PER_KEY",
    ],
)
def test_positive_runtime_bounds_reject_zero_with_variable_name(variable: str) -> None:
    environment = os.environ.copy()
    environment[variable] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert variable in result.stderr


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "invalid"])
def test_sampling_rate_rejects_invalid_values(value: str) -> None:
    environment = os.environ.copy()
    environment["MCP_REQUEST_SUCCESS_SAMPLE_RATE"] = value
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MCP_REQUEST_SUCCESS_SAMPLE_RATE" in result.stderr


def test_admission_wait_allows_zero_but_rejects_negative() -> None:
    environment = os.environ.copy()
    environment["MCP_ADMISSION_WAIT_MS"] = "-1"
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MCP_ADMISSION_WAIT_MS" in result.stderr


@pytest.mark.parametrize(
    "worker_variable",
    ["MCP_SYNC_TOOL_WORKERS", "MCP_COUNT_TOOL_WORKERS"],
)
def test_worker_pool_cannot_exceed_process_admission_capacity(worker_variable: str) -> None:
    environment = os.environ.copy()
    environment["MCP_ADMISSION_ACTOR_CAPACITY"] = "4"
    environment["MCP_ADMISSION_PROCESS_CAPACITY"] = "4"
    environment["MCP_SYNC_TOOL_WORKERS"] = "4"
    environment["MCP_COUNT_TOOL_WORKERS"] = "4"
    environment[worker_variable] = "5"
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert worker_variable in result.stderr


@pytest.mark.parametrize(
    "variable",
    [
        "MCP_HOSTED_MODE",
        "MCP_ADMISSION_CONTROL_ENABLED",
        "MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS",
    ],
)
def test_runtime_booleans_reject_typos(variable: str) -> None:
    environment = os.environ.copy()
    environment[variable] = "treu"
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert variable in result.stderr


@pytest.mark.parametrize(
    "value",
    [
        "http://wb-agent.example",
        "wb-agent.example",
        "https://user:secret@wb-agent.example",
        "https://wb-agent.example:not-a-port",
        "https://exa mple.com",
        "https://-invalid.example",
        "https://wb-agent.example/api",
        "https://wb-agent.example?",
        "https://wb-agent.example?token=secret",
        "https://wb-agent.example#",
        "https://wb-agent.example#fragment",
    ],
)
def test_enabled_aria_rejects_unsafe_base_urls(value: str) -> None:
    environment = os.environ.copy()
    environment["WANDB_MCP_TOOL_PROFILE"] = "models-weave-agents-aria"
    environment["MCP_WORKLOAD_PROFILE"] = "local"
    environment["WB_AGENT_BASE_URL"] = value
    result = subprocess.run(
        [sys.executable, "-c", _RESOLVE_RUNTIME],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "WB_AGENT_BASE_URL" in result.stderr


def test_disabled_aria_does_not_require_network_endpoint_configuration() -> None:
    environment = os.environ.copy()
    environment["WANDB_MCP_TOOL_PROFILE"] = "models-weave"
    environment["MCP_WORKLOAD_PROFILE"] = "local"
    environment["WB_AGENT_BASE_URL"] = "http://legacy.invalid"
    result = subprocess.run(
        [sys.executable, "-c", _RESOLVE_RUNTIME],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_enabled_aria_requires_response_budget_large_enough_for_safe_error() -> None:
    environment = os.environ.copy()
    environment["WANDB_MCP_TOOL_PROFILE"] = "models-weave-agents-aria"
    environment["MCP_WORKLOAD_PROFILE"] = "local"
    environment["WB_AGENT_BASE_URL"] = "https://wb-agent.wandb.ai"
    environment["MAX_RESPONSE_TOKENS"] = "63"
    result = subprocess.run(
        [sys.executable, "-c", _RESOLVE_RUNTIME],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "MAX_RESPONSE_TOKENS" in result.stderr


@pytest.mark.parametrize(
    "variable",
    ["MCP_ADMISSION_ACTOR_CAPACITY", "MCP_ADMISSION_PROCESS_CAPACITY"],
)
def test_enabled_admission_requires_capacity_for_heavy_tools(variable: str) -> None:
    environment = os.environ.copy()
    environment["MCP_ADMISSION_CONTROL_ENABLED"] = "true"
    environment["MCP_ADMISSION_ACTOR_CAPACITY"] = "3" if variable == "MCP_ADMISSION_ACTOR_CAPACITY" else "4"
    environment["MCP_ADMISSION_PROCESS_CAPACITY"] = "3" if variable == "MCP_ADMISSION_PROCESS_CAPACITY" else "4"
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert variable in result.stderr


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("MAX_RESPONSE_TOKENS", "100001"),
        ("MAX_ACCUMULATED_BYTES", str(1024 * 1024 * 1024 + 1)),
        ("MCP_TOOL_TIMEOUT_SECONDS", "301"),
        ("MCP_WANDB_REQUEST_TIMEOUT_SECONDS", "121"),
        ("MCP_ADMISSION_PROCESS_CAPACITY", "65"),
        ("MCP_ADMISSION_WAIT_MS", "30001"),
        ("MCP_MAX_WANDB_QUERY_ITEMS", "10001"),
        ("MCP_MAX_GQL_ITEMS", "1001"),
        ("MCP_MAX_GQL_ITEMS_PER_PAGE", "201"),
        ("MCP_SYNC_TOOL_WORKERS", "17"),
        ("MCP_COUNT_TOOL_WORKERS", "17"),
        ("MCP_ANALYTICS_QUEUE_CAPACITY", "257"),
        ("MCP_ANALYTICS_TEST_BUFFER_CAPACITY", "1001"),
        ("SESSION_TTL_SECONDS", "86401"),
        ("MAX_SESSIONS_PER_KEY", "1001"),
    ],
)
def test_runtime_bounds_reject_oversized_values(variable: str, value: str) -> None:
    environment = os.environ.copy()
    environment[variable] = value
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_CONFIG],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert variable in result.stderr
