from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wandb_mcp_server.runtime_contract import (
    load_runtime_contract,
    resolve_runtime_selection,
    runtime_contract_sha256,
    tools_for_profile,
    validate_runtime_contract,
)


EXPECTED_COUNTS = {
    "models-only": {"read-write": 17, "read-only": 15},
    "models-weave": {"read-write": 22, "read-only": 20},
    "models-weave-agents": {"read-write": 30, "read-only": 28},
    "models-weave-agents-aria": {"read-write": 33, "read-only": 30},
    "models-weave-graphql-compat": {"read-write": 23, "read-only": 21},
}
LEGACY_VARIABLES = (
    "WANDB_MCP_ENABLE_WEAVE_TOOLS",
    "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
    "WANDB_MCP_ENABLE_ARIA_TOOLS",
    "WANDB_MCP_ENABLE_RAW_GRAPHQL",
    "WANDB_MCP_READ_ONLY",
)


def test_packaged_contract_has_exact_profiles_and_complete_tool_metadata():
    contract = load_runtime_contract()
    validate_runtime_contract(contract)

    assert set(contract["tool_profiles"]) == set(EXPECTED_COUNTS)
    names: list[str] = []
    for group in contract["tool_groups"].values():
        for tool in group["tools"]:
            assert set(tool) == {"name", "access", "risk", "prerequisite"}
            names.append(tool["name"])
    assert len(names) == len(set(names)) == 34

    for profile, access_counts in EXPECTED_COUNTS.items():
        for access_mode, expected in access_counts.items():
            assert len(tools_for_profile(contract, profile, access_mode)) == expected


def test_contract_hash_is_canonical_and_tamper_changes_identity():
    contract = load_runtime_contract()
    original = runtime_contract_sha256(contract)
    reordered = json.loads(json.dumps(contract, sort_keys=True))
    assert runtime_contract_sha256(reordered) == original

    tampered = copy.deepcopy(contract)
    tampered["tool_profiles"]["models-only"]["expected_tools"]["read-write"] = 18
    with pytest.raises(ValueError, match="count mismatch"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["workload_profiles"]["dedicated"]["limits"]["MCP_TOOL_TIMEOUT_SECONDS"] = -1
    with pytest.raises(ValueError, match="workload limit"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["capacity_classes"]["small"]["process_capacity"] = 2
    with pytest.raises(ValueError, match="actor capacity"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["tool_groups"]["models"]["tools"][0]["risk"] = "write"
    with pytest.raises(ValueError, match="access and risk metadata disagree"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["tool_groups"]["models"]["tools"][2]["access"] = "read"
    with pytest.raises(ValueError, match="access and risk metadata disagree"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["workload_profiles"]["dedicated"]["limits"]["MCP_MAX_HISTORY_SAMPLES"] = 1_000_000
    with pytest.raises(ValueError, match="workload limit"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["capacity_classes"]["small"]["process_capacity"] = 100_000
    with pytest.raises(ValueError, match="capacity values"):
        validate_runtime_contract(tampered)


@pytest.mark.parametrize("malformed", [None, [], "contract", 1])
def test_contract_rejects_non_object_root_with_bounded_error(malformed: object):
    with pytest.raises(ValueError, match="root must be an object"):
        validate_runtime_contract(malformed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: contract.update({"tool_groups": {"models": None}}),
        lambda contract: contract["tool_groups"]["models"].update({"tools": [None]}),
        lambda contract: contract["tool_groups"]["models"]["tools"][0].update({"access": []}),
        lambda contract: contract["tool_groups"]["models"]["tools"][0].update({"prerequisite": []}),
        lambda contract: contract["tool_profiles"].update({"models-only": None}),
        lambda contract: contract["tool_profiles"]["models-only"].update({"groups": [{}]}),
        lambda contract: contract["tool_profiles"]["models-only"].update({"managed_workloads": [{}]}),
        lambda contract: contract.update({"legacy_tool_environment_variables": [{}]}),
        lambda contract: contract["workload_profiles"].update({"shared": None}),
        lambda contract: contract["capacity_classes"].update({"small": None}),
    ],
)
def test_contract_rejects_malformed_nested_policy_with_value_error(mutate):
    contract = copy.deepcopy(load_runtime_contract())
    mutate(contract)
    with pytest.raises(ValueError):
        validate_runtime_contract(contract)


def test_default_selection_is_public_models_weave(monkeypatch: pytest.MonkeyPatch):
    for name in LEGACY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name in (
        "WANDB_MCP_TOOL_PROFILE",
        "WANDB_MCP_ACCESS_MODE",
        "MCP_WORKLOAD_PROFILE",
        "MCP_CAPACITY_CLASS",
    ):
        monkeypatch.delenv(name, raising=False)

    selection = resolve_runtime_selection()

    assert selection.tool_profile == "models-weave"
    assert selection.access_mode == "read-write"
    assert selection.workload_profile == "local"
    assert selection.capacity_class == "small"
    assert len(selection.tools) == 22
    assert selection.runtime_contract_sha256.startswith("sha256:")


def test_trace_profile_requires_backend_but_endpoint_never_enables_a_group(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave")
    monkeypatch.setenv("WF_TRACE_SERVER_URL", "")
    monkeypatch.delenv("WEAVE_TRACE_SERVER_URL", raising=False)
    with pytest.raises(ValueError, match="non-empty WF_TRACE_SERVER_URL"):
        resolve_runtime_selection()

    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-only")
    selection = resolve_runtime_selection()
    assert selection.groups == frozenset({"models"})


@pytest.mark.parametrize("workload", ["shared", "dedicated"])
def test_managed_trace_profile_requires_explicit_backend(monkeypatch: pytest.MonkeyPatch, workload: str):
    monkeypatch.setenv("WANDB_MCP_TOOL_PROFILE", "models-weave")
    monkeypatch.setenv("MCP_WORKLOAD_PROFILE", workload)
    monkeypatch.delenv("WF_TRACE_SERVER_URL", raising=False)
    monkeypatch.delenv("WEAVE_TRACE_SERVER_URL", raising=False)

    with pytest.raises(ValueError, match="explicit WF_TRACE_SERVER_URL"):
        resolve_runtime_selection()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MCP_WORKLOAD_PROFILE", "SHARED"),
        ("MCP_CAPACITY_CLASS", "SMALL"),
    ],
)
def test_managed_selectors_require_canonical_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolve_runtime_selection()


@pytest.mark.parametrize("name", LEGACY_VARIABLES)
def test_each_legacy_variable_is_rejected_with_bounded_migration(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError) as raised:
        resolve_runtime_selection()
    message = str(raised.value)
    assert "WANDB_MCP_TOOL_PROFILE" in message
    assert "WANDB_MCP_ACCESS_MODE" in message
    assert name in message
    assert len(message) < 400


def test_managed_workload_rejects_release_critical_numeric_override():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "MCP_WORKLOAD_PROFILE": "dedicated",
            "MCP_CAPACITY_CLASS": "small",
            "MCP_MAX_HISTORY_SAMPLES": "1",
        }
    )
    process = subprocess.run(
        [sys.executable, "-c", "import wandb_mcp_server.config"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert "cannot override managed MCP_WORKLOAD_PROFILE=dedicated" in process.stderr


@pytest.mark.parametrize("name", ["SESSION_TTL_SECONDS", "MAX_SESSIONS_PER_KEY"])
def test_managed_workload_owns_session_limits(name: str):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "MCP_WORKLOAD_PROFILE": "shared",
            "MCP_CAPACITY_CLASS": "large",
            name: "1",
        }
    )
    process = subprocess.run(
        [sys.executable, "-c", "import wandb_mcp_server.config"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert f"{name} cannot override managed MCP_WORKLOAD_PROFILE=shared" in process.stderr


def test_local_workload_retains_bounded_advanced_override():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "MCP_WORKLOAD_PROFILE": "local",
            "MCP_CAPACITY_CLASS": "small",
            "MCP_MAX_HISTORY_SAMPLES": "777",
        }
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from wandb_mcp_server.config import MCP_MAX_HISTORY_SAMPLES; print(MCP_MAX_HISTORY_SAMPLES)",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "777"
