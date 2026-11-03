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
    managed_workloads_for,
    profile_combinations,
)


EXPECTED_COUNTS = {
    "models-only": {"read-write": 17, "read-only": 15},
    "models-weave": {"read-write": 22, "read-only": 20},
    "models-weave-graphql-compat": {"read-write": 23, "read-only": 21},
    "models-weave-agents": {"read-write": 30, "read-only": 28},
    "models-weave-agents-agent-lens": {"read-write": 39, "read-only": 37},
    "models-weave-agents-aria": {"read-write": 34, "read-only": 31},
    "models-weave-agents-raw-graphql": {"read-write": 31, "read-only": 29},
    "models-weave-agents-agent-lens-aria": {"read-write": 43, "read-only": 40},
    "models-weave-agents-agent-lens-raw-graphql": {"read-write": 40, "read-only": 38},
    "models-weave-agents-aria-raw-graphql": {"read-write": 35, "read-only": 32},
    "models-weave-agents-agent-lens-aria-raw-graphql": {"read-write": 44, "read-only": 41},
}
EXPECTED_MANAGED = {
    "models-only": ("dedicated", "shared"),
    "models-weave": ("dedicated", "shared"),
    "models-weave-agents": ("shared",),
    "models-weave-agents-agent-lens": ("shared",),
    "models-weave-agents-aria": ("shared",),
    "models-weave-agents-agent-lens-aria": ("shared",),
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

    combinations = {resolution.name: resolution for resolution in profile_combinations(contract)}
    assert set(combinations) == set(EXPECTED_COUNTS)
    assert set(contract["tool_profiles"]) == {"models-only", "models-weave", "models-weave-agents"}
    assert set(contract["tool_profile_aliases"]) == {
        "models-weave-agents-aria",
        "models-weave-graphql-compat",
        "models-weave-agents-agent-lens",
    }
    for name, resolution in combinations.items():
        assert managed_workloads_for(contract, resolution.groups) == EXPECTED_MANAGED.get(name, ())
    names: list[str] = []
    for group in contract["tool_groups"].values():
        for tool in group["tools"]:
            assert set(tool) == {"name", "access", "risk", "prerequisite"}
            names.append(tool["name"])
    assert len(names) == len(set(names)) == 44

    for profile, access_counts in EXPECTED_COUNTS.items():
        resolution = combinations[profile]
        for access_mode, expected in access_counts.items():
            assert (
                len(tools_for_profile(contract, resolution.base_profile, access_mode, resolution.extensions))
                == expected
            )
    for alias in contract["tool_profile_aliases"]:
        assert tools_for_profile(contract, alias, "read-write") == tools_for_profile(
            contract, combinations[alias].base_profile, "read-write", combinations[alias].extensions
        )


def test_contract_hash_is_canonical_and_tamper_changes_identity():
    contract = load_runtime_contract()
    original = runtime_contract_sha256(contract)
    reordered = json.loads(json.dumps(contract, sort_keys=True))
    assert runtime_contract_sha256(reordered) == original

    tampered = copy.deepcopy(contract)
    tampered["tool_groups"]["aria"]["requires"] = ["models"]
    tampered["tool_profile_aliases"]["models-weave-agents-aria"]["base"] = "models-only"
    with pytest.raises(ValueError, match="ambiguous"):
        validate_runtime_contract(tampered)

    tampered = copy.deepcopy(contract)
    tampered["tool_groups"]["agents"]["managed_workloads"] = ["everywhere"]
    with pytest.raises(ValueError, match="Managed workload allowlist"):
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
        lambda contract: contract["tool_groups"]["models"].update({"managed_workloads": [{}]}),
        lambda contract: contract["tool_profile_aliases"].update(
            {"models-only": {"base": "models-weave", "extensions": ["aria"]}}
        ),
        lambda contract: contract["tool_groups"]["agent-lens"].update({"kind": "base"}),
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


def _shared_environment(**overrides: str) -> dict[str, str]:
    environment = {
        "MCP_WORKLOAD_PROFILE": "shared",
        "WF_TRACE_SERVER_URL": "https://trace.wandb.ai",
        "WB_AGENT_BASE_URL": "https://wb-agent.wandb.ai",
        "AGENT_LENS_BASE_URL": "https://agent-lens.example.com",
    }
    environment.update(overrides)
    return environment


def test_extensions_compose_onto_a_base_profile_regardless_of_order():
    from wandb_mcp_server.runtime_contract import resolve_runtime_selection

    contract = load_runtime_contract()
    first = resolve_runtime_selection(
        _shared_environment(WANDB_MCP_TOOL_PROFILE="models-weave-agents", WANDB_MCP_TOOL_EXTENSIONS="aria, agent-lens")
    )
    second = resolve_runtime_selection(
        _shared_environment(WANDB_MCP_TOOL_PROFILE="models-weave-agents", WANDB_MCP_TOOL_EXTENSIONS="agent-lens,aria")
    )
    aliased = resolve_runtime_selection(
        _shared_environment(WANDB_MCP_TOOL_PROFILE="models-weave-agents-agent-lens", WANDB_MCP_TOOL_EXTENSIONS="aria")
    )

    assert first == second == aliased
    assert first.tool_profile == "models-weave-agents-agent-lens-aria"
    assert first.base_profile == "models-weave-agents"
    assert first.extensions == frozenset({"agent-lens", "aria"})
    assert first.groups == frozenset({"models", "weave", "agents", "agent-lens", "aria"})
    assert len(first.tools) == 43
    assert first.tools == tools_for_profile(contract, "models-weave-agents", "read-write", ("agent-lens", "aria"))

    plain = resolve_runtime_selection(_shared_environment(WANDB_MCP_TOOL_PROFILE="models-weave-agents"))
    assert plain.extensions == frozenset()
    assert plain.tool_profile == "models-weave-agents"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"WANDB_MCP_TOOL_PROFILE": "models-weave", "WANDB_MCP_TOOL_EXTENSIONS": "aria"}, "requires the agents"),
        ({"WANDB_MCP_TOOL_PROFILE": "models-only", "WANDB_MCP_TOOL_EXTENSIONS": "raw-graphql"}, "requires the weave"),
        (
            {"WANDB_MCP_TOOL_PROFILE": "models-weave-agents", "WANDB_MCP_TOOL_EXTENSIONS": "weave"},
            "Unknown WANDB_MCP_TOOL_EXTENSIONS",
        ),
        (
            {"WANDB_MCP_TOOL_PROFILE": "models-weave-agents", "WANDB_MCP_TOOL_EXTENSIONS": "aria,aria"},
            "repeats an extension",
        ),
        (
            {"WANDB_MCP_TOOL_PROFILE": "models-weave-agents", "WANDB_MCP_TOOL_EXTENSIONS": "raw-graphql"},
            "group\\(s\\) not allowed: raw-graphql",
        ),
        (
            {
                "WANDB_MCP_TOOL_PROFILE": "models-weave",
                "WANDB_MCP_TOOL_EXTENSIONS": "",
                "MCP_WORKLOAD_PROFILE": "dedicated",
            },
            None,
        ),
    ],
)
def test_extension_selection_fails_closed(environment: dict[str, str], message: str | None):
    from wandb_mcp_server.runtime_contract import resolve_runtime_selection

    if message is None:
        assert resolve_runtime_selection(_shared_environment(**environment)).tool_profile == "models-weave"
        return
    with pytest.raises(ValueError, match=message):
        resolve_runtime_selection(_shared_environment(**environment))


def test_dedicated_rejects_every_extension_group():
    from wandb_mcp_server.runtime_contract import resolve_runtime_selection

    for extension in ("agent-lens", "aria"):
        with pytest.raises(ValueError, match=f"not allowed: .*{extension}"):
            resolve_runtime_selection(
                _shared_environment(
                    MCP_WORKLOAD_PROFILE="dedicated",
                    WANDB_MCP_TOOL_PROFILE="models-weave-agents",
                    WANDB_MCP_TOOL_EXTENSIONS=extension,
                )
            )
