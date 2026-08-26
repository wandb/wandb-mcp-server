"""Canonical MCP runtime-profile contract and startup resolution.

The packaged JSON file is the sole source of truth for tool manifests and
managed workload defaults.  This module deliberately resolves environment
selectors at server construction time, after the CLI has loaded dotenv and
before any tools are registered.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from typing import Any, Mapping
from urllib.parse import urlsplit


_CONTRACT_RESOURCE = "runtime-contract.json"
_MIGRATION_MESSAGE = (
    "Legacy MCP tool flags are not supported in v0.4; configure WANDB_MCP_TOOL_PROFILE and WANDB_MCP_ACCESS_MODE"
)

# These are parser safety ceilings, not deployment defaults. The JSON contract
# chooses values within them; both the release controller and the installed
# runtime reject a corrupted or accidentally unbounded policy.
WORKLOAD_LIMIT_BOUNDS: dict[str, tuple[int, int]] = {
    "MAX_RESPONSE_TOKENS": (1, 100_000),
    "MAX_ACCUMULATED_BYTES": (1, 1024 * 1024 * 1024),
    "MCP_TOOL_TIMEOUT_SECONDS": (1, 300),
    "MCP_WANDB_REQUEST_TIMEOUT_SECONDS": (1, 120),
    "MCP_ADMISSION_WAIT_MS": (0, 30_000),
    "MCP_MAX_QUERY_LIMIT": (1, 10_000),
    "MCP_MAX_FULL_TRACE_LIMIT": (1, 1_000),
    "MCP_MAX_HISTORY_SAMPLES": (1, 10_000),
    "MCP_MAX_HISTORY_KEYS": (1, 500),
    "MCP_MAX_HISTORY_RANGE_STEPS": (1, 1_000_000),
    "MCP_MAX_WANDB_QUERY_ITEMS": (1, 10_000),
    "MCP_MAX_FULL_DETAIL_ITEMS": (1, 100),
    "MCP_MAX_PROJECT_FIELDS": (1, 10_000),
    "MCP_MAX_PROBE_RUNS": (1, 100),
    "MCP_MAX_EVALUATION_ROWS": (1, 10_000),
    "MCP_MAX_SCHEMA_SAMPLE_ROWS": (1, 1_000),
    "MCP_MAX_WANDB_QUERY_ITEMS_PER_PAGE": (1, 500),
    "MCP_MAX_GQL_ITEMS": (1, 1_000),
    "MCP_MAX_GQL_ITEMS_PER_PAGE": (1, 200),
    "SESSION_TTL_SECONDS": (1, 86_400),
    "MAX_SESSIONS_PER_KEY": (1, 1_000),
}
CAPACITY_BOUNDS: dict[str, tuple[int, int]] = {
    "actor_capacity": (1, 64),
    "process_capacity": (1, 64),
    "sync_workers": (1, 16),
    "count_workers": (1, 16),
}
HTTP_RATE_MAXIMA = {"per_key_per_minute": 10_000, "global_per_minute": 100_000}


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """One validated deployment envelope resolved from orthogonal selectors."""

    tool_profile: str
    access_mode: str
    workload_profile: str
    capacity_class: str
    tools: frozenset[str]
    groups: frozenset[str]
    runtime_contract_sha256: str

    @property
    def read_only(self) -> bool:
        return self.access_mode == "read-only"


def canonical_json(value: Any) -> bytes:
    """Return the canonical representation used by release evidence."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def load_runtime_contract() -> dict[str, Any]:
    """Load and structurally validate the contract packaged in the wheel."""
    resource = files("wandb_mcp_server").joinpath(_CONTRACT_RESOURCE)
    try:
        contract = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Packaged MCP runtime contract is unreadable") from error
    validate_runtime_contract(contract)
    return contract


def runtime_contract_sha256(contract: Mapping[str, Any] | None = None) -> str:
    """Return the stable, algorithm-qualified identity of the contract."""
    effective = dict(contract) if contract is not None else load_runtime_contract()
    return f"sha256:{hashlib.sha256(canonical_json(effective)).hexdigest()}"


def _tool_index(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tools: dict[str, Mapping[str, Any]] = {}
    for group_name, group in contract["tool_groups"].items():
        for tool in group["tools"]:
            name = tool["name"]
            if name in tools:
                raise ValueError(f"Runtime contract assigns tool more than once: {name}")
            tools[name] = {**tool, "group": group_name}
    return tools


def tools_for_profile(
    contract: Mapping[str, Any],
    tool_profile: str,
    access_mode: str,
) -> frozenset[str]:
    """Resolve one exact tools/list manifest from the canonical contract."""
    try:
        profile = contract["tool_profiles"][tool_profile]
    except KeyError as error:
        raise ValueError(f"Unknown WANDB_MCP_TOOL_PROFILE: {tool_profile}") from error
    if access_mode not in contract["selectors"]["access_mode"]["values"]:
        raise ValueError("WANDB_MCP_ACCESS_MODE must be read-write or read-only")

    tools: set[str] = set()
    for group_name in profile["groups"]:
        tools.update(tool["name"] for tool in contract["tool_groups"][group_name]["tools"])
    if access_mode == "read-only":
        write_tools = {name for name, metadata in _tool_index(contract).items() if metadata["access"] == "write"}
        tools.difference_update(write_tools)
    return frozenset(tools)


def validate_runtime_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if profile metadata is incomplete, ambiguous, or tampered."""
    if not isinstance(contract, Mapping):
        raise ValueError("MCP runtime contract root must be an object")
    if contract.get("schema_version") != 1 or contract.get("package") != "wandb_mcp_server":
        raise ValueError("Unsupported MCP runtime contract")
    selectors = contract.get("selectors")
    required_selectors = {"tool_profile", "access_mode", "workload_profile", "capacity_class"}
    if not isinstance(selectors, dict) or set(selectors) != required_selectors:
        raise ValueError("Runtime contract selectors are incomplete")

    expected_selector_environments = {
        "tool_profile": "WANDB_MCP_TOOL_PROFILE",
        "access_mode": "WANDB_MCP_ACCESS_MODE",
        "workload_profile": "MCP_WORKLOAD_PROFILE",
        "capacity_class": "MCP_CAPACITY_CLASS",
    }
    for selector_name, environment_name in expected_selector_environments.items():
        definition = selectors[selector_name]
        if not isinstance(definition, dict) or definition.get("environment") != environment_name:
            raise ValueError(f"Runtime contract selector is invalid: {selector_name}")
    if selectors["tool_profile"] != {
        "environment": "WANDB_MCP_TOOL_PROFILE",
        "default": "models-weave",
    }:
        raise ValueError("Runtime contract tool-profile selector is invalid")
    if selectors["access_mode"] != {
        "environment": "WANDB_MCP_ACCESS_MODE",
        "default": "read-write",
        "values": ["read-write", "read-only"],
    }:
        raise ValueError("Runtime contract access selector is invalid")

    groups = contract.get("tool_groups")
    profiles = contract.get("tool_profiles")
    if not isinstance(groups, dict) or not groups or not isinstance(profiles, dict) or not profiles:
        raise ValueError("Runtime contract must define tool groups and profiles")
    if set(groups) != {"models", "weave", "agents", "aria", "raw-graphql"}:
        raise ValueError("Runtime contract tool groups are incomplete")
    allowed_access = {"read", "write"}
    allowed_risks = {"read", "write", "credential-forwarding", "non-idempotent-credential-forwarding", "compatibility"}
    allowed_prerequisites = {"none", "wandb", "trace-backend", "aria-https-origin"}
    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not isinstance(group, dict):
            raise ValueError("Every runtime tool group must be non-empty")
        group_tools = group.get("tools")
        if set(group) != {"tools"} or not isinstance(group_tools, list) or not group_tools:
            raise ValueError("Every runtime tool group must be non-empty")
        for tool in group_tools:
            if not isinstance(tool, dict):
                raise ValueError(f"Tool metadata is incomplete in group {group_name}")
            if set(tool) != {"name", "access", "risk", "prerequisite"}:
                raise ValueError(f"Tool metadata is incomplete in group {group_name}")
            if not isinstance(tool["name"], str) or not tool["name"]:
                raise ValueError("Runtime contract tool names must be non-empty")
            if (
                not isinstance(tool["access"], str)
                or not isinstance(tool["risk"], str)
                or tool["access"] not in allowed_access
                or tool["risk"] not in allowed_risks
            ):
                raise ValueError(f"Tool risk metadata is invalid: {tool['name']}")
            write_risks = {"write", "non-idempotent-credential-forwarding"}
            if (tool["access"] == "write") != (tool["risk"] in write_risks):
                raise ValueError(f"Tool access and risk metadata disagree: {tool['name']}")
            if not isinstance(tool["prerequisite"], str) or tool["prerequisite"] not in allowed_prerequisites:
                raise ValueError(f"Tool prerequisite is invalid: {tool['name']}")
    _tool_index(contract)

    access_modes = selectors["access_mode"]["values"]
    expected_profiles = {
        "models-only",
        "models-weave",
        "models-weave-agents",
        "models-weave-agents-aria",
        "models-weave-graphql-compat",
    }
    if set(profiles) != expected_profiles:
        raise ValueError("Runtime contract tool profiles are incomplete")
    expected_profile_groups = {
        "models-only": ["models"],
        "models-weave": ["models", "weave"],
        "models-weave-agents": ["models", "weave", "agents"],
        "models-weave-agents-aria": ["models", "weave", "agents", "aria"],
        "models-weave-graphql-compat": ["models", "weave", "raw-graphql"],
    }
    expected_managed_workloads = {
        "models-only": ["shared", "dedicated"],
        "models-weave": ["shared", "dedicated"],
        "models-weave-agents": ["shared"],
        "models-weave-agents-aria": [],
        "models-weave-graphql-compat": [],
    }
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Tool profile metadata is incomplete: {profile_name}")
        if set(profile) != {"groups", "managed_workloads", "expected_tools"}:
            raise ValueError(f"Tool profile metadata is incomplete: {profile_name}")
        profile_groups = profile.get("groups")
        if (
            not isinstance(profile_groups, list)
            or not profile_groups
            or not all(isinstance(group, str) for group in profile_groups)
            or not set(profile_groups) <= set(groups)
        ):
            raise ValueError(f"Tool profile groups are invalid: {profile_name}")
        if len(profile_groups) != len(set(profile_groups)):
            raise ValueError(f"Tool profile repeats a group: {profile_name}")
        if profile_groups != expected_profile_groups[profile_name]:
            raise ValueError(f"Tool profile groups do not match the reviewed contract: {profile_name}")
        managed = profile.get("managed_workloads")
        if (
            not isinstance(managed, list)
            or not all(isinstance(workload, str) for workload in managed)
            or not set(managed) <= {"shared", "dedicated"}
        ):
            raise ValueError(f"Managed workload allowlist is invalid: {profile_name}")
        if managed != expected_managed_workloads[profile_name]:
            raise ValueError(f"Managed workload allowlist does not match the reviewed contract: {profile_name}")
        expected = profile.get("expected_tools")
        if not isinstance(expected, dict) or set(expected) != set(access_modes):
            raise ValueError(f"Tool profile counts are incomplete: {profile_name}")
        for access_mode in access_modes:
            actual = len(tools_for_profile(contract, profile_name, access_mode))
            if expected[access_mode] != actual:
                raise ValueError(f"Tool profile count mismatch: {profile_name}/{access_mode}")

    legacy = contract.get("legacy_tool_environment_variables")
    if (
        not isinstance(legacy, list)
        or not all(isinstance(value, str) for value in legacy)
        or len(legacy) != len(set(legacy))
    ):
        raise ValueError("Legacy tool environment denylist is invalid")
    if set(legacy) != {
        "WANDB_MCP_ENABLE_WEAVE_TOOLS",
        "WANDB_MCP_ENABLE_WEAVE_AGENT_TOOLS",
        "WANDB_MCP_ENABLE_ARIA_TOOLS",
        "WANDB_MCP_ENABLE_RAW_GRAPHQL",
        "WANDB_MCP_READ_ONLY",
    }:
        raise ValueError("Legacy tool environment denylist is incomplete")

    workloads = contract.get("workload_profiles")
    capacities = contract.get("capacity_classes")
    if not isinstance(workloads, dict) or set(workloads) != {"shared", "dedicated", "local"}:
        raise ValueError("Runtime workload profiles are incomplete")
    if not isinstance(capacities, dict) or set(capacities) != {"small", "medium", "large"}:
        raise ValueError("Runtime capacity classes are incomplete")
    if selectors["workload_profile"] != {
        "environment": "MCP_WORKLOAD_PROFILE",
        "default": "local",
        "values": ["shared", "dedicated", "local"],
    }:
        raise ValueError("Runtime contract workload selector is invalid")
    if selectors["capacity_class"] != {
        "environment": "MCP_CAPACITY_CLASS",
        "default": "small",
        "values": ["small", "medium", "large"],
    }:
        raise ValueError("Runtime contract capacity selector is invalid")
    required_limits = set(WORKLOAD_LIMIT_BOUNDS) | {"MCP_ADMISSION_CONTROL_ENABLED"}
    for workload_name, workload in workloads.items():
        if not isinstance(workload, dict):
            raise ValueError(f"Runtime workload policy is invalid: {workload_name}")
        if set(workload) != {"managed", "limits", "http_rate_policy"}:
            raise ValueError(f"Runtime workload policy is invalid: {workload_name}")
        limits = workload["limits"]
        rate_policy = workload["http_rate_policy"]
        if not isinstance(limits, dict) or set(limits) != required_limits:
            raise ValueError(f"Runtime workload limits are incomplete: {workload_name}")
        if not isinstance(rate_policy, dict) or set(rate_policy) != {
            "enabled",
            "per_key_per_minute",
            "global_per_minute",
        }:
            raise ValueError(f"Runtime HTTP rate policy is invalid: {workload_name}")
        if workload["managed"] is not (workload_name != "local"):
            raise ValueError(f"Runtime managed-workload marker is invalid: {workload_name}")
        for name, value in limits.items():
            if name == "MCP_ADMISSION_CONTROL_ENABLED":
                if not isinstance(value, bool):
                    raise ValueError(f"Runtime workload boolean is invalid: {workload_name}/{name}")
            else:
                minimum, maximum = WORKLOAD_LIMIT_BOUNDS[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > maximum:
                    raise ValueError(f"Runtime workload limit is invalid: {workload_name}/{name}")
        if not isinstance(rate_policy["enabled"], bool):
            raise ValueError(f"Runtime HTTP rate enablement is invalid: {workload_name}")
        rates = (rate_policy["per_key_per_minute"], rate_policy["global_per_minute"])
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in rates):
            raise ValueError(f"Runtime HTTP rates are invalid: {workload_name}")
        if any(rate_policy[name] > maximum for name, maximum in HTTP_RATE_MAXIMA.items()):
            raise ValueError(f"Runtime HTTP rates exceed safety bounds: {workload_name}")
        if rate_policy["enabled"] != all(value > 0 for value in rates):
            raise ValueError(f"Runtime HTTP rate policy is inconsistent: {workload_name}")
    required_capacity = set(CAPACITY_BOUNDS)
    for capacity_name, capacity in capacities.items():
        if not isinstance(capacity, dict) or set(capacity) != required_capacity:
            raise ValueError("Runtime capacity policy is invalid")
        for name, value in capacity.items():
            minimum, maximum = CAPACITY_BOUNDS[name]
            if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
                raise ValueError(f"Runtime capacity values are invalid: {capacity_name}")
        if capacity["actor_capacity"] > capacity["process_capacity"]:
            raise ValueError(f"Runtime actor capacity exceeds process capacity: {capacity_name}")
        if capacity["sync_workers"] > capacity["process_capacity"]:
            raise ValueError(f"Runtime sync workers exceed process capacity: {capacity_name}")
        if capacity["count_workers"] > capacity["process_capacity"]:
            raise ValueError(f"Runtime count workers exceed process capacity: {capacity_name}")


def _validate_trace_backend_prerequisite(environment: Mapping[str, str], workload_profile: str) -> None:
    """Require a usable trace origin only when a selected group needs one."""
    if "WF_TRACE_SERVER_URL" in environment:
        value = environment["WF_TRACE_SERVER_URL"].strip()
    elif "WEAVE_TRACE_SERVER_URL" in environment:
        value = environment["WEAVE_TRACE_SERVER_URL"].strip()
    elif workload_profile != "local":
        raise ValueError("Managed trace tool profiles require an explicit WF_TRACE_SERVER_URL")
    else:
        value = "https://trace.wandb.ai"
    if not value:
        raise ValueError("The selected tool profile requires a non-empty WF_TRACE_SERVER_URL")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as error:
        raise ValueError("WF_TRACE_SERVER_URL must be a valid absolute HTTP(S) URL") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("WF_TRACE_SERVER_URL must be a valid absolute HTTP(S) URL without credentials or fragments")


def resolve_runtime_selection(environment: Mapping[str, str] | None = None) -> RuntimeSelection:
    """Resolve and validate selectors immediately before MCP construction."""
    env = os.environ if environment is None else environment
    contract = load_runtime_contract()
    legacy_present = sorted(name for name in contract["legacy_tool_environment_variables"] if name in env)
    if legacy_present:
        raise ValueError(f"{_MIGRATION_MESSAGE} (found: {', '.join(legacy_present)})")

    selectors = contract["selectors"]

    def selected(key: str) -> str:
        definition = selectors[key]
        raw = env.get(definition["environment"], definition["default"])
        value = raw.strip() if isinstance(raw, str) else ""
        if not value:
            raise ValueError(f"{definition['environment']} must not be empty")
        values = definition.get("values")
        if values is not None and value not in values:
            allowed = ", ".join(values)
            raise ValueError(f"{definition['environment']} must be one of: {allowed}")
        return value

    tool_profile = selected("tool_profile")
    access_mode = selected("access_mode")
    workload_profile = selected("workload_profile")
    capacity_class = selected("capacity_class")
    if tool_profile not in contract["tool_profiles"]:
        raise ValueError(f"Unknown WANDB_MCP_TOOL_PROFILE: {tool_profile}")

    managed = contract["workload_profiles"][workload_profile]["managed"]
    allowed = contract["tool_profiles"][tool_profile]["managed_workloads"]
    if managed and workload_profile not in allowed:
        raise ValueError(
            f"WANDB_MCP_TOOL_PROFILE={tool_profile} is not allowed with managed {workload_profile} workload"
        )

    groups = frozenset(contract["tool_profiles"][tool_profile]["groups"])
    if groups & {"weave", "agents"}:
        _validate_trace_backend_prerequisite(env, workload_profile)
    if "aria" in groups:
        configured_aria_url = env.get("WB_AGENT_BASE_URL", "").strip()
        if not configured_aria_url:
            raise ValueError("The ARIA tool profile requires an explicit WB_AGENT_BASE_URL HTTPS origin")
        from wandb_mcp_server.config import validate_aria_base_url

        validate_aria_base_url(configured_aria_url)
        response_budget = env.get("MAX_RESPONSE_TOKENS")
        if response_budget is not None:
            try:
                response_budget_value = int(response_budget)
            except (TypeError, ValueError):
                raise ValueError("MAX_RESPONSE_TOKENS must be an integer") from None
            if response_budget_value < 64:
                raise ValueError("MAX_RESPONSE_TOKENS must be at least 64 for the ARIA tool profile")

    tools = tools_for_profile(contract, tool_profile, access_mode)
    return RuntimeSelection(
        tool_profile=tool_profile,
        access_mode=access_mode,
        workload_profile=workload_profile,
        capacity_class=capacity_class,
        tools=tools,
        groups=groups,
        runtime_contract_sha256=runtime_contract_sha256(contract),
    )


def workload_policy(selection: RuntimeSelection) -> Mapping[str, Any]:
    """Return the immutable workload/capacity policy for one selection."""
    contract = load_runtime_contract()
    return {
        **contract["workload_profiles"][selection.workload_profile],
        "capacity": contract["capacity_classes"][selection.capacity_class],
    }
