#!/usr/bin/env python3
"""Validate and build the public W&B MCP release artifact.

This utility is intentionally provider-neutral. It creates deterministic JSON
that a protected release workflow can sign and store as provenance; it does not
publish, tag, promote, or obtain credentials.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "release" / "public-contract.json"
ATTESTATION_TYPE = "https://wandb.ai/attestations/mcp-public-release/v1"
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)?$")
_GENERATED_FEATURES_START = "<!-- BEGIN GENERATED: PUBLIC FEATURE PROFILES -->"
_GENERATED_FEATURES_END = "<!-- END GENERATED: PUBLIC FEATURE PROFILES -->"


class ReleaseError(RuntimeError):
    """A fail-closed release-contract violation."""


@dataclass(frozen=True)
class Profile:
    """One exact feature/read-only registration contract."""

    name: str
    features: dict[str, bool]
    read_only: bool
    environment: dict[str, str]
    tools: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": self.features,
            "read_only": self.read_only,
            "environment": self.environment,
            "tools": list(self.tools),
        }


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read release contract {path}: {error}") from error
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise ReleaseError("public release contract schema_version must be 1")
    if contract.get("package") != "wandb_mcp_server":
        raise ReleaseError("public release contract package must be wandb_mcp_server")
    if not isinstance(contract.get("build_requirements"), str) or not contract["build_requirements"]:
        raise ReleaseError("public release contract must define build_requirements")
    qualification_tests = contract.get("qualification_tests")
    if (
        not isinstance(qualification_tests, list)
        or not qualification_tests
        or not all(isinstance(path, str) and path for path in qualification_tests)
    ):
        raise ReleaseError("public release contract must define qualification_tests")
    required_pr_checks = contract.get("required_pr_checks")
    if (
        not isinstance(required_pr_checks, list)
        or not required_pr_checks
        or len(required_pr_checks) != len(set(required_pr_checks))
        or not all(isinstance(name, str) and name for name in required_pr_checks)
    ):
        raise ReleaseError("public release contract must define unique required_pr_checks")
    locked_runtime = contract.get("locked_runtime")
    if (
        not isinstance(locked_runtime, dict)
        or not locked_runtime
        or not all(isinstance(name, str) and isinstance(version, str) for name, version in locked_runtime.items())
    ):
        raise ReleaseError("public release contract must define locked_runtime versions")

    groups = contract.get("feature_groups")
    if not isinstance(groups, dict) or not groups:
        raise ReleaseError("public release contract must define feature_groups")

    seen_tools: set[str] = set()
    seen_environments: set[str] = set()
    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not isinstance(group, dict):
            raise ReleaseError("feature group names and definitions must be mappings")
        environment = group.get("environment")
        if not isinstance(environment, str) or not environment:
            raise ReleaseError(f"feature group {group_name} has no environment variable")
        if environment in seen_environments:
            raise ReleaseError(f"duplicate feature environment variable: {environment}")
        seen_environments.add(environment)
        if not isinstance(group.get("default"), bool):
            raise ReleaseError(f"feature group {group_name} default must be boolean")
        tools = group.get("tools")
        if not isinstance(tools, list) or not tools or not all(isinstance(tool, str) and tool for tool in tools):
            raise ReleaseError(f"feature group {group_name} must define tool names")
        duplicates = seen_tools.intersection(tools)
        if duplicates:
            raise ReleaseError(f"tools occur in more than one group: {sorted(duplicates)}")
        seen_tools.update(tools)

    base_tools = contract.get("base_tools")
    if not isinstance(base_tools, list) or not base_tools:
        raise ReleaseError("public release contract must define base_tools")
    duplicates = seen_tools.intersection(base_tools)
    if duplicates or len(base_tools) != len(set(base_tools)):
        raise ReleaseError(f"duplicate public tools: {sorted(duplicates)}")
    seen_tools.update(base_tools)

    write_tools = contract.get("write_tools")
    if not isinstance(write_tools, list) or not set(write_tools) <= seen_tools:
        raise ReleaseError("write_tools must be a subset of public tools")
    if not isinstance(contract.get("read_only_environment"), str):
        raise ReleaseError("read_only_environment must be defined")

    named_profiles = contract.get("named_profiles")
    if not isinstance(named_profiles, dict) or not named_profiles:
        raise ReleaseError("named_profiles must be defined")
    allowed_profile_keys = set(groups) | {"read_only"}
    for name, overrides in named_profiles.items():
        if not isinstance(name, str) or not isinstance(overrides, dict):
            raise ReleaseError("named profile names and overrides must be mappings")
        unknown = set(overrides) - allowed_profile_keys
        if unknown:
            raise ReleaseError(f"named profile {name} has unknown keys: {sorted(unknown)}")
        if not all(isinstance(value, bool) for value in overrides.values()):
            raise ReleaseError(f"named profile {name} values must be boolean")


def _profile(contract: dict[str, Any], name: str, features: dict[str, bool], read_only: bool) -> Profile:
    tools = set(contract["base_tools"])
    environment: dict[str, str] = {}
    for group_name, group in contract["feature_groups"].items():
        enabled = features[group_name]
        environment[group["environment"]] = str(enabled).lower()
        if enabled:
            tools.update(group["tools"])
    environment[contract["read_only_environment"]] = str(read_only).lower()
    if read_only:
        tools.difference_update(contract["write_tools"])
    return Profile(
        name=name,
        features=dict(features),
        read_only=read_only,
        environment=environment,
        tools=tuple(sorted(tools)),
    )


def named_profile(contract: dict[str, Any], name: str) -> Profile:
    try:
        overrides = contract["named_profiles"][name]
    except KeyError as error:
        raise ReleaseError(f"unknown named profile: {name}") from error
    features = {
        group_name: overrides.get(group_name, group["default"])
        for group_name, group in contract["feature_groups"].items()
    }
    return _profile(contract, name, features, overrides.get("read_only", False))


def exhaustive_profiles(contract: dict[str, Any]) -> tuple[Profile, ...]:
    group_names = tuple(contract["feature_groups"])
    profiles: list[Profile] = []
    for values in itertools.product((False, True), repeat=len(group_names) + 1):
        feature_values = dict(zip(group_names, values[:-1], strict=True))
        read_only = values[-1]
        feature_slug = "-".join(f"{name}={int(feature_values[name])}" for name in group_names)
        name = f"{feature_slug}-read_only={int(read_only)}"
        profiles.append(_profile(contract, name, feature_values, read_only))
    return tuple(profiles)


def _read_version_facts(root: Path) -> dict[str, str]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    pyproject_version = pyproject["project"]["version"]

    init_text = (root / "src" / "wandb_mcp_server" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    if match is None:
        raise ReleaseError("src/wandb_mcp_server/__init__.py does not define __version__")

    lock = tomllib.loads((root / "uv.lock").read_text())
    package_versions = {
        package["version"]
        for package in lock.get("package", [])
        if package.get("name", "").replace("-", "_") == "wandb_mcp_server"
    }
    if len(package_versions) != 1:
        raise ReleaseError("uv.lock must contain exactly one wandb-mcp-server package entry")

    return {
        "pyproject": pyproject_version,
        "package": match.group(1),
        "lock": package_versions.pop(),
    }


def verify_version(root: Path, version: str) -> dict[str, str]:
    if not _VERSION_PATTERN.fullmatch(version):
        raise ReleaseError(f"invalid release version: {version}")
    facts = _read_version_facts(root)
    mismatched = {source: value for source, value in facts.items() if value != version}
    if mismatched:
        raise ReleaseError(f"release version {version} does not match package metadata: {mismatched}")

    note = root / "docs" / "releases" / f"v{version}.md"
    index = root / "docs" / "releases" / "README.md"
    if not note.is_file():
        raise ReleaseError(f"missing release note: {note.relative_to(root)}")
    if not index.is_file() or f"(v{version}.md)" not in index.read_text():
        raise ReleaseError(f"release index does not link v{version}.md")
    return facts


def policy_identity(root: Path, contract_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Bind qualification evidence to its versioned policy and executable tests."""

    build_requirements = root / contract["build_requirements"]
    if not build_requirements.is_file():
        raise ReleaseError(f"missing build dependency lock: {build_requirements}")
    test_digests: dict[str, str] = {}
    for relative_path in contract["qualification_tests"]:
        path = root / relative_path
        if not path.is_file():
            raise ReleaseError(f"missing qualification test: {relative_path}")
        test_digests[relative_path] = sha256_file(path)
    return {
        "contract_sha256": sha256_file(contract_path),
        "build_requirements": {
            "path": contract["build_requirements"],
            "sha256": sha256_file(build_requirements),
        },
        "qualification_tests": test_digests,
    }


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or "git command failed"
        raise ReleaseError(detail)
    return process.stdout.strip()


def _json_command(command: list[str], root: Path) -> Any:
    process = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or f"command failed: {command[0]}"
        raise ReleaseError(detail)
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"command did not return JSON: {command[0]}") from error


def verify_clean_tree(root: Path) -> None:
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ReleaseError("release source tree is dirty; commit or remove every change before release")


def verify_source_pr_gate(root: Path, contract_path: Path, repository: str, sha: str) -> dict[str, Any]:
    """Verify that a source tag resolves to one approved, green release PR."""

    started = time.monotonic()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ReleaseError("repository must be formatted as owner/name")
    if sha != _git(root, "rev-parse", "HEAD"):
        raise ReleaseError("source-gate SHA does not match HEAD")
    parents = _git(root, "rev-list", "--parents", "-n", "1", sha).split()
    if len(parents) != 3:
        raise ReleaseError("source release commit must be a two-parent merge commit")

    pulls = _json_command(
        ["gh", "api", f"/repos/{repository}/commits/{sha}/pulls", "--paginate"],
        root,
    )
    matching = [
        pull
        for pull in pulls
        if pull.get("base", {}).get("ref") == "main" and pull.get("merged_at") and pull.get("merge_commit_sha") == sha
    ]
    if len(matching) != 1:
        raise ReleaseError("source release commit must map to exactly one merged PR targeting main")
    pull = matching[0]
    head_sha = pull.get("head", {}).get("sha")
    if head_sha != parents[2]:
        raise ReleaseError("merged PR head does not match the release merge commit's second parent")
    if _git(root, "rev-parse", f"{sha}^{{tree}}") != _git(root, "rev-parse", f"{head_sha}^{{tree}}"):
        raise ReleaseError("merged source tree differs from the reviewed PR head tree")

    owner, name = repository.split("/", 1)
    query = """
      query($owner:String!,$name:String!,$number:Int!){
        repository(owner:$owner,name:$name){
          pullRequest(number:$number){
            reviewDecision
            reviewThreads(first:100){nodes{isResolved} pageInfo{hasNextPage}}
            commits(last:1){nodes{commit{oid statusCheckRollup{state contexts(first:100){
              nodes{__typename ... on CheckRun{name status conclusion} ... on StatusContext{context state}}
              pageInfo{hasNextPage}
            }}}}}
          }
        }
      }
    """
    response = _json_command(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pull['number']}",
        ],
        root,
    )
    pr = response.get("data", {}).get("repository", {}).get("pullRequest")
    if not isinstance(pr, dict):
        raise ReleaseError("cannot read release PR gate state")
    if pr.get("reviewDecision") != "APPROVED":
        raise ReleaseError("release PR does not have an approved review decision")
    threads = pr.get("reviewThreads", {})
    if threads.get("pageInfo", {}).get("hasNextPage"):
        raise ReleaseError("release PR has more review threads than the bounded gate can verify")
    unresolved = sum(not node.get("isResolved", False) for node in threads.get("nodes", []))
    if unresolved:
        raise ReleaseError(f"release PR has {unresolved} unresolved review conversations")

    commits = pr.get("commits", {}).get("nodes", [])
    if len(commits) != 1 or commits[0].get("commit", {}).get("oid") != head_sha:
        raise ReleaseError("release PR status rollup is not bound to its reviewed head SHA")
    rollup = commits[0]["commit"].get("statusCheckRollup")
    if not isinstance(rollup, dict) or rollup.get("state") != "SUCCESS":
        raise ReleaseError("release PR current-head status rollup is not successful")
    contexts = rollup.get("contexts", {})
    if contexts.get("pageInfo", {}).get("hasNextPage"):
        raise ReleaseError("release PR has more checks than the bounded gate can verify")
    passed: set[str] = set()
    observed: dict[str, str] = {}
    for node in contexts.get("nodes", []):
        if node.get("__typename") == "CheckRun":
            check_name = node.get("name")
            outcome = node.get("conclusion")
            successful = node.get("status") == "COMPLETED" and outcome == "SUCCESS"
        else:
            check_name = node.get("context")
            outcome = node.get("state")
            successful = outcome == "SUCCESS"
        if isinstance(check_name, str):
            observed[check_name] = str(outcome)
            if successful:
                passed.add(check_name)
    contract = load_contract(contract_path)
    missing = set(contract["required_pr_checks"]) - passed
    if missing:
        raise ReleaseError(f"release PR is missing successful required checks: {sorted(missing)}")

    issued_at = datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "gate": "candidate-pr",
        "status": "passed",
        "repository": repository,
        "pull_request": pull["number"],
        "reviewed_head_sha": head_sha,
        "reviewed_tree": _git(root, "rev-parse", f"{head_sha}^{{tree}}"),
        "merge_sha": sha,
        "merge_tree": _git(root, "rev-parse", f"{sha}^{{tree}}"),
        "review_decision": "APPROVED",
        "unresolved_conversations": 0,
        "required_checks": {name: observed[name] for name in contract["required_pr_checks"]},
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "expires_at": (issued_at + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "duration_ms": round((time.monotonic() - started) * 1000),
    }


def source_identity(root: Path) -> dict[str, str]:
    return {
        "sha": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
    }


def verify_signed_tag(root: Path, version: str, verification: str = "local") -> str:
    tag = f"v{version}"
    if _git(root, "cat-file", "-t", f"refs/tags/{tag}", check=False) != "tag":
        raise ReleaseError(f"{tag} must be an annotated tag")
    if _git(root, "rev-list", "-n", "1", tag) != _git(root, "rev-parse", "HEAD"):
        raise ReleaseError(f"{tag} does not point to HEAD")
    if verification == "local":
        process = subprocess.run(["git", "verify-tag", tag], cwd=root, check=False, capture_output=True, text=True)
        if process.returncode:
            raise ReleaseError(f"{tag} does not have a locally verifiable signature")
    elif verification == "github":
        repository = _git(root, "config", "--get", "remote.origin.url")
        match = re.search(r"github\.com[/:]([^/]+/[^/.]+)(?:\.git)?$", repository)
        if match is None:
            raise ReleaseError("cannot derive GitHub repository from origin")
        tag_object = _git(root, "rev-parse", f"refs/tags/{tag}")
        process = subprocess.run(
            ["gh", "api", f"/repos/{match.group(1)}/git/tags/{tag_object}", "--jq", ".verification.verified"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode or process.stdout.strip() != "true":
            raise ReleaseError(f"{tag} does not have a GitHub-verified signature")
    else:
        raise ReleaseError(f"unsupported tag verification mode: {verification}")
    return tag


def preflight(
    root: Path,
    contract_path: Path,
    version: str,
    gate: str,
    *,
    tag_verification: str = "local",
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    facts = verify_version(root, version)
    verify_clean_tree(root)
    identity = source_identity(root)
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": gate,
        "version": version,
        "source": identity,
        "metadata_versions": facts,
        "policy": policy_identity(root, contract_path, contract),
        "profile_count": len(exhaustive_profiles(contract)),
        "status": "passed",
    }
    if gate == "source-released":
        result["tag"] = verify_signed_tag(root, version, tag_verification)
    return result


def validate_output_directory(root: Path, output_dir: Path) -> None:
    resolved_root = root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == (resolved_root / "dist"):
        raise ReleaseError("release builds may not use the repository dist/ directory")
    if resolved_output == resolved_root or resolved_root in resolved_output.parents:
        raise ReleaseError("release output must be outside the source checkout")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReleaseError(f"release output directory is not empty: {output_dir}")


def build_artifacts(root: Path, contract_path: Path, version: str, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    verify_version(root, version)
    verify_clean_tree(root)
    validate_output_directory(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = policy_identity(root, contract_path, contract)
    build_requirements = root / contract["build_requirements"]
    started = time.monotonic()
    process = subprocess.run(
        [
            "uv",
            "build",
            "--build-constraints",
            str(build_requirements),
            "--require-hashes",
            "--no-create-gitignore",
            "--out-dir",
            str(output_dir),
        ],
        cwd=root,
        check=False,
        text=True,
    )
    if process.returncode:
        raise ReleaseError("uv build failed")
    verify_clean_tree(root)
    artifacts = sorted((*output_dir.glob("*.whl"), *output_dir.glob("*.tar.gz")))
    if len(artifacts) != 2:
        raise ReleaseError("isolated build must produce exactly one wheel and one source distribution")
    records = [{"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size} for path in artifacts]
    checksums = "".join(f"{record['sha256']}  {record['name']}\n" for record in records)
    (output_dir / "SHA256SUMS").write_text(checksums)
    result = {
        "schema_version": 1,
        "version": version,
        "source": source_identity(root),
        "policy": policy,
        "artifacts": records,
        "status": "built",
        "duration_ms": round((time.monotonic() - started) * 1000),
    }
    (output_dir / "build-manifest.json").write_bytes(canonical_json(result))
    return result


def _artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        if not path.is_file():
            raise ReleaseError(f"release evidence file does not exist: {path}")
        records.append({"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size})
    return records


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be a JSON object: {path}")
    return value


def _artifact_by_suffix(records: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    matches = [record for record in records if record["name"].endswith(suffix)]
    if len(matches) != 1:
        raise ReleaseError(f"release evidence must contain exactly one {suffix} artifact")
    return matches[0]


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ReleaseError(f"cannot read build checksums {path}: {error}") from error
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/]+)", line)
        if match is None or match.group(2) in checksums:
            raise ReleaseError(f"invalid build checksum line: {line!r}")
        checksums[match.group(2)] = match.group(1)
    return checksums


def validate_release_artifacts(
    root: Path,
    contract_path: Path,
    version: str,
    artifact_paths: Iterable[Path],
    build_manifest_path: Path,
    build_checksums_path: Path,
) -> list[dict[str, Any]]:
    """Bind qualified inputs to one build and reject fabricated security reports."""

    artifact_records = _artifact_records(artifact_paths)
    if len(artifact_records) != 4:
        raise ReleaseError("release evidence must contain exactly wheel, sdist, SBOM, and vulnerability report")
    wheel = _artifact_by_suffix(artifact_records, ".whl")
    sdist = _artifact_by_suffix(artifact_records, ".tar.gz")
    sbom_record = _artifact_by_suffix(artifact_records, ".spdx.json")
    vulnerability_record = _artifact_by_suffix(artifact_records, ".vulnerabilities.json")

    normalized_version = version.replace("-", "_")
    if wheel["name"] != f"wandb_mcp_server-{normalized_version}-py3-none-any.whl":
        raise ReleaseError("wheel name does not match the qualified version")
    if sdist["name"] != f"wandb_mcp_server-{normalized_version}.tar.gz":
        raise ReleaseError("source distribution name does not match the qualified version")

    identity = source_identity(root)
    contract = load_contract(contract_path)
    manifest = _read_json_object(build_manifest_path, "build manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "built"
        or manifest.get("version") != version
        or manifest.get("source") != identity
        or manifest.get("policy") != policy_identity(root, contract_path, contract)
        or not isinstance(manifest.get("duration_ms"), int)
        or manifest["duration_ms"] < 0
    ):
        raise ReleaseError("build manifest does not match the qualified source and policy")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list) or len(manifest_artifacts) != 2:
        raise ReleaseError("build manifest must contain exactly one wheel and one source distribution")
    expected_build_records = sorted((wheel, sdist), key=lambda record: record["name"])
    if sorted(manifest_artifacts, key=lambda record: record.get("name", "")) != expected_build_records:
        raise ReleaseError("wheel or source distribution does not match the build manifest")
    checksums = _read_checksums(build_checksums_path)
    expected_checksums = {record["name"]: record["sha256"] for record in expected_build_records}
    if checksums != expected_checksums:
        raise ReleaseError("wheel or source distribution does not match the build checksums")

    sbom_path = next(path for path in artifact_paths if path.name == sbom_record["name"])
    sbom = _read_json_object(sbom_path, "SPDX SBOM")
    if (
        not str(sbom.get("spdxVersion", "")).startswith("SPDX-2.")
        or sbom.get("SPDXID") != "SPDXRef-DOCUMENT"
        or not isinstance(sbom.get("packages"), list)
    ):
        raise ReleaseError("SBOM is not a complete SPDX JSON document")

    vulnerability_path = next(path for path in artifact_paths if path.name == vulnerability_record["name"])
    vulnerability_report = _read_json_object(vulnerability_path, "Grype vulnerability report")
    if vulnerability_report.get("descriptor", {}).get("name") != "grype" or not isinstance(
        vulnerability_report.get("matches"), list
    ):
        raise ReleaseError("vulnerability report is not complete Grype JSON")
    blocking = sorted(
        {
            str(match.get("vulnerability", {}).get("id", "unknown"))
            for match in vulnerability_report["matches"]
            if str(match.get("vulnerability", {}).get("severity", "")).lower() in {"high", "critical"}
        }
    )
    if blocking:
        raise ReleaseError(f"vulnerability report contains High/Critical findings: {blocking}")
    return artifact_records


def validate_parent_evidence(root: Path, path: Path, contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    evidence = _read_json_object(path, "candidate gate evidence")
    identity = source_identity(root)
    contract = load_contract(contract_path)
    try:
        issued_at = datetime.fromisoformat(str(evidence.get("issued_at", "")).replace("Z", "+00:00"))
        expires_at = datetime.fromisoformat(str(evidence.get("expires_at", "")).replace("Z", "+00:00"))
    except ValueError as error:
        raise ReleaseError("candidate gate evidence has invalid validity timestamps") from error
    now = datetime.now(timezone.utc)
    if issued_at.tzinfo is None or expires_at.tzinfo is None or issued_at > now or expires_at <= now:
        raise ReleaseError("candidate gate evidence is not currently valid")
    if expires_at - issued_at > timedelta(hours=1):
        raise ReleaseError("candidate gate evidence validity exceeds one hour")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("gate") != "candidate-pr"
        or evidence.get("status") != "passed"
        or evidence.get("merge_sha") != identity["sha"]
        or evidence.get("merge_tree") != identity["tree"]
        or evidence.get("reviewed_tree") != identity["tree"]
        or evidence.get("review_decision") != "APPROVED"
        or evidence.get("unresolved_conversations") != 0
        or not isinstance(evidence.get("required_checks"), dict)
        or not isinstance(evidence.get("duration_ms"), int)
        or evidence["duration_ms"] < 0
        or set(evidence["required_checks"]) != set(contract["required_pr_checks"])
        or any(outcome != "SUCCESS" for outcome in evidence["required_checks"].values())
    ):
        raise ReleaseError("candidate gate evidence does not qualify this source")
    return evidence


def create_attestation(
    root: Path,
    contract_path: Path,
    version: str,
    profile_evidence_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    build_manifest_path: Path,
    build_checksums_path: Path,
    parent_evidence_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    verify_version(root, version)
    verify_clean_tree(root)
    expected_profiles = [profile.as_dict() for profile in exhaustive_profiles(contract)]
    profile_evidence_records: list[dict[str, Any]] = []
    evidence_by_python: dict[str, dict[str, Any]] = {}
    for profile_evidence_path in profile_evidence_paths:
        try:
            evidence = json.loads(profile_evidence_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ReleaseError(f"invalid profile evidence {profile_evidence_path}: {error}") from error
        if evidence.get("status") != "passed" or evidence.get("profiles") != expected_profiles:
            raise ReleaseError("profile evidence does not prove every exact public profile")
        if evidence.get("contract_sha256") != sha256_file(contract_path):
            raise ReleaseError("profile evidence was produced from a different release contract")
        harness = root / "scripts" / "mcp_stdio_smoke.py"
        if evidence.get("harness_sha256") != sha256_file(harness):
            raise ReleaseError("profile evidence was produced by a different installed-wheel harness")
        if evidence.get("version") != version:
            raise ReleaseError("profile evidence version does not match the release")
        if evidence.get("mcp_version") != contract["mcp"]["locked"]:
            raise ReleaseError("profile evidence does not use the locked MCP SDK version")
        if evidence.get("locked_runtime") != contract["locked_runtime"]:
            raise ReleaseError("profile evidence does not use the locked W&B runtime")
        if not isinstance(evidence.get("duration_ms"), int) or evidence["duration_ms"] < 0:
            raise ReleaseError("profile evidence does not contain a valid duration")
        python_version = evidence.get("python_version")
        if python_version not in contract["python_versions"]:
            raise ReleaseError(f"profile evidence has unsupported Python version: {python_version}")
        if python_version in evidence_by_python:
            raise ReleaseError(f"duplicate profile evidence for Python {python_version}")
        evidence_by_python[python_version] = evidence
        profile_evidence_records.append(
            {
                "name": profile_evidence_path.name,
                "sha256": sha256_file(profile_evidence_path),
                "python_version": python_version,
                "mcp_version": evidence.get("mcp_version"),
                "profile_count": len(expected_profiles),
            }
        )
    missing_python = set(contract["python_versions"]) - set(evidence_by_python)
    if missing_python:
        raise ReleaseError(f"missing exact-profile evidence for Python versions: {sorted(missing_python)}")

    artifact_paths = tuple(artifact_paths)
    artifact_records = validate_release_artifacts(
        root,
        contract_path,
        version,
        artifact_paths,
        build_manifest_path,
        build_checksums_path,
    )
    parent_evidence = validate_parent_evidence(root, parent_evidence_path, contract_path)

    identity = source_identity(root)
    if any(evidence.get("source_sha") != identity["sha"] for evidence in evidence_by_python.values()):
        raise ReleaseError("profile evidence source SHA does not match the release source")
    wheel_hashes = {record["sha256"] for record in artifact_records if record["name"].endswith(".whl")}
    if any(evidence.get("wheel_sha256") not in wheel_hashes for evidence in evidence_by_python.values()):
        raise ReleaseError("profile evidence was not produced from the release wheel")
    build_manifest = _read_json_object(build_manifest_path, "build manifest")
    return {
        "_type": ATTESTATION_TYPE,
        "schema_version": 1,
        "release": {"version": version, "source_sha": identity["sha"], "source_tree": identity["tree"]},
        "policy": policy_identity(root, contract_path, contract),
        "parent_evidence": {
            "gate": parent_evidence["gate"],
            "sha256": sha256_file(parent_evidence_path),
            "pull_request": parent_evidence["pull_request"],
            "reviewed_head_sha": parent_evidence["reviewed_head_sha"],
        },
        "subjects": artifact_records,
        "profile_evidence": sorted(profile_evidence_records, key=lambda item: item["python_version"]),
        "compatibility": {
            "python": contract["python_versions"],
            "mcp": contract["mcp"],
        },
        "timings": {
            "candidate_gate_ms": parent_evidence["duration_ms"],
            "artifact_build_ms": build_manifest["duration_ms"],
            "exact_profiles_ms": {
                python_version: evidence_by_python[python_version]["duration_ms"]
                for python_version in sorted(evidence_by_python)
            },
        },
        "status": "qualified",
        "approval_state": "source-pr-approved",
        "rollback": {"strategy": "fix-forward", "mutable_artifact": False},
    }


def _write_json(value: Any, output: Path | None) -> None:
    payload = canonical_json(value)
    if output is None:
        sys.stdout.buffer.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def _profiles_payload(contract_path: Path, *, all_profiles: bool, names: list[str]) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if all_profiles:
        profiles = exhaustive_profiles(contract)
    elif names:
        profiles = tuple(named_profile(contract, name) for name in names)
    else:
        profiles = tuple(named_profile(contract, name) for name in contract["named_profiles"])
    return {
        "schema_version": 1,
        "contract_sha256": sha256_file(contract_path),
        "profiles": [profile.as_dict() for profile in profiles],
    }


def _generated_feature_docs(contract: dict[str, Any]) -> str:
    environment_rows = [
        "| Variable | Default | Effect |",
        "|---|---:|---|",
    ]
    for group_name, group in contract["feature_groups"].items():
        environment_rows.append(
            f"| `{group['environment']}` | `{str(group['default']).lower()}` | "
            f"Registers the `{group_name}` group ({len(group['tools'])} tools). |"
        )
    environment_rows.append(
        f"| `{contract['read_only_environment']}` | `false` | Removes all tools classified as writes. |"
    )

    profile_rows = [
        "| Profile | Weave | Agents | ARIA | Raw GraphQL | Read-only | Exact tools |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in contract["named_profiles"]:
        profile = named_profile(contract, name)
        values = [str(profile.features[key]).lower() for key in ("weave", "agents", "aria", "raw_graphql")]
        profile_rows.append(
            f"| `{name}` | "
            + " | ".join(f"`{value}`" for value in values)
            + f" | `{str(profile.read_only).lower()}` | {len(profile.tools)} |"
        )
    body = "\n".join(
        [
            _GENERATED_FEATURES_START,
            "<!-- Generated by scripts/public_release.py docs. Do not edit this block. -->",
            "",
            "Release-controlled feature variables:",
            "",
            *environment_rows,
            "",
            "Named exact tool profiles:",
            "",
            *profile_rows,
            "",
            "Use `python scripts/public_release.py profiles --all` for every exact feature/read-only combination and tool name.",
            _GENERATED_FEATURES_END,
        ]
    )
    return body


def update_generated_docs(contract_path: Path, readme: Path, *, check: bool) -> dict[str, Any]:
    contract = load_contract(contract_path)
    try:
        current = readme.read_text()
    except OSError as error:
        raise ReleaseError(f"cannot read documentation target {readme}: {error}") from error
    if current.count(_GENERATED_FEATURES_START) != 1 or current.count(_GENERATED_FEATURES_END) != 1:
        raise ReleaseError("README must contain exactly one generated public-feature block")
    before, remainder = current.split(_GENERATED_FEATURES_START, 1)
    _, after = remainder.split(_GENERATED_FEATURES_END, 1)
    rendered = before + _generated_feature_docs(contract) + after
    changed = rendered != current
    if check and changed:
        raise ReleaseError("generated README feature profiles are stale; run public_release.py docs --write")
    if not check and changed:
        readme.write_text(rendered)
    return {"schema_version": 1, "target": str(readme), "changed": changed, "status": "passed"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles = subparsers.add_parser("profiles", help="Print exact public tool profiles")
    profiles.add_argument("--all", action="store_true", dest="all_profiles")
    profiles.add_argument("--name", action="append", default=[])
    profiles.add_argument("--output", type=Path)

    preflight_parser = subparsers.add_parser("preflight", help="Validate source release invariants")
    preflight_parser.add_argument("--version", required=True)
    preflight_parser.add_argument("--gate", choices=("candidate", "source-released"), default="candidate")
    preflight_parser.add_argument("--tag-verification", choices=("local", "github"), default="local")
    preflight_parser.add_argument("--output", type=Path)

    source_gate = subparsers.add_parser("source-gate", help="Verify the approved release PR behind a source tag")
    source_gate.add_argument("--repository", required=True)
    source_gate.add_argument("--sha", required=True)
    source_gate.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build", help="Build wheel and sdist into a fresh external directory")
    build.add_argument("--version", required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    attest = subparsers.add_parser("attest", help="Create the predicate for protected keyless signing")
    attest.add_argument("--version", required=True)
    attest.add_argument("--profile-evidence", type=Path, action="append", required=True)
    attest.add_argument("--artifact", type=Path, action="append", required=True)
    attest.add_argument("--build-manifest", type=Path, required=True)
    attest.add_argument("--build-checksums", type=Path, required=True)
    attest.add_argument("--parent-evidence", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)

    docs = subparsers.add_parser("docs", help="Generate or verify public feature documentation")
    docs.add_argument("--readme", type=Path, default=REPOSITORY_ROOT / "README.md")
    mode = docs.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "profiles":
            result = _profiles_payload(args.contract, all_profiles=args.all_profiles, names=args.name)
            _write_json(result, args.output)
        elif args.command == "preflight":
            _write_json(
                preflight(
                    args.root,
                    args.contract,
                    args.version,
                    args.gate,
                    tag_verification=args.tag_verification,
                ),
                args.output,
            )
        elif args.command == "source-gate":
            _write_json(
                verify_source_pr_gate(args.root, args.contract, args.repository, args.sha),
                args.output,
            )
        elif args.command == "build":
            _write_json(build_artifacts(args.root, args.contract, args.version, args.output_dir), None)
        elif args.command == "attest":
            attestation = create_attestation(
                args.root,
                args.contract,
                args.version,
                args.profile_evidence,
                args.artifact,
                args.build_manifest,
                args.build_checksums,
                args.parent_evidence,
            )
            _write_json(attestation, args.output)
        elif args.command == "docs":
            _write_json(update_generated_docs(args.contract, args.readme, check=args.check), None)
        else:  # pragma: no cover - argparse enforces the command set.
            parser.error("unknown command")
    except ReleaseError as error:
        parser.exit(2, f"release precondition failed: {error}\n")


if __name__ == "__main__":
    main()
