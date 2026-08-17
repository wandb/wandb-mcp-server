"""Tests for the version-neutral public release contract and evidence builder."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

import pytest
from mcp.server.fastmcp import FastMCP

from scripts import public_release
from scripts.mcp_stdio_smoke import _server_environment
from wandb_mcp_server.server import register_tools


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "release" / "public-contract.json"
GENERIC_RELEASE_FILES = (
    REPOSITORY_ROOT / "RELEASING.md",
    REPOSITORY_ROOT / "CONTRIBUTING.md",
    REPOSITORY_ROOT / "docs" / "releases" / "TEMPLATE.md",
    REPOSITORY_ROOT / ".agents" / "skills" / "release-wandb-mcp-server" / "SKILL.md",
    REPOSITORY_ROOT / ".github" / "workflows" / "release-source.yml",
    REPOSITORY_ROOT / "scripts" / "public_release.py",
    CONTRACT_PATH,
)


def _contract() -> dict:
    return public_release.load_contract(CONTRACT_PATH)


def test_contract_generates_every_exact_feature_and_read_only_profile():
    contract = _contract()
    profiles = public_release.exhaustive_profiles(contract)

    assert len(profiles) == 32
    assert len({profile.name for profile in profiles}) == 32
    assert all(profile.tools == tuple(sorted(set(profile.tools))) for profile in profiles)

    by_shape = {(tuple(profile.features.values()), profile.read_only): profile for profile in profiles}
    assert len(by_shape) == 32
    for profile in profiles:
        write_tools = set(contract["write_tools"])
        if profile.read_only:
            assert write_tools.isdisjoint(profile.tools)
        else:
            expected_writes = write_tools.intersection(
                set(contract["base_tools"])
                | {
                    tool
                    for feature, enabled in profile.features.items()
                    if enabled
                    for tool in contract["feature_groups"][feature]["tools"]
                }
            )
            assert expected_writes <= set(profile.tools)


def test_contract_matches_runtime_for_every_profile(monkeypatch):
    contract = _contract()
    import wandb_mcp_server.config as config

    try:
        for profile in public_release.exhaustive_profiles(contract):
            for name, value in profile.environment.items():
                monkeypatch.setenv(name, value)
            importlib.reload(config)
            server = FastMCP("release-contract-test")
            register_tools(server)
            assert set(server._tool_manager._tools) == set(profile.tools), profile.name
    finally:
        for group in contract["feature_groups"].values():
            monkeypatch.delenv(group["environment"], raising=False)
        monkeypatch.delenv(contract["read_only_environment"], raising=False)
        importlib.reload(config)


def test_named_profiles_are_shortcuts_to_exact_contracts():
    contract = _contract()

    default = public_release.named_profile(contract, "default")
    full = public_release.named_profile(contract, "full")
    strict = public_release.named_profile(contract, "strict-read-only")

    assert default.features == {"weave": True, "agents": False, "aria": False, "raw_graphql": False}
    assert full.features == {"weave": True, "agents": True, "aria": True, "raw_graphql": True}
    assert strict.read_only is True
    assert set(contract["write_tools"]).isdisjoint(strict.tools)


def test_release_version_validation_generalizes_to_future_versions(tmp_path):
    version = "9.8.7"
    (tmp_path / "src" / "wandb_mcp_server").mkdir(parents=True)
    (tmp_path / "docs" / "releases").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "wandb_mcp_server"\nversion = "{version}"\n')
    (tmp_path / "src" / "wandb_mcp_server" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (tmp_path / "uv.lock").write_text(f'version = 1\n\n[[package]]\nname = "wandb-mcp-server"\nversion = "{version}"\n')
    (tmp_path / "docs" / "releases" / f"v{version}.md").write_text(f"# v{version}\n")
    (tmp_path / "docs" / "releases" / "README.md").write_text(f"- [v{version}](v{version}.md)\n")

    assert public_release.verify_version(tmp_path, version) == {
        "pyproject": version,
        "package": version,
        "lock": version,
    }


def test_generic_release_process_has_no_current_version_or_candidate_sha():
    forbidden = (
        "0.4.0",
        "305e5a640ea370ba10d8aab7ff034802eb5f37d0",
    )
    for path in GENERIC_RELEASE_FILES:
        text = path.read_text()
        for value in forbidden:
            assert value not in text, f"{path.relative_to(REPOSITORY_ROOT)} hard-codes {value}"


def test_isolated_build_rejects_repository_and_nonempty_outputs(tmp_path):
    with pytest.raises(public_release.ReleaseError, match="dist"):
        public_release.validate_output_directory(REPOSITORY_ROOT, REPOSITORY_ROOT / "dist")
    with pytest.raises(public_release.ReleaseError, match="outside"):
        public_release.validate_output_directory(REPOSITORY_ROOT, REPOSITORY_ROOT / "another-output")

    output = tmp_path / "release-output"
    output.mkdir()
    (output / "stale.whl").write_text("stale")
    with pytest.raises(public_release.ReleaseError, match="not empty"):
        public_release.validate_output_directory(REPOSITORY_ROOT, output)


def test_profile_payload_is_deterministic(tmp_path):
    first = public_release._profiles_payload(CONTRACT_PATH, all_profiles=True, names=[])
    second = public_release._profiles_payload(CONTRACT_PATH, all_profiles=True, names=[])

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_bytes(public_release.canonical_json(first))
    second_path.write_bytes(public_release.canonical_json(second))

    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text())["profiles"] == [
        profile.as_dict() for profile in public_release.exhaustive_profiles(_contract())
    ]


def test_installed_wheel_harness_is_loopback_only():
    environment = _server_environment(
        "http://127.0.0.1:43210",
        "/tmp/hermetic-release-home",
        unsafe_stdout_override=False,
    )

    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["no_proxy"] == "127.0.0.1,localhost"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert environment[name] == "http://127.0.0.1:9"


def test_attestation_rejects_profile_evidence_from_another_source(monkeypatch, tmp_path):
    contract = _contract()
    version = "0.4.0"
    wheel = tmp_path / "wandb_mcp_server-0.4.0-py3-none-any.whl"
    sdist = tmp_path / "wandb_mcp_server-0.4.0.tar.gz"
    sbom = tmp_path / "wandb_mcp_server-0.4.0.spdx.json"
    vulnerabilities = tmp_path / "wandb_mcp_server-0.4.0.vulnerabilities.json"
    wheel.write_text("wheel")
    sdist.write_text("sdist")
    sbom.write_text(json.dumps({"spdxVersion": "SPDX-2.3", "SPDXID": "SPDXRef-DOCUMENT", "packages": []}))
    vulnerabilities.write_text(json.dumps({"descriptor": {"name": "grype"}, "matches": []}))

    source = {"sha": "expected-source", "tree": "tree"}
    policy = public_release.policy_identity(REPOSITORY_ROOT, CONTRACT_PATH, contract)
    build_records = [
        {"name": path.name, "sha256": public_release.sha256_file(path), "size": path.stat().st_size}
        for path in sorted((wheel, sdist))
    ]
    build_manifest = tmp_path / "build-manifest.json"
    build_manifest.write_bytes(
        public_release.canonical_json(
            {
                "schema_version": 1,
                "version": version,
                "source": source,
                "policy": policy,
                "artifacts": build_records,
                "status": "built",
                "duration_ms": 1,
            }
        )
    )
    build_checksums = tmp_path / "SHA256SUMS"
    build_checksums.write_text("".join(f"{record['sha256']}  {record['name']}\n" for record in build_records))
    parent_evidence = tmp_path / "candidate-pr-evidence.json"
    issued_at = datetime.now(timezone.utc)
    parent_evidence.write_bytes(
        public_release.canonical_json(
            {
                "schema_version": 1,
                "gate": "candidate-pr",
                "status": "passed",
                "merge_sha": source["sha"],
                "merge_tree": source["tree"],
                "pull_request": 999,
                "reviewed_head_sha": "reviewed",
                "reviewed_tree": source["tree"],
                "review_decision": "APPROVED",
                "unresolved_conversations": 0,
                "required_checks": {name: "SUCCESS" for name in contract["required_pr_checks"]},
                "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
                "expires_at": (issued_at + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                "duration_ms": 1,
            }
        )
    )

    evidence_paths = []
    for python_version in contract["python_versions"]:
        evidence = {
            "schema_version": 1,
            "status": "passed",
            "version": version,
            "source_sha": "wrong-source",
            "wheel_sha256": public_release.sha256_file(wheel),
            "contract_sha256": public_release.sha256_file(CONTRACT_PATH),
            "harness_sha256": public_release.sha256_file(REPOSITORY_ROOT / "scripts" / "mcp_stdio_smoke.py"),
            "python_version": python_version,
            "mcp_version": contract["mcp"]["locked"],
            "locked_runtime": contract["locked_runtime"],
            "profiles": [profile.as_dict() for profile in public_release.exhaustive_profiles(contract)],
            "duration_ms": 1,
        }
        evidence_path = tmp_path / f"profiles-{python_version}.json"
        evidence_path.write_bytes(public_release.canonical_json(evidence))
        evidence_paths.append(evidence_path)

    monkeypatch.setattr(public_release, "verify_clean_tree", lambda _root: None)
    monkeypatch.setattr(public_release, "source_identity", lambda _root: source)
    with pytest.raises(public_release.ReleaseError, match="source SHA"):
        public_release.create_attestation(
            REPOSITORY_ROOT,
            CONTRACT_PATH,
            version,
            evidence_paths,
            (wheel, sdist, sbom, vulnerabilities),
            build_manifest,
            build_checksums,
            parent_evidence,
        )


def test_source_release_workflow_is_tag_only_pinned_and_draft():
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "release-source.yml").read_text()

    assert "workflow_dispatch" not in workflow
    assert 'tags:\n      - "v*.*.*"' in workflow
    assert "--tag-verification github" in workflow
    assert "--draft" in workflow
    assert "--verify-tag" in workflow
    assert "pypi" not in workflow.lower()
    assert "group: public-source-release" in workflow
    assert "--clobber" not in workflow
    assert "candidate-pr-evidence.json" in workflow
    assert "only-fixed" not in workflow
    assert "--predicate-type https://spdx.dev/Document/v2.3" in workflow
    assert "--predicate-type https://wandb.ai/attestations/mcp-public-release/v1" in workflow
    assert "uv export --frozen --no-dev --no-emit-project" in workflow
    assert 'uv pip install --python "$TEST_ENV/bin/python" --no-deps "$WHEEL"' in workflow

    unpinned_actions = []
    for line in workflow.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.split("uses:", 1)[1].strip().split()[0]
        revision = reference.rsplit("@", 1)[-1]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            unpinned_actions.append(reference)
    assert not unpinned_actions


def test_workflows_do_not_reference_pat_secrets():
    secret_reference = re.compile(r"\bsecrets\.([A-Za-z0-9_]+)")
    pat_segment = re.compile(r"(?:^|_)PAT(?:_|$)", re.IGNORECASE)
    violations: dict[str, list[str]] = {}

    for workflow in sorted((REPOSITORY_ROOT / ".github" / "workflows").glob("*.y*ml")):
        secret_names = {
            match.group(1)
            for match in secret_reference.finditer(workflow.read_text())
            if pat_segment.search(match.group(1))
        }
        if secret_names:
            violations[workflow.name] = sorted(secret_names)

    assert not violations, f"workflow PAT secret references are forbidden: {violations}"


def test_generated_feature_documentation_is_current():
    result = public_release.update_generated_docs(CONTRACT_PATH, REPOSITORY_ROOT / "README.md", check=True)

    assert result["status"] == "passed"
    assert result["changed"] is False
    readme = (REPOSITORY_ROOT / "README.md").read_text()
    assert "Generated by scripts/public_release.py docs" in readme
    for name in _contract()["named_profiles"]:
        assert f"| `{name}` |" in readme


def test_vulnerability_evidence_rejects_high_or_critical_findings(monkeypatch, tmp_path):
    contract = _contract()
    version = "0.4.0"
    wheel = tmp_path / "wandb_mcp_server-0.4.0-py3-none-any.whl"
    sdist = tmp_path / "wandb_mcp_server-0.4.0.tar.gz"
    sbom = tmp_path / "wandb_mcp_server-0.4.0.spdx.json"
    vulnerabilities = tmp_path / "wandb_mcp_server-0.4.0.vulnerabilities.json"
    wheel.write_text("wheel")
    sdist.write_text("sdist")
    sbom.write_text(json.dumps({"spdxVersion": "SPDX-2.3", "SPDXID": "SPDXRef-DOCUMENT", "packages": []}))
    vulnerabilities.write_text(
        json.dumps(
            {
                "descriptor": {"name": "grype"},
                "matches": [{"vulnerability": {"id": "CVE-test", "severity": "High"}}],
            }
        )
    )
    source = {"sha": "source", "tree": "tree"}
    records = [
        {"name": path.name, "sha256": public_release.sha256_file(path), "size": path.stat().st_size}
        for path in sorted((wheel, sdist))
    ]
    manifest = tmp_path / "build-manifest.json"
    manifest.write_bytes(
        public_release.canonical_json(
            {
                "schema_version": 1,
                "version": version,
                "source": source,
                "policy": public_release.policy_identity(REPOSITORY_ROOT, CONTRACT_PATH, contract),
                "artifacts": records,
                "status": "built",
                "duration_ms": 1,
            }
        )
    )
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text("".join(f"{record['sha256']}  {record['name']}\n" for record in records))
    monkeypatch.setattr(public_release, "source_identity", lambda _root: source)

    with pytest.raises(public_release.ReleaseError, match="High/Critical"):
        public_release.validate_release_artifacts(
            REPOSITORY_ROOT,
            CONTRACT_PATH,
            version,
            (wheel, sdist, sbom, vulnerabilities),
            manifest,
            checksums,
        )

    checksums.write_text("".join(f"{'0' * 64}  {record['name']}\n" for record in records))
    with pytest.raises(public_release.ReleaseError, match="build checksums"):
        public_release.validate_release_artifacts(
            REPOSITORY_ROOT,
            CONTRACT_PATH,
            version,
            (wheel, sdist, sbom, vulnerabilities),
            manifest,
            checksums,
        )


def test_candidate_gate_evidence_rejects_expiry_and_wrong_source(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    evidence = {
        "schema_version": 1,
        "gate": "candidate-pr",
        "status": "passed",
        "merge_sha": "source",
        "merge_tree": "tree",
        "pull_request": 1,
        "reviewed_head_sha": "head",
        "reviewed_tree": "tree",
        "review_decision": "APPROVED",
        "unresolved_conversations": 0,
        "required_checks": {name: "SUCCESS" for name in _contract()["required_pr_checks"]},
        "issued_at": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "expires_at": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "duration_ms": 1,
    }
    path = tmp_path / "candidate.json"
    path.write_bytes(public_release.canonical_json(evidence))
    monkeypatch.setattr(public_release, "source_identity", lambda _root: {"sha": "source", "tree": "tree"})

    with pytest.raises(public_release.ReleaseError, match="currently valid"):
        public_release.validate_parent_evidence(REPOSITORY_ROOT, path)

    evidence["issued_at"] = now.isoformat().replace("+00:00", "Z")
    evidence["expires_at"] = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    evidence["merge_sha"] = "wrong"
    path.write_bytes(public_release.canonical_json(evidence))
    with pytest.raises(public_release.ReleaseError, match="does not qualify"):
        public_release.validate_parent_evidence(REPOSITORY_ROOT, path)


def test_source_gate_binds_approved_pr_tree_and_required_checks(monkeypatch):
    contract = _contract()
    merge_sha = "a" * 40
    head_sha = "b" * 40
    first_parent = "c" * 40
    tree = "d" * 40

    def fake_git(_root, *arguments, **_kwargs):
        if arguments == ("rev-parse", "HEAD"):
            return merge_sha
        if arguments == ("rev-list", "--parents", "-n", "1", merge_sha):
            return f"{merge_sha} {first_parent} {head_sha}"
        if arguments in {
            ("rev-parse", f"{merge_sha}^{{tree}}"),
            ("rev-parse", f"{head_sha}^{{tree}}"),
        }:
            return tree
        raise AssertionError(arguments)

    pulls = [
        {
            "number": 987,
            "base": {"ref": "main"},
            "merged_at": "2026-08-13T00:00:00Z",
            "merge_commit_sha": merge_sha,
            "head": {"sha": head_sha},
        }
    ]
    contexts = [
        {"__typename": "CheckRun", "name": name, "status": "COMPLETED", "conclusion": "SUCCESS"}
        for name in contract["required_pr_checks"]
    ]
    graphql = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewDecision": "APPROVED",
                    "reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}},
                    "commits": {
                        "nodes": [
                            {
                                "commit": {
                                    "oid": head_sha,
                                    "statusCheckRollup": {
                                        "state": "SUCCESS",
                                        "contexts": {"nodes": contexts, "pageInfo": {"hasNextPage": False}},
                                    },
                                }
                            }
                        ]
                    },
                }
            }
        }
    }
    responses = iter((pulls, graphql))
    monkeypatch.setattr(public_release, "_git", fake_git)
    monkeypatch.setattr(public_release, "_json_command", lambda _command, _root: next(responses))

    evidence = public_release.verify_source_pr_gate(
        REPOSITORY_ROOT,
        CONTRACT_PATH,
        "wandb/wandb-mcp-server",
        merge_sha,
    )

    assert evidence["pull_request"] == 987
    assert evidence["reviewed_head_sha"] == head_sha
    assert evidence["reviewed_tree"] == evidence["merge_tree"] == tree
    assert set(evidence["required_checks"]) == set(contract["required_pr_checks"])
    assert evidence["expires_at"] > evidence["issued_at"]


def test_release_build_uses_hashed_build_inputs(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(public_release, "verify_version", lambda _root, _version: {})
    monkeypatch.setattr(public_release, "verify_clean_tree", lambda _root: None)
    monkeypatch.setattr(public_release, "source_identity", lambda _root: {"sha": "source", "tree": "tree"})

    class Result:
        returncode = 0

    def fake_run(command, **_kwargs):
        commands.append(command)
        wheel = tmp_path / "artifacts" / "wandb_mcp_server-9.8.7-py3-none-any.whl"
        sdist = tmp_path / "artifacts" / "wandb_mcp_server-9.8.7.tar.gz"
        wheel.write_text("wheel")
        sdist.write_text("sdist")
        return Result()

    monkeypatch.setattr(public_release.subprocess, "run", fake_run)
    output = tmp_path / "artifacts"
    public_release.build_artifacts(REPOSITORY_ROOT, CONTRACT_PATH, "9.8.7", output)

    assert commands == [
        [
            "uv",
            "build",
            "--build-constraints",
            str(REPOSITORY_ROOT / "release" / "build-requirements.lock"),
            "--require-hashes",
            "--no-create-gitignore",
            "--out-dir",
            str(output),
        ]
    ]

    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    assert 'requires = ["hatchling==1.27.0"]' in pyproject


def test_official_release_channels_exclude_pypi():
    releasing = (REPOSITORY_ROOT / "RELEASING.md").read_text()
    normalized = " ".join(releasing.split())

    assert "signed source tag" in releasing
    assert "GitHub Release" in releasing
    assert "immutable container digest" in releasing
    assert "PyPI and mutable container tags are not release channels" in normalized
