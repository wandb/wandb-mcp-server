"""Exercise secret failures without credentials or an installed cloud SDK."""

from concurrent.futures import ThreadPoolExecutor
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wandb_mcp_server import secrets_resolver as secrets
from wandb_mcp_server import session_manager as sessions


@pytest.fixture
def cloud(monkeypatch):
    errors = ModuleType("google.api_core.exceptions")
    for name in (
        "DeadlineExceeded",
        "ServiceUnavailable",
        "TooManyRequests",
        "PermissionDenied",
        "Unauthenticated",
        "NotFound",
        "FailedPrecondition",
    ):
        setattr(errors, name, type(name, (Exception,), {}))
    api_core = ModuleType("google.api_core")
    api_core.exceptions = errors
    secretmanager = ModuleType("google.cloud.secretmanager")
    client = MagicMock()
    client.__enter__.return_value = client
    secretmanager.SecretManagerServiceClient = MagicMock(return_value=client)
    google_cloud = ModuleType("google.cloud")
    google_cloud.secretmanager = secretmanager
    for name, module in {
        "google.api_core": api_core,
        "google.api_core.exceptions": errors,
        "google.cloud": google_cloud,
        "google.cloud.secretmanager": secretmanager,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    client.access_secret_version.return_value = SimpleNamespace(payload=SimpleNamespace(data=b"fixture-key"))
    return client, errors


@pytest.mark.parametrize("version", ["latest", "7"])
def test_exact_version_and_explicit_rpc_budget(cloud, version):
    client, _ = cloud
    assert secrets.SecretsResolver("gcp", "fixture-project").fetch_secret("fixture", version=version) == b"fixture-key"
    client.access_secret_version.assert_called_once_with(
        name=f"projects/fixture-project/secrets/fixture/versions/{version}", retry=None, timeout=3.0
    )
    client.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "error,reason,calls",
    [
        ("PermissionDenied", "access_denied", 1),
        ("Unauthenticated", "access_denied", 1),
        ("NotFound", "not_found", 1),
        ("FailedPrecondition", "disabled", 1),
        ("DeadlineExceeded", "timeout", 2),
        ("ServiceUnavailable", "unavailable", 2),
        ("TooManyRequests", "unavailable", 2),
    ],
)
def test_failures_are_bounded_and_sanitized(cloud, caplog, error, reason, calls):
    client, errors = cloud
    client.access_secret_version.side_effect = getattr(errors, error)("PRIVATE_CREDENTIAL_CANARY")
    with pytest.raises(secrets.SecretResolutionError) as caught:
        secrets.SecretsResolver("gcp", "PRIVATE_PROJECT_CANARY").fetch_secret("PRIVATE_SECRET_CANARY")
    assert caught.value.reason == reason
    assert client.access_secret_version.call_count == calls
    assert "CANARY" not in str(caught.value) + caplog.text


def test_transient_failure_can_recover(cloud):
    client, errors = cloud
    client.access_secret_version.side_effect = [
        errors.ServiceUnavailable("private"),
        SimpleNamespace(payload=SimpleNamespace(data=b"key")),
    ]
    assert secrets.SecretsResolver("gcp", "fixture").fetch_secret("fixture") == b"key"
    assert client.access_secret_version.call_count == 2


@pytest.mark.parametrize("payload", [b"", None, "not-bytes"])
def test_empty_or_invalid_payload_is_unavailable(cloud, payload):
    client, _ = cloud
    client.access_secret_version.return_value.payload.data = payload
    with pytest.raises(secrets.SecretResolutionError, match="empty"):
        secrets.SecretsResolver("gcp", "fixture").fetch_secret("fixture")
    assert client.access_secret_version.call_count == 1


@pytest.mark.parametrize(
    "secret_id,version", [("../private", "1"), ("", "1"), ("valid", "0"), ("valid", ""), ("valid", "1/private")]
)
def test_invalid_reference_never_calls_provider(cloud, secret_id, version):
    client, _ = cloud
    with pytest.raises(secrets.SecretResolutionError, match="configuration"):
        secrets.SecretsResolver("gcp", "fixture").fetch_secret(secret_id, version=version)
    client.access_secret_version.assert_not_called()


@pytest.fixture
def session_dependency(monkeypatch):
    import wandb_mcp_server.config as config

    sessions.reset_session_manager()
    resolver = MagicMock()
    monkeypatch.setattr(sessions, "get_secrets_resolver_from_env", lambda: resolver)
    monkeypatch.setattr(sessions.MultiTenantSessionManager, "_start_cleanup_task", lambda self: None)
    monkeypatch.setattr(config, "MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS", True)
    yield resolver
    sessions.reset_session_manager()


def test_sessions_use_pinned_reference_and_cache_key(session_dependency, monkeypatch):
    monkeypatch.setenv("MCP_SERVER_HMAC_SECRET_ID", "fixture-session")
    monkeypatch.setenv("MCP_SERVER_HMAC_SECRET_VERSION", "7")
    session_dependency.fetch_secret.return_value = b"fixture-key"
    with ThreadPoolExecutor(max_workers=8) as executor:
        managers = list(executor.map(lambda _: sessions.get_session_manager(), range(20)))
    assert all(manager is managers[0] for manager in managers)
    session_dependency.fetch_secret.assert_called_once_with("fixture-session", version="7")


def test_session_failure_cooldown_and_recovery(session_dependency, monkeypatch, caplog):
    clock = [100.0]
    monkeypatch.setattr(sessions.time, "monotonic", lambda: clock[0])
    session_dependency.fetch_secret.side_effect = secrets.SecretResolutionError("access_denied")
    for _ in range(3):
        with pytest.raises(sessions.SessionInitializationError) as caught:
            sessions.get_session_manager()
        assert caught.value.reason == "access_denied"
    assert session_dependency.fetch_secret.call_count == 1
    clock[0] = 105.0
    session_dependency.fetch_secret.side_effect = None
    session_dependency.fetch_secret.return_value = b"new-key"
    assert sessions.get_session_manager()._enable_hmac_sha256_sessions
    assert session_dependency.fetch_secret.call_count == 2


def test_unexpected_initialization_error_is_sanitized(session_dependency, caplog):
    session_dependency.fetch_secret.side_effect = RuntimeError("PRIVATE_CREDENTIAL_CANARY")
    with pytest.raises(sessions.SessionInitializationError) as caught:
        sessions.get_session_manager()
    assert "CANARY" not in str(caught.value) + caplog.text


def test_rotation_rejects_old_sessions_and_accepts_new_across_workers(session_dependency):
    def manager(key):
        return sessions.MultiTenantSessionManager(enable_hmac_sha256_sessions=True, hmac_sha256_key=key)

    old, new, peer = manager(b"old-key"), manager(b"new-key"), manager(b"new-key")
    old_id = old.create_session("fixture-api-key")
    with pytest.raises(sessions.SessionValidationError, match="signature"):
        new.restore_portable_session(old_id, "fixture-api-key")
    new_id = new.create_session("fixture-api-key")
    peer.restore_portable_session(new_id, "fixture-api-key")


def test_datadog_reference_is_configurable_and_failure_nonfatal(monkeypatch, session_dependency):
    from wandb_mcp_server.analytics_datadog import _resolve_dd_api_key

    monkeypatch.delenv("DD_API_KEY", raising=False)
    monkeypatch.setattr(secrets, "get_secrets_resolver_from_env", lambda: session_dependency)
    monkeypatch.setenv("MCP_DATADOG_SECRET_ID", "fixture-telemetry")
    monkeypatch.setenv("MCP_DATADOG_SECRET_VERSION", "9")
    session_dependency.fetch_secret.return_value = b"fixture-key"
    assert _resolve_dd_api_key() == "fixture-key"
    session_dependency.fetch_secret.assert_called_once_with("fixture-telemetry", version="9")
    session_dependency.fetch_secret.side_effect = secrets.SecretResolutionError("access_denied")
    assert _resolve_dd_api_key() == ""
