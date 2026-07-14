"""Portable session IDs preserve safe harness context across workers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wandb_mcp_server.session_manager import MultiTenantSessionManager


def _manager(*, ttl: int = 3600) -> MultiTenantSessionManager:
    return MultiTenantSessionManager(
        session_ttl_seconds=ttl,
        enable_hmac_sha256_sessions=True,
        hmac_sha256_key=b"test-hmac-key-with-enough-entropy",
    )


def test_portable_session_restores_metadata_in_another_manager() -> None:
    first_worker = _manager()
    session_id = first_worker.create_session(
        "api-key-a",
        metadata={
            "agent_harness": "codex",
            "mcp_client_app": "codex",
            "mcp_protocol_version": "2025-06-18",
        },
    )

    second_worker = _manager()
    assert session_id.startswith("sess2_")
    assert len(session_id) < 128
    assert second_worker.create_session("api-key-a", session_id=session_id) == session_id
    restored = second_worker.get_session(session_id)
    assert restored is not None
    assert restored.metadata["agent_harness"] == "codex"
    assert restored.metadata["mcp_protocol_version"] == "2025-06-18"


def test_portable_session_rejects_tampering() -> None:
    manager = _manager()
    session_id = manager.create_session("api-key-a", metadata={"agent_harness": "cursor"})
    tampered = f"{session_id[:-1]}{'A' if session_id[-1] != 'A' else 'B'}"

    with pytest.raises(ValueError, match="signature"):
        _manager().restore_portable_session(tampered, "api-key-a")


def test_portable_session_rejects_another_api_key() -> None:
    manager = _manager()
    session_id = manager.create_session("api-key-a", metadata={"agent_harness": "cursor"})

    with pytest.raises(ValueError, match="API key mismatch"):
        _manager().restore_portable_session(session_id, "api-key-b")


def test_portable_session_expires() -> None:
    with patch("wandb_mcp_server.session_manager.time.time", return_value=1000):
        session_id = _manager(ttl=60).create_session(
            "api-key-a",
            metadata={"agent_harness": "claude_code"},
        )
    with patch("wandb_mcp_server.session_manager.time.time", return_value=1061):
        with pytest.raises(ValueError, match="expired"):
            _manager(ttl=60).restore_portable_session(session_id, "api-key-a")


def test_legacy_session_is_accepted_without_attribution() -> None:
    manager = _manager()
    legacy_id = "sess_0123456789abcdef"

    assert manager.restore_portable_session(legacy_id, "api-key-a") == {}
    manager.create_session(
        "api-key-a",
        session_id=legacy_id,
        metadata={"agent_harness": "codex"},
    )
    session = manager.get_session(legacy_id)
    assert session is not None
    assert session.metadata == {}


def test_non_hmac_manager_keeps_legacy_session_format() -> None:
    manager = MultiTenantSessionManager(enable_hmac_sha256_sessions=False)
    assert manager.create_session("api-key-a").startswith("sess_")
