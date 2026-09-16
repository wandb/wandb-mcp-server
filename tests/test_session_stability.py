"""Session lifecycle regressions; all actors and signing keys are synthetic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import hmac
from threading import Barrier
from unittest.mock import patch

import pytest

from wandb_mcp_server import session_manager as sessions


@pytest.fixture(autouse=True)
def no_background_cleanup():
    with patch.object(sessions.MultiTenantSessionManager, "_start_cleanup_task"):
        yield


def manager(*, limit=2, key=b"synthetic-session-signing-key"):
    return sessions.MultiTenantSessionManager(
        session_ttl_seconds=60,
        max_sessions_per_key=limit,
        enable_hmac_sha256_sessions=key is not None,
        hmac_sha256_key=key,
    )


def test_capacity_never_evicts_active_requests():
    mgr = manager()
    first = mgr.create_session("actor-a")
    second = mgr.create_session("actor-a")
    mgr.start_request(first, "request-one")
    mgr.start_request(second, "request-two")
    with pytest.raises(sessions.SessionCapacityError):
        mgr.create_session("actor-a")
    assert mgr.get_stats()["active_requests"] == 2
    assert mgr.get_session(first) is not None
    assert mgr.get_session(second) is not None


def test_capacity_evicts_only_idle_even_if_active_is_older():
    mgr = manager()
    active = mgr.create_session("actor-a")
    idle = mgr.create_session("actor-a")
    mgr.start_request(active, "request-one")
    mgr.get_session(active).last_accessed -= timedelta(hours=1)
    replacement = mgr.create_session("actor-a")
    assert mgr.get_session(active) is not None
    assert mgr.get_session(idle) is None
    assert mgr.get_session(replacement) is not None


def test_request_completion_starts_idle_ttl_and_is_idempotent():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    mgr.start_request(sid, "request-one")
    state = mgr.get_session(sid)
    state.last_accessed -= timedelta(hours=1)
    mgr.end_request(sid, "request-one")
    completed = state.last_accessed
    assert completed > datetime.now() - timedelta(seconds=1)
    mgr.end_request(sid, "request-one")
    assert state.last_accessed == completed
    mgr._cleanup_expired_sessions()
    assert mgr.get_session(sid) is state


def test_warm_validation_rechecks_signature_and_verification_key():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    mgr._hmac_sha256_key = b"different-verification-key"
    # Preserve the cached actor hash to isolate signature verification.
    mgr.get_session(sid).api_key_hash = mgr._hash_api_key("actor-a")
    assert not mgr.validate_session(sid, "actor-a")
    mgr._hmac_sha256_key = None
    mgr.get_session(sid).api_key_hash = mgr._hash_api_key("actor-a")
    assert not mgr.validate_session(sid, "actor-a")


@pytest.mark.parametrize("cache_present", [True, False])
def test_old_signed_session_acquires_same_id_and_tracks_request(cache_present):
    mgr = manager()
    with patch.object(sessions.time, "time", return_value=1000):
        sid = mgr.create_session("actor-a")
    if not cache_present:
        mgr = manager()
    with patch.object(sessions.time, "time", return_value=100_000):
        actual, created = mgr.acquire_request("actor-a", "request-one", session_id=sid)
    assert actual == sid
    assert created is not cache_present
    assert mgr.get_session(sid).active_requests == {"request-one"}


def test_acquire_restores_after_idle_cache_cleanup():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    mgr.get_session(sid).last_accessed -= timedelta(hours=2)
    mgr._cleanup_expired_sessions()
    assert mgr.get_session(sid) is None
    assert mgr.acquire_request("actor-a", "request-one", session_id=sid) == (sid, True)


@pytest.mark.parametrize("cache_present", [True, False])
def test_foreign_actor_rejected_without_request_or_mapping_leaks(cache_present):
    issuer = manager()
    sid = issuer.create_session("actor-a")
    mgr = issuer if cache_present else manager()
    baseline = mgr.get_stats()
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.acquire_request("actor-b", "request-one", session_id=sid)
    assert exc.value.reason == "session_actor_mismatch"
    assert mgr.get_stats() == baseline


@pytest.mark.parametrize("cache_present", [True, False])
def test_future_issued_session_rejected_warm_and_cold(cache_present):
    issuer = manager()
    with patch.object(sessions.time, "time", return_value=10_000):
        sid = issuer.create_session("actor-a")
    mgr = issuer if cache_present else manager()
    with patch.object(sessions.time, "time", return_value=1000):
        with pytest.raises(sessions.SessionValidationError) as exc:
            mgr.acquire_request("actor-a", "request-one", session_id=sid)
    assert exc.value.reason == "session_future"
    assert mgr.get_stats()["active_requests"] == 0


@pytest.mark.parametrize(
    "bad_session", ["sess2_", "sess2_not-signed", "sess2_" + "A" * 10_000], ids=["empty", "invalid", "oversized"]
)
def test_malformed_session_rejected_without_state(bad_session):
    mgr = manager()
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.acquire_request("actor-a", "request-one", session_id=bad_session)
    assert exc.value.reason == "session_invalid"
    assert mgr.get_stats()["total_sessions"] == 0
    assert mgr.get_stats()["unique_api_keys"] == 0


def test_absent_signed_delete_is_idempotent_without_creation():
    issuer = manager()
    sid = issuer.create_session("actor-a")
    mgr = manager()
    assert mgr.cleanup_authorized_session(sid, "actor-a") is None
    assert mgr.get_stats()["total_sessions"] == 0
    assert mgr.get_stats()["unique_api_keys"] == 0
    with pytest.raises(sessions.SessionValidationError):
        mgr.cleanup_authorized_session(sid, "actor-b")


def test_absent_unsigned_delete_cannot_adopt_actor():
    mgr = manager()
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.cleanup_authorized_session("sess_legacy", "actor-a")
    assert exc.value.reason == "session_unknown"
    assert mgr.get_stats()["total_sessions"] == 0


@pytest.mark.parametrize("signed", [True, False])
def test_delete_requires_owner_and_idle_session(signed):
    mgr = manager(key=b"synthetic-session-signing-key" if signed else None)
    sid, _ = mgr.acquire_request("actor-a", "request-one")
    with pytest.raises(sessions.SessionValidationError):
        mgr.cleanup_authorized_session(sid, "actor-b")
    with pytest.raises(sessions.SessionBusyError):
        mgr.cleanup_authorized_session(sid, "actor-a")
    assert mgr.get_session(sid).active_requests == {"request-one"}
    mgr.end_request(sid, "request-one")
    mgr.cleanup_authorized_session(sid, "actor-a")
    assert mgr.get_session(sid) is None
    assert mgr.get_stats()["unique_api_keys"] == 0


def test_capacity_failure_does_not_create_empty_actor_mapping():
    mgr = manager(limit=0)
    with pytest.raises(sessions.SessionCapacityError):
        mgr.acquire_request("actor-a", "request-one")
    assert mgr.get_stats()["unique_api_keys"] == 0


def test_concurrent_acquisition_is_atomic_with_eviction():
    mgr = manager(limit=1)

    def acquire(index):
        try:
            return mgr.acquire_request("actor-a", f"request-{index}")[0]
        except sessions.SessionCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        acquired = list(pool.map(acquire, range(24)))
    successful = [sid for sid in acquired if sid is not None]
    assert len(successful) == 1
    assert mgr.get_stats()["active_requests"] == 1
    assert mgr.get_session(successful[0]) is not None


def test_concurrent_same_session_acquisition_preserves_every_request():
    mgr = manager(limit=1)
    sid = mgr.create_session("actor-a")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: mgr.acquire_request("actor-a", f"request-{i}", session_id=sid), range(24)))
    assert all(result == (sid, False) for result in results)
    assert mgr.get_stats()["active_requests"] == 24
    with pytest.raises(sessions.SessionBusyError):
        mgr.cleanup_authorized_session(sid, "actor-a")


def test_missing_signing_key_rejects_restoration_without_state():
    sid = manager().create_session("actor-a")
    mgr = manager(key=None)
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.acquire_request("actor-a", "request-one", session_id=sid)
    assert exc.value.reason == "session_unverifiable"
    assert mgr.get_stats()["total_sessions"] == 0


def test_tampered_signature_rejects_acquisition_and_delete():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    tampered = sid[:-1] + ("A" if sid[-1] != "A" else "B")
    for operation in (
        lambda: mgr.acquire_request("actor-a", "request-one", session_id=tampered),
        lambda: mgr.cleanup_authorized_session(tampered, "actor-a"),
    ):
        with pytest.raises(sessions.SessionValidationError) as exc:
            operation()
        assert exc.value.reason == "session_signature"
    assert mgr.get_session(sid) is not None
    assert mgr.get_stats()["active_requests"] == 0


def signed_payload(mgr, payload):
    signature = hmac.new(mgr._hmac_sha256_key, payload, hashlib.sha256).digest()[:12]
    return f"sess2_{mgr._urlsafe_encode(payload)}_{mgr._urlsafe_encode(signature)}"


@pytest.mark.parametrize("field,value", [(0, "v2"), (1, "-1"), (2, "bad"), (3, "bad"), (4, "xx"), (5, "xx"), (6, "x")])
def test_even_signed_malformed_payloads_fail_closed(field, value):
    mgr = manager()
    fields = ["v1", "3e8", "a" * 16, mgr._hash_api_key("actor-a")[:24], "c", "c", "p"]
    fields[field] = value
    sid = signed_payload(mgr, "|".join(fields).encode("ascii"))
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.acquire_request("actor-a", "request-one", session_id=sid)
    assert exc.value.reason == "session_invalid"
    assert mgr.get_stats()["total_sessions"] == 0


def test_existing_six_field_signed_format_still_restores():
    mgr = manager()
    payload = f"v1|3e8|{'a' * 16}|{mgr._hash_api_key('actor-a')[:24]}|c|c".encode("ascii")
    sid = signed_payload(mgr, payload)
    assert mgr.acquire_request("actor-a", "request-one", session_id=sid) == (sid, True)
    assert mgr.get_session(sid).metadata["session_event_emitted"] is False


def test_noncanonical_base64_encoding_is_not_an_alias():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    encoded = sid[len("sess2_") :]
    alias = "sess2_" + encoded[:-17] + "=_" + encoded[-16:]
    with pytest.raises(sessions.SessionValidationError) as exc:
        mgr.acquire_request("actor-a", "request-one", session_id=alias)
    assert exc.value.reason == "session_invalid"
    assert mgr.get_stats()["total_sessions"] == 1


def test_concurrent_delete_cannot_remove_an_acquired_request():
    mgr = manager()
    sid = mgr.create_session("actor-a")
    for index in range(20):
        barrier = Barrier(2)

        def acquire():
            barrier.wait()
            return mgr.acquire_request("actor-a", f"request-{index}", session_id=sid)

        def delete():
            barrier.wait()
            try:
                mgr.cleanup_authorized_session(sid, "actor-a")
            except sessions.SessionBusyError:
                pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            acquired = pool.submit(acquire)
            deleted = pool.submit(delete)
            assert acquired.result()[0] == sid
            deleted.result()
        assert mgr.get_session(sid).active_requests == {f"request-{index}"}
        mgr.end_request(sid, f"request-{index}")


def test_validation_reasons_do_not_echo_untrusted_details():
    rejected = sessions.SessionValidationError("secret-credential-canary")
    assert rejected.reason == "session_invalid"
    assert "secret-credential-canary" not in str(rejected)
