"""
Session manager for multi-tenant API key isolation.

This module provides enhanced session management for multi-tenant environments,
ensuring complete isolation between concurrent requests with different API keys.

Key features:
- Session-based API key isolation
- Request tracking and auditing
- Automatic cleanup of expired sessions
- Validation to prevent cross-tenant leakage
"""

import base64
import hashlib
import hmac
import os
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any
import threading
import logging

from wandb_mcp_server.utils import get_rich_logger, get_session_prefix_from_session
from wandb_mcp_server.secrets_resolver import get_secrets_resolver_from_env

logger = get_rich_logger(__name__)

_PORTABLE_SESSION_PREFIX = "sess2_"
_HARNESS_CODES = {
    "unknown": "u",
    "codex": "c",
    "claude_code": "k",
    "claude_desktop": "d",
    "claude_ai": "a",
    "cursor": "r",
    "gemini_cli": "g",
    "lechat": "l",
    "linear": "n",
    "vscode": "v",
    "mcp_inspector": "i",
    "load_test": "t",
}
_HARNESSES_BY_CODE = {value: key for key, value in _HARNESS_CODES.items()}
_PROTOCOL_CODES = {
    "unknown": "u",
    "2024-11-05": "a",
    "2025-03-26": "b",
    "2025-06-18": "c",
}
_PROTOCOLS_BY_CODE = {value: key for key, value in _PROTOCOL_CODES.items()}


# Context variables for session management
current_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
current_api_key_hash: ContextVar[Optional[str]] = ContextVar("api_key_hash", default=None)


class SessionCapacityError(ValueError):
    """Raised when a key has reached its maximum number of concurrent sessions."""


@dataclass
class Session:
    """Represents an isolated session for a specific API key."""

    session_id: str
    api_key_hash: str  # Store hash for comparison
    created_at: datetime
    last_accessed: datetime
    request_count: int = 0
    active_requests: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self):
        """Update last access time and increment request count."""
        self.last_accessed = datetime.now()
        self.request_count += 1


class MultiTenantSessionManager:
    """
    Enhanced session manager for multi-tenant environments.

    Provides strong isolation between concurrent requests with different API keys.
    """

    def __init__(
        self,
        session_ttl_seconds: int = 3600,  # 1 hour default
        max_sessions_per_key: int = 10,
        enable_hmac_sha256_sessions: bool = False,
        hmac_sha256_key: Optional[bytes] = None,
    ):
        """
        Initialize the session manager.

        Args:
            session_ttl_seconds: Time-to-live for idle sessions
            max_sessions_per_key: Maximum concurrent sessions per API key
            enable_hmac_sha256_sessions: Use HMAC-SHA256 for session verification
        """
        self._sessions: Dict[str, Session] = {}
        self._api_key_sessions: Dict[str, Set[str]] = defaultdict(set)  # api_key_hash -> session_ids
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._session_ttl = session_ttl_seconds
        self._max_sessions_per_key = max_sessions_per_key
        self._enable_hmac_sha256_sessions = enable_hmac_sha256_sessions
        self._hmac_sha256_key: Optional[bytes] = None

        # Initialize HMAC-SHA256 key if enabled
        if self._enable_hmac_sha256_sessions and hmac_sha256_key:
            self._hmac_sha256_key = hmac_sha256_key
            logger.info("HMAC-SHA256 sessions enabled")
        elif self._enable_hmac_sha256_sessions:
            try:
                resolver = get_secrets_resolver_from_env()
                if resolver is None:
                    raise RuntimeError("SecretsResolver not configured but HMAC sessions enabled")
                key_bytes = resolver.fetch_secret("mcp-server-secret-hmac-key")
                if not key_bytes:
                    raise RuntimeError("Fetched empty HMAC key")
                self._hmac_sha256_key = key_bytes
                logger.info("HMAC-SHA256 sessions enabled")
            except Exception as e:
                logger.error("Failed to initialize HMAC-SHA256 sessions: %s", e)
                raise
        else:
            logger.warning(
                "MultiTenantSessionManager using plain SHA-256 hashing. For stronger security, set MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS=true and configure a secrets provider."
            )

        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

        logger.info(
            "MultiTenantSessionManager initialized (TTL: %ss, Max sessions/key: %s, HMAC sessions: %s)",
            session_ttl_seconds,
            max_sessions_per_key,
            self._enable_hmac_sha256_sessions,
        )

    def _hash_api_key(self, api_key: str) -> str:
        """Create a secure hash of the API key for storage."""
        api_key_bytes = api_key.encode()
        if self._hmac_sha256_key:
            return hmac.new(self._hmac_sha256_key, api_key_bytes, hashlib.sha256).hexdigest()
        return hashlib.sha256(api_key_bytes).hexdigest()

    @staticmethod
    def _urlsafe_encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _urlsafe_decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))

    def _portable_session_id(
        self,
        api_key_hash: str,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        if self._hmac_sha256_key is None:
            raise RuntimeError("portable sessions require an HMAC key")
        metadata = metadata or {}
        harness = str(metadata.get("agent_harness") or metadata.get("mcp_client_app") or "unknown").lower()
        protocol = str(metadata.get("mcp_protocol_version") or "unknown").lower()
        issued_at = format(int(time.time()), "x")
        payload = "|".join(
            (
                "v1",
                issued_at,
                uuid.uuid4().hex[:16],
                api_key_hash[:24],
                _HARNESS_CODES.get(harness, "u"),
                _PROTOCOL_CODES.get(protocol, "u"),
            )
        ).encode("ascii")
        signature = hmac.new(self._hmac_sha256_key, payload, hashlib.sha256).digest()[:12]
        return f"{_PORTABLE_SESSION_PREFIX}{self._urlsafe_encode(payload)}_{self._urlsafe_encode(signature)}"

    def restore_portable_session(self, session_id: str, api_key: str) -> Dict[str, Any]:
        """Verify a portable session and return its safe cross-worker metadata.

        Legacy ``sess_`` IDs return an empty mapping for one session TTL.  Their
        identity is intentionally not trusted because it was not signed.
        """
        if not session_id.startswith(_PORTABLE_SESSION_PREFIX):
            return {}
        if self._hmac_sha256_key is None:
            raise ValueError("Portable session cannot be verified")
        try:
            encoded = session_id[len(_PORTABLE_SESSION_PREFIX) :]
            # A 12-byte signature is always 16 unpadded base64url characters.
            # Parse by fixed width because '_' is itself part of base64url.
            if len(encoded) <= 17 or encoded[-17] != "_":
                raise ValueError("invalid portable session framing")
            encoded_payload = encoded[:-17]
            encoded_signature = encoded[-16:]
            payload = self._urlsafe_decode(encoded_payload)
            signature = self._urlsafe_decode(encoded_signature)
        except Exception as exc:
            raise ValueError("Malformed portable session") from exc
        expected = hmac.new(self._hmac_sha256_key, payload, hashlib.sha256).digest()[:12]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Portable session signature mismatch")
        try:
            version, issued_hex, _nonce, key_prefix, harness_code, protocol_code = payload.decode("ascii").split("|")
            issued_at = int(issued_hex, 16)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Malformed portable session payload") from exc
        if version != "v1":
            raise ValueError("Unsupported portable session version")
        now = int(time.time())
        if issued_at > now + 300 or now - issued_at > self._session_ttl:
            raise ValueError("Portable session expired")
        api_key_hash = self._hash_api_key(api_key)
        if not hmac.compare_digest(key_prefix, api_key_hash[:24]):
            raise ValueError("Portable session API key mismatch")
        harness = _HARNESSES_BY_CODE.get(harness_code, "unknown")
        protocol = _PROTOCOLS_BY_CODE.get(protocol_code, "unknown")
        return {
            "agent_harness": harness,
            "mcp_client_app": harness,
            "mcp_protocol_version": protocol,
            "mcp_client_source": "portable_session",
        }

    def create_session(
        self,
        api_key: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session for an API key.

        Args:
            api_key: The W&B API key
            session_id: Optional session ID (will generate if not provided)

        Returns:
            The session ID

        Raises:
            ValueError: If max sessions per key is exceeded
        """
        with self._lock:
            api_key_hash = self._hash_api_key(api_key)

            # Generate a signed, cross-worker session when HMAC is configured.
            generated_session = not session_id
            if not session_id:
                if self._hmac_sha256_key is not None:
                    session_id = self._portable_session_id(api_key_hash, metadata)
                else:
                    session_id = f"sess_{uuid.uuid4().hex}"

            restored_metadata: Dict[str, Any] = {}
            if session_id.startswith(_PORTABLE_SESSION_PREFIX):
                restored_metadata = self.restore_portable_session(session_id, api_key)
            elif generated_session and metadata:
                restored_metadata = dict(metadata)

            _session_prefix = get_session_prefix_from_session(session_id)
            _log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{_session_prefix}] " if _session_prefix else ""}
            )

            # Check if session already exists
            if session_id in self._sessions:
                session = self._sessions[session_id]
                # Validate API key matches
                if session.api_key_hash != api_key_hash:
                    _log.error("API key mismatch!")
                    raise ValueError("Session API key mismatch!")
                session.update_access()
                return session_id

            # Check max sessions per API key
            existing_sessions = self._api_key_sessions[api_key_hash]
            if len(existing_sessions) >= self._max_sessions_per_key:
                # Clean up old sessions for this key
                self._cleanup_api_key_sessions(api_key_hash)

                # Check again after cleanup
                if len(existing_sessions) >= self._max_sessions_per_key:
                    _log.warning(
                        f"Max sessions ({self._max_sessions_per_key}) reached for API key hash {api_key_hash[:8]}..."
                    )
                    raise SessionCapacityError(
                        f"Maximum concurrent sessions ({self._max_sessions_per_key}) exceeded for this API key"
                    )

            # Create new session
            session = Session(
                session_id=session_id,
                api_key_hash=api_key_hash,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                metadata=restored_metadata,
            )

            self._sessions[session_id] = session
            self._api_key_sessions[api_key_hash].add(session_id)

            _log.info("Created session %s...", get_session_prefix_from_session(session_id))
            return session_id

    def validate_session(self, session_id: str, api_key: str) -> bool:
        """
        Validate that a session ID matches the provided API key.

        Args:
            session_id: The session ID to validate
            api_key: The API key to validate against

        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            _session_prefix = get_session_prefix_from_session(session_id)
            _log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{_session_prefix}] " if _session_prefix else ""}
            )
            if session_id not in self._sessions:
                _log.warning("Session not found")
                return False

            session = self._sessions[session_id]
            api_key_hash = self._hash_api_key(api_key)

            if session.api_key_hash != api_key_hash:
                _log.error("API key validation failed!")
                return False

            # Update access time
            session.update_access()
            return True

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def start_request(self, session_id: str, request_id: str) -> bool:
        """
        Mark a request as active in a session.

        Args:
            session_id: The session ID
            request_id: Unique request identifier

        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            _session_prefix = get_session_prefix_from_session(session_id)
            _log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{_session_prefix}] " if _session_prefix else ""}
            )
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]
            session.active_requests.add(request_id)
            session.update_access()

            _log.debug("Started request %s...", get_session_prefix_from_session(request_id) or request_id)
            return True

    def end_request(self, session_id: str, request_id: str):
        """
        Mark a request as completed in a session.

        Args:
            session_id: The session ID
            request_id: Unique request identifier
        """
        with self._lock:
            _session_prefix = get_session_prefix_from_session(session_id)
            _log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{_session_prefix}] " if _session_prefix else ""}
            )
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.active_requests.discard(request_id)
                _log.debug("Ended request %s...", get_session_prefix_from_session(request_id) or request_id)

    def cleanup_session(self, session_id: str):
        """
        Explicitly cleanup a session.

        Args:
            session_id: The session ID to cleanup
        """
        with self._lock:
            _session_prefix = get_session_prefix_from_session(session_id)
            _log = logging.LoggerAdapter(
                logger, {"session_id_prefix": f"[{_session_prefix}] " if _session_prefix else ""}
            )
            if session_id not in self._sessions:
                return

            session = self._sessions[session_id]

            # Check for active requests
            if session.active_requests:
                _log.warning(
                    "Cleaning up session %s... with %s active requests",
                    get_session_prefix_from_session(session_id),
                    len(session.active_requests),
                )

            # Remove from api_key_sessions
            self._api_key_sessions[session.api_key_hash].discard(session_id)
            if not self._api_key_sessions[session.api_key_hash]:
                del self._api_key_sessions[session.api_key_hash]

            # Remove session
            del self._sessions[session_id]

            _log.info(
                "Cleaned up session %s... (requests: %s)",
                get_session_prefix_from_session(session_id),
                session.request_count,
            )

    def _cleanup_api_key_sessions(self, api_key_hash: str):
        """Cleanup old sessions for a specific API key."""
        with self._lock:
            session_ids = list(self._api_key_sessions[api_key_hash])

            # Sort by last accessed time
            sessions_by_age = sorted(
                [(sid, self._sessions[sid]) for sid in session_ids if sid in self._sessions],
                key=lambda x: x[1].last_accessed,
            )

            # Remove oldest sessions until we're under the limit
            while len(sessions_by_age) > self._max_sessions_per_key - 1:
                old_session_id, _ = sessions_by_age.pop(0)
                self.cleanup_session(old_session_id)

    def _cleanup_expired_sessions(self):
        """Cleanup expired sessions based on TTL."""
        with self._lock:
            now = datetime.now()
            ttl_delta = timedelta(seconds=self._session_ttl)

            expired_sessions = [
                sid
                for sid, session in self._sessions.items()
                if (now - session.last_accessed) > ttl_delta and not session.active_requests
            ]

            for session_id in expired_sessions:
                self.cleanup_session(session_id)

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def _start_cleanup_task(self):
        """Start the background cleanup task."""

        def cleanup_loop():
            while True:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        import threading

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        with self._lock:
            total_sessions = len(self._sessions)
            unique_api_keys = len(self._api_key_sessions)
            active_requests = sum(len(s.active_requests) for s in self._sessions.values())

            return {
                "total_sessions": total_sessions,
                "unique_api_keys": unique_api_keys,
                "active_requests": active_requests,
                "session_ttl_seconds": self._session_ttl,
                "max_sessions_per_key": self._max_sessions_per_key,
            }


# Global session manager instance
_session_manager: Optional[MultiTenantSessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> MultiTenantSessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is not None:
        return _session_manager
    with _session_manager_lock:
        if _session_manager is not None:
            return _session_manager
        try:
            ttl = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
        except ValueError:
            raise ValueError("SESSION_TTL_SECONDS must be an integer (got: %r)" % os.environ.get("SESSION_TTL_SECONDS"))
        try:
            max_sessions = int(os.environ.get("MAX_SESSIONS_PER_KEY", "10"))
        except ValueError:
            raise ValueError(
                "MAX_SESSIONS_PER_KEY must be an integer (got: %r)" % os.environ.get("MAX_SESSIONS_PER_KEY")
            )
        enable_hmac_sessions = os.environ.get("MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS", "false").lower() == "true"
        try:
            _session_manager = MultiTenantSessionManager(
                session_ttl_seconds=ttl,
                max_sessions_per_key=max_sessions,
                enable_hmac_sha256_sessions=enable_hmac_sessions,
            )
        except Exception as e:
            raise RuntimeError(f"Unable to initialize session manager: {e}") from e

    return _session_manager


def reset_session_manager():
    """Reset the global session manager (for testing)."""
    global _session_manager
    if _session_manager:
        # Cleanup all sessions
        with _session_manager._lock:
            session_ids = list(_session_manager._sessions.keys())
            for sid in session_ids:
                _session_manager.cleanup_session(sid)
    _session_manager = None
