"""MCP client harness classification for analytics.

The values produced here are intentionally low-cardinality and untrusted.
They are for product analytics and operational debugging only.
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping

_SAFE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_UNSAFE_CHARS_RE = re.compile(r"[^a-z0-9_.-]+")
_MAX_DEBUG_FIELD_LENGTH = 64

_DEFAULT_CLIENT_FAMILY = "unknown"
_DEFAULT_CLIENT_APP = "unknown"
_DEFAULT_CLIENT_SOURCE = "unknown"
_DEFAULT_CONFIDENCE = "low"

_DEBUG_FIELDS_ENV = "MCP_HARNESS_DEBUG_FIELDS"


@dataclass(frozen=True)
class HarnessContext:
    """Low-cardinality MCP client context for analytics events."""

    mcp_client_family: str = _DEFAULT_CLIENT_FAMILY
    mcp_client_app: str = _DEFAULT_CLIENT_APP
    mcp_client_source: str = _DEFAULT_CLIENT_SOURCE
    mcp_protocol_version: str = "unknown"
    mcp_jsonrpc_method: str = "unknown"
    mcp_client_name: str = ""
    mcp_client_version: str = ""
    mcp_user_agent_product: str = ""
    mcp_client_confidence: str = _DEFAULT_CONFIDENCE
    mcp_client_mismatch: str = ""

    def default_fields(self) -> dict[str, str]:
        """Return fields safe for default analytics sinks."""
        return {
            "mcp_client_family": self.mcp_client_family,
            "mcp_client_app": self.mcp_client_app,
            "mcp_client_source": self.mcp_client_source,
            "mcp_protocol_version": self.mcp_protocol_version,
            "mcp_jsonrpc_method": self.mcp_jsonrpc_method,
        }

    def debug_fields(self) -> dict[str, str]:
        """Return bounded fields useful for temporary classifier debugging."""
        fields = {
            "mcp_client_name": self.mcp_client_name,
            "mcp_client_version": self.mcp_client_version,
            "mcp_user_agent_product": self.mcp_user_agent_product,
            "mcp_client_confidence": self.mcp_client_confidence,
            "mcp_client_mismatch": self.mcp_client_mismatch,
        }
        return {key: value for key, value in fields.items() if value}

    def analytics_fields(
        self,
        *,
        include_debug: bool | None = None,
    ) -> dict[str, str]:
        """Return harness fields for analytics events."""
        fields = self.default_fields()
        if include_debug is None:
            include_debug = _env_bool(_DEBUG_FIELDS_ENV)
        if include_debug:
            fields.update(self.debug_fields())
        return fields

    def session_metadata(self) -> dict[str, str]:
        """Return normalized fields safe to persist in per-session metadata."""
        return self.default_fields()


current_harness_context: ContextVar[HarnessContext | None] = ContextVar(
    "mcp_harness_context",
    default=None,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_value(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    text = text.replace("/", ".")
    text = _UNSAFE_CHARS_RE.sub(".", text).strip(".")
    if not _SAFE_RE.match(text):
        return default
    return text[:_MAX_DEBUG_FIELD_LENGTH]


def _debug_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("/", ".")
    text = _UNSAFE_CHARS_RE.sub(".", text).strip(".")
    if not text:
        return ""
    return text[:_MAX_DEBUG_FIELD_LENGTH]


def _header(headers: Mapping[str, str] | None, name: str) -> str:
    if not headers:
        return ""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return str(value).strip()
    return ""


def _first_user_agent_product(user_agent: str) -> str:
    if not user_agent:
        return ""
    return _debug_value(user_agent.split()[0])


def _nested_dict(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    child = value.get(key)
    return child if isinstance(child, dict) else {}


def _client_info_from_meta(
    params: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    meta = _nested_dict(params, "_meta")
    client_info = meta.get("io.modelcontextprotocol/clientInfo")
    if isinstance(client_info, dict):
        return client_info, "meta_client_info"
    return {}, ""


def _protocol_from_meta(params: dict[str, Any]) -> str:
    meta = _nested_dict(params, "_meta")
    return _sanitize_value(meta.get("io.modelcontextprotocol/protocolVersion"))


def _classify_client(value: object) -> tuple[str, str]:
    text = str(value or "").lower()
    if not text:
        return _DEFAULT_CLIENT_FAMILY, _DEFAULT_CLIENT_APP
    if "wandb-mcp-load-test" in text or "compatibility-check" in text or "load_test" in text:
        return "load_test", "load_test"
    if "linear" in text:
        return "linear", "linear_agent"
    if "mistral" in text or "lechat" in text or "le-chat" in text:
        return "mistral", "lechat"
    if "claude-code" in text or ("claude" in text and "code" in text):
        return "claude", "claude_code"
    if "claude-desktop" in text or ("claude" in text and "desktop" in text):
        return "claude", "claude_desktop"
    if "claude" in text or "anthropic" in text:
        return "claude", "claude"
    if "codex" in text:
        return "openai", "codex_cli"
    if "openai" in text or "chatgpt" in text:
        return "openai", "openai_responses"
    if "cursor" in text:
        return "cursor", "cursor"
    if "gemini" in text:
        return "gemini", "gemini_cli"
    if "vscode" in text or "visual-studio-code" in text:
        return "vscode", "vscode"
    return _DEFAULT_CLIENT_FAMILY, _DEFAULT_CLIENT_APP


def _context_from_client_info(
    *,
    client_info: dict[str, Any],
    source: str,
    protocol_version: str,
    jsonrpc_method: str,
    user_agent_product: str,
) -> HarnessContext:
    name = _debug_value(client_info.get("name"))
    version = _debug_value(client_info.get("version"))
    family, app = _classify_client(name)
    return HarnessContext(
        mcp_client_family=family,
        mcp_client_app=app,
        mcp_client_source=source,
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=jsonrpc_method,
        mcp_client_name=name,
        mcp_client_version=version,
        mcp_user_agent_product=user_agent_product,
        mcp_client_confidence="high" if family != "unknown" else "low",
    )


def _context_from_session_metadata(
    *,
    session_metadata: Mapping[str, Any],
    protocol_version: str,
    jsonrpc_method: str,
    user_agent_product: str,
    mismatch: str = "",
) -> HarnessContext:
    return HarnessContext(
        mcp_client_family=_sanitize_value(session_metadata.get("mcp_client_family")),
        mcp_client_app=_sanitize_value(session_metadata.get("mcp_client_app")),
        mcp_client_source="session_metadata",
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=jsonrpc_method,
        mcp_user_agent_product=user_agent_product,
        mcp_client_confidence="high",
        mcp_client_mismatch=mismatch,
    )


def _context_from_user_agent(
    *,
    user_agent: str,
    protocol_version: str,
    jsonrpc_method: str,
) -> HarnessContext:
    product = _first_user_agent_product(user_agent)
    family, app = _classify_client(user_agent)
    return HarnessContext(
        mcp_client_family=family,
        mcp_client_app=app,
        mcp_client_source="user_agent" if user_agent else "unknown",
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=jsonrpc_method,
        mcp_user_agent_product=product,
        mcp_client_confidence="medium" if family != "unknown" else "low",
    )


def extract_harness_context(
    headers: Mapping[str, str] | None,
    body_json: Mapping[str, Any] | None = None,
    session_metadata: Mapping[str, Any] | None = None,
) -> HarnessContext:
    """Extract safe MCP client context from allowlisted request signals."""
    body = body_json if isinstance(body_json, Mapping) else {}
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    method = _sanitize_value(body.get("method"))
    protocol_version = _sanitize_value(params.get("protocolVersion")) if params else _DEFAULT_CLIENT_SOURCE
    if protocol_version == "unknown" and params:
        protocol_version = _protocol_from_meta(params)
    if protocol_version == "unknown":
        protocol_version = _sanitize_value(_header(headers, "MCP-Protocol-Version"))

    user_agent = _header(headers, "User-Agent")
    user_agent_product = _first_user_agent_product(user_agent)

    if method != "initialize" and session_metadata:
        meta_client_info, _ = _client_info_from_meta(params)
        mismatch = ""
        if meta_client_info:
            _, meta_app = _classify_client(meta_client_info.get("name"))
            session_app = _sanitize_value(session_metadata.get("mcp_client_app"))
            if meta_app != "unknown" and session_app != "unknown" and meta_app != session_app:
                mismatch = "client_info"
        return _context_from_session_metadata(
            session_metadata=session_metadata,
            protocol_version=protocol_version,
            jsonrpc_method=method,
            user_agent_product=user_agent_product,
            mismatch=mismatch,
        )

    if method == "initialize":
        meta_client_info, meta_source = _client_info_from_meta(params)
        if meta_client_info:
            return _context_from_client_info(
                client_info=meta_client_info,
                source=meta_source,
                protocol_version=protocol_version,
                jsonrpc_method=method,
                user_agent_product=user_agent_product,
            )
        raw_client_info = params.get("clientInfo")
        client_info = raw_client_info if isinstance(raw_client_info, dict) else {}
        if client_info:
            return _context_from_client_info(
                client_info=client_info,
                source="initialize_client_info",
                protocol_version=protocol_version,
                jsonrpc_method=method,
                user_agent_product=user_agent_product,
            )

    return _context_from_user_agent(
        user_agent=user_agent,
        protocol_version=protocol_version,
        jsonrpc_method=method,
    )
