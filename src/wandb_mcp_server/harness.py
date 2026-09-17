"""MCP client harness classification for analytics.

The canonical fields in this module are deliberately low-cardinality.  Raw
client names, versions, and user agents are bounded debug/session attributes;
they are never tags.
"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Mapping

_SAFE_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_CALL_TYPE_RE = re.compile(r"^[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*$")
_UNSAFE_CHARS_RE = re.compile(r"[^a-z0-9_.-]+")
_MAX_DEBUG_FIELD_LENGTH = 64

_DEFAULT_CLIENT_FAMILY = "unknown"
_DEFAULT_CLIENT_APP = "unknown"
_DEFAULT_CLIENT_SOURCE = "unknown"
_DEFAULT_CONFIDENCE = "low"

_DEBUG_FIELDS_ENV = "MCP_HARNESS_DEBUG_FIELDS"

_VENDOR_BY_FAMILY = {
    "openai": "openai",
    "claude": "anthropic",
    "cursor": "cursor",
    "gemini": "google",
    "mistral": "mistral",
    "linear": "linear",
    "vscode": "microsoft",
    "mcp_inspector": "modelcontextprotocol",
    "load_test": "internal",
}


@dataclass(frozen=True)
class HarnessContext:
    """Low-cardinality MCP client context for analytics events.

    The ``mcp_*`` attributes are retained for one-release compatibility.  New
    analytics should use ``agent_harness``, ``client_vendor``, and ``call_type``.
    """

    mcp_client_family: str = _DEFAULT_CLIENT_FAMILY
    mcp_client_app: str = _DEFAULT_CLIENT_APP
    mcp_client_source: str = _DEFAULT_CLIENT_SOURCE
    mcp_protocol_version: str = "unknown"
    mcp_jsonrpc_method: str = "unknown"
    canonical_call_type: str = ""
    mcp_client_name: str = ""
    mcp_client_version: str = ""
    mcp_user_agent_product: str = ""
    mcp_client_confidence: str = _DEFAULT_CONFIDENCE
    mcp_client_mismatch: str = ""

    @property
    def agent_harness(self) -> str:
        """Return the exact client product, not its vendor family."""
        return self.mcp_client_app or _DEFAULT_CLIENT_APP

    @property
    def client_vendor(self) -> str:
        """Return the canonical vendor associated with the client family."""
        return _VENDOR_BY_FAMILY.get(self.mcp_client_family, "unknown")

    @property
    def call_type(self) -> str:
        """Return the MCP method using the protocol's slash spelling."""
        if self.canonical_call_type:
            return self.canonical_call_type
        if self.mcp_jsonrpc_method == "unknown":
            return "unknown"
        return self.mcp_jsonrpc_method.replace(".", "/")

    def with_call_type(self, call_type: str) -> "HarnessContext":
        """Return a copy scoped to a specific JSON-RPC method."""
        canonical = _sanitize_call_type(call_type)
        return replace(
            self,
            canonical_call_type=canonical,
            mcp_jsonrpc_method=_sanitize_value(canonical),
        )

    def default_fields(self) -> dict[str, str]:
        """Return canonical fields plus one-release compatibility aliases."""
        return {
            "agent_harness": self.agent_harness,
            "client_vendor": self.client_vendor,
            "call_type": self.call_type,
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
        """Return normalized fields safe to persist in session metadata."""
        fields = self.default_fields()
        if self.mcp_client_version:
            fields["mcp_client_version"] = self.mcp_client_version
        if self.mcp_client_confidence:
            fields["mcp_client_confidence"] = self.mcp_client_confidence
        return fields


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


def _sanitize_call_type(value: object, *, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()[:_MAX_DEBUG_FIELD_LENGTH]
    if not text or not _CALL_TYPE_RE.fullmatch(text):
        return default
    return text


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


def _client_info_from_meta(params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    meta = _nested_dict(params, "_meta")
    client_info = meta.get("io.modelcontextprotocol/clientInfo")
    if isinstance(client_info, dict):
        return client_info, "meta_client_info"
    return {}, ""


def _protocol_from_meta(params: dict[str, Any]) -> str:
    meta = _nested_dict(params, "_meta")
    return _sanitize_value(meta.get("io.modelcontextprotocol/protocolVersion"))


def _classify_client(value: object) -> tuple[str, str]:
    """Return the legacy family and canonical product name."""
    text = str(value or "").lower()
    if not text:
        return _DEFAULT_CLIENT_FAMILY, _DEFAULT_CLIENT_APP
    if "wandb-mcp-load-test" in text or "compatibility-check" in text or "load_test" in text:
        return "load_test", "load_test"
    if "mcp-inspector" in text or "modelcontextprotocol/inspector" in text:
        return "mcp_inspector", "mcp_inspector"
    if "linear" in text:
        return "linear", "linear"
    if "mistral" in text or "lechat" in text or "le-chat" in text:
        return "mistral", "lechat"
    if "claude-code" in text or ("claude" in text and "code" in text):
        return "claude", "claude_code"
    if "claude-desktop" in text or ("claude" in text and "desktop" in text):
        return "claude", "claude_desktop"
    if "claude-ai" in text or "claude.ai" in text or "claude" in text or "anthropic" in text:
        return "claude", "claude_ai"
    if "codex" in text:
        return "openai", "codex"
    if "openai" in text or "chatgpt" in text:
        return "openai", "unknown"
    if "cursor" in text:
        return "cursor", "cursor"
    if "gemini" in text:
        return "gemini", "gemini_cli"
    if "vscode" in text or "visual-studio-code" in text:
        return "vscode", "vscode"
    return _DEFAULT_CLIENT_FAMILY, _DEFAULT_CLIENT_APP


def _lower_priority_mismatch(
    trusted_app: str,
    *,
    meta_client_info: Mapping[str, Any] | None = None,
    user_agent: str = "",
) -> str:
    """Flag a known conflicting signal without changing trusted identity."""
    if meta_client_info:
        _, meta_app = _classify_client(meta_client_info.get("name") or meta_client_info.get("title"))
        if meta_app != "unknown" and trusted_app != "unknown" and meta_app != trusted_app:
            return "client_info"
    _, user_agent_app = _classify_client(user_agent)
    if user_agent_app != "unknown" and trusted_app != "unknown" and user_agent_app != trusted_app:
        return "user_agent"
    return ""


def _context_from_client_info(
    *,
    client_info: dict[str, Any],
    source: str,
    protocol_version: str,
    call_type: str,
    user_agent_product: str,
) -> HarnessContext:
    name = _debug_value(client_info.get("name"))
    version = _debug_value(client_info.get("version"))
    family, app = _classify_client(name or client_info.get("title"))
    return HarnessContext(
        mcp_client_family=family,
        mcp_client_app=app,
        mcp_client_source=source,
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=_sanitize_value(call_type),
        canonical_call_type=call_type,
        mcp_client_name=name,
        mcp_client_version=version,
        mcp_user_agent_product=user_agent_product,
        mcp_client_confidence="high" if family != "unknown" else "low",
    )


def _context_from_session_metadata(
    *,
    session_metadata: Mapping[str, Any],
    protocol_version: str,
    call_type: str,
    user_agent_product: str,
    mismatch: str = "",
) -> HarnessContext:
    app = _sanitize_value(session_metadata.get("agent_harness") or session_metadata.get("mcp_client_app"))
    derived_family, _ = _classify_client(app)
    family = _sanitize_value(session_metadata.get("mcp_client_family"), default=derived_family)
    if family == "unknown":
        family = derived_family
    stored_protocol = _sanitize_value(session_metadata.get("mcp_protocol_version"))
    if protocol_version == "unknown":
        protocol_version = stored_protocol
    return HarnessContext(
        mcp_client_family=family,
        mcp_client_app=app,
        mcp_client_source="session_metadata",
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=_sanitize_value(call_type),
        canonical_call_type=call_type,
        mcp_client_version=_debug_value(session_metadata.get("mcp_client_version")),
        mcp_user_agent_product=user_agent_product,
        mcp_client_confidence="high",
        mcp_client_mismatch=mismatch,
    )


def _context_from_user_agent(
    *,
    user_agent: str,
    protocol_version: str,
    call_type: str,
) -> HarnessContext:
    product = _first_user_agent_product(user_agent)
    family, app = _classify_client(user_agent)
    return HarnessContext(
        mcp_client_family=family,
        mcp_client_app=app,
        mcp_client_source="user_agent" if user_agent else "unknown",
        mcp_protocol_version=protocol_version,
        mcp_jsonrpc_method=_sanitize_value(call_type),
        canonical_call_type=call_type,
        mcp_user_agent_product=product,
        mcp_client_confidence="medium" if family != "unknown" else "low",
    )


def extract_harness_context(
    headers: Mapping[str, str] | None,
    body_json: Mapping[str, Any] | None = None,
    session_metadata: Mapping[str, Any] | None = None,
) -> HarnessContext:
    """Extract safe MCP client context from allowlisted request signals.

    Official initialize ``clientInfo`` wins for initialize requests.  On later
    requests verified session metadata wins over request ``_meta`` and user
    agent hints; disagreements are recorded without changing attribution.
    """
    body = body_json if isinstance(body_json, Mapping) else {}
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    call_type = _sanitize_call_type(body.get("method"))
    protocol_version = _sanitize_value(params.get("protocolVersion")) if params else "unknown"
    if protocol_version == "unknown" and params:
        protocol_version = _protocol_from_meta(params)
    if protocol_version == "unknown":
        protocol_version = _sanitize_value(_header(headers, "MCP-Protocol-Version"))

    user_agent = _header(headers, "User-Agent")
    user_agent_product = _first_user_agent_product(user_agent)
    meta_client_info, meta_source = _client_info_from_meta(params)

    if call_type == "initialize":
        raw_client_info = params.get("clientInfo")
        client_info = raw_client_info if isinstance(raw_client_info, dict) else {}
        if client_info:
            context = _context_from_client_info(
                client_info=client_info,
                source="initialize_client_info",
                protocol_version=protocol_version,
                call_type=call_type,
                user_agent_product=user_agent_product,
            )
            return replace(
                context,
                mcp_client_mismatch=_lower_priority_mismatch(
                    context.agent_harness,
                    meta_client_info=meta_client_info,
                    user_agent=user_agent,
                ),
            )
        if meta_client_info:
            context = _context_from_client_info(
                client_info=meta_client_info,
                source=meta_source,
                protocol_version=protocol_version,
                call_type=call_type,
                user_agent_product=user_agent_product,
            )
            return replace(
                context,
                mcp_client_mismatch=_lower_priority_mismatch(
                    context.agent_harness,
                    user_agent=user_agent,
                ),
            )

    if call_type != "initialize" and session_metadata:
        session_app = _sanitize_value(session_metadata.get("agent_harness") or session_metadata.get("mcp_client_app"))
        return _context_from_session_metadata(
            session_metadata=session_metadata,
            protocol_version=protocol_version,
            call_type=call_type,
            user_agent_product=user_agent_product,
            mismatch=_lower_priority_mismatch(
                session_app,
                meta_client_info=meta_client_info,
                user_agent=user_agent,
            ),
        )

    if meta_client_info:
        context = _context_from_client_info(
            client_info=meta_client_info,
            source=meta_source,
            protocol_version=protocol_version,
            call_type=call_type,
            user_agent_product=user_agent_product,
        )
        return replace(
            context,
            mcp_client_mismatch=_lower_priority_mismatch(
                context.agent_harness,
                user_agent=user_agent,
            ),
        )

    return _context_from_user_agent(
        user_agent=user_agent,
        protocol_version=protocol_version,
        call_type=call_type,
    )


def context_from_sdk_request(request_context: Any, *, call_type: str) -> HarnessContext:
    """Resolve initialized client information from an MCP SDK request context."""
    try:
        session = getattr(request_context, "session", None)
        client_params = getattr(session, "client_params", None)
        if client_params is None:
            raise AttributeError("client params unavailable")
        if hasattr(client_params, "model_dump"):
            dumped = client_params.model_dump(by_alias=True)
        elif isinstance(client_params, Mapping):
            dumped = dict(client_params)
        else:
            dumped = {
                "clientInfo": getattr(client_params, "client_info", None),
                "protocolVersion": getattr(client_params, "protocol_version", None),
            }
        client_info = dumped.get("clientInfo") or dumped.get("client_info")
        if hasattr(client_info, "model_dump"):
            client_info = client_info.model_dump(by_alias=True)
        if not isinstance(client_info, dict):
            raise AttributeError("client info unavailable")
        protocol_version = _sanitize_value(dumped.get("protocolVersion") or dumped.get("protocol_version"))
        return _context_from_client_info(
            client_info=client_info,
            source="sdk_client_params",
            protocol_version=protocol_version,
            call_type=_sanitize_call_type(call_type),
            user_agent_product="",
        )
    except Exception:
        return HarnessContext().with_call_type(call_type)
