"""Redact infrastructure and credentials from externally visible error data."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

_REDACTED_INTERNAL = "<internal W&B API>"
_REDACTED_SECRET = "<redacted>"
_REDACTION_MARKER_RE = re.compile(r"(<internal W&B API>|<redacted>)")
MAX_EXTERNAL_ERROR_CHARS = 4_096
_KUBERNETES_URL_RE = re.compile(
    r"(?i)\bhttps?://(?:[^/\s@]+@)?"
    r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.svc(?:\.cluster\.local)?"
    r"(?::\d{1,5})?(?:/[^\s\"'<>]*)?"
)
_KUBERNETES_HOST_RE = re.compile(
    r"(?i)\b[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?"
    r"\.svc(?:\.cluster\.local)?(?::\d{1,5})?\b"
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)\b(https?://)[^/\s:@]+:[^/\s@]+@")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_AUTH_RE = re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=_-]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)"
    r"(\s*[=:]\s*[\"']?)([^\s,;}\"']+)"
)


def _configured_secrets() -> tuple[str, ...]:
    values: set[str] = set()
    for name, value in os.environ.items():
        upper_name = name.upper()
        if (
            "API_KEY" in upper_name
            or upper_name.endswith("_TOKEN")
            or upper_name.endswith("_SECRET")
            or upper_name.endswith("_PASSWORD")
        ) and len(value) >= 8:
            values.add(value)
    # Hosted authentication stores the customer W&B key in a request-local
    # ContextVar rather than WANDB_API_KEY. Resolve it lazily to avoid an import
    # cycle during api_client initialization, and redact it if an upstream SDK
    # exception happens to echo the credential.
    try:
        from wandb_mcp_server.api_client import WandBApiManager

        request_api_key = WandBApiManager.get_api_key()
    except (ImportError, RuntimeError, ValueError):
        request_api_key = None
    if request_api_key and len(request_api_key) >= 8:
        values.add(request_api_key)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_plain_parts(text: str, redact: Callable[[str], str]) -> str:
    """Keep application redaction markers stable across repeated boundaries."""
    return "".join(part if index % 2 else redact(part) for index, part in enumerate(_REDACTION_MARKER_RE.split(text)))


def sanitize_sensitive_text(
    value: object,
    *,
    max_chars: int | None = None,
) -> str:
    """Return text with known secrets and internal service addresses removed."""
    text = str(value)
    secrets = _configured_secrets()
    # A real configured secret may itself contain a marker spelling. Remove
    # the complete secret before treating standalone markers as safe output.
    for secret in secrets:
        if _REDACTION_MARKER_RE.search(secret):
            text = text.replace(secret, _REDACTED_SECRET)
    internal_url = (os.environ.get("WANDB_INTERNAL_BASE_URL") or "").rstrip("/")
    if internal_url:
        text = _redact_plain_parts(text, lambda part: part.replace(internal_url, _REDACTED_INTERNAL))
        parsed = urlparse(internal_url if "://" in internal_url else f"http://{internal_url}")
        if parsed.netloc:
            text = _redact_plain_parts(text, lambda part: part.replace(parsed.netloc, _REDACTED_INTERNAL))
        if parsed.hostname:
            text = _redact_plain_parts(text, lambda part: part.replace(parsed.hostname, _REDACTED_INTERNAL))
    text = _KUBERNETES_URL_RE.sub(_REDACTED_INTERNAL, text)
    text = _KUBERNETES_HOST_RE.sub(_REDACTED_INTERNAL, text)
    text = _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", text)
    text = _BEARER_RE.sub(f"Bearer {_REDACTED_SECRET}", text)
    text = _BASIC_AUTH_RE.sub(f"Basic {_REDACTED_SECRET}", text)
    text = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED_SECRET}",
        text,
    )
    for secret in secrets:
        text = _redact_plain_parts(text, lambda part, secret=secret: part.replace(secret, _REDACTED_SECRET))
    if max_chars is not None and len(text) > max_chars:
        omitted = len(text) - max_chars
        text = f"{text[:max_chars]}… <truncated {omitted} chars>"
    return text


def sanitize_sensitive_value(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
    _error_context: bool = False,
) -> Any:
    """Recursively sanitize common MCP result and telemetry value shapes."""
    if isinstance(value, str):
        return sanitize_sensitive_text(
            value,
            max_chars=MAX_EXTERNAL_ERROR_CHARS if _error_context else None,
        )
    if isinstance(value, (bytes, bytearray)):
        decoded = bytes(value).decode("utf-8", errors="replace")
        return sanitize_sensitive_text(
            decoded,
            max_chars=MAX_EXTERNAL_ERROR_CHARS if _error_context else None,
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if _depth >= 12:
        return "<value omitted>"

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return "<cyclic value>"
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                sanitized_key = sanitize_sensitive_text(key)
                result[sanitized_key] = sanitize_sensitive_value(
                    child,
                    _seen=seen,
                    _depth=_depth + 1,
                    _error_context=(
                        _error_context or sanitized_key.lower() in {"error", "message", "detail", "reason", "exception"}
                    ),
                )
            return result
        if isinstance(value, tuple):
            return tuple(
                sanitize_sensitive_value(
                    child,
                    _seen=seen,
                    _depth=_depth + 1,
                    _error_context=_error_context,
                )
                for child in value
            )
        if isinstance(value, list):
            return [
                sanitize_sensitive_value(
                    child,
                    _seen=seen,
                    _depth=_depth + 1,
                    _error_context=_error_context,
                )
                for child in value
            ]
        if isinstance(value, Sequence):
            return [
                sanitize_sensitive_value(
                    child,
                    _seen=seen,
                    _depth=_depth + 1,
                    _error_context=_error_context,
                )
                for child in value
            ]

        model_copy = getattr(value, "model_copy", None)
        model_fields = getattr(type(value), "model_fields", None)
        if callable(model_copy) and isinstance(model_fields, Mapping):
            updates = {
                field_name: sanitize_sensitive_value(
                    getattr(value, field_name),
                    _seen=seen,
                    _depth=_depth + 1,
                    _error_context=(
                        _error_context or field_name.lower() in {"error", "message", "detail", "reason", "exception"}
                    ),
                )
                for field_name in model_fields
                if hasattr(value, field_name)
            }
            return model_copy(update=updates)
        return value
    finally:
        seen.discard(identity)


__all__ = [
    "MAX_EXTERNAL_ERROR_CHARS",
    "sanitize_sensitive_text",
    "sanitize_sensitive_value",
]
