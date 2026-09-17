"""Keep internal W&B transport addresses out of public MCP responses."""

from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit

from wandb_mcp_server.config import WANDB_API_BASE_URL, WANDB_BASE_URL


def public_app_base_url() -> str:
    """Return the user-facing W&B application origin."""
    parsed = urlsplit(WANDB_BASE_URL)
    hostname = parsed.hostname or ""
    netloc = parsed.netloc
    if hostname == "api.wandb.ai":
        netloc = "wandb.ai"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")).rstrip("/")


def public_wandb_url(*segments: object) -> str:
    """Build a public W&B link from already-known resource identifiers."""
    suffix = "/".join(quote(str(segment).strip("/"), safe="") for segment in segments)
    return f"{public_app_base_url()}/{suffix}" if suffix else public_app_base_url()


def publicize_wandb_url(value: object, *, fallback_segments: tuple[object, ...] = ()) -> str | None:
    """Rewrite an SDK URL from the internal transport origin to the public app."""
    if value is None or not str(value).strip():
        has_complete_fallback = fallback_segments and all(
            segment is not None and str(segment).strip() for segment in fallback_segments
        )
        return public_wandb_url(*fallback_segments) if has_complete_fallback else None
    raw = str(value).strip()
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return f"{public_app_base_url()}/{raw.lstrip('/')}"

    internal = urlsplit(WANDB_API_BASE_URL)
    internal_hostname = (internal.hostname or "").lower()
    value_hostname = (parsed.hostname or "").lower()
    is_internal_origin = (
        parsed.scheme == internal.scheme and parsed.netloc == internal.netloc and WANDB_API_BASE_URL != WANDB_BASE_URL
    )
    is_kubernetes_service = value_hostname.endswith(".svc") or ".svc." in value_hostname
    if (
        is_internal_origin
        or is_kubernetes_service
        or (internal_hostname and value_hostname == internal_hostname and WANDB_API_BASE_URL != WANDB_BASE_URL)
    ):
        public = urlsplit(public_app_base_url())
        return urlunsplit((public.scheme, public.netloc, parsed.path, parsed.query, parsed.fragment))
    return raw


__all__ = ["public_app_base_url", "public_wandb_url", "publicize_wandb_url"]
