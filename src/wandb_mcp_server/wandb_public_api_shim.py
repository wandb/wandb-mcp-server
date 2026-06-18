"""Temporary W&B Public API relogin shim.

This mirrors the header behavior from wandb/wandb#12009 until MCP can depend on
a released W&B SDK that includes it. Remove this module after that release is
pinned in ``pyproject.toml`` and WB-35715 is resolved.
"""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any

import wandb

DISABLE_ENV_VAR = "WANDB_MCP_DISABLE_PUBLIC_API_SHIM"
USER_AGENT_PREFIX = "W&B Public Client "
APPLIED_ATTR = "_wandb_mcp_public_api_shim_applied"
ORIGINAL_ATTR = "_wandb_mcp_public_api_shim_original"


def apply_public_api_relogin_shim(
    public_api_module: ModuleType | None = None,
) -> bool:
    """Apply the W&B Public API relogin shim.

    Returns:
        True when the module was wrapped, False when it was disabled or already
        applied.
    """
    if os.getenv(DISABLE_ENV_VAR):
        return False

    if public_api_module is None:
        import wandb.apis.public.api as public_api_module

    if getattr(public_api_module, APPLIED_ATTR, False):
        return False

    original_service_api = getattr(public_api_module, "ServiceApi")

    def wrapped_service_api(
        *args: Any,
        settings: Any = None,
        **kwargs: Any,
    ) -> Any:
        if settings is not None:
            headers = dict(getattr(settings, "x_extra_http_headers", None) or {})
            headers.setdefault("Use-Admin-Privileges", "true")
            headers.setdefault(
                "User-Agent",
                f"{USER_AGENT_PREFIX}{wandb.__version__}",
            )
            settings.x_extra_http_headers = headers
        return original_service_api(*args, settings=settings, **kwargs)

    wrapped_service_api.__name__ = getattr(
        original_service_api,
        "__name__",
        "ServiceApi",
    )
    wrapped_service_api.__doc__ = getattr(
        original_service_api,
        "__doc__",
        None,
    )

    setattr(public_api_module, ORIGINAL_ATTR, original_service_api)
    setattr(public_api_module, "ServiceApi", wrapped_service_api)
    setattr(public_api_module, APPLIED_ATTR, True)
    return True
