"""Internal provenance for server-authored tool failures.

The marker is not serialized. Plain data, including fields named error/result,
must never become a protocol error based on its shape alone.
"""

from __future__ import annotations

import json
from typing import Any


class ToolErrorPayload(dict[str, Any]):
    """A server-authored failure that retains the existing dictionary API."""


class ToolErrorJSON(str):
    """A server-authored failure that retains the existing JSON-string API."""


def tool_error_json(payload: dict[str, Any], **kwargs: Any) -> ToolErrorJSON:
    """Serialize an owned failure without losing its provenance in process."""
    return ToolErrorJSON(json.dumps(payload, **kwargs))
