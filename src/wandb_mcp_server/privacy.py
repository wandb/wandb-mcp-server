"""Shared privacy configuration validation without logging configuration values."""

import os


def resolve_privacy_level() -> str:
    """Keep the default when unset; reject an explicitly invalid configuration."""
    level = os.environ.get("MCP_LOG_PRIVACY_LEVEL", "off").strip().lower()
    if level not in {"off", "standard", "strict"}:
        raise ValueError("MCP_LOG_PRIVACY_LEVEL must be off, standard, or strict")
    return level
