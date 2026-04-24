"""Tests for MCP_LOG_FORMAT-driven logger configuration.

These cover the opt-in JSON formatter path added alongside the Datadog Agent-mode
chart work. The default (rich) path must remain unchanged so Cloud Run production
keeps the log shape the existing dashboards expect.
"""

from __future__ import annotations

import io
import json
import logging
import os
from unittest import mock

import pytest
from rich.logging import RichHandler


def _reload_utils():
    """Reimport wandb_mcp_server.utils to pick up env-var changes at import time.

    get_rich_logger reads MCP_LOG_FORMAT each call, so a fresh import is not strictly
    required, but we bounce the logger state so handlers don't stack across tests.
    """
    import importlib

    import wandb_mcp_server.utils as utils

    importlib.reload(utils)
    return utils


@pytest.fixture(autouse=True)
def _isolate_logger_between_tests():
    """Each test gets a fresh logger with no inherited handlers."""
    name = "wandb_mcp_server.tests.log_format"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    yield name
    logger.handlers.clear()


def test_default_is_rich_when_env_unset(_isolate_logger_between_tests):
    """MCP_LOG_FORMAT unset -> RichHandler (preserves today's Cloud Run behavior)."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MCP_LOG_FORMAT", None)
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) == 1
    assert isinstance(handlers[0], RichHandler)


def test_rich_is_explicit_default(_isolate_logger_between_tests):
    """MCP_LOG_FORMAT=rich -> RichHandler."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "rich"}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) == 1
    assert isinstance(handlers[0], RichHandler)


def test_json_format_emits_structured_lines(_isolate_logger_between_tests, capsys):
    """MCP_LOG_FORMAT=json -> one JSON object per record with the required fields."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "json"}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)

        # Swap handler stream for an in-memory buffer so we can read it back.
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        buf = io.StringIO()
        handler.stream = buf

        logger.info("hello %s", "world")
        logger.error("boom")

    raw_lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(raw_lines) == 2

    first = json.loads(raw_lines[0])
    assert first["level"] == "info"
    assert first["logger"] == name
    assert first["message"] == "hello world"
    assert first["timestamp"].endswith("Z")

    second = json.loads(raw_lines[1])
    assert second["level"] == "error"
    assert second["message"] == "boom"


def test_json_format_is_case_insensitive(_isolate_logger_between_tests):
    """MCP_LOG_FORMAT=JSON (caps) and extraneous whitespace still mean JSON."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "  JSON  "}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert not isinstance(handler, RichHandler)


def test_json_format_includes_exception_info(_isolate_logger_between_tests):
    """Exceptions serialize into exc_info field so DD captures the stack trace."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "json"}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
        handler = logger.handlers[0]
        buf = io.StringIO()
        handler.stream = buf
        try:
            raise ValueError("intentional")
        except ValueError:
            logger.exception("it blew up")

    payload = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["level"] == "error"
    assert "it blew up" in payload["message"]
    assert "ValueError" in payload["exc_info"]
    assert "intentional" in payload["exc_info"]


def test_json_format_preserves_session_prefix(_isolate_logger_between_tests):
    """The session_id_prefix LoggerAdapter contract still works in JSON mode."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "json"}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
        handler = logger.handlers[0]
        buf = io.StringIO()
        handler.stream = buf

        adapter = logging.LoggerAdapter(logger, {"session_id_prefix": "sess_abc "})
        adapter.info("hello")

    payload = json.loads(buf.getvalue().strip())
    assert payload["session_id_prefix"] == "sess_abc "
    assert payload["message"].startswith("sess_abc ")


def test_unknown_log_format_falls_back_to_rich(_isolate_logger_between_tests):
    """Unknown values (typos, etc.) fall back to rich, don't blow up startup."""
    name = _isolate_logger_between_tests
    with mock.patch.dict(os.environ, {"MCP_LOG_FORMAT": "rainbow"}, clear=False):
        utils = _reload_utils()
        logger = utils.get_rich_logger(name)
    assert isinstance(logger.handlers[0], RichHandler)
