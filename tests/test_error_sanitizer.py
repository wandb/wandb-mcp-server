"""Infrastructure and credential canaries stay out of every output boundary."""

from __future__ import annotations

import base64
import io
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from uvicorn.logging import AccessFormatter

from wandb_mcp_server.analytics import _prepare_event
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.error_sanitizer import (
    MAX_EXTERNAL_ERROR_CHARS,
    sanitize_sensitive_text,
    sanitize_sensitive_value,
)
from wandb_mcp_server.instrumented_server import InstrumentedFastMCP
from wandb_mcp_server.utils import _SensitiveDataFilter

INTERNAL_URL = "http://wandb-api.default.svc.cluster.local:8081/graphql"
SECRET = "test-secret-api-key-123456"


@pytest.fixture(autouse=True)
def _canaries(monkeypatch):
    monkeypatch.setenv(
        "WANDB_INTERNAL_BASE_URL",
        "http://wandb-api.default.svc.cluster.local:8081",
    )
    monkeypatch.setenv("WANDB_API_KEY", SECRET)


def _assert_canaries_absent(value: object) -> None:
    rendered = str(value)
    assert INTERNAL_URL not in rendered
    assert "wandb-api.default.svc.cluster.local" not in rendered
    assert SECRET not in rendered


def test_sanitizes_nested_cyclic_and_credential_values() -> None:
    value: dict[str, object] = {
        "error": f"POST {INTERNAL_URL} Authorization: Bearer {SECRET}",
        "secret": SECRET,
    }
    value["cycle"] = value

    sanitized = sanitize_sensitive_value(value)

    _assert_canaries_absent(sanitized)
    assert sanitized["cycle"] == "<cyclic value>"


def test_log_filter_sanitizes_message_arguments_and_exception() -> None:
    try:
        raise RuntimeError(f"upstream {INTERNAL_URL} api_key={SECRET}")
    except RuntimeError:
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "failed %s",
            (INTERNAL_URL,),
            __import__("sys").exc_info(),
        )

    assert _SensitiveDataFilter().filter(record)
    _assert_canaries_absent(record.getMessage())
    _assert_canaries_absent(record._wandb_mcp_sanitized_exc_info)
    assert record.exc_info is None


def _uvicorn_access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:8080", "POST", path, "1.1", 200),
        None,
    )


def _uvicorn_access_formatter() -> AccessFormatter:
    return AccessFormatter(
        '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        use_colors=False,
    )


def test_log_filter_preserves_uvicorn_access_record_contract() -> None:
    record = _uvicorn_access_record(f"/mcp?upstream={INTERNAL_URL}&api_key={SECRET}")

    assert _SensitiveDataFilter().filter(record)

    assert isinstance(record.args, tuple)
    assert len(record.args) == 5
    rendered = _uvicorn_access_formatter().format(record)
    _assert_canaries_absent(rendered)
    assert "POST" in rendered
    assert "200 OK" in rendered


def test_sanitized_uvicorn_access_record_remains_queue_safe() -> None:
    record = _uvicorn_access_record(f"/mcp?upstream={INTERNAL_URL}&api_key={SECRET}")

    assert _SensitiveDataFilter().filter(record)
    restored = pickle.loads(pickle.dumps(record))

    rendered = _uvicorn_access_formatter().format(restored)
    _assert_canaries_absent(rendered)
    assert len(restored.args) == 5


def test_uvicorn_access_logger_formats_through_process_record_factory() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_uvicorn_access_formatter())
    handler.addFilter(_SensitiveDataFilter())
    logger = logging.getLogger("uvicorn.access.sanitizer-regression")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:8080",
            "POST",
            f"/mcp?upstream={INTERNAL_URL}&api_key={SECRET}",
            "1.1",
            200,
        )
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    rendered = stream.getvalue()
    _assert_canaries_absent(rendered)
    assert "POST" in rendered
    assert "200 OK" in rendered


def test_log_filter_preserves_mapping_arguments_and_sanitizes_literals() -> None:
    record = logging.LogRecord(
        "structured.logger",
        logging.INFO,
        __file__,
        1,
        f"backend={INTERNAL_URL} request=%(path)s api_key=%(api_key)s",
        ({"path": INTERNAL_URL, "api_key": SECRET},),
        None,
    )

    assert _SensitiveDataFilter().filter(record)

    assert isinstance(record.args, dict)
    assert set(record.args) == {"path", "api_key"}
    _assert_canaries_absent(record.getMessage())


def test_custom_argument_record_sanitizes_literals_without_flattening() -> None:
    class CustomLogRecord(logging.LogRecord):
        pass

    record = CustomLogRecord(
        "custom.structured.logger",
        logging.INFO,
        __file__,
        1,
        f"backend={INTERNAL_URL} api_key=%s literal={SECRET} progress=100%%",
        (SECRET,),
        None,
    )

    assert _SensitiveDataFilter().filter(record)

    assert type(record) is CustomLogRecord
    assert isinstance(record.args, tuple)
    assert len(record.args) == 1
    assert "%s" in record.msg
    assert "%%" in record.msg
    rendered = record.getMessage()
    _assert_canaries_absent(rendered)
    assert "api_key=<redacted>" in rendered
    assert "progress=100%" in rendered


def test_log_filter_formats_uvicorn_access_records_concurrently() -> None:
    sanitizer = _SensitiveDataFilter()
    formatter = _uvicorn_access_formatter()

    def _format(index: int) -> str:
        record = _uvicorn_access_record(f"/mcp/{index}?upstream={INTERNAL_URL}&token={SECRET}")
        assert sanitizer.filter(record)
        return formatter.format(record)

    with ThreadPoolExecutor(max_workers=8) as pool:
        rendered = list(pool.map(_format, range(100)))

    assert len(rendered) == 100
    for message in rendered:
        _assert_canaries_absent(message)
        assert "200 OK" in message


def test_analytics_preparation_sanitizes_before_forwarding() -> None:
    event = _prepare_event(
        {
            "schema_version": "1.1",
            "event_type": "tool_call",
            "error": f"upstream {INTERNAL_URL}",
            "usage_dimensions": {"token": SECRET},
        }
    )

    _assert_canaries_absent(event)


@pytest.mark.asyncio
async def test_public_tool_result_is_sanitized(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("sanitizer-test")

    @server.tool(name="list_entities_tool")
    def leaking_tool() -> dict[str, str]:
        return {
            "error": "upstream_error",
            "message": f"{INTERNAL_URL} Bearer {SECRET}",
        }

    result = await server.call_tool("list_entities_tool", {})

    _assert_canaries_absent(result)
    assert "<internal W&B API>" in str(result)


@pytest.mark.asyncio
async def test_oversized_structured_tool_error_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("bounded-error-test")

    @server.tool(name="list_entities_tool")
    def oversized_error() -> dict[str, str]:
        return {
            "error": "upstream_error",
            "message": f"{SECRET}{'x' * (MAX_EXTERNAL_ERROR_CHARS * 3)}",
        }

    result = await server.call_tool("list_entities_tool", {})

    _assert_canaries_absent(result)
    assert len(str(result)) <= MAX_EXTERNAL_ERROR_CHARS
    assert "details_truncated" in str(result)


@pytest.mark.asyncio
async def test_raised_tool_error_is_sanitized(monkeypatch) -> None:
    monkeypatch.setenv("MCP_ANALYTICS_DISABLED", "true")
    server = InstrumentedFastMCP("sanitizer-error-test")

    @server.tool(name="list_entities_tool")
    def leaking_tool() -> None:
        raise ToolError(f"upstream {INTERNAL_URL} api_key={SECRET}")

    with pytest.raises(ToolError) as caught:
        await server.call_tool("list_entities_tool", {})

    _assert_canaries_absent(caught.value)
    assert "<internal W&B API>" in str(caught.value)


def test_request_context_api_key_is_sanitized_without_environment(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    token = WandBApiManager.set_context_api_key(SECRET)
    try:
        sanitized = sanitize_sensitive_text(f"upstream echoed {SECRET}")
    finally:
        WandBApiManager.reset_context_api_key(token)

    _assert_canaries_absent(sanitized)


def test_sanitizes_internal_host_without_url_scheme() -> None:
    sanitized = sanitize_sensitive_text("dial wandb-api.default.svc.cluster.local:8081 failed")

    _assert_canaries_absent(sanitized)


def test_sanitizes_basic_auth_and_binary_error_payloads(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    encoded = base64.b64encode(f":{SECRET}".encode()).decode()
    token = WandBApiManager.set_context_api_key(SECRET)
    try:
        sanitized = sanitize_sensitive_value(
            {
                "message": f"Authorization: Basic {encoded}".encode(),
                "detail": f"api_key={SECRET}".encode(),
            }
        )
    finally:
        WandBApiManager.reset_context_api_key(token)

    _assert_canaries_absent(sanitized)
    assert encoded not in str(sanitized)


def test_external_error_text_is_bounded_after_redaction() -> None:
    sanitized = sanitize_sensitive_value({"error": f"{SECRET}{'x' * (MAX_EXTERNAL_ERROR_CHARS * 2)}"})

    _assert_canaries_absent(sanitized)
    assert "<truncated" in sanitized["error"]
    assert len(sanitized["error"]) < MAX_EXTERNAL_ERROR_CHARS + 100
