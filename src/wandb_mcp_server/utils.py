from __future__ import annotations

import json as _json
import logging
import netrc
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import simple_parsing
from rich.logging import RichHandler
from rich.console import Console

os.environ["WANDB_SILENT"] = "True"
os.environ["WEAVE_SILENT"] = "True"


# Define a handler to redirect logs
class RedirectLoggerHandler(logging.Handler):
    """A handler that redirects log records to another logger."""

    def __init__(self, target_logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_logger = target_logger

    def emit(self, record):
        # Format the message using the handler's formatter if it has one
        # otherwise use the record's message. This ensures consistency
        # if formatters are used elsewhere.
        try:
            msg = self.format(record)
            new_record = logging.makeLogRecord(
                {
                    **record.__dict__,
                    "msg": msg,
                    "args": [],  # Args are already incorporated into msg by format()
                }
            )
            self.target_logger.handle(new_record)
        except Exception:
            self.handleError(record)


# JSON formatter for structured logs (e.g. when running behind a Datadog Agent that
# parses JSON log payloads). Keeps a stable set of top-level fields so downstream
# log backends (Datadog, GCP Cloud Logging) extract `level`, `timestamp`, `logger`,
# `message` automatically without custom pipelines.
class _JsonLogFormatter(logging.Formatter):
    """One JSON object per record. stdlib-only, no extra deps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Preserve session prefix context for correlation across request chains
        session_prefix = getattr(record, "session_id_prefix", "")
        if session_prefix:
            payload["session_id_prefix"] = session_prefix
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # stack_info is separate from exc_info; include if present
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return _json.dumps(payload, default=str)


def _build_log_handler() -> logging.Handler:
    """Build the log handler based on MCP_LOG_FORMAT.

    - "rich" (default): RichHandler for pretty local dev output. Also the Cloud Run
      production default today -- do NOT change without validating DD dashboards.
    - "json": Structured JSON lines on stderr. Recommended when running under a DD Agent
      (agent auto-parses level/timestamp) or in any container where downstream parsers
      expect structured logs. Opt-in via env var; chart 0.42.2+ sets this when
      mcp-server.datadog.enabled=true.
    """
    log_format = os.environ.get("MCP_LOG_FORMAT", "rich").strip().lower()
    if log_format == "json":
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonLogFormatter())
        return handler
    # Default: rich output, preserves today's behavior for local dev + Cloud Run.
    stderr_console = Console(stderr=True)
    return RichHandler(
        console=stderr_console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )


# Third-party loggers we explicitly reconfigure in JSON mode so every line emitted by
# the process reaches Datadog (or any other container-log backend) as structured JSON,
# not just ones from wandb_mcp_server.* modules that go through get_rich_logger().
#
# Excludes wandb_mcp_server.analytics: it has its own GCP-Logging-friendly formatter
# (_StructuredJsonFormatter in analytics.py) that downstream BigQuery pipelines depend
# on. Reconfiguring it would silently break analytics ingestion.
_JSON_MODE_THIRD_PARTY_LOGGERS = (
    "",  # root
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "mcp",  # MCP SDK; covers mcp.server.streamable_http etc.
)


def configure_process_logging() -> None:
    """Idempotent: install the JSON handler on root + third-party loggers when enabled.

    Called once from the CLI entrypoint. No-op when MCP_LOG_FORMAT=rich (default) so
    local dev and Cloud Run keep today's behavior. When MCP_LOG_FORMAT=json, replaces
    handlers on root + uvicorn.* + mcp so access logs and SDK logs also emit
    structured JSON lines -- closes the gap where get_rich_logger() only covers our
    own modules.

    Skips wandb_mcp_server.analytics intentionally: it owns its own formatter contract
    with downstream BigQuery pipelines and must not be reconfigured here.
    """
    if os.environ.get("MCP_LOG_FORMAT", "rich").strip().lower() != "json":
        return

    for name in _JSON_MODE_THIRD_PARTY_LOGGERS:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.addHandler(_build_log_handler())
        # Third-party loggers should not propagate up to root -- we already installed
        # the JSON handler on each, and propagation would duplicate every record.
        # The root logger itself keeps propagate=True (it's the root, nothing above).
        if name != "":
            lg.propagate = False


# Moved get_rich_logger here
def get_rich_logger(
    name: str,
    propagate: bool = False,
    default_level_str: str = "INFO",
    env_var_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger. Output format is controlled by MCP_LOG_FORMAT
    (values: "rich" default, or "json" for structured one-line-per-record output
    suitable for containerized deployments behind a Datadog Agent).

    The log level can be set via an environment variable if `env_var_name` is provided.
    Otherwise, it defaults to `default_level_str`.
    """
    logger = logging.getLogger(name)
    _log_handler = _build_log_handler()

    # Inject session prefix into message if provided via LoggerAdapter extra
    class _SessionPrefixInjectFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            prefix = getattr(record, "session_id_prefix", "")
            try:
                if prefix and isinstance(record.msg, str) and not record.msg.startswith(prefix):
                    record.msg = f"{prefix}{record.msg}"
            except Exception:
                pass
            # Ensure attribute exists for formatters that reference it
            if not hasattr(record, "session_id_prefix"):
                record.session_id_prefix = ""
            return True

    logger.addFilter(_SessionPrefixInjectFilter())
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(_log_handler)

    # Determine the effective log level string
    # Start with the function's default_level_str (e.g., "INFO")
    effective_level_str = default_level_str.upper()
    source_of_level = f"function default ('{default_level_str}')"

    if env_var_name:
        env_level_value = os.environ.get(env_var_name)
        if env_level_value:
            effective_level_str = env_level_value.upper()
            source_of_level = f"environment variable '{env_var_name}' ('{env_level_value}')"

    # Attempt to convert the string to a logging level integer
    final_log_level = getattr(logging, effective_level_str, None)

    # If conversion failed, issue a warning and determine a fallback level.
    if not isinstance(final_log_level, int):
        warning_msg_parts = [
            f"Warning: Invalid log level string '{effective_level_str}' from {source_of_level} for logger '{name}'.",
            "Valid levels are DEBUG, INFO, WARNING, ERROR, CRITICAL.",
        ]

        # Check if the issue was with an environment variable and if the original default_level_str is valid
        if env_var_name and os.environ.get(env_var_name) and effective_level_str != default_level_str.upper():
            fallback_to_default_level = getattr(logging, default_level_str.upper(), None)
            if isinstance(fallback_to_default_level, int):
                final_log_level = fallback_to_default_level
                warning_msg_parts.append(f"Falling back to function default '{default_level_str.upper()}'.")
            else:  # Function default is also bad, use hardcoded INFO
                final_log_level = logging.INFO
                warning_msg_parts.append(
                    f"Function default '{default_level_str.upper()}' also invalid. Falling back to INFO."
                )
        else:  # No env var was specified, or env var was not set, or default_level_str itself was bad
            final_log_level = logging.INFO  # Hardcoded ultimate fallback
            warning_msg_parts.append("Falling back to INFO.")

        print(" ".join(warning_msg_parts), file=sys.stderr)

    logger.setLevel(final_log_level)
    logger.propagate = propagate
    return logger


# Setup module-level logger now that get_rich_logger is defined
utils_logger = get_rich_logger(__name__)


# Define server arguments using a dataclass for simple_parsing
@dataclass
class ServerMCPArgs:
    """Arguments for the Weave MCP Server."""

    wandb_api_key: Optional[str] = field(default=None, metadata=dict(help="Weights & Biases API key"))
    weave_entity: Optional[str] = field(
        default=None,
        metadata=dict(help="The Weights & Biases entity to log traced MCP server calls to"),
    )
    weave_project: Optional[str] = field(
        default="weave-mcp-server",
        metadata=dict(help="The Weights & Biases project to log traced MCP server calls to"),
    )
    transport: str = field(
        default="stdio",
        metadata=dict(help="Transport type: 'stdio' for local MCP client communication or 'http' for HTTP server"),
    )
    port: Optional[int] = field(
        default=None,
        metadata=dict(help="Port to run the HTTP server on. Defaults to 8080 when using HTTP transport."),
    )
    host: str = field(
        default="localhost",
        metadata=dict(help="Host to bind HTTP server to"),
    )


# Initialize server args global variable
_server_args = None


# Moved helper functions
def _wandb_base_url() -> str:
    return os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")


def _wandb_api_key_via_netrc_file(filepath: str) -> str | None:
    netrc_path = os.path.expanduser(filepath)
    if not os.path.exists(netrc_path):
        return None
    nrc = netrc.netrc(netrc_path)
    res = nrc.authenticators(urlparse(_wandb_base_url()).netloc)
    api_key = None
    if res:
        _, _, api_key = res
    return api_key


def _wandb_api_key_via_netrc() -> str | None:
    for filepath in ("~/.netrc", "~/_netrc"):
        api_key = _wandb_api_key_via_netrc_file(filepath)
        if api_key:
            return api_key
    return None


def get_server_args():
    """Get the server arguments, parsing them if not already done."""
    global _server_args
    if _server_args is None:
        _server_args = ServerMCPArgs()  # wandb_api_key is None by default

        # Only parse args when explicitly requested, not at import time
        if os.environ.get("PARSE_ARGS_AT_IMPORT", "0") == "1":
            # This potentially updates _server_args with values from command line,
            # including wandb_api_key if provided as an argument.
            _server_args = simple_parsing.parse(ServerMCPArgs, dest=_server_args)

        # Check netrc file first, if API key not already set (e.g., by CLI)
        if _server_args.wandb_api_key is None:
            netrc_api_key = _wandb_api_key_via_netrc()
            if netrc_api_key:
                os.environ["WANDB_API_KEY"] = netrc_api_key  # Set for other modules
                _server_args.wandb_api_key = netrc_api_key
                # utils_logger.info("W&B API key loaded from .netrc file.")

        # If not set via netrc or CLI, try environment variable
        if _server_args.wandb_api_key is None:
            env_api_key = os.getenv("WANDB_API_KEY")
            if env_api_key:
                _server_args.wandb_api_key = env_api_key
                # utils_logger.info("W&B API key loaded from WANDB_API_KEY environment variable.")

        # If after all methods (CLI, netrc, env var), API key is still None or effectively empty,
        # set to empty string to match previous behavior and log a warning.
        if not _server_args.wandb_api_key:  # Covers None or empty string
            _server_args.wandb_api_key = ""  # Ensure it's an empty string if not found
            utils_logger.warning(
                "W&B API key was not found through command-line arguments, .netrc, or WANDB_API_KEY environment variable. "
                "Services requiring W&B authentication may not function correctly or may fail."
            )

    return _server_args


def merge_metadata(metadata_list: List[Dict]) -> Dict:
    """Merge metadata from multiple query results."""
    if not metadata_list:
        return {}

    merged = {
        "total_traces": 0,
        "token_counts": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "average_tokens_per_trace": 0,
        },
        "time_range": {"earliest": None, "latest": None},
        "status_summary": {"success": 0, "error": 0, "other": 0},
        "op_distribution": {},
    }

    for metadata in metadata_list:
        # Sum up trace counts
        merged["total_traces"] += metadata.get("total_traces", 0)

        # Sum up token counts
        token_counts = metadata.get("token_counts", {})
        merged["token_counts"]["total_tokens"] += token_counts.get("total_tokens", 0)
        merged["token_counts"]["input_tokens"] += token_counts.get("input_tokens", 0)
        merged["token_counts"]["output_tokens"] += token_counts.get("output_tokens", 0)

        # Update time range
        time_range = metadata.get("time_range", {})
        if time_range.get("earliest"):
            if not merged["time_range"]["earliest"] or time_range["earliest"] < merged["time_range"]["earliest"]:
                merged["time_range"]["earliest"] = time_range["earliest"]
        if time_range.get("latest"):
            if not merged["time_range"]["latest"] or time_range["latest"] > merged["time_range"]["latest"]:
                merged["time_range"]["latest"] = time_range["latest"]

        # Sum up status counts
        status_summary = metadata.get("status_summary", {})
        merged["status_summary"]["success"] += status_summary.get("success", 0)
        merged["status_summary"]["error"] += status_summary.get("error", 0)
        merged["status_summary"]["other"] += status_summary.get("other", 0)

        # Merge op distributions
        for op, count in metadata.get("op_distribution", {}).items():
            merged["op_distribution"][op] = merged["op_distribution"].get(op, 0) + count

    # Calculate average tokens per trace
    if merged["total_traces"] > 0:
        merged["token_counts"]["average_tokens_per_trace"] = (
            merged["token_counts"]["total_tokens"] / merged["total_traces"]
        )

    return merged


def get_git_commit():
    logger = get_rich_logger(__name__)
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return str(result.stdout.strip())[:8]
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
        return "unknown"


def _get_session_prefix_length() -> int:
    """
    Read the desired session prefix length from SESSION_PREFIX_LENGTH env var.
    Defaults to 12 if not set or invalid.
    """
    try:
        value = int(os.environ.get("SESSION_PREFIX_LENGTH", "12"))
        return value if value > 0 else 12
    except Exception:
        return 12


def get_session_prefix_from_session(session_id: Optional[str]) -> Optional[str]:
    """
    Return the session prefix (first N chars) for a given session_id using
    SESSION_PREFIX_LENGTH (default 12). Returns None if session_id is falsy.
    """
    if not session_id:
        return ""
    length = _get_session_prefix_length()
    return str(session_id)[:length]


def get_request_session(request) -> Optional[str]:
    """
    Return the current request's session_id if available.
    Prefers request.state.session_id, falls back to 'mcp-session-id' header.
    """
    try:
        session_id = getattr(request.state, "session_id", None)
        if session_id:
            return str(session_id)
        # Prefer canonical casing, fall back to lowercase if present
        return request.headers.get("Mcp-Session-Id") or request.headers.get("mcp-session-id")
    except Exception:
        return ""


def get_request_session_prefix(request, length: int = 12) -> Optional[str]:
    """
    Return the current request's session_id prefix if available.
    Uses get_request_session() to resolve the id.
    """
    try:
        session_id = get_request_session(request)
        return get_session_prefix_from_session(session_id)
    except Exception:
        return ""
