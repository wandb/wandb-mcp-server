import netrc
import os
from dataclasses import field
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

import simple_parsing
from dataclasses import dataclass
import weave

logger = logging.getLogger(__name__)

# Define server arguments using a dataclass for simple_parsing
@dataclass
class ServerMCPArgs:
    """Arguments for the Weave MCP Server."""

    wandb_api_key: Optional[str] = field(
        default=None, metadata=dict(help="Weights & Biases API key")
    )
    product_name: str = field(
        default="weave", metadata=dict(help="Product name (weave, wandb, or all)")
    )
    use_weave: bool = field(
        default=True,
        metadata=dict(help="Whether or not to trace MCP server calls to weave"),
    )
    weave_entity: Optional[str] = field(
        default=None,
        metadata=dict(
            help="The Weights & Biases entity to log traced MCP server calls to"
        ),
    )
    weave_project: Optional[str] = field(
        default="weave-mcp-server",
        metadata=dict(
            help="The Weights & Biases project to log traced MCP server calls to"
        ),
    )


def _wandb_base_url() -> str:
    # TODO: make configurable
    return "https://api.wandb.ai"


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



# Initialize server args global variable
_server_args = None


def get_server_args():
    """Get the server arguments, parsing them if not already done."""
    global _server_args
    if _server_args is None:
        _server_args = ServerMCPArgs()
        # Only parse args when explicitly requested, not at import time
        if os.environ.get("PARSE_ARGS_AT_IMPORT", "0") == "1":
            _server_args = simple_parsing.parse(ServerMCPArgs)

        # Get API key from environment if not provided via CLI
        if not _server_args.wandb_api_key:
            _server_args.wandb_api_key = os.getenv("WANDB_API_KEY", "")

        if not _server_args.wandb_api_key:
            _server_args.wandb_api_key = _wandb_api_key_via_netrc()

    return _server_args


if get_server_args().use_weave and weave is not None:
    if get_server_args().weave_entity is not None:
        weave_entity_project = (
            f"{get_server_args().weave_entity}/{get_server_args().weave_project}"
        )
    else:
        weave_entity_project = f"{get_server_args().weave_project}"
    weave.init(weave_entity_project)

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
            if (
                not merged["time_range"]["earliest"]
                or time_range["earliest"] < merged["time_range"]["earliest"]
            ):
                merged["time_range"]["earliest"] = time_range["earliest"]
        if time_range.get("latest"):
            if (
                not merged["time_range"]["latest"]
                or time_range["latest"] > merged["time_range"]["latest"]
            ):
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

