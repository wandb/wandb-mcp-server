"""Batch-resolve root spans for child trace_ids in a single query."""

import json
from typing import List

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.query_weave import get_trace_service
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

RESOLVE_TRACE_ROOTS_TOOL_DESCRIPTION = """Batch-resolve root spans for a list of child trace_ids.

Given trace_ids from child traces, returns the root span (top-level parent)
for each in a SINGLE batched query. This eliminates the N+1 problem of
looking up root spans one at a time.

<when_to_use>
Call AFTER query_weave_traces_tool when you have found child traces and need
to know which root session, conversation, or workflow each belongs to.

Typical two-step workflow:
1. query_weave_traces_tool(filters={...}) -> find matching child traces
2. resolve_trace_roots_tool(trace_ids=[unique trace_ids from results])
   -> get root span name/context for each in one call

Common use cases:
- Content search: find LLM calls containing text, then map to sessions
- Error triage: find traces with exceptions, then identify root workflows
- Cost analysis: find expensive child calls, then identify parent pipelines
</when_to_use>

<critical_info>
Performance: O(1) API calls regardless of how many trace_ids are passed.
Each trace_id maps to exactly one root span (the trace's top-level parent).
Root spans have parent_id=null and share the same trace_id as their children.
</critical_info>

Parameters
----------
entity_name : str
    W&B entity (username or team).
project_name : str
    W&B project name.
trace_ids : list of str
    List of trace_id values from child traces to resolve.

Returns
-------
JSON with:
  - roots: dict mapping trace_id -> root span {id, trace_id, op_name, display_name, started_at}
  - resolved: number of trace_ids successfully resolved
  - total_requested: number of unique trace_ids requested
"""


def resolve_trace_roots(
    entity_name: str,
    project_name: str,
    trace_ids: List[str],
) -> str:
    """Batch-resolve root spans for child trace_ids."""
    api = WandBApiManager.get_api()
    with track_tool_execution(
        "resolve_trace_roots",
        api.viewer,
        {
            "entity_name": entity_name,
            "project_name": project_name,
            "trace_id_count": len(trace_ids),
        },
    ) as ctx:
        if not trace_ids:
            return json.dumps({"roots": {}, "resolved": 0, "total_requested": 0})

        try:
            service = get_trace_service()
            root_map = service.resolve_trace_roots(
                entity_name=entity_name,
                project_name=project_name,
                trace_ids=trace_ids,
            )

            roots = {}
            for tid, root in root_map.items():
                roots[tid] = {
                    "id": root.get("id"),
                    "trace_id": root.get("trace_id"),
                    "op_name": root.get("op_name"),
                    "display_name": root.get("display_name"),
                    "started_at": root.get("started_at"),
                }

            unique_count = len(set(trace_ids))
            return json.dumps(
                {
                    "roots": roots,
                    "resolved": len(roots),
                    "total_requested": unique_count,
                },
                default=str,
            )

        except Exception as e:
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "resolve_failed", "message": str(e)[:500]})
