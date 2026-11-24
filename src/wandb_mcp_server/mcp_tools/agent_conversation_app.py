"""Interactive Weave Agent Conversation & Trajectory Inspector MCP App (`ui://wandb/agent-conversation.html`)."""

from __future__ import annotations

from importlib.resources import files
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from wandb_mcp_server.config import resolve_agent_lens_base_url, structured_error
from wandb_mcp_server.error_sanitizer import sanitize_sensitive_text
from wandb_mcp_server.mcp_tools.agents import get_agent_conversation, get_agent_trace
from wandb_mcp_server.tool_results import ToolErrorJSON
from wandb_mcp_server.wandb_urls import public_forge_agent_lens_url

CONVERSATION_RESOURCE_URI = "ui://wandb/agent-conversation.html"
CONVERSATION_SCHEMA = "wandb.agent_conversation/v1"


def register_agent_conversation_resource(mcp: FastMCP) -> None:
    """Register the offline single-file Conversation & Span Trajectory Inspector MCP App resource."""
    if not hasattr(mcp, "resource"):
        return
    ui: dict[str, Any] = {
        "prefersBorder": True,
        "csp": {"connectDomains": [], "resourceDomains": ["data:"]},
    }
    if domain := os.environ.get("WANDB_MCP_APP_DOMAIN"):
        ui["domain"] = domain

    @mcp.resource(CONVERSATION_RESOURCE_URI, mime_type="text/html;profile=mcp-app", meta={"ui": ui})
    def agent_conversation_html() -> str:
        return files("wandb_mcp_server").joinpath("data/agent-conversation.html").read_text(encoding="utf-8")


def _default_access_mode() -> Literal["read-only", "read-write"]:
    raw = (os.environ.get("WANDB_MCP_ACCESS_MODE") or "read-write").strip().lower()
    return "read-only" if raw == "read-only" else "read-write"


def _default_agent_lens_available() -> bool:
    try:
        resolve_agent_lens_base_url()
        return True
    except ValueError:
        return False


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _safe_optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _stringify_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Handle OpenAI / Anthropic content-block lists cleanly when text blocks exist
        parts: List[str] = []
        all_blocks = True
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                all_blocks = False
                break
        if all_blocks and parts:
            return "\n".join(parts)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _canonical_trace_hex(raw: Any) -> Optional[str]:
    text = str(raw or "").strip().lower().replace("-", "")
    if len(text) == 32 and all(ch in "0123456789abcdef" for ch in text):
        return text
    return None


def _trace_id_from_weave_ref(raw_ref: Any) -> Optional[str]:
    if not isinstance(raw_ref, str) or "/agent_turn/" not in raw_ref:
        return None
    match = re.search(r"/agent_turn/([0-9a-fA-F-]{32,36})$", raw_ref.strip())
    if not match:
        return None
    raw_id = match.group(1).strip()
    hex_id = _canonical_trace_hex(raw_id)
    if hex_id and "-" not in raw_id:
        return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"
    return raw_id


def _align_trace_id(candidate: Optional[str], known_trace_ids: Dict[str, str]) -> Optional[str]:
    if not candidate:
        return None
    hex_key = _canonical_trace_hex(candidate)
    if hex_key and hex_key in known_trace_ids:
        return known_trace_ids[hex_key]
    return candidate


def _normalize_role(raw_role: Any, msg: Dict[str, Any]) -> str:
    role = str(raw_role or "").strip().lower()
    if role in {"user", "human", "user_message"}:
        return "user"
    if role in {"assistant", "ai", "model", "agent", "assistant_message"}:
        return "assistant"
    if role in {"tool", "function", "tool_result", "tool_call"}:
        return "tool"
    if role in {"system", "developer", "agent_start", "context_compacted"}:
        return "system"
    if role in {"handoff", "transfer", "agent_handoff"} or msg.get("handoff_target"):
        return "handoff"
    if "tool_result" in msg or "tool_call_id" in msg:
        return "tool"
    if "tool_calls" in msg:
        return "assistant"
    return role if role else "other"


def _normalize_feedback_list(
    raw_feedback: Any,
    fallback_trace_id: Optional[str] = None,
    known_trace_ids: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_feedback, list):
        return []
    trace_lookup = known_trace_ids or {}
    out: List[Dict[str, Any]] = []
    for item in raw_feedback:
        if not isinstance(item, dict):
            continue
        fb_id = str(item.get("id") or "").strip() or None
        fb_type = str(item.get("feedback_type") or item.get("type") or "feedback").strip()
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        note_raw = payload.get("note") or item.get("note") or payload.get("comment")
        note = sanitize_sensitive_text(note_raw, max_chars=1_000) if note_raw else None
        emoji: Optional[str] = None
        if isinstance(payload.get("emoji"), str) and payload["emoji"].strip():
            emoji = payload["emoji"].strip()
        elif isinstance(item.get("emoji"), str) and item["emoji"].strip():
            emoji = item["emoji"].strip()
        elif isinstance(item.get("scorer_tags"), list) and item["scorer_tags"]:
            first_tag = item["scorer_tags"][0]
            if isinstance(first_tag, str) and first_tag.strip():
                emoji = first_tag.strip()
        raw_trace_id = (
            str(item.get("span_trace_id") or item.get("trace_id") or "").strip()
            or _trace_id_from_weave_ref(item.get("weave_ref"))
            or (str(fallback_trace_id).strip() if fallback_trace_id else "")
            or None
        )
        trace_id = _align_trace_id(raw_trace_id, trace_lookup)
        span_id = str(item.get("span_id") or "").strip() or None
        created_at = str(item.get("created_at") or "").strip() or None
        created_by = str(item.get("wb_user_id") or item.get("creator") or "").strip() or None
        out.append(
            {
                "id": fb_id,
                "feedbackType": fb_type,
                "note": note,
                "emoji": emoji,
                "traceId": trace_id,
                "spanId": span_id,
                "createdAt": created_at,
                "createdBy": created_by,
            }
        )
    return out


def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    calls: List[Dict[str, Any]] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(tc.get("tool_name") or tc.get("name") or fn.get("name") or "tool").strip()
        tc_id = str(tc.get("id") or tc.get("tool_call_id") or "").strip() or None
        args_val = tc.get("arguments") if "arguments" in tc else fn.get("arguments", tc.get("input"))
        res_val = tc.get("tool_result") if "tool_result" in tc else tc.get("result", tc.get("output"))
        status = str(tc.get("status_code") or tc.get("status") or "").strip().upper() or None
        err = _stringify_value(tc.get("error") or tc.get("status_message"))
        calls.append(
            {
                "id": tc_id,
                "name": name,
                "arguments": _stringify_value(args_val),
                "result": _stringify_value(res_val),
                "spanId": str(tc.get("span_id") or "").strip() or None,
                "statusCode": status,
                "durationMs": _safe_optional_float(tc.get("duration_ms")),
                "error": err,
            }
        )
    return calls


def _normalize_message(msg: Any, idx: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(msg, dict):
        return (
            {
                "index": idx,
                "role": "other",
                "content": _stringify_value(msg),
                "toolName": None,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": None,
                "spanName": None,
                "statusCode": None,
                "error": None,
                "durationMs": None,
                "model": None,
            },
            {},
        )

    msg_type = str(msg.get("type") or "").strip()
    span_id = str(msg.get("span_id") or "").strip() or None
    span_name = str(msg.get("span_name") or msg.get("operation_name") or "").strip() or None
    top_status_raw = str(msg.get("status_code") or msg.get("status") or "").strip().upper()

    # Unpack Weave AgentChatMessage tagged union (`weave-utils.ts` normalizeChatMessage)
    if msg_type == "user_message" and isinstance(msg.get("user_message"), dict):
        um = msg["user_message"]
        status_code = top_status_raw or None
        err_val = msg.get("error") or (msg.get("status_message") if status_code == "ERROR" else None)
        return (
            {
                "index": idx,
                "role": "user",
                "content": _stringify_value(um.get("text")),
                "toolName": None,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": span_id,
                "spanName": span_name,
                "statusCode": status_code,
                "error": _stringify_value(err_val),
                "durationMs": None,
                "model": None,
            },
            {},
        )

    if msg_type == "assistant_message" and isinstance(msg.get("assistant_message"), dict):
        am = msg["assistant_message"]
        sub_status = str(am.get("status") or "").strip().upper()
        status_code = (sub_status if sub_status and sub_status != "UNSET" else top_status_raw) or None
        err_val = msg.get("error") or (
            (msg.get("status_message") or msg.get("error_type")) if status_code == "ERROR" else None
        )
        model = str(am.get("model") or "").strip() or None
        return (
            {
                "index": idx,
                "role": "assistant",
                "content": _stringify_value(am.get("text")),
                "toolName": None,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": span_id,
                "spanName": span_name,
                "statusCode": status_code,
                "error": _stringify_value(err_val),
                "durationMs": _safe_optional_float(am.get("duration_ms")),
                "model": model,
            },
            {
                "inputTokens": _safe_optional_int(am.get("input_tokens")),
                "outputTokens": _safe_optional_int(am.get("output_tokens")),
                "costUsd": _safe_optional_float(am.get("total_cost_usd")),
            },
        )

    if msg_type == "tool_call" and isinstance(msg.get("tool_call"), dict):
        tc = msg["tool_call"]
        sub_status = str(tc.get("status") or "").strip().upper()
        status_code = (sub_status if sub_status and sub_status != "UNSET" else top_status_raw) or None
        err_val = msg.get("error") or (
            (msg.get("status_message") or msg.get("error_type")) if status_code == "ERROR" else None
        )
        tool_name = str(tc.get("tool_name") or "").strip() or "tool"
        args_str = _stringify_value(tc.get("tool_arguments"))
        if args_str == "":
            args_str = None
        return (
            {
                "index": idx,
                "role": "tool",
                "content": None,
                "toolName": tool_name,
                "toolCallId": str(msg.get("tool_call_id") or "").strip() or None,
                "arguments": args_str,
                "toolCalls": [],
                "toolResult": _stringify_value(tc.get("tool_result")),
                "spanId": span_id,
                "spanName": span_name or tool_name,
                "statusCode": status_code,
                "error": _stringify_value(err_val),
                "durationMs": _safe_optional_float(tc.get("duration_ms")),
                "model": None,
            },
            {},
        )

    if msg_type == "agent_start" and isinstance(msg.get("agent_start"), dict):
        ast = msg["agent_start"]
        sub_status = str(ast.get("status") or "").strip().upper()
        status_code = (sub_status if sub_status and sub_status != "UNSET" else top_status_raw) or None
        model = str(ast.get("model") or "").strip() or None
        sys_inst = _stringify_value(ast.get("system_instructions"))
        content = sys_inst if sys_inst else (f"Agent started ({model})" if model else "Agent started")
        return (
            {
                "index": idx,
                "role": "system",
                "content": content,
                "toolName": None,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": span_id,
                "spanName": span_name,
                "statusCode": status_code,
                "error": None,
                "durationMs": None,
                "model": model,
            },
            {},
        )

    if msg_type == "agent_handoff" and isinstance(msg.get("agent_handoff"), dict):
        ah = msg["agent_handoff"]
        target = str(
            ah.get("target_agent") or ah.get("to_agent") or ah.get("agent_name") or msg.get("agent_name") or "agent"
        ).strip()
        reason = _stringify_value(ah.get("reason") or ah.get("summary") or ah.get("message"))
        content = f"Handoff to {target}: {reason}" if reason else f"Handoff to {target}"
        return (
            {
                "index": idx,
                "role": "handoff",
                "content": content,
                "toolName": target,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": span_id,
                "spanName": span_name,
                "statusCode": top_status_raw or None,
                "error": None,
                "durationMs": None,
                "model": None,
            },
            {},
        )

    if msg_type == "context_compacted" and isinstance(msg.get("context_compacted"), dict):
        cc = msg["context_compacted"]
        before = _safe_optional_int(cc.get("compaction_items_before"))
        after = _safe_optional_int(cc.get("compaction_items_after"))
        summary = _stringify_value(cc.get("compaction_summary"))
        counts = f" ({before} -> {after} items)" if before is not None and after is not None else ""
        content = f"Context compacted{counts}: {summary}" if summary else f"Context compacted{counts}"
        return (
            {
                "index": idx,
                "role": "system",
                "content": content,
                "toolName": None,
                "toolCallId": None,
                "arguments": None,
                "toolCalls": [],
                "toolResult": None,
                "spanId": span_id,
                "spanName": span_name,
                "statusCode": top_status_raw or None,
                "error": None,
                "durationMs": None,
                "model": None,
            },
            {},
        )

    role = _normalize_role(msg.get("role") or msg.get("type"), msg)
    content = _stringify_value(msg.get("content") if "content" in msg else msg.get("text"))
    tool_name = str(msg.get("tool_name") or msg.get("name") or msg.get("handoff_target") or "").strip() or None
    tool_call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "").strip() or None
    arguments = _stringify_value(
        msg.get("arguments")
        if "arguments" in msg
        else (msg.get("tool_arguments") if "tool_arguments" in msg else msg.get("tool_input"))
    )
    tool_calls = _normalize_tool_calls(msg.get("tool_calls"))
    tool_result = _stringify_value(msg.get("tool_result") if "tool_result" in msg else msg.get("output"))
    status_code = top_status_raw or None
    err_val = msg.get("error") or (msg.get("status_message") if status_code == "ERROR" else None)
    error = _stringify_value(err_val)
    duration_ms = _safe_optional_float(msg.get("duration_ms") or msg.get("total_duration_ms"))
    model = str(msg.get("response_model") or msg.get("request_model") or msg.get("model") or "").strip() or None
    return (
        {
            "index": idx,
            "role": role,
            "content": content,
            "toolName": tool_name,
            "toolCallId": tool_call_id,
            "arguments": arguments,
            "toolCalls": tool_calls,
            "toolResult": tool_result,
            "spanId": span_id,
            "spanName": span_name,
            "statusCode": status_code,
            "error": error,
            "durationMs": duration_ms,
            "model": model,
        },
        {},
    )


def _normalize_turn(raw_turn: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    trace_id = str(raw_turn.get("trace_id") or raw_turn.get("id") or "").strip() or None
    span_id = str(raw_turn.get("span_id") or raw_turn.get("root_span_id") or "").strip() or None
    agent_name = str(raw_turn.get("agent_name") or "").strip() or None
    agent_version = str(raw_turn.get("agent_version") or "").strip() or None
    status_raw = str(raw_turn.get("status_code") or raw_turn.get("status") or "OK").strip().upper()
    raw_messages = raw_turn.get("messages") if isinstance(raw_turn.get("messages"), list) else []

    messages: List[Dict[str, Any]] = []
    msg_usage: List[Dict[str, Any]] = []
    msg_feedback_raw: List[Any] = []
    for i, raw_m in enumerate(raw_messages):
        norm_m, usage_m = _normalize_message(raw_m, i)
        messages.append(norm_m)
        if usage_m:
            msg_usage.append(usage_m)
        if isinstance(raw_m, dict) and isinstance(raw_m.get("feedback"), list):
            msg_feedback_raw.extend(raw_m["feedback"])

    has_msg_error = any(
        m.get("statusCode") == "ERROR"
        or bool(m.get("error"))
        or any(tc.get("statusCode") == "ERROR" or bool(tc.get("error")) for tc in m.get("toolCalls", []))
        for m in messages
    )
    is_error = status_raw == "ERROR" or bool(raw_turn.get("is_error")) or bool(raw_turn.get("error")) or has_msg_error
    status_code = "ERROR" if is_error else (status_raw if status_raw and status_raw != "UNSET" else "OK")

    turn_err_raw = raw_turn.get("error") or raw_turn.get("failure_reason") or raw_turn.get("status_message")
    error_message = _stringify_value(turn_err_raw) if (is_error or turn_err_raw) else None
    if not error_message and is_error:
        for m in messages:
            if m.get("error"):
                error_message = m["error"]
                break
            for tc in m.get("toolCalls", []):
                if tc.get("error"):
                    error_message = tc["error"]
                    break
            if error_message:
                break

    in_tok = _safe_optional_int(
        raw_turn.get("input_tokens") if "input_tokens" in raw_turn else raw_turn.get("total_input_tokens")
    )
    if in_tok is None:
        msg_in = [u["inputTokens"] for u in msg_usage if u.get("inputTokens") is not None]
        if msg_in:
            in_tok = sum(msg_in)

    out_tok = _safe_optional_int(
        raw_turn.get("output_tokens") if "output_tokens" in raw_turn else raw_turn.get("total_output_tokens")
    )
    if out_tok is None:
        msg_out = [u["outputTokens"] for u in msg_usage if u.get("outputTokens") is not None]
        if msg_out:
            out_tok = sum(msg_out)

    tot_tok = _safe_optional_int(raw_turn.get("total_tokens"))
    if tot_tok is None and (in_tok is not None or out_tok is not None):
        tot_tok = (in_tok or 0) + (out_tok or 0)

    duration_ms = _safe_optional_float(
        raw_turn.get("total_duration_ms") if "total_duration_ms" in raw_turn else raw_turn.get("duration_ms")
    )
    cost_usd = _safe_optional_float(
        raw_turn.get("total_cost_usd") if "total_cost_usd" in raw_turn else raw_turn.get("cost_usd")
    )
    if cost_usd is None:
        msg_costs = [u["costUsd"] for u in msg_usage if u.get("costUsd") is not None]
        if msg_costs:
            cost_usd = round(sum(msg_costs), 6)

    started_at = str(raw_turn.get("started_at") or raw_turn.get("created_at") or "").strip() or next(
        (
            str(rm.get("started_at")).strip()
            for rm in raw_messages
            if isinstance(rm, dict) and isinstance(rm.get("started_at"), str) and rm.get("started_at").strip()
        ),
        None,
    )
    model = str(
        raw_turn.get("response_model") or raw_turn.get("request_model") or raw_turn.get("model") or ""
    ).strip() or next((m["model"] for m in messages if m.get("model")), None)

    known_turn_map = {_canonical_trace_hex(trace_id): trace_id} if _canonical_trace_hex(trace_id) and trace_id else {}
    combined_turn_fb = [
        *(raw_turn.get("feedback") if isinstance(raw_turn.get("feedback"), list) else []),
        *msg_feedback_raw,
    ]
    feedback = _normalize_feedback_list(
        combined_turn_fb,
        fallback_trace_id=trace_id,
        known_trace_ids=known_turn_map,
    )

    return {
        "turnIndex": turn_index,
        "traceId": trace_id,
        "spanId": span_id,
        "agentName": agent_name,
        "agentVersion": agent_version,
        "statusCode": status_code,
        "isError": is_error,
        "errorMessage": error_message,
        "startedAt": started_at,
        "durationMs": duration_ms,
        "inputTokens": in_tok,
        "outputTokens": out_tok,
        "totalTokens": tot_tok,
        "costUsd": cost_usd,
        "model": model,
        "messages": messages,
        "feedback": feedback,
    }


def _normalize_selected_failure(raw: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        return None
    trace_id = str(raw.get("trace_id") or raw.get("traceId") or "").strip()
    failure_id = str(raw.get("failure_id") or raw.get("failureId") or "").strip()
    if not trace_id or not failure_id:
        return None
    return {"traceId": trace_id, "failureId": failure_id}


def build_agent_conversation_result(
    raw_json: str,
    *,
    mode: Literal["conversation", "trace"],
    entity_name: str,
    project_name: str,
    conversation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    offset: int = 0,
    selected_failure: Optional[Dict[str, str]] = None,
    access_mode: Optional[Literal["read-only", "read-write"]] = None,
    agent_lens_available: Optional[bool] = None,
) -> CallToolResult:
    """Wrap raw get_agent_conversation / get_agent_trace JSON in a backward-compatible MCP App CallToolResult."""
    try:
        decoded = json.loads(raw_json)
    except (TypeError, ValueError):
        err_payload = structured_error("agents_query_failed", "Received invalid JSON from Weave Agents endpoint.")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(err_payload, separators=(",", ":")))],
            structuredContent=err_payload,
            isError=True,
        )

    if isinstance(raw_json, ToolErrorJSON) or (isinstance(decoded, dict) and decoded.get("error")):
        err_dict = decoded if isinstance(decoded, dict) else structured_error("agents_query_failed", str(raw_json))
        return CallToolResult(
            content=[TextContent(type="text", text=raw_json)],
            structuredContent=err_dict,
            isError=True,
        )

    data = decoded if isinstance(decoded, dict) else {}
    effective_access = access_mode or _default_access_mode()
    effective_al = _default_agent_lens_available() if agent_lens_available is None else bool(agent_lens_available)

    if mode == "trace":
        resolved_trace_id = str(data.get("trace_id") or trace_id or "").strip() or None
        resolved_conv_id = str(data.get("conversation_id") or conversation_id or "").strip() or None
        turns = [_normalize_turn(data, 1)]
        total_turns = 1
        has_more = False
    else:
        resolved_conv_id = str(data.get("conversation_id") or conversation_id or "").strip() or None
        raw_turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        start_index = max(0, int(offset or 0))
        turns = [
            _normalize_turn(turn, start_index + i + 1) for i, turn in enumerate(raw_turns) if isinstance(turn, dict)
        ]
        resolved_trace_id = trace_id or (turns[0]["traceId"] if len(turns) == 1 else None)
        total_turns = _safe_int(data.get("total_turns"), len(turns))
        has_more = bool(data.get("has_more")) or (start_index + len(turns) < total_turns)

    agent_name = str(data.get("agent_name") or "").strip() or next(
        (t["agentName"] for t in turns if t.get("agentName")), None
    )
    agent_version = str(data.get("agent_version") or "").strip() or next(
        (t["agentVersion"] for t in turns if t.get("agentVersion")), None
    )

    known_trace_ids: Dict[str, str] = {}
    for t in turns:
        tid = t.get("traceId")
        hex_key = _canonical_trace_hex(tid)
        if hex_key and isinstance(tid, str):
            known_trace_ids[hex_key] = tid

    raw_top_fb = [
        *(data.get("feedback") if isinstance(data.get("feedback"), list) else []),
        *(data.get("conversation_feedback") if isinstance(data.get("conversation_feedback"), list) else []),
    ]
    top_feedback = _normalize_feedback_list(
        raw_top_fb,
        fallback_trace_id=resolved_trace_id,
        known_trace_ids=known_trace_ids,
    )
    seen_fb_keys: set[str] = set()
    merged_feedback: List[Dict[str, Any]] = []
    for fb in [*top_feedback, *(fb_item for t in turns for fb_item in t["feedback"])]:
        if fb.get("traceId"):
            fb["traceId"] = _align_trace_id(fb["traceId"], known_trace_ids)
        dedupe_key = fb["id"] or f"{fb['feedbackType']}:{fb['traceId']}:{fb['emoji']}:{fb['note']}"
        if dedupe_key in seen_fb_keys:
            continue
        seen_fb_keys.add(dedupe_key)
        merged_feedback.append(fb)

    sum_in = sum(t["inputTokens"] or 0 for t in turns)
    sum_out = sum(t["outputTokens"] or 0 for t in turns)
    sum_tot = sum(t["totalTokens"] or 0 for t in turns)
    total_in = _safe_int(data.get("total_input_tokens"), sum_in)
    total_out = _safe_int(data.get("total_output_tokens"), sum_out)
    total_tokens = _safe_int(data.get("total_tokens"), sum_tot or (total_in + total_out))

    sum_dur = round(sum(t["durationMs"] or 0.0 for t in turns), 2)
    top_dur = _safe_optional_float(data.get("total_duration_ms"))
    total_duration_ms = top_dur if top_dur is not None else sum_dur

    top_cost = _safe_optional_float(data.get("total_cost_usd"))
    if top_cost is None:
        turn_costs = [t["costUsd"] for t in turns if t["costUsd"] is not None]
        total_cost_usd = round(sum(turn_costs), 6) if turn_costs else None
    else:
        total_cost_usd = top_cost

    error_count = sum(1 for t in turns if t["isError"])
    compacted = bool(
        (isinstance(data.get("_field_compaction"), dict) and data["_field_compaction"].get("applied"))
        or (isinstance(data.get("_truncation"), dict) and data["_truncation"].get("applied"))
    )

    structured_payload: Dict[str, Any] = {
        "schema": CONVERSATION_SCHEMA,
        "mode": mode,
        "entityName": entity_name.strip(),
        "projectName": project_name.strip(),
        "conversationId": resolved_conv_id,
        "traceId": resolved_trace_id,
        "agentName": agent_name,
        "agentVersion": agent_version,
        "forgeUrl": public_forge_agent_lens_url(entity_name.strip(), project_name.strip()),
        "accessMode": effective_access,
        "agentLensAvailable": effective_al,
        "compacted": compacted,
        "summary": {
            "turnCount": len(turns),
            "totalTurns": max(total_turns, len(turns)),
            "hasMore": has_more,
            "totalInputTokens": total_in,
            "totalOutputTokens": total_out,
            "totalTokens": total_tokens,
            "totalDurationMs": total_duration_ms,
            "totalCostUsd": total_cost_usd,
            "errorCount": error_count,
        },
        "turns": turns,
        "feedback": merged_feedback,
        "selectedFailure": _normalize_selected_failure(selected_failure),
    }

    return CallToolResult(
        content=[TextContent(type="text", text=raw_json)],
        structuredContent=structured_payload,
        isError=False,
    )


def render_agent_trace(
    entity_name: str,
    project_name: str,
    trace_id: str,
    include_feedback: bool = False,
    include_model_tool_calls: bool = False,
    selected_failure: Optional[Dict[str, str]] = None,
    *,
    _access_mode: Optional[Literal["read-only", "read-write"]] = None,
    _agent_lens_available: Optional[bool] = None,
) -> CallToolResult:
    """Execute get_agent_trace and return a CallToolResult with structuredContent for the Conversation Inspector."""
    raw = get_agent_trace(
        entity_name=entity_name,
        project_name=project_name,
        trace_id=trace_id,
        include_feedback=include_feedback,
        include_model_tool_calls=include_model_tool_calls,
    )
    return build_agent_conversation_result(
        raw,
        mode="trace",
        entity_name=entity_name,
        project_name=project_name,
        trace_id=trace_id,
        selected_failure=selected_failure,
        access_mode=_access_mode,
        agent_lens_available=_agent_lens_available,
    )


def render_agent_conversation(
    entity_name: str,
    project_name: str,
    conversation_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    sort_direction: Literal["asc", "desc"] = "asc",
    include_feedback: bool = False,
    include_model_tool_calls: bool = False,
    selected_failure: Optional[Dict[str, str]] = None,
    *,
    _access_mode: Optional[Literal["read-only", "read-write"]] = None,
    _agent_lens_available: Optional[bool] = None,
) -> CallToolResult:
    """Execute get_agent_conversation and return a CallToolResult with structuredContent for the Conversation Inspector."""
    raw = get_agent_conversation(
        entity_name=entity_name,
        project_name=project_name,
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
        sort_direction=sort_direction,
        include_feedback=include_feedback,
        include_model_tool_calls=include_model_tool_calls,
    )
    return build_agent_conversation_result(
        raw,
        mode="conversation",
        entity_name=entity_name,
        project_name=project_name,
        conversation_id=conversation_id,
        offset=offset,
        selected_failure=selected_failure,
        access_mode=_access_mode,
        agent_lens_available=_agent_lens_available,
    )


__all__ = [
    "CONVERSATION_RESOURCE_URI",
    "CONVERSATION_SCHEMA",
    "build_agent_conversation_result",
    "register_agent_conversation_resource",
    "render_agent_conversation",
    "render_agent_trace",
]
