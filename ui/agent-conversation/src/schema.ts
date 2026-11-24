import { z } from "zod";

export const ToolErrorPayloadSchema = z.object({
  error: z.union([z.string(), z.record(z.string(), z.unknown())]),
  message: z.string(),
});

export const NormalizedFeedbackSchema = z.object({
  id: z.string().nullable().default(null),
  feedbackType: z.string(),
  note: z.string().nullable().default(null),
  emoji: z.string().nullable().default(null),
  traceId: z.string().nullable().default(null),
  spanId: z.string().nullable().default(null),
  createdAt: z.string().nullable().default(null),
  createdBy: z.string().nullable().default(null),
});

export const NormalizedToolCallSchema = z.object({
  id: z.string().nullable().default(null),
  name: z.string(),
  arguments: z.string().nullable().default(null),
  result: z.string().nullable().default(null),
  spanId: z.string().nullable().default(null),
  statusCode: z.string().nullable().default(null),
  durationMs: z.number().nullable().default(null),
  error: z.string().nullable().default(null),
});

export const NormalizedMessageSchema = z.object({
  index: z.number(),
  role: z.string(),
  content: z.string().nullable().default(null),
  toolName: z.string().nullable().default(null),
  toolCallId: z.string().nullable().default(null),
  arguments: z.string().nullable().default(null),
  toolCalls: z.array(NormalizedToolCallSchema).default([]),
  toolResult: z.string().nullable().default(null),
  spanId: z.string().nullable().default(null),
  spanName: z.string().nullable().default(null),
  statusCode: z.string().nullable().default(null),
  error: z.string().nullable().default(null),
  durationMs: z.number().nullable().default(null),
  model: z.string().nullable().default(null),
});

export const NormalizedTurnSchema = z.object({
  turnIndex: z.number(),
  traceId: z.string().nullable().default(null),
  spanId: z.string().nullable().default(null),
  agentName: z.string().nullable().default(null),
  agentVersion: z.string().nullable().default(null),
  statusCode: z.string().default("OK"),
  isError: z.boolean().default(false),
  errorMessage: z.string().nullable().default(null),
  startedAt: z.string().nullable().default(null),
  durationMs: z.number().nullable().default(null),
  inputTokens: z.number().nullable().default(null),
  outputTokens: z.number().nullable().default(null),
  totalTokens: z.number().nullable().default(null),
  costUsd: z.number().nullable().default(null),
  model: z.string().nullable().default(null),
  messages: z.array(NormalizedMessageSchema).default([]),
  feedback: z.array(NormalizedFeedbackSchema).default([]),
});

export const ConversationSummarySchema = z.object({
  turnCount: z.number().default(0),
  totalTurns: z.number().default(0),
  hasMore: z.boolean().default(false),
  totalInputTokens: z.number().default(0),
  totalOutputTokens: z.number().default(0),
  totalTokens: z.number().default(0),
  totalDurationMs: z.number().default(0),
  totalCostUsd: z.number().nullable().default(null),
  errorCount: z.number().default(0),
});

export const ConversationPayloadSchema = z.object({
  schema: z.literal("wandb.agent_conversation/v1"),
  mode: z.enum(["conversation", "trace"]),
  entityName: z.string(),
  projectName: z.string(),
  conversationId: z.string().nullable().default(null),
  traceId: z.string().nullable().default(null),
  agentName: z.string().nullable().default(null),
  agentVersion: z.string().nullable().default(null),
  forgeUrl: z.string(),
  accessMode: z.enum(["read-only", "read-write"]).default("read-only"),
  agentLensAvailable: z.boolean().default(false),
  compacted: z.boolean().default(false),
  summary: ConversationSummarySchema,
  turns: z.array(NormalizedTurnSchema).default([]),
  feedback: z.array(NormalizedFeedbackSchema).default([]),
});

export type NormalizedFeedback = z.infer<typeof NormalizedFeedbackSchema>;
export type NormalizedToolCall = z.infer<typeof NormalizedToolCallSchema>;
export type NormalizedMessage = z.infer<typeof NormalizedMessageSchema>;
export type NormalizedTurn = z.infer<typeof NormalizedTurnSchema>;
export type ConversationPayload = z.infer<typeof ConversationPayloadSchema>;

export interface AppliedConversationTag {
  tagId: string | null;
  name: string;
  color: string | null;
  traceId: string | null;
  source: string | null;
}

export interface TagCatalogEntry {
  id: string;
  name: string;
  color: string | null;
}

export type SpanKind = "agent" | "turn" | "llm" | "tool" | "span";

export interface SpanWaterfallNode {
  spanId: string;
  parentSpanId: string | null;
  traceId: string | null;
  name: string;
  kind: SpanKind;
  toolName: string | null;
  model: string | null;
  statusCode: string;
  isError: boolean;
  errorPreview: string | null;
  depth: number;
  offsetMs: number;
  durationMs: number;
  leftPct: number;
  widthPct: number;
  inputTokens: number | null;
  outputTokens: number | null;
  attributesJson: string | null;
}

function parseTimestampMs(raw: unknown): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return raw > 1e12 ? raw : raw * 1000;
  }
  if (typeof raw === "string" && raw.trim()) {
    const parsed = Date.parse(raw.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function classifySpanKind(rec: Record<string, unknown>, name: string): SpanKind {
  const kindRaw = String(rec.span_kind ?? rec.kind ?? rec.operation_name ?? "").toLowerCase();
  const lowerName = name.toLowerCase();
  if (rec.tool_name || kindRaw.includes("tool") || lowerName.startsWith("tool.") || lowerName.includes("tool_call")) {
    return "tool";
  }
  if (
    rec.request_model ||
    rec.response_model ||
    kindRaw.includes("llm") ||
    kindRaw.includes("chat") ||
    lowerName.includes("chat.completions") ||
    lowerName.includes("responses") ||
    lowerName.includes("messages")
  ) {
    return "llm";
  }
  if (rec.is_invocation === true || kindRaw.includes("turn") || lowerName.includes("turn")) {
    return "turn";
  }
  if (kindRaw.includes("agent") || lowerName.includes("agent")) {
    return "agent";
  }
  return "span";
}

function stringifyCompactDetails(rec: Record<string, unknown>): string | null {
  const detailObj: Record<string, unknown> = {};
  for (const key of [
    "inputs",
    "outputs",
    "attributes",
    "custom_attrs",
    "tool_input",
    "tool_output",
    "status_message",
    "error",
  ]) {
    if (rec[key] !== undefined && rec[key] !== null) {
      detailObj[key] = rec[key];
    }
  }
  if (Object.keys(detailObj).length === 0) {
    return null;
  }
  try {
    return JSON.stringify(detailObj, null, 2);
  } catch {
    return null;
  }
}

export function buildSpanWaterfall(rawJsonOrObj: unknown): SpanWaterfallNode[] {
  let root: unknown = rawJsonOrObj;
  if (typeof rawJsonOrObj === "string") {
    try {
      root = JSON.parse(rawJsonOrObj);
    } catch {
      return [];
    }
  }
  if (!root || typeof root !== "object") {
    return [];
  }
  const obj = root as Record<string, unknown>;
  let rawSpans: unknown[] = [];
  if (Array.isArray(obj.conversations)) {
    for (const conv of obj.conversations) {
      if (conv && typeof conv === "object" && Array.isArray((conv as Record<string, unknown>).spans)) {
        rawSpans.push(...((conv as Record<string, unknown>).spans as unknown[]));
      }
    }
  } else if (Array.isArray(obj.spans)) {
    rawSpans = obj.spans;
  }

  interface RawSpanItem {
    spanId: string;
    parentSpanId: string | null;
    traceId: string | null;
    name: string;
    kind: SpanKind;
    toolName: string | null;
    model: string | null;
    statusCode: string;
    isError: boolean;
    errorPreview: string | null;
    startMs: number | null;
    durationMs: number;
    inputTokens: number | null;
    outputTokens: number | null;
    attributesJson: string | null;
  }

  const parsed: RawSpanItem[] = [];
  rawSpans.forEach((entry, idx) => {
    if (!entry || typeof entry !== "object") {
      return;
    }
    const rec = entry as Record<string, unknown>;
    const spanId = String(rec.span_id ?? rec.id ?? `span-${idx}`).trim();
    const parentRaw = rec.parent_id ?? rec.parent_span_id ?? null;
    const parentSpanId = typeof parentRaw === "string" && parentRaw.trim() ? parentRaw.trim() : null;
    const traceId = typeof rec.trace_id === "string" && rec.trace_id.trim() ? rec.trace_id.trim() : null;
    const toolName = typeof rec.tool_name === "string" && rec.tool_name.trim() ? rec.tool_name.trim() : null;
    const modelRaw = rec.response_model ?? rec.request_model ?? rec.model;
    const model = typeof modelRaw === "string" && modelRaw.trim() ? modelRaw.trim() : null;
    const name = String(
      rec.name ?? rec.span_name ?? rec.operation_name ?? toolName ?? model ?? spanId,
    ).trim();
    const statusRaw = String(rec.status_code ?? rec.status ?? "OK").toUpperCase();
    const errRaw = rec.error ?? (statusRaw === "ERROR" ? rec.status_message : null);
    const errorPreview =
      errRaw !== undefined && errRaw !== null
        ? (typeof errRaw === "string" ? errRaw : JSON.stringify(errRaw)).slice(0, 320)
        : null;
    const isError = statusRaw === "ERROR" || Boolean(rec.is_error) || Boolean(errorPreview);
    const startMs = parseTimestampMs(rec.started_at ?? rec.start_time ?? rec.created_at);
    const endMs = parseTimestampMs(rec.ended_at ?? rec.end_time);
    let durationMs =
      typeof rec.duration_ms === "number" && Number.isFinite(rec.duration_ms)
        ? Math.max(0, rec.duration_ms)
        : typeof rec.total_duration_ms === "number" && Number.isFinite(rec.total_duration_ms)
          ? Math.max(0, rec.total_duration_ms)
          : startMs !== null && endMs !== null && endMs >= startMs
            ? endMs - startMs
            : 0;
    durationMs = Math.round(durationMs * 100) / 100;
    const inTok = typeof rec.input_tokens === "number" ? rec.input_tokens : null;
    const outTok = typeof rec.output_tokens === "number" ? rec.output_tokens : null;

    parsed.push({
      spanId,
      parentSpanId,
      traceId,
      name,
      kind: classifySpanKind(rec, name),
      toolName,
      model,
      statusCode: isError ? "ERROR" : statusRaw || "OK",
      isError,
      errorPreview,
      startMs,
      durationMs,
      inputTokens: inTok,
      outputTokens: outTok,
      attributesJson: stringifyCompactDetails(rec),
    });
  });

  if (parsed.length === 0) {
    return [];
  }

  const knownStarts = parsed.map((s) => s.startMs).filter((v): v is number => v !== null);
  const baseStartMs = knownStarts.length > 0 ? Math.min(...knownStarts) : 0;

  const byId = new Map<string, RawSpanItem>();
  for (const item of parsed) {
    byId.set(item.spanId, item);
  }

  const childrenByParent = new Map<string | null, RawSpanItem[]>();
  for (const item of parsed) {
    const parentKey = item.parentSpanId && byId.has(item.parentSpanId) ? item.parentSpanId : null;
    const list = childrenByParent.get(parentKey) ?? [];
    list.push(item);
    childrenByParent.set(parentKey, list);
  }
  for (const list of childrenByParent.values()) {
    list.sort((a, b) => (a.startMs ?? baseStartMs) - (b.startMs ?? baseStartMs));
  }

  const ordered: Array<{ item: RawSpanItem; depth: number; offsetMs: number }> = [];
  const visited = new Set<string>();
  const visit = (node: RawSpanItem, depth: number) => {
    if (visited.has(node.spanId)) {
      return;
    }
    visited.add(node.spanId);
    const offsetMs =
      node.startMs !== null ? Math.max(0, Math.round((node.startMs - baseStartMs) * 100) / 100) : 0;
    ordered.push({ item: node, depth, offsetMs });
    const kids = childrenByParent.get(node.spanId) ?? [];
    for (const kid of kids) {
      visit(kid, Math.min(depth + 1, 8));
    }
  };

  for (const rootItem of childrenByParent.get(null) ?? []) {
    visit(rootItem, 0);
  }
  for (const item of parsed) {
    if (!visited.has(item.spanId)) {
      visit(item, 0);
    }
  }

  const totalHorizonMs = Math.max(
    1,
    ...ordered.map(({ item, offsetMs }) => offsetMs + Math.max(item.durationMs, 1)),
  );

  return ordered.map(({ item, depth, offsetMs }) => {
    const leftPct = Math.min(96, Math.max(0, (offsetMs / totalHorizonMs) * 100));
    const rawWidth = (Math.max(item.durationMs, totalHorizonMs * 0.015) / totalHorizonMs) * 100;
    const widthPct = Math.max(2, Math.min(100 - leftPct, rawWidth));
    return {
      spanId: item.spanId,
      parentSpanId: item.parentSpanId,
      traceId: item.traceId,
      name: item.name,
      kind: item.kind,
      toolName: item.toolName,
      model: item.model,
      statusCode: item.statusCode,
      isError: item.isError,
      errorPreview: item.errorPreview,
      depth,
      offsetMs,
      durationMs: item.durationMs,
      leftPct: Math.round(leftPct * 100) / 100,
      widthPct: Math.round(widthPct * 100) / 100,
      inputTokens: item.inputTokens,
      outputTokens: item.outputTokens,
      attributesJson: item.attributesJson,
    };
  });
}

export function parseConversationTags(rawJsonOrObj: unknown): AppliedConversationTag[] {
  let root: unknown = rawJsonOrObj;
  if (typeof rawJsonOrObj === "string") {
    try {
      root = JSON.parse(rawJsonOrObj);
    } catch {
      return [];
    }
  }
  if (!root || typeof root !== "object") {
    return [];
  }
  const data = (root as Record<string, unknown>).data ?? root;
  const items: unknown[] = [];
  if (Array.isArray(data)) {
    for (const entry of data) {
      if (entry && typeof entry === "object" && Array.isArray((entry as Record<string, unknown>).tags)) {
        items.push(...((entry as Record<string, unknown>).tags as unknown[]));
      } else {
        items.push(entry);
      }
    }
  } else if (data && typeof data === "object") {
    for (const val of Object.values(data as Record<string, unknown>)) {
      if (Array.isArray(val)) {
        items.push(...val);
      }
    }
  }

  const seen = new Set<string>();
  const out: AppliedConversationTag[] = [];
  for (const raw of items) {
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const rec = raw as Record<string, unknown>;
    const tagId = typeof (rec.tag_id ?? rec.id) === "string" ? String(rec.tag_id ?? rec.id).trim() : null;
    const name = String(rec.tag_name ?? rec.name ?? tagId ?? "").trim();
    if (!name) {
      continue;
    }
    const key = `${tagId ?? ""}:${name}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push({
      tagId: tagId || null,
      name,
      color: typeof rec.color === "string" && rec.color.trim() ? rec.color.trim() : null,
      traceId: typeof rec.trace_id === "string" && rec.trace_id.trim() ? rec.trace_id.trim() : null,
      source: typeof (rec.source ?? rec.applied_by) === "string" ? String(rec.source ?? rec.applied_by) : null,
    });
  }
  return out;
}

export function parseTagCatalog(rawJsonOrObj: unknown): TagCatalogEntry[] {
  let root: unknown = rawJsonOrObj;
  if (typeof rawJsonOrObj === "string") {
    try {
      root = JSON.parse(rawJsonOrObj);
    } catch {
      return [];
    }
  }
  if (!root || typeof root !== "object") {
    return [];
  }
  const data = (root as Record<string, unknown>).data;
  if (!Array.isArray(data)) {
    return [];
  }
  const entries: TagCatalogEntry[] = [];
  for (const item of data) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const rec = item as Record<string, unknown>;
    const id = typeof rec.id === "string" ? rec.id.trim() : "";
    const name = typeof rec.name === "string" ? rec.name.trim() : "";
    if (!id || !name) {
      continue;
    }
    entries.push({
      id,
      name,
      color: typeof rec.color === "string" && rec.color.trim() ? rec.color.trim() : null,
    });
  }
  return entries;
}
