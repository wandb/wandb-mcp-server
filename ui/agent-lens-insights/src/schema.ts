import { z } from "zod";

export const ToolErrorPayloadSchema = z.object({
  error: z.union([z.string(), z.record(z.string(), z.unknown())]),
  message: z.string(),
});

export const ClusterBreakdownItemSchema = z.object({
  id: z.string(),
  title: z.string().nullable().default(null),
  count: z.number().default(0),
  share: z.number().nullable().default(null),
  severity: z.string().nullable().default(null),
});

export const InsightsCategorySchema = z.object({
  categoryId: z.string(),
  label: z.string(),
  count: z.number().default(0),
  share: z.number().default(0),
  severity: z.string().nullable().default(null),
  clusterBreakdowns: z.array(ClusterBreakdownItemSchema).default([]),
});

export const InsightsClusterSchema = z.object({
  id: z.string(),
  title: z.string(),
  summary: z.string().nullable().default(null),
  categoryId: z.string().nullable().default(null),
  count: z.number().nullable().default(null),
  share: z.number().default(0),
  severity: z.string().nullable().default(null),
  attribution: z.string().nullable().default(null),
});

export const SignatureSectionSchema = z.object({
  turnCount: z.number().default(0),
  categories: z.array(InsightsCategorySchema).default([]),
  clusters: z.array(InsightsClusterSchema).default([]),
});

export const FailureAttributionItemSchema = z.object({
  traceId: z.string().nullable().default(null),
  failureId: z.string().nullable().default(null),
  rootCause: z.string(),
  severity: z.string().nullable().default(null),
  summary: z.string().nullable().default(null),
  spanName: z.string().nullable().default(null),
});

export const InsightsPayloadSchema = z.object({
  schema: z.literal("wandb.agent_lens_insights/v1"),
  entityName: z.string(),
  projectName: z.string(),
  forgeUrl: z.string(),
  accessMode: z.enum(["read-only", "read-write"]).default("read-only"),
  window: z.object({
    startAt: z.string(),
    endAt: z.string(),
    windowDays: z.number().nullable().default(null),
    agentName: z.string().nullable().default(null),
  }),
  coverage: z.object({
    firstWeek: z.string().nullable().default(null),
    latestWeek: z.string().nullable().default(null),
    classifiedTurnCount: z.number().nullable().default(null),
    totalTurnCount: z.number().nullable().default(null),
    coveragePct: z.number().nullable().default(null),
  }),
  clusteringStatus: z
    .array(
      z.object({
        signatureType: z.string(),
        status: z.string(),
        completedAt: z.string().nullable().default(null),
      }),
    )
    .default([]),
  failure: SignatureSectionSchema,
  intent: SignatureSectionSchema,
  failureAttributions: z.array(FailureAttributionItemSchema).default([]),
  warnings: z.array(z.string()).default([]),
});

export type InsightsCategory = z.infer<typeof InsightsCategorySchema>;
export type InsightsCluster = z.infer<typeof InsightsClusterSchema>;
export type FailureAttributionItem = z.infer<typeof FailureAttributionItemSchema>;
export type InsightsPayload = z.infer<typeof InsightsPayloadSchema>;

export interface ExampleTurnItem {
  conversationId: string | null;
  traceId: string | null;
  topicId: string | null;
  failureReason: string | null;
  severity: string | null;
  durationMs: number | null;
  totalTokens: number | null;
  costUsd: number | null;
  startedAt: string | null;
  attribution: string | null;
  attributionSummary: string | null;
}

export interface EmbeddedDrawerMessage {
  role: string;
  content: string | null;
  toolName: string | null;
  arguments: string | null;
  toolResult: string | null;
  error: string | null;
}

export interface EmbeddedDrawerTurn {
  turnIndex: number;
  traceId: string | null;
  isError: boolean;
  errorMessage: string | null;
  durationMs: number | null;
  totalTokens: number | null;
  messages: EmbeddedDrawerMessage[];
}

export interface EmbeddedConversationData {
  conversationId: string | null;
  traceId: string | null;
  agentName: string | null;
  turns: EmbeddedDrawerTurn[];
}

export function parseExampleTurnsResponse(raw: unknown): ExampleTurnItem[] {
  let root: unknown = raw;
  if (typeof raw === "string") {
    try {
      root = JSON.parse(raw);
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
  const items: ExampleTurnItem[] = [];
  for (const entry of data) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const rec = entry as Record<string, unknown>;
    const convId = typeof rec.conversation_id === "string" && rec.conversation_id.trim() ? rec.conversation_id.trim() : null;
    const traceId = typeof rec.trace_id === "string" && rec.trace_id.trim() ? rec.trace_id.trim() : null;
    if (!convId && !traceId) {
      continue;
    }
    const reasonRaw = rec.failure_reason ?? rec.reason ?? rec.summary ?? rec.intent_summary;
    items.push({
      conversationId: convId,
      traceId,
      topicId: typeof rec.topic_id === "string" && rec.topic_id.trim() ? rec.topic_id.trim() : null,
      failureReason: typeof reasonRaw === "string" && reasonRaw.trim() ? reasonRaw.trim() : null,
      severity: typeof rec.severity === "string" && rec.severity.trim() ? rec.severity.trim().toLowerCase() : null,
      durationMs: typeof rec.duration_ms === "number" ? rec.duration_ms : null,
      totalTokens: typeof rec.total_tokens === "number" ? rec.total_tokens : null,
      costUsd: typeof rec.cost_usd === "number" ? rec.cost_usd : null,
      startedAt: typeof (rec.started_at ?? rec.created_at) === "string" ? String(rec.started_at ?? rec.created_at) : null,
      attribution: null,
      attributionSummary: null,
    });
  }
  return items;
}

export function enrichTurnsWithAttributions(
  turns: ExampleTurnItem[],
  rawAttributions: unknown,
): ExampleTurnItem[] {
  let root: unknown = rawAttributions;
  if (typeof rawAttributions === "string") {
    try {
      root = JSON.parse(rawAttributions);
    } catch {
      return turns;
    }
  }
  if (!root || typeof root !== "object") {
    return turns;
  }
  const data = (root as Record<string, unknown>).data ?? root;
  const list = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray((data as Record<string, unknown>).attributions)
      ? ((data as Record<string, unknown>).attributions as unknown[])
      : [];
  if (list.length === 0) {
    return turns;
  }
  const byTrace = new Map<string, { cause: string; summary: string | null }>();
  for (const item of list) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const rec = item as Record<string, unknown>;
    const tid = typeof rec.trace_id === "string" ? rec.trace_id.trim() : "";
    if (!tid) {
      continue;
    }
    const cause = String(
      rec.root_cause ?? rec.attribution ?? rec.failure_type ?? rec.category ?? "attributed",
    ).trim();
    const sumRaw = rec.summary ?? rec.reason ?? rec.explanation;
    byTrace.set(tid, {
      cause,
      summary: typeof sumRaw === "string" && sumRaw.trim() ? sumRaw.trim() : null,
    });
  }
  return turns.map((t) => {
    const match = t.traceId ? byTrace.get(t.traceId) : undefined;
    if (!match) {
      return t;
    }
    return {
      ...t,
      attribution: match.cause,
      attributionSummary: match.summary,
    };
  });
}

export function parseEmbeddedConversation(rawStructuredOrJson: unknown): EmbeddedConversationData | null {
  let obj: unknown = rawStructuredOrJson;
  if (typeof rawStructuredOrJson === "string") {
    try {
      obj = JSON.parse(rawStructuredOrJson);
    } catch {
      return null;
    }
  }
  if (!obj || typeof obj !== "object") {
    return null;
  }
  const rec = obj as Record<string, unknown>;

  // Path 1: normalized wandb.agent_conversation/v1 structuredContent
  if (rec.schema === "wandb.agent_conversation/v1" && Array.isArray(rec.turns)) {
    return {
      conversationId: typeof rec.conversationId === "string" ? rec.conversationId : null,
      traceId: typeof rec.traceId === "string" ? rec.traceId : null,
      agentName: typeof rec.agentName === "string" ? rec.agentName : null,
      turns: (rec.turns as Array<Record<string, unknown>>).map((t, i) => ({
        turnIndex: typeof t.turnIndex === "number" ? t.turnIndex : i + 1,
        traceId: typeof t.traceId === "string" ? t.traceId : null,
        isError: Boolean(t.isError),
        errorMessage: typeof t.errorMessage === "string" ? t.errorMessage : null,
        durationMs: typeof t.durationMs === "number" ? t.durationMs : null,
        totalTokens: typeof t.totalTokens === "number" ? t.totalTokens : null,
        messages: Array.isArray(t.messages)
          ? (t.messages as Array<Record<string, unknown>>).map((m) => ({
              role: String(m.role ?? "other"),
              content: typeof m.content === "string" ? m.content : null,
              toolName: typeof m.toolName === "string" ? m.toolName : null,
              arguments: typeof m.arguments === "string" ? m.arguments : null,
              toolResult: typeof m.toolResult === "string" ? m.toolResult : null,
              error: typeof m.error === "string" ? m.error : null,
            }))
          : [],
      })),
    };
  }

  // Path 2: raw Weave /agents/conversations/chat or /agents/traces/chat JSON
  const rawTurns = Array.isArray(rec.turns)
    ? (rec.turns as Array<Record<string, unknown>>)
    : Array.isArray(rec.messages)
      ? [rec]
      : [];
  return {
    conversationId: typeof rec.conversation_id === "string" ? rec.conversation_id : null,
    traceId: typeof rec.trace_id === "string" ? rec.trace_id : null,
    agentName: typeof rec.agent_name === "string" ? rec.agent_name : null,
    turns: rawTurns.map((t, i) => ({
      turnIndex: i + 1,
      traceId: typeof t.trace_id === "string" ? t.trace_id : null,
      isError: String(t.status_code ?? "").toUpperCase() === "ERROR" || Boolean(t.error),
      errorMessage: typeof t.error === "string" ? t.error : null,
      durationMs: typeof (t.total_duration_ms ?? t.duration_ms) === "number"
        ? Number(t.total_duration_ms ?? t.duration_ms)
        : null,
      totalTokens: typeof t.total_tokens === "number" ? t.total_tokens : null,
      messages: Array.isArray(t.messages)
        ? (t.messages as Array<Record<string, unknown>>).map((m) => ({
            role: String(m.role ?? "other"),
            content: typeof m.content === "string" ? m.content : null,
            toolName: typeof (m.tool_name ?? m.name) === "string" ? String(m.tool_name ?? m.name) : null,
            arguments: m.arguments !== undefined ? (typeof m.arguments === "string" ? m.arguments : JSON.stringify(m.arguments, null, 2)) : null,
            toolResult: m.tool_result !== undefined ? (typeof m.tool_result === "string" ? m.tool_result : JSON.stringify(m.tool_result, null, 2)) : null,
            error: typeof m.error === "string" ? m.error : null,
          }))
        : [],
    })),
  };
}
