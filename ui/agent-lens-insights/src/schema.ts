import { z } from "zod";
import {
  normalizeFailureSeverity,
  type FailureSeverityId,
} from "../../shared/failure-cluster-markdown";

export const ToolErrorPayloadSchema = z.object({
  error: z.union([z.string(), z.record(z.string(), z.unknown())]),
  message: z.string(),
});

export const FailureSeveritySchema = z
  .enum(["info", "minor", "major"])
  .nullable()
  .catch(null)
  .default(null);

export const ClusterBreakdownItemSchema = z.object({
  id: z.string(),
  title: z.string(),
  label: z.string().nullable().default(null),
  description: z.string().nullable().default(null),
  severity: FailureSeveritySchema,
  count: z.number().default(0),
  share: z.number().default(0),
  conversationCount: z.number().default(0),
  failedCount: z.number().default(0),
  frustratedCount: z.number().default(0),
  failureRate: z.number().default(0),
  frustrationRate: z.number().default(0),
  averageLatencyMs: z.number().default(0),
  averageCostPerTurnUsd: z.number().default(0),
});

export const InsightsCategorySchema = z.object({
  categoryId: z.string(),
  label: z.string(),
  count: z.number().default(0),
  share: z.number().default(0),
  conversationCount: z.number().default(0),
  failedCount: z.number().default(0),
  frustratedCount: z.number().default(0),
  failureRate: z.number().default(0),
  frustrationRate: z.number().default(0),
  averageLatencyMs: z.number().default(0),
  averageCostPerTurnUsd: z.number().default(0),
  clusterBreakdowns: z.array(ClusterBreakdownItemSchema).default([]),
});

export const InsightsClusterSchema = z.object({
  id: z.string(),
  title: z.string(),
  label: z.string().nullable().default(null),
  description: z.string().nullable().default(null),
  categoryId: z.string().nullable().default(null),
  categoryLabel: z.string().nullable().default(null),
  count: z.number().nullable().default(null),
  share: z.number().default(0),
  severity: FailureSeveritySchema,
  conversationCount: z.number().nullable().default(null),
  failedCount: z.number().nullable().default(null),
  frustratedCount: z.number().nullable().default(null),
  failureRate: z.number().nullable().default(null),
  frustrationRate: z.number().nullable().default(null),
  averageLatencyMs: z.number().nullable().default(null),
  averageCostPerTurnUsd: z.number().nullable().default(null),
});

export const SignatureSectionSchema = z.object({
  turnCount: z.number().default(0),
  failingTurnCount: z.number().nullable().default(null),
  categories: z.array(InsightsCategorySchema).default([]),
  clusters: z.array(InsightsClusterSchema).default([]),
});

export const FailureAttributionItemSchema = z.object({
  traceId: z.string(),
  sourceTraceId: z.string().nullable().default(null),
  failureId: z.string().nullable().default(null),
  failureCategory: z.string().nullable().default(null),
  categoryLabel: z.string().nullable().default(null),
  failureReason: z.string().nullable().default(null),
  failureSeverity: FailureSeveritySchema,
  failureEvidenceSpanIds: z.array(z.string()).default([]),
  failureCluster: z
    .object({
      id: z.string(),
      title: z.string().nullable().default(null),
    })
    .nullable()
    .default(null),
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
  }),
  clusteringStatus: z
    .array(
      z.object({
        signatureType: z.string(),
        status: z.string(),
        completedAt: z.string().nullable().default(null),
        windowStart: z.string().nullable().default(null),
        windowEnd: z.string().nullable().default(null),
      }),
    )
    .default([]),
  failure: SignatureSectionSchema,
  intent: SignatureSectionSchema,
  failureAttributions: z.array(FailureAttributionItemSchema).default([]),
  warnings: z.array(z.string()).default([]),
});

export type ClusterBreakdownItem = z.infer<typeof ClusterBreakdownItemSchema>;
export type InsightsCategory = z.infer<typeof InsightsCategorySchema>;
export type InsightsCluster = z.infer<typeof InsightsClusterSchema>;
export type FailureAttributionItem = z.infer<typeof FailureAttributionItemSchema>;
export type InsightsPayload = z.infer<typeof InsightsPayloadSchema>;

export interface ExampleTurnItem {
  conversationId: string;
  traceId: string;
  turnNumber: number;
  failureId: string | null;
  failureCategory: string | null;
  failureReason: string | null;
  failureSeverity: FailureSeverityId | null;
  failureEvidenceSpanIds: string[];
  clusterId: string | null;
  clusterTitle: string | null;
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

function parseEvidenceSpanIds(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter((s): s is string => typeof s === "string")
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Parse `/insights/{type}/categories/{id}/example-turns`, which returns
 * `{conversation_id, trace_id, turn_number, failure_id, failure_category, failure_reason, failure_severity, failure_evidence_span_ids}`.
 */
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
  for (let i = 0; i < data.length; i += 1) {
    const entry = data[i];
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const rec = entry as Record<string, unknown>;
    const convId = typeof rec.conversation_id === "string" ? rec.conversation_id.trim() : "";
    const traceId = typeof rec.trace_id === "string" ? rec.trace_id.trim() : "";
    if (!convId || !traceId) {
      continue;
    }
    const turnNumber =
      typeof rec.turn_number === "number" && Number.isFinite(rec.turn_number) && rec.turn_number > 0
        ? Math.trunc(rec.turn_number)
        : i + 1;
    const failureId =
      typeof rec.failure_id === "string" && rec.failure_id.trim() ? rec.failure_id.trim() : null;
    const failureCategory =
      typeof rec.failure_category === "string" && rec.failure_category.trim()
        ? rec.failure_category.trim()
        : null;
    const failureReason =
      typeof rec.failure_reason === "string" && rec.failure_reason.trim()
        ? rec.failure_reason.trim()
        : null;
    items.push({
      conversationId: convId,
      traceId,
      turnNumber,
      failureId,
      failureCategory,
      failureReason,
      failureSeverity: normalizeFailureSeverity(rec.failure_severity),
      failureEvidenceSpanIds: parseEvidenceSpanIds(rec.failure_evidence_span_ids),
      clusterId: null,
      clusterTitle: null,
    });
  }
  return items;
}

/**
 * Enrich example turns with `/insights/failure-attributions/query` `TurnFailureAttribution` rows
 * (`{trace_id, source_trace_id, failure_id, failure_category, failure_reason, failure_severity, failure_evidence_span_ids, failure_cluster: {id, title}}`).
 */
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
  const data = (root as Record<string, unknown>).data;
  if (!Array.isArray(data) || data.length === 0) {
    return turns;
  }

  interface AttributionRow {
    failureId: string | null;
    failureCategory: string | null;
    failureReason: string | null;
    failureSeverity: FailureSeverityId | null;
    evidenceSpanIds: string[];
    clusterId: string | null;
    clusterTitle: string | null;
  }

  const byTrace = new Map<string, AttributionRow>();
  const byTraceAndFailure = new Map<string, AttributionRow>();

  for (const item of data) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const rec = item as Record<string, unknown>;
    const tid = typeof rec.trace_id === "string" ? rec.trace_id.trim() : "";
    if (!tid) {
      continue;
    }
    const fid =
      typeof rec.failure_id === "string" && rec.failure_id.trim() ? rec.failure_id.trim() : null;
    const fc = rec.failure_cluster;
    let clusterId: string | null = null;
    let clusterTitle: string | null = null;
    if (fc && typeof fc === "object") {
      const fcRec = fc as Record<string, unknown>;
      clusterId = typeof fcRec.id === "string" && fcRec.id.trim() ? fcRec.id.trim() : null;
      clusterTitle =
        typeof fcRec.title === "string" && fcRec.title.trim() ? fcRec.title.trim() : null;
    }
    const parsed: AttributionRow = {
      failureId: fid,
      failureCategory:
        typeof rec.failure_category === "string" && rec.failure_category.trim()
          ? rec.failure_category.trim()
          : null,
      failureReason:
        typeof rec.failure_reason === "string" && rec.failure_reason.trim()
          ? rec.failure_reason.trim()
          : null,
      failureSeverity: normalizeFailureSeverity(rec.failure_severity),
      evidenceSpanIds: parseEvidenceSpanIds(rec.failure_evidence_span_ids),
      clusterId,
      clusterTitle,
    };
    if (fid) {
      byTraceAndFailure.set(`${tid}:${fid}`, parsed);
    }
    if (!byTrace.has(tid)) {
      byTrace.set(tid, parsed);
    }
  }

  return turns.map((t) => {
    const match =
      (t.failureId ? byTraceAndFailure.get(`${t.traceId}:${t.failureId}`) : undefined) ??
      byTrace.get(t.traceId);
    if (!match) {
      return t;
    }
    return {
      ...t,
      failureId: t.failureId ?? match.failureId,
      failureCategory: t.failureCategory ?? match.failureCategory,
      failureReason: t.failureReason ?? match.failureReason,
      failureSeverity: t.failureSeverity ?? match.failureSeverity,
      failureEvidenceSpanIds:
        t.failureEvidenceSpanIds.length > 0 ? t.failureEvidenceSpanIds : match.evidenceSpanIds,
      clusterId: match.clusterId ?? t.clusterId,
      clusterTitle: match.clusterTitle ?? t.clusterTitle,
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
      durationMs:
        typeof (t.total_duration_ms ?? t.duration_ms) === "number"
          ? Number(t.total_duration_ms ?? t.duration_ms)
          : null,
      totalTokens: typeof t.total_tokens === "number" ? t.total_tokens : null,
      messages: Array.isArray(t.messages)
        ? (t.messages as Array<Record<string, unknown>>).map((m) => ({
            role: String(m.role ?? "other"),
            content: typeof m.content === "string" ? m.content : null,
            toolName:
              typeof (m.tool_name ?? m.name) === "string"
                ? String(m.tool_name ?? m.name)
                : null,
            arguments:
              m.arguments !== undefined
                ? typeof m.arguments === "string"
                  ? m.arguments
                  : JSON.stringify(m.arguments, null, 2)
                : null,
            toolResult:
              m.tool_result !== undefined
                ? typeof m.tool_result === "string"
                  ? m.tool_result
                  : JSON.stringify(m.tool_result, null, 2)
                : null,
            error: typeof m.error === "string" ? m.error : null,
          }))
        : [],
    })),
  };
}
