import { z } from "zod";

export const TurnStateSchema = z.enum([
  "queued",
  "in_progress",
  "completed",
  "errored",
  "cancelled",
]);
export type TurnState = z.infer<typeof TurnStateSchema>;

export const AccessModeSchema = z.enum(["read-only", "read-write"]);
export type AccessMode = z.infer<typeof AccessModeSchema>;

export const ToolStepSchema = z.object({
  call_id: z.string(),
  name: z.string(),
  status: z.enum(["running", "ok", "recovered_error"]),
  started_at: z.string().nullable().optional(),
  ended_at: z.string().nullable().optional(),
});
export type AriaToolStep = z.infer<typeof ToolStepSchema>;

export const PermissionRequestSchema = z
  .object({
    type: z.string().optional(),
    reason: z.string().nullable().optional(),
    domains: z.array(z.string()).nullable().optional(),
  })
  .passthrough();
export type AriaPermissionRequest = z.infer<typeof PermissionRequestSchema>;

export const AgentQuestionSchema = z
  .object({
    id: z.string().nullable().optional(),
    type: z.string().nullable().optional(),
    question: z.string(),
    options: z.array(z.string()).nullable().optional(),
  })
  .passthrough();
export type AriaAgentQuestion = z.infer<typeof AgentQuestionSchema>;

export const LatestToolActivitySchema = z
  .object({
    type: z.string().optional(),
    name: z.string().optional(),
    call_id: z.string().optional(),
    is_error: z.boolean().optional(),
    recovered_error: z.boolean().optional(),
    timestamp: z.string().optional(),
  })
  .passthrough();

export const ProgressSummarySchema = z
  .object({
    phase: z.string().optional(),
    status_text: z.string().optional(),
    elapsed_seconds: z.number().nullable().optional(),
    message_count: z.number().default(0),
    tool_call_record_count: z.number().default(0),
    tool_error_count: z.number().optional(),
    internal_error_note: z.string().optional(),
    requested_network_domains: z.array(z.string()).optional(),
    rejected_network_domains: z.array(z.string()).optional(),
    requires_permission: z.boolean().optional(),
    requires_user_input: z.boolean().optional(),
    latest_tool_activity: LatestToolActivitySchema.optional(),
    poll_interrupted: z
      .object({
        type: z.string().optional(),
        message: z.string().optional(),
        retryable: z.boolean().optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();
export type AriaProgressSummary = z.infer<typeof ProgressSummarySchema>;

export const ScopeSchema = z.object({
  entity: z.string().nullable().optional(),
  project: z.string().nullable().optional(),
});
export type AriaScope = z.infer<typeof ScopeSchema>;

export const AriaTurnPayloadSchema = z
  .object({
    ok: z.boolean().default(true),
    turn_id: z.string().min(1),
    thread_id: z.string().nullable().optional(),
    parent_turn_id: z.string().nullable().optional(),
    resumed_from_turn_id: z.string().nullable().optional(),
    continued_in_turn_id: z.string().nullable().optional(),
    title: z.string().nullable().optional(),
    user_prompt: z.string().nullable().optional(),
    scope: ScopeSchema.default({ entity: null, project: null }),
    state: TurnStateSchema,
    is_terminal: z.boolean(),
    created_at: z.string().nullable().optional(),
    updated_at: z.string().nullable().optional(),
    poll_after_seconds: z.number().nullable().optional(),
    latest_response: z.string().nullable().optional(),
    error_info: z.string().nullable().optional(),
    agent_questions: z.array(AgentQuestionSchema).nullable().optional(),
    permission_requests: z.array(PermissionRequestSchema).nullable().optional(),
    active_allowed_network_domains: z.array(z.string()).nullable().optional(),
    requires_permission: z.boolean().optional(),
    requires_user_input: z.boolean().optional(),
    tool_steps: z.array(ToolStepSchema).optional(),
    progress: ProgressSummarySchema.optional(),
    next_action: z.string().nullable().optional(),
    forge_url: z.string().nullable().optional(),
    forge_home_url: z.string().nullable().optional(),
    access_mode: AccessModeSchema.optional(),
    error: z
      .object({
        type: z.string().optional(),
        message: z.string(),
        retryable: z.boolean().optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();
export type AriaTurnPayload = z.infer<typeof AriaTurnPayloadSchema>;

export const ParallelSummarySchema = z.object({
  total_count: z.number(),
  completed_count: z.number(),
  active_count: z.number(),
  awaiting_permission_count: z.number().default(0),
  awaiting_user_input_count: z.number().default(0),
  errored_count: z.number().default(0),
  cancelled_count: z.number().default(0),
  all_terminal: z.boolean(),
});
export type AriaParallelSummary = z.infer<typeof ParallelSummarySchema>;

export const AriaParallelBatchPayloadSchema = AriaTurnPayloadSchema.extend({
  parallel_turns: z.array(AriaTurnPayloadSchema).min(1),
  parallel_summary: ParallelSummarySchema.optional(),
});
export type AriaParallelBatchPayload = z.infer<typeof AriaParallelBatchPayloadSchema>;

export const AriaThreadSummarySchema = z
  .object({
    thread_id: z.string().min(1),
    title: z.string().nullable().optional(),
    prompt_preview: z.string().nullable().optional(),
    scope: ScopeSchema.default({ entity: null, project: null }),
    is_pinned: z.boolean().nullable().optional(),
    visibility: z.string().nullable().optional(),
    created_at: z.string().nullable().optional(),
    last_turn_created_at: z.string().nullable().optional(),
    forge_url: z.string().nullable().optional(),
  })
  .passthrough();
export type AriaThreadSummary = z.infer<typeof AriaThreadSummarySchema>;

export const AriaThreadTurnSummarySchema = z.object({
  turn_id: z.string().min(1),
  parent_turn_id: z.string().nullable().optional(),
  state: TurnStateSchema,
  is_terminal: z.boolean(),
  created_at: z.string().nullable().optional(),
  updated_at: z.string().nullable().optional(),
});
export type AriaThreadTurnSummary = z.infer<typeof AriaThreadTurnSummarySchema>;

export const AriaThreadPayloadSchema = z
  .object({
    ok: z.literal(true),
    thread: AriaThreadSummarySchema,
    turns: z.array(AriaThreadTurnSummarySchema),
    newest_completed_turn_id: z.string().nullable().optional(),
    count: z.number().default(0),
    offset: z.number().default(0),
    has_more: z.boolean().default(false),
    next_offset: z.number().nullable().optional(),
    access_mode: AccessModeSchema.optional(),
    forge_home_url: z.string().nullable().optional(),
  })
  .passthrough();
export type AriaThreadPayload = z.infer<typeof AriaThreadPayloadSchema>;

export const AriaThreadsListPayloadSchema = z
  .object({
    ok: z.literal(true),
    threads: z.array(AriaThreadSummarySchema),
    count: z.number().default(0),
    offset: z.number().default(0),
    has_more: z.boolean().default(false),
    next_offset: z.number().nullable().optional(),
    access_mode: AccessModeSchema.optional(),
    forge_home_url: z.string().nullable().optional(),
  })
  .passthrough();
export type AriaThreadsListPayload = z.infer<typeof AriaThreadsListPayloadSchema>;

export type ParsedAriaToolEnvelope =
  | { kind: "parallel_turns"; data: AriaParallelBatchPayload }
  | { kind: "turn"; data: AriaTurnPayload }
  | { kind: "thread"; data: AriaThreadPayload }
  | { kind: "threads_list"; data: AriaThreadsListPayload }
  | { kind: "error"; message: string; retryable: boolean; turnId: string | null };

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function unwrapRawPayload(result: {
  structuredContent?: unknown;
  content?: Array<{ type: string; text?: string }>;
}): unknown {
  if (isPlainRecord(result.structuredContent)) {
    if (isPlainRecord(result.structuredContent.result)) {
      return result.structuredContent.result;
    }
    return result.structuredContent;
  }
  const textItem = result.content?.find(
    (item) => item.type === "text" && typeof item.text === "string",
  );
  if (textItem?.text) {
    const rawText = textItem.text.trim();
    const jsonCandidate = rawText.includes(": {")
      ? rawText.slice(rawText.indexOf("{"))
      : rawText;
    try {
      const parsed: unknown = JSON.parse(jsonCandidate);
      if (isPlainRecord(parsed) && isPlainRecord(parsed.result)) {
        return parsed.result;
      }
      return parsed;
    } catch {
      return rawText;
    }
  }
  return null;
}

export function parseAriaToolEnvelope(result: {
  isError?: boolean;
  structuredContent?: unknown;
  content?: Array<{ type: string; text?: string }>;
}): ParsedAriaToolEnvelope {
  const unwrapped = unwrapRawPayload(result);

  if (result.isError || (isPlainRecord(unwrapped) && unwrapped.ok === false)) {
    if (isPlainRecord(unwrapped)) {
      const errObj = isPlainRecord(unwrapped.error) ? unwrapped.error : null;
      const msg =
        (errObj && typeof errObj.message === "string" && errObj.message) ||
        (typeof unwrapped.message === "string" && unwrapped.message) ||
        "ARIA tool call failed.";
      const retryable = Boolean(errObj && errObj.retryable === true);
      const turnId = typeof unwrapped.turn_id === "string" ? unwrapped.turn_id : null;
      return { kind: "error", message: msg, retryable, turnId };
    }
    if (typeof unwrapped === "string" && unwrapped) {
      return {
        kind: "error",
        message: unwrapped.slice(0, 320),
        retryable: false,
        turnId: null,
      };
    }
    return {
      kind: "error",
      message: "ARIA request failed.",
      retryable: false,
      turnId: null,
    };
  }

  if (isPlainRecord(unwrapped) && Array.isArray(unwrapped.parallel_turns)) {
    const batch = AriaParallelBatchPayloadSchema.safeParse(unwrapped);
    if (batch.success) {
      return { kind: "parallel_turns", data: batch.data };
    }
  }

  if (isPlainRecord(unwrapped) && "turn_id" in unwrapped && "state" in unwrapped) {
    const single = AriaTurnPayloadSchema.safeParse(unwrapped);
    if (single.success) {
      return { kind: "turn", data: single.data };
    }
  }

  if (isPlainRecord(unwrapped) && isPlainRecord(unwrapped.thread) && Array.isArray(unwrapped.turns)) {
    const thread = AriaThreadPayloadSchema.safeParse(unwrapped);
    if (thread.success) {
      return { kind: "thread", data: thread.data };
    }
  }

  if (isPlainRecord(unwrapped) && Array.isArray(unwrapped.threads)) {
    const list = AriaThreadsListPayloadSchema.safeParse(unwrapped);
    if (list.success) {
      return { kind: "threads_list", data: list.data };
    }
  }

  return {
    kind: "error",
    message: "Received an unrecognized ARIA response payload.",
    retryable: false,
    turnId: null,
  };
}
