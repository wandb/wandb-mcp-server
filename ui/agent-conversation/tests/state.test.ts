import { describe, expect, it } from "vitest";
import { isAllowedHttpsUrl } from "../../shared/url-guard";
import { buildSpanWaterfall, parseConversationTags, type ConversationPayload } from "../src/schema";
import {
  buildConversationModelContext,
  canPerformWrites,
  createInitialConversationState,
  getVisibleTurns,
  reduceConversationState,
} from "../src/state";

function makeSamplePayload(overrides: Partial<ConversationPayload> = {}): ConversationPayload {
  return {
    schema: "wandb.agent_conversation/v1",
    mode: "conversation",
    entityName: "acme",
    projectName: "support-agent",
    conversationId: "conv-42",
    traceId: null,
    agentName: "billing-router",
    agentVersion: "v3",
    forgeUrl: "https://forge.coreweave.com/acme/support-agent/agent-lens",
    accessMode: "read-write",
    agentLensAvailable: true,
    compacted: false,
    summary: {
      turnCount: 2,
      totalTurns: 2,
      hasMore: false,
      totalInputTokens: 520,
      totalOutputTokens: 180,
      totalTokens: 700,
      totalDurationMs: 1450,
      totalCostUsd: 0.0042,
      errorCount: 1,
    },
    turns: [
      {
        turnIndex: 1,
        traceId: "trace-ok",
        spanId: "span-turn-1",
        agentName: "billing-router",
        agentVersion: "v3",
        statusCode: "OK",
        isError: false,
        errorMessage: null,
        startedAt: "2026-11-23T18:00:00Z",
        durationMs: 450,
        inputTokens: 200,
        outputTokens: 80,
        totalTokens: 280,
        costUsd: 0.0015,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "user",
            content: "Check invoice #99",
            toolName: null,
            toolCallId: null,
            arguments: null,
            toolCalls: [],
            toolResult: null,
            spanId: null,
            spanName: null,
            statusCode: null,
            error: null,
            durationMs: null,
            model: null,
          },
        ],
        feedback: [],
      },
      {
        turnIndex: 2,
        traceId: "trace-err",
        spanId: "span-turn-2",
        agentName: "billing-router",
        agentVersion: "v3",
        statusCode: "ERROR",
        isError: true,
        errorMessage: "Stripe lookup timed out after 1000ms",
        startedAt: "2026-11-23T18:01:00Z",
        durationMs: 1000,
        inputTokens: 320,
        outputTokens: 100,
        totalTokens: 420,
        costUsd: 0.0027,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "tool",
            content: null,
            toolName: "lookup_invoice",
            toolCallId: "call-1",
            arguments: '{"invoice_id":"99"}',
            toolCalls: [],
            toolResult: null,
            spanId: "span-tool-99",
            spanName: "tool.lookup_invoice",
            statusCode: "ERROR",
            error: "TimeoutError",
            durationMs: 990,
            model: null,
          },
        ],
        feedback: [],
      },
    ],
    feedback: [],
    ...overrides,
  };
}

describe("url-guard", () => {
  it("allows valid https deep links and rejects non-https schemes", () => {
    expect(isAllowedHttpsUrl("https://forge.coreweave.com/acme/proj/agent-lens")).toBe(true);
    expect(isAllowedHttpsUrl("http://forge.coreweave.com/acme/proj")).toBe(false);
    expect(isAllowedHttpsUrl("javascript:alert(1)")).toBe(false);
    expect(isAllowedHttpsUrl("data:text/html,<h1>hi</h1>")).toBe(false);
    expect(isAllowedHttpsUrl("/relative/path")).toBe(false);
  });
});

describe("Conversation state reducer & context sync", () => {
  it("auto-selects first failing turn and publishes structured context", () => {
    let state = createInitialConversationState();
    state = reduceConversationState(state, { type: "loaded", payload: makeSamplePayload() });

    expect(state.status).toBe("ready");
    expect(state.selectedTarget?.turnIndex).toBe(2);
    expect(state.selectedTarget?.traceId).toBe("trace-err");
    expect(state.selectedTarget?.toolName).toBe("lookup_invoice");

    const ctx = buildConversationModelContext(state);
    expect(ctx?.content[0]?.text).toContain("[MCP Agent Conversation Context]");
    expect(ctx?.structuredContent).toMatchObject({
      entity_name: "acme",
      project_name: "support-agent",
      conversation_id: "conv-42",
      trace_id: "trace-err",
      tool_name: "lookup_invoice",
    });
  });

  it("filters errors-only turns and gates write actions in read-only mode", () => {
    let state = reduceConversationState(createInitialConversationState(), {
      type: "loaded",
      payload: makeSamplePayload({ accessMode: "read-only" }),
    });
    expect(canPerformWrites(state, true)).toBe(false);

    state = reduceConversationState(state, { type: "toggleErrorsOnly" });
    const visible = getVisibleTurns(state);
    expect(visible).toHaveLength(1);
    expect(visible[0]?.turnIndex).toBe(2);
  });

  it("records inline feedback and tag updates cleanly", () => {
    let state = reduceConversationState(createInitialConversationState(), {
      type: "loaded",
      payload: makeSamplePayload(),
    });
    expect(canPerformWrites(state, true)).toBe(true);

    state = reduceConversationState(state, {
      type: "feedbackAdded",
      turnIndex: 2,
      item: {
        id: "fb-1",
        feedbackType: "wandb.note.1",
        note: "Increase timeout",
        emoji: null,
        traceId: "trace-err",
        spanId: "span-turn-2",
        createdAt: "2026-11-23T18:02:00Z",
        createdBy: null,
      },
    });
    expect(state.payload?.turns[1]?.feedback).toHaveLength(1);

    const parsedTags = parseConversationTags({
      data: [{ tag_id: "tag-uuid-1", tag_name: "escalated", trace_id: "trace-err" }],
    });
    state = reduceConversationState(state, { type: "tagsLoaded", tags: parsedTags });
    expect(state.tags).toHaveLength(1);
    expect(state.tags[0]?.name).toBe("escalated");
  });

  it("builds nested OTel span waterfall and updates context on span click", () => {
    const spans = buildSpanWaterfall({
      conversations: [
        {
          conversation_id: "conv-42",
          spans: [
            {
              span_id: "root-turn",
              trace_id: "trace-err",
              name: "agent.turn",
              is_invocation: true,
              started_at: "2026-11-23T18:01:00.000Z",
              duration_ms: 1000,
              status_code: "ERROR",
            },
            {
              span_id: "child-llm",
              parent_id: "root-turn",
              trace_id: "trace-err",
              name: "openai.responses",
              request_model: "gpt-5.4",
              started_at: "2026-11-23T18:01:00.050Z",
              duration_ms: 350,
              status_code: "OK",
            },
            {
              span_id: "child-tool",
              parent_id: "root-turn",
              trace_id: "trace-err",
              name: "tool.lookup_invoice",
              tool_name: "lookup_invoice",
              started_at: "2026-11-23T18:01:00.420Z",
              duration_ms: 580,
              status_code: "ERROR",
              error: "Stripe socket timeout",
            },
          ],
        },
      ],
    });

    expect(spans).toHaveLength(3);
    expect(spans[0]?.depth).toBe(0);
    expect(spans[1]?.depth).toBe(1);
    expect(spans[1]?.kind).toBe("llm");
    expect(spans[2]?.kind).toBe("tool");
    expect(spans[2]?.isError).toBe(true);

    let state = reduceConversationState(createInitialConversationState(), {
      type: "loaded",
      payload: makeSamplePayload(),
    });
    state = reduceConversationState(state, { type: "spansLoaded", spans });
    state = reduceConversationState(state, { type: "selectSpan", spanId: "child-tool" });

    const ctx = buildConversationModelContext(state);
    expect(ctx?.structuredContent).toMatchObject({
      span_id: "child-tool",
      tool_name: "lookup_invoice",
      error_preview: "Stripe socket timeout",
    });
  });
});
