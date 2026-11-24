import { describe, expect, it } from "vitest";
import {
  enrichTurnsWithAttributions,
  parseEmbeddedConversation,
  parseExampleTurnsResponse,
  type InsightsPayload,
} from "../src/schema";
import {
  buildInsightsModelContext,
  createInitialInsightsState,
  reduceInsightsState,
} from "../src/state";

function makeInsightsPayload(overrides: Partial<InsightsPayload> = {}): InsightsPayload {
  return {
    schema: "wandb.agent_lens_insights/v1",
    entityName: "acme",
    projectName: "support-agent",
    forgeUrl: "https://forge.coreweave.com/acme/support-agent/agent-lens",
    accessMode: "read-write",
    window: {
      startAt: "2026-11-16T00:00:00Z",
      endAt: "2026-11-23T00:00:00Z",
      windowDays: 7,
      agentName: null,
    },
    coverage: {
      firstWeek: "2026-10-05",
      latestWeek: "2026-11-16",
      classifiedTurnCount: 940,
      totalTurnCount: 1000,
      coveragePct: 94.0,
    },
    clusteringStatus: [
      { signatureType: "failure", status: "ready", completedAt: "2026-11-23T02:00:00Z" },
    ],
    failure: {
      turnCount: 40,
      categories: [
        {
          categoryId: "tool_failure",
          label: "Tool Failure",
          count: 28,
          share: 0.7,
          severity: "high",
          clusterBreakdowns: [{ id: "topic-stripe", title: "Stripe Timeout", count: 18, share: 0.45, severity: "high" }],
        },
      ],
      clusters: [
        {
          id: "topic-stripe",
          title: "Stripe Timeout",
          summary: "Invoice lookup calls timing out on large accounts",
          categoryId: "tool_failure",
          count: 18,
          share: 0.45,
          severity: "high",
          attribution: "tool_error",
        },
      ],
    },
    intent: {
      turnCount: 900,
      categories: [
        {
          categoryId: "billing_inquiry",
          label: "Billing Inquiry",
          count: 540,
          share: 0.6,
          severity: null,
          clusterBreakdowns: [],
        },
      ],
      clusters: [],
    },
    failureAttributions: [],
    warnings: [],
    ...overrides,
  };
}

describe("Insights schema & state reducer", () => {
  it("selects top failure category on initial load and builds model context", () => {
    let state = createInitialInsightsState();
    state = reduceInsightsState(state, { type: "loaded", payload: makeInsightsPayload() });

    expect(state.status).toBe("ready");
    expect(state.activeTab).toBe("failures");
    expect(state.selectedContext?.categoryId).toBe("tool_failure");

    const ctx = buildInsightsModelContext(state);
    expect(ctx?.content[0]?.text).toContain("[MCP Agent Lens Insights Context]");
    expect(ctx?.structuredContent).toMatchObject({
      entity_name: "acme",
      project_name: "support-agent",
      signature_type: "failure",
      category_id: "tool_failure",
    });
  });

  it("parses example turns, enriches failure attributions, and parses embedded conversation", () => {
    const rawTurns = {
      data: [
        {
          conversation_id: "conv-1",
          trace_id: "trace-1",
          failure_reason: "Stripe 504 gateway timeout",
          duration_ms: 1250,
          total_tokens: 410,
        },
      ],
    };
    const turns = parseExampleTurnsResponse(rawTurns);
    expect(turns).toHaveLength(1);

    const enriched = enrichTurnsWithAttributions(turns, {
      data: [
        {
          trace_id: "trace-1",
          root_cause: "tool_error",
          summary: "lookup_invoice exceeded upstream timeout",
        },
      ],
    });
    expect(enriched[0]?.attribution).toBe("tool_error");
    expect(enriched[0]?.attributionSummary).toBe("lookup_invoice exceeded upstream timeout");

    const embedded = parseEmbeddedConversation({
      schema: "wandb.agent_conversation/v1",
      conversationId: "conv-1",
      traceId: "trace-1",
      agentName: "billing-router",
      turns: [
        {
          turnIndex: 1,
          traceId: "trace-1",
          isError: true,
          errorMessage: "Timeout",
          durationMs: 1250,
          totalTokens: 410,
          messages: [{ role: "user", content: "Refund order #12" }],
        },
      ],
    });
    expect(embedded?.conversationId).toBe("conv-1");
    expect(embedded?.turns[0]?.messages[0]?.content).toBe("Refund order #12");
  });

  it("preserves existing payload non-destructively when window update fails", () => {
    let state = reduceInsightsState(createInitialInsightsState(), {
      type: "loaded",
      payload: makeInsightsPayload(),
    });
    state = reduceInsightsState(state, { type: "windowUpdateStarted" });
    expect(state.isWindowUpdating).toBe(true);

    state = reduceInsightsState(state, { type: "loadFailed", message: "Upstream 503" });
    expect(state.status).toBe("ready");
    expect(state.payload?.projectName).toBe("support-agent");
    expect(state.banner?.message).toBe("Upstream 503");
  });
});
