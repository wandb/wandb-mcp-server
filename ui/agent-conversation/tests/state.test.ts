import { describe, expect, it } from "vitest";
import contractFixture from "../../shared/fixtures/agent-lens-contract.json";
import { isAllowedHttpsUrl } from "../../shared/url-guard";
import {
  buildSpanWaterfall,
  parseConversationTags,
  parseFailureAttributions,
  type ConversationPayload,
} from "../src/schema";
import {
  buildActiveConversationForgeLink,
  buildConversationModelContext,
  canPerformWrites,
  createInitialConversationState,
  getEvidenceSpanIdsSet,
  getVisibleTurns,
  reduceConversationState,
} from "../src/state";

function makeSamplePayload(overrides: Partial<ConversationPayload> = {}): ConversationPayload {
  return {
    schema: "wandb.agent_conversation/v1",
    mode: "conversation",
    entityName: "acme",
    projectName: "support-agent",
    conversationId: "conv-stripe-8841",
    traceId: null,
    agentName: "billing-router",
    agentVersion: "v4.2",
    forgeUrl: "https://forge.coreweave.com/agent-lens/acme/support-agent",
    accessMode: "read-write",
    agentLensAvailable: true,
    compacted: false,
    selectedFailure: {
      traceId: "trace-stripe-8841",
      failureId: "77869ce8-2917-5d33-bc74-ab63603e278e",
    },
    summary: {
      turnCount: 2,
      totalTurns: 2,
      hasMore: false,
      totalInputTokens: 610,
      totalOutputTokens: 180,
      totalTokens: 790,
      totalDurationMs: 1530,
      totalCostUsd: 0.0046,
      errorCount: 1,
    },
    turns: [
      {
        turnIndex: 1,
        traceId: "trace-stripe-8840",
        spanId: "span-turn-8840",
        agentName: "billing-router",
        agentVersion: "v4.2",
        statusCode: "OK",
        isError: false,
        errorMessage: null,
        startedAt: "2026-11-23T21:14:00Z",
        durationMs: 410,
        inputTokens: 220,
        outputTokens: 68,
        totalTokens: 288,
        costUsd: 0.0014,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "user",
            content: "Can you check why EU-West invoice #8841 failed to settle?",
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
        traceId: "trace-stripe-8841",
        spanId: "span-turn-8841",
        agentName: "billing-router",
        agentVersion: "v4.2",
        statusCode: "ERROR",
        isError: true,
        errorMessage: "StripeInvalidRequestError: 422 Unprocessable Entity — Missing required param: currency.",
        startedAt: "2026-11-23T21:14:05Z",
        durationMs: 1120,
        inputTokens: 390,
        outputTokens: 112,
        totalTokens: 502,
        costUsd: 0.0032,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "tool",
            content: null,
            toolName: "lookup_invoice",
            toolCallId: "call_8841",
            arguments: '{"invoice_id":"8841","ledger":"eu-west"}',
            toolCalls: [],
            toolResult: null,
            spanId: "span-tool-8841",
            spanName: "tool.lookup_invoice",
            statusCode: "ERROR",
            error: "422 Unprocessable Entity: Missing required param: currency.",
            durationMs: 640,
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
    expect(isAllowedHttpsUrl("https://forge.coreweave.com/agent-lens/acme/proj")).toBe(true);
    expect(isAllowedHttpsUrl("http://forge.coreweave.com/agent-lens/acme/proj")).toBe(false);
    expect(isAllowedHttpsUrl("javascript:alert(1)")).toBe(false);
    expect(isAllowedHttpsUrl("data:text/html,<h1>hi</h1>")).toBe(false);
    expect(isAllowedHttpsUrl("/relative/path")).toBe(false);
  });
});

describe("Conversation state reducer, agent failures & evidence span sync", () => {
  it("parses real conversation_tags (tag field) and TurnFailureAttribution rows", () => {
    const tags = parseConversationTags(contractFixture.conversation_tags);
    expect(tags).toHaveLength(1);
    expect(tags[0]?.name).toBe("eu-ledger");

    const failures = parseFailureAttributions(contractFixture.failure_attributions);
    expect(failures).toHaveLength(2);
    expect(failures[0]).toMatchObject({
      traceId: "trace-stripe-8841",
      failureId: "77869ce8-2917-5d33-bc74-ab63603e278e",
      failureCategory: "tool_misuse",
      categoryLabel: "Tool misuse",
      failureSeverity: "major",
      evidenceSpanIds: ["span-tool-8841"],
      clusterTitle: "lookup_invoice omits required currency",
    });
  });

  it("pins selected_failure, focuses evidence span in waterfall, and includes failure reason in context", () => {
    const failures = parseFailureAttributions(contractFixture.failure_attributions);
    const spans = buildSpanWaterfall(contractFixture.weave_conversation_spans);

    let state = reduceConversationState(createInitialConversationState(), {
      type: "loaded",
      payload: makeSamplePayload(),
    });
    state = reduceConversationState(state, { type: "failuresLoaded", failures });
    expect(state.activeFailureId).toBe("77869ce8-2917-5d33-bc74-ab63603e278e");
    expect(Array.from(getEvidenceSpanIdsSet(state))).toEqual(["span-tool-8841"]);

    state = reduceConversationState(state, {
      type: "focusFailure",
      failureId: "77869ce8-2917-5d33-bc74-ab63603e278e",
    });
    expect(state.activeTab).toBe("spans");

    state = reduceConversationState(state, { type: "spansLoaded", spans });
    expect(state.selectedSpanId).toBe("span-tool-8841");

    const ctx = buildConversationModelContext(state);
    expect(ctx?.content[0]?.text).toContain("evidence_span=true");
    expect(ctx?.content[0]?.text).toContain("failure_reason=");
    expect(ctx?.structuredContent).toMatchObject({
      span_id: "span-tool-8841",
      tool_name: "lookup_invoice",
      failure_id: "77869ce8-2917-5d33-bc74-ab63603e278e",
      failure_severity: "major",
      is_evidence_span: true,
      url: "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/spans/turns/trace-stripe-8841/spans/span-tool-8841",
    });
    expect(buildActiveConversationForgeLink(state)).toBe(
      "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/spans/turns/trace-stripe-8841/spans/span-tool-8841",
    );
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
});
