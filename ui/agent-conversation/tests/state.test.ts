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

  it("builds nested span waterfall from /agents/spans/query AgentSpanSchema rows and matches 32-hex vs UUID traceIds", () => {
    const spansQueryResponse = {
      total_count: 2,
      groups: [],
      spans: [
        {
          project_id: "acme/composer-prod",
          trace_id: "019a4f10884170008000abcdef123456",
          span_id: "span-root-1",
          parent_span_id: null,
          span_name: "studio_turn",
          span_kind: "INTERNAL",
          started_at: "2026-11-23T18:10:00.000Z",
          ended_at: "2026-11-23T18:10:03.620Z",
          status_code: "UNSET",
          status_message: null,
          operation_name: "agent.turn",
        },
        {
          project_id: "acme/composer-prod",
          trace_id: "019a4f10884170008000abcdef123456",
          span_id: "span-tool-edit",
          parent_span_id: "span-root-1",
          span_name: "edit_track_region",
          span_kind: "CLIENT",
          started_at: "2026-11-23T18:10:01.100Z",
          ended_at: "2026-11-23T18:10:01.510Z",
          status_code: "OK",
          status_message: null,
          tool_name: "edit_track_region",
          tool_call_arguments: '{"stem":"lead_vox","bars":[9,12]}',
          tool_call_result: '{"ok":true,"mutated_stem":"lead_vox"}',
          custom_attrs_string: { focused_track_id: "chorus_synth" },
        },
      ],
    };

    const waterfall = buildSpanWaterfall(spansQueryResponse);
    expect(waterfall).toHaveLength(2);
    expect(waterfall[0]).toMatchObject({
      spanId: "span-root-1",
      depth: 0,
      statusCode: "OK",
      durationMs: 3620,
    });
    expect(waterfall[1]).toMatchObject({
      spanId: "span-tool-edit",
      parentSpanId: "span-root-1",
      depth: 1,
      offsetMs: 1100,
      durationMs: 410,
      kind: "tool",
      toolName: "edit_track_region",
    });
    expect(waterfall[1]?.attributesJson).toContain('"focused_track_id": "chorus_synth"');
    expect(waterfall[1]?.attributesJson).toContain('"mutated_stem": "lead_vox"');

    let state = reduceConversationState(createInitialConversationState(), {
      type: "loaded",
      payload: makeSamplePayload({
        turns: [
          {
            turnIndex: 1,
            traceId: "019a4f10-8841-7000-8000-abcdef123456",
            spanId: "span-root-1",
            agentName: "composer-v2",
            agentVersion: "2.4.0",
            statusCode: "OK",
            isError: false,
            errorMessage: null,
            startedAt: "2026-11-23T18:10:00Z",
            durationMs: 3620,
            inputTokens: 1200,
            outputTokens: 180,
            totalTokens: 1380,
            costUsd: 0.0065,
            model: "claude-sonnet-4.5",
            messages: [],
            feedback: [],
          },
        ],
      }),
    });
    state = reduceConversationState(state, {
      type: "failuresLoaded",
      failures: [
        {
          traceId: "019a4f10884170008000abcdef123456",
          sourceTraceId: null,
          failureId: "fail-hex-1",
          failureCategory: "tool_misuse",
          categoryLabel: "Tool misuse",
          failureReason: "Mutated lead_vox instead of chorus_synth",
          failureSeverity: "major",
          evidenceSpanIds: ["span-tool-edit"],
          clusterId: "cl-1",
          clusterTitle: "Wrong stem mutated",
        },
      ],
    });
    state = reduceConversationState(state, { type: "toggleErrorsOnly" });
    expect(getVisibleTurns(state)).toHaveLength(1);
  });
});
