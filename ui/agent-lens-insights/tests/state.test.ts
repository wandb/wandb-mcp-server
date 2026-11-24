import { describe, expect, it } from "vitest";
import contractFixture from "../../shared/fixtures/agent-lens-contract.json";
import {
  InsightsPayloadSchema,
  enrichTurnsWithAttributions,
  parseEmbeddedConversation,
  parseExampleTurnsResponse,
  type InsightsPayload,
} from "../src/schema";
import {
  buildActiveForgeLink,
  buildInsightsModelContext,
  createInitialInsightsState,
  reduceInsightsState,
} from "../src/state";

function makeInsightsPayload(overrides: Partial<InsightsPayload> = {}): InsightsPayload {
  const f = contractFixture.failure_breakdowns.data;
  const i = contractFixture.intent_breakdowns.data;
  const toolMisuse = f.categories.find((c) => c.category === "tool_misuse")!;
  const clusterRow = toolMisuse.cluster_breakdowns[0]!;

  return InsightsPayloadSchema.parse({
    schema: "wandb.agent_lens_insights/v1",
    entityName: "acme",
    projectName: "support-agent",
    forgeUrl: "https://forge.coreweave.com/agent-lens/acme/support-agent",
    accessMode: "read-write",
    window: {
      startAt: contractFixture.window_7d.start_at,
      endAt: contractFixture.window_7d.end_at,
      windowDays: 7,
      agentName: contractFixture.project.agent_name,
    },
    coverage: {
      firstWeek: contractFixture.latest_week.data.first_week,
      latestWeek: contractFixture.latest_week.data.latest_week,
    },
    clusteringStatus: contractFixture.clustering_status.data.map((row) => ({
      signatureType: row.signature_type,
      status: "completed",
      completedAt: row.completed_at,
      windowStart: row.window_start,
      windowEnd: row.window_end,
    })),
    failure: {
      turnCount: f.turn_count,
      failingTurnCount: 612,
      categories: [
        {
          categoryId: "tool_misuse",
          label: "Tool misuse",
          count: toolMisuse.count,
          share: toolMisuse.count / f.turn_count,
          conversationCount: toolMisuse.conversation_count,
          failedCount: toolMisuse.failed_count,
          frustratedCount: toolMisuse.frustrated_count,
          failureRate: toolMisuse.failed_count / toolMisuse.count,
          frustrationRate: toolMisuse.frustrated_count / toolMisuse.count,
          averageLatencyMs: toolMisuse.average_latency_ms,
          averageCostPerTurnUsd: toolMisuse.average_cost_per_turn_usd,
          clusterBreakdowns: [
            {
              id: clusterRow.id,
              title: clusterRow.title,
              label: clusterRow.label,
              description: clusterRow.description,
              severity: clusterRow.severity,
              count: clusterRow.count,
              share: clusterRow.count / f.turn_count,
              conversationCount: clusterRow.conversation_count,
              failedCount: clusterRow.failed_count,
              frustratedCount: clusterRow.frustrated_count,
              failureRate: 1,
              frustrationRate: clusterRow.frustrated_count / clusterRow.count,
              averageLatencyMs: clusterRow.average_latency_ms,
              averageCostPerTurnUsd: clusterRow.average_cost_per_turn_usd,
            },
          ],
        },
      ],
      clusters: [],
    },
    intent: {
      turnCount: i.turn_count,
      failingTurnCount: null,
      categories: [
        {
          categoryId: "action_request",
          label: "Task request",
          count: i.categories[0]!.count,
          share: i.categories[0]!.count / i.turn_count,
          conversationCount: i.categories[0]!.conversation_count,
          failedCount: i.categories[0]!.failed_count,
          frustratedCount: i.categories[0]!.frustrated_count,
          failureRate: i.categories[0]!.failed_count / i.categories[0]!.count,
          frustrationRate: i.categories[0]!.frustrated_count / i.categories[0]!.count,
          averageLatencyMs: i.categories[0]!.average_latency_ms,
          averageCostPerTurnUsd: i.categories[0]!.average_cost_per_turn_usd,
          clusterBreakdowns: [],
        },
      ],
      clusters: [],
    },
    failureAttributions: [],
    warnings: [],
    ...overrides,
  });
}

describe("Insights schema, markdown brief & state reducer", () => {
  it("parses real example-turn & attribution payloads without invented latency/token fields", () => {
    const turns = parseExampleTurnsResponse(contractFixture.example_turns);
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({
      conversationId: "conv-stripe-8841",
      traceId: "trace-stripe-8841",
      turnNumber: 2,
      failureId: "77869ce8-2917-5d33-bc74-ab63603e278e",
      failureCategory: "tool_misuse",
      failureSeverity: "major",
      failureEvidenceSpanIds: ["span-tool-8841"],
    });
    expect("durationMs" in turns[0]!).toBe(false);
    expect("totalTokens" in turns[0]!).toBe(false);

    const enriched = enrichTurnsWithAttributions(turns, contractFixture.failure_attributions);
    expect(enriched[0]?.clusterId).toBe("01a1fc1d-f6a4-7e82-9d62-fbdeb71f5d12");
    expect(enriched[0]?.clusterTitle).toBe("lookup_invoice omits required currency");

    const embedded = parseEmbeddedConversation(contractFixture.weave_conversation);
    expect(embedded?.conversationId).toBe("conv-stripe-8841");
    expect(embedded?.turns).toHaveLength(2);
  });

  it("emits the product failureClusterToMarkdown brief and Forge deep links when a cluster + turn are selected", () => {
    const payload = makeInsightsPayload();
    const cat = payload.failure.categories[0]!;
    const cluster = cat.clusterBreakdowns[0]!;
    const examplesKey = `cluster:failure:${cat.categoryId}:${cluster.id}`;

    let state = reduceInsightsState(createInitialInsightsState(), { type: "loaded", payload });
    expect(state.activeTab).toBe("failure");

    state = reduceInsightsState(state, {
      type: "toggleCategory",
      categoryKey: `cat:failure:${cat.categoryId}`,
      signatureType: "failure",
      category: cat,
    });
    state = reduceInsightsState(state, {
      type: "toggleClusterExamples",
      examplesKey,
      signatureType: "failure",
      category: cat,
      cluster,
    });

    const turns = enrichTurnsWithAttributions(
      parseExampleTurnsResponse(contractFixture.example_turns),
      contractFixture.failure_attributions,
    );
    state = reduceInsightsState(state, {
      type: "examplesLoaded",
      examplesKey,
      items: turns,
    });
    state = reduceInsightsState(state, {
      type: "selectTurn",
      signatureType: "failure",
      category: cat,
      cluster,
      examplesKey,
      turn: turns[0]!,
    });

    const ctx = buildInsightsModelContext(state);
    const md = ctx?.content[0]?.text ?? "";
    expect(md).toContain("# lookup_invoice omits required currency");
    expect(md).toContain("Agent Lens detected 284 tool misuse failures over 7 days:");
    expect(md).toContain("- Severity: High");
    expect(md).toContain("- Impact: 284 turns in 196 conversations");
    expect(md).toContain("## Examples");
    expect(md).toContain("span span-tool-8841");
    expect(md).not.toContain("[[span:");
    expect(md).toContain(
      "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/thread/turns/trace-stripe-8841",
    );

    expect(ctx?.structuredContent).toMatchObject({
      category_id: "tool_misuse",
      topic_id: "01a1fc1d-f6a4-7e82-9d62-fbdeb71f5d12",
      severity: "major",
      selected_failure: {
        trace_id: "trace-stripe-8841",
        failure_id: "77869ce8-2917-5d33-bc74-ab63603e278e",
      },
    });
    expect(buildActiveForgeLink(state)).toBe(
      "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/thread/turns/trace-stripe-8841",
    );
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
