import { describe, expect, it } from "vitest";
import { parseAriaToolEnvelope } from "../src/schema";
import {
  applyOptimisticSendInput,
  applyParallelBatchToState,
  buildModelContextPayload,
  computeLiveElapsedSeconds,
  createInitialState,
  deriveSessionPhase,
  extractRequestedDomains,
  formatElapsedClock,
  getPollableTurnIds,
  summarizeFleet,
  upsertTurnIntoState,
} from "../src/state";

describe("ARIA MCP App schema & state engine", () => {
  it("unwraps FastMCP structuredContent.result for single turns and parallel batches", () => {
    const single = parseAriaToolEnvelope({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "turn-1",
          thread_id: "thread-1",
          title: "Chorus Extrapolation Audit",
          user_prompt: "Audit 16-bar chorus drift",
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          state: "in_progress",
          is_terminal: false,
          progress: {
            phase: "working",
            status_text: "ARIA is running shell.",
            elapsed_seconds: 18.4,
            message_count: 2,
            tool_call_record_count: 3,
          },
          access_mode: "read-write",
        },
      },
    });
    expect(single.kind).toBe("turn");
    if (single.kind === "turn") {
      expect(single.data.turn_id).toBe("turn-1");
      expect(single.data.title).toBe("Chorus Extrapolation Audit");
    }

    const parallel = parseAriaToolEnvelope({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "turn-a",
          state: "completed",
          is_terminal: true,
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          parallel_turns: [
            {
              ok: true,
              turn_id: "turn-a",
              thread_id: "th-a",
              title: "Arm A · Baseline",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              latest_response: "Arm A finished.",
            },
            {
              ok: true,
              turn_id: "turn-b",
              thread_id: "th-b",
              title: "Arm B · Gated Loop",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "in_progress",
              is_terminal: false,
            },
          ],
          parallel_summary: {
            total_count: 2,
            completed_count: 1,
            active_count: 1,
            awaiting_permission_count: 0,
            awaiting_user_input_count: 0,
            errored_count: 0,
            cancelled_count: 0,
            all_terminal: false,
          },
        },
      },
    });
    expect(parallel.kind).toBe("parallel_turns");
  });

  it("maintains 1-to-1 chronological lineage across optimistic input, permission pause, and continuation", () => {
    let state = createInitialState();
    state = applyOptimisticSendInput(
      state,
      {
        message: "Diagnose chorus attention collapse",
        entity: "studio-ai",
        project: "looped-synth-study",
      },
      1_000,
    );
    expect(state.sessions).toHaveLength(1);
    expect(state.sessions[0]!.turns[0]!.optimistic).toBe(true);

    // Real root turn replaces optimistic turn and pauses on arxiv.org network_access
    state = upsertTurnIntoState(
      state,
      {
        ok: true,
        turn_id: "turn-root",
        thread_id: "thread-root",
        title: "Chorus Attention Collapse",
        user_prompt: "Diagnose chorus attention collapse",
        scope: { entity: "studio-ai", project: "looped-synth-study" },
        state: "completed",
        is_terminal: true,
        requires_permission: true,
        permission_requests: [
          {
            type: "network_access",
            reason: "Read primary arXiv paper on looped attention.",
            domains: ["arxiv.org"],
          },
        ],
        progress: {
          phase: "awaiting_permission",
          message_count: 4,
          tool_call_record_count: 6,
          elapsed_seconds: 42,
        },
      },
      2_000,
    );

    expect(state.sessions).toHaveLength(1);
    expect(state.sessions[0]!.turns).toHaveLength(1);
    expect(deriveSessionPhase(state.sessions[0]!)).toBe("awaiting_permission");
    expect(extractRequestedDomains(state.sessions[0]!.turns[0]!.turn)).toEqual({
      domains: ["arxiv.org"],
      reasons: ["Read primary arXiv paper on looped attention."],
    });

    // Continuation turn with granted domain appends to the same 1-to-1 session
    state = upsertTurnIntoState(
      state,
      {
        ok: true,
        turn_id: "turn-cont",
        thread_id: "thread-root",
        parent_turn_id: "turn-root",
        user_prompt: "Outbound network access granted for arxiv.org.",
        scope: { entity: "studio-ai", project: "looped-synth-study" },
        state: "completed",
        is_terminal: true,
        active_allowed_network_domains: ["arxiv.org"],
        latest_response: "| Arm | Extrapolation |\n| --- | --- |\n| Arm C | 0.941 |",
      },
      5_000,
    );

    expect(state.sessions).toHaveLength(1);
    expect(state.sessions[0]!.turns.map((t) => t.turn.turn_id)).toEqual([
      "turn-root",
      "turn-cont",
    ]);
    expect(deriveSessionPhase(state.sessions[0]!)).toBe("completed");

    const ctx = buildModelContextPayload(state);
    expect(ctx?.structuredContent.mode).toBe("single");
    expect(ctx?.text).toContain("Chorus Attention Collapse");
  });

  it("tracks N parallel sub-agents, computes pollable subset, and ticks live elapsed clocks", () => {
    let state = createInitialState();
    const batch = parseAriaToolEnvelope({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "turn-1",
          state: "in_progress",
          is_terminal: false,
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          parallel_turns: [
            {
              ok: true,
              turn_id: "turn-1",
              thread_id: "th-1",
              title: "Arm A · 4-Layer Baseline",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              latest_response: "Baseline val_loss 0.312",
            },
            {
              ok: true,
              turn_id: "turn-2",
              thread_id: "th-2",
              title: "Arm B · 1-Layer Control",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "in_progress",
              is_terminal: false,
              progress: {
                phase: "working",
                elapsed_seconds: 65,
                message_count: 2,
                tool_call_record_count: 4,
              },
            },
            {
              ok: true,
              turn_id: "turn-3",
              thread_id: "th-3",
              title: "Arm C · Gated Loop",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              requires_permission: true,
              permission_requests: [{ type: "network_access", domains: ["arxiv.org"] }],
            },
            {
              ok: true,
              turn_id: "turn-4",
              thread_id: "th-4",
              title: "Arm D · Weight-Tied",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "queued",
              is_terminal: false,
            },
          ],
        },
      },
    });
    expect(batch.kind).toBe("parallel_turns");
    if (batch.kind === "parallel_turns") {
      state = applyParallelBatchToState(state, batch.data, 10_000);
    }

    expect(state.sessions).toHaveLength(4);
    expect(state.parallelLayout).toBe("grid");
    expect(getPollableTurnIds(state)).toEqual(["turn-2", "turn-4"]);

    const summary = summarizeFleet(state.sessions);
    expect(summary).toEqual({
      total: 4,
      active: 2,
      awaitingPermission: 1,
      awaitingInput: 0,
      completed: 1,
      errored: 0,
      allTerminal: false,
    });

    const liveElapsed = computeLiveElapsedSeconds(state.sessions[1]!.turns[0]!, 15_000);
    expect(liveElapsed).toBeCloseTo(70, 1);
    expect(formatElapsedClock(liveElapsed)).toBe("1m 10s");

    const ctx = buildModelContextPayload(state);
    expect(ctx?.structuredContent.mode).toBe("parallel");
    expect(ctx?.text).toContain("ARIA Parallel Sub-Agents (4)");
  });
});
