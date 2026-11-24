import { describe, expect, it } from "vitest";
import { parseAriaToolEnvelope } from "../src/schema";
import {
  applyOptimisticSendInput,
  applyParallelBatchToState,
  applyThreadPayloadToState,
  applyThreadsListToState,
  buildModelContextPayload,
  computeLiveElapsedSeconds,
  createInitialState,
  deriveSessionPhase,
  extractRequestedDomains,
  formatElapsedClock,
  formatLiveStepLine,
  getHydratableCompletedTurnIds,
  getPollableTurnIds,
  isLongInlineAnswer,
  summarizeBatch,
  upsertTurnIntoState,
} from "../src/state";

describe("ARIA compact run card schema & state engine", () => {
  it("unwraps FastMCP structuredContent.result for single turns and batch polls", () => {
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

  it("maintains chronological lineage across optimistic input, permission gate, and continuation", () => {
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
    expect(state.view).toBe("run");
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

    // Continuation turn with granted domain appends to the same session
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

  it("formats compact live step lines and detects long inline answers", () => {
    const line = formatLiveStepLine(
      {
        syncedAtMs: 10_000,
        turn: {
          ok: true,
          turn_id: "t-live",
          scope: { entity: "wandb", project: "lab" },
          state: "in_progress",
          is_terminal: false,
          tool_steps: [
            { call_id: "c1", name: "query_wandb", status: "ok" },
            { call_id: "c2", name: "shell", status: "recovered_error" },
            { call_id: "c3", name: "shell", status: "running" },
          ],
          progress: {
            phase: "working",
            elapsed_seconds: 35,
            message_count: 2,
            tool_call_record_count: 5,
          },
        },
      },
      12_000,
    );
    expect(line).toBe("Running shell, 3 steps, 37s");
    expect(isLongInlineAnswer("Short summary.")).toBe(false);
    expect(isLongInlineAnswer("Line\n".repeat(14))).toBe(true);
  });

  it("renders parallel_turns into RunList view, auto-expands gated run, and preserves RunList on single-turn poll", () => {
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
          ],
        },
      },
    });
    expect(batch.kind).toBe("parallel_turns");
    if (batch.kind === "parallel_turns") {
      state = applyParallelBatchToState(state, batch.data, 10_000);
    }

    expect(state.view).toBe("runList");
    expect(state.sessions).toHaveLength(3);
    expect(state.expandedBatchSessionId).toBe("th-3");
    expect(getPollableTurnIds(state)).toEqual(["turn-2"]);

    const summary = summarizeBatch(state.sessions);
    expect(summary).toEqual({
      total: 3,
      active: 1,
      awaitingPermission: 1,
      awaitingInput: 0,
      completed: 1,
      errored: 0,
      allTerminal: false,
    });

    const liveElapsed = computeLiveElapsedSeconds(state.sessions[1]!.turns[0]!, 15_000);
    expect(liveElapsed).toBeCloseTo(70, 1);
    expect(formatElapsedClock(liveElapsed)).toBe("1m 10s");

    // Single remaining active turn finishes via background poll — RunList view stays open
    state = upsertTurnIntoState(
      state,
      {
        ok: true,
        turn_id: "turn-2",
        thread_id: "th-2",
        title: "Arm B · 1-Layer Control",
        scope: { entity: "studio-ai", project: "looped-synth-study" },
        state: "completed",
        is_terminal: true,
        latest_response: "Control finished.",
      },
      16_000,
    );
    expect(state.view).toBe("runList");

    const ctx = buildModelContextPayload(state);
    expect(ctx?.structuredContent.mode).toBe("batch");
    expect(ctx?.text).toContain("ARIA Batch (3 runs)");
  });

  it("opens ThreadList as the card view and transitions to RunCard with back-navigation flag on thread load", () => {
    let state = createInitialState();
    state = applyThreadsListToState(state, {
      ok: true,
      threads: [
        {
          thread_id: "thread-older",
          title: "8x RoPE Base Frequency Sweep",
          prompt_preview: "Sweep RoPE theta across 10k, 50k, and 250k",
          scope: { entity: "wandb", project: "looped-transformer-lab" },
        },
      ],
      count: 1,
      offset: 0,
      has_more: false,
    });
    expect(state.view).toBe("threadList");
    expect(state.recentThreads).toHaveLength(1);

    state = applyThreadPayloadToState(
      state,
      {
        ok: true,
        thread: {
          thread_id: "thread-older",
          root_turn_id: "turn-old-1",
          latest_turn_id: "turn-old-2",
          turn_count: 2,
          title: "8x RoPE Base Frequency Sweep",
          prompt_preview: "Sweep RoPE theta across 10k, 50k, and 250k",
          scope: { entity: "wandb", project: "looped-transformer-lab" },
          state: "completed",
          is_terminal: true,
        },
        turns: [
          {
            turn_id: "turn-old-2",
            parent_turn_id: "turn-old-1",
            state: "completed",
            is_terminal: true,
          },
          {
            turn_id: "turn-old-1",
            parent_turn_id: null,
            state: "completed",
            is_terminal: true,
          },
        ],
        count: 2,
        offset: 0,
        has_more: false,
      },
      2_000,
    );

    expect(state.view).toBe("run");
    expect(state.cameFromThreadList).toBe(true);
    expect(state.sessions).toHaveLength(1);
    expect(state.sessions[0]!.sessionId).toBe("thread-older");
    expect(state.sessions[0]!.label).toBe("8x RoPE Base Frequency Sweep");
    expect(getHydratableCompletedTurnIds(state)).toEqual(["turn-old-2"]);
  });
});
