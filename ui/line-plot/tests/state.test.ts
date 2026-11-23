import { describe, expect, it } from "vitest";
import type { GraphPayload } from "../src/schema";
import { smoothPoints } from "../src/smoothing";
import {
  buildLiveRefreshPlan,
  computeStepZoomWindow,
  createInitialState,
  formatModelContextUpdate,
  mergeRefreshedPayload,
  projectDisplaySeries,
  reduceChartState,
} from "../src/state";

function makePayload(overrides: Partial<GraphPayload> = {}): GraphPayload {
  return {
    schema: "wandb.graph/v1",
    series: [
      {
        seriesId: '["run-done","loss","primary"]',
        metadataId: '["run-done","loss"]',
        role: "primary",
        kind: "line",
        metricName: "loss",
        historySeries: [
          { x: 0, y: 2 },
          { x: 1, y: 1 },
        ],
      },
      {
        seriesId: '["run-live","loss","primary"]',
        metadataId: '["run-live","loss"]',
        role: "primary",
        kind: "line",
        metricName: "loss",
        historySeries: [
          { x: 0, y: 4 },
          { x: 1, y: 3 },
        ],
      },
    ],
    seriesStylesById: {
      '["run-done","loss","primary"]': {
        color: "#00a99d",
        lineWidth: 1.6,
        mark: "solid",
        colorKey: "run-done:loss",
      },
      '["run-live","loss","primary"]': {
        color: "#e87535",
        lineWidth: 1.6,
        mark: "solid",
        colorKey: "run-live:loss",
      },
    },
    seriesMetadataById: {
      '["run-done","loss"]': {
        displayName: "Baseline",
        title: "Baseline · loss",
        runName: "run-done",
        runState: "finished",
        legendHref: "https://wandb.ai/team/proj/runs/run-done",
        entityName: "team",
        projectName: "proj",
      },
      '["run-live","loss"]': {
        displayName: "Candidate",
        title: "Candidate · loss",
        runName: "run-live",
        runState: "running",
        legendHref: "https://wandb.ai/team/proj/runs/run-live",
        entityName: "team",
        projectName: "proj",
      },
    },
    runsById: {
      "run-done": {
        displayName: "Baseline",
        state: "finished",
        active: false,
        url: "https://wandb.ai/team/proj/runs/run-done",
      },
      "run-live": {
        displayName: "Candidate",
        state: "running",
        active: true,
        url: "https://wandb.ai/team/proj/runs/run-live",
      },
    },
    hasActiveRuns: true,
    axis: {
      xKey: "_step",
      xIsDatetime: false,
      xScale: "linear",
      yScale: "linear",
      xTitle: "_step",
      yTitle: "",
    },
    xDomain: [null, null],
    yDomain: [null, null],
    title: "loss",
    legend: { show: true, position: "north" },
    effectiveConfig: {
      metrics: ["loss"],
      xAxis: "_step",
      smoothingWeight: 0,
      smoothingType: "exponential",
    },
    coverage: {
      sampled: true,
      truncated: false,
      ignoredConfig: [],
      perRunErrors: [],
      grouped: false,
      autoDiscoveredRuns: true,
    },
    url: "https://wandb.ai/team/proj",
    workspaceUrl: "https://wandb.ai/team/proj/workspace",
    range_cap: 20_000,
    arguments: {
      entity_name: "team",
      project_name: "proj",
      run_ids: ["run-done", "run-live"],
      config: { metrics: ["loss"] },
      samples: 500,
      min_step: null,
      max_step: null,
    },
    ...overrides,
  };
}

describe("computeStepZoomWindow", () => {
  it("rounds fractional selections outward to whole non-negative steps", () => {
    expect(computeStepZoomWindow(1.8, 7.1, 1000)).toEqual({
      minStep: 1,
      maxStep: 8,
      canRequery: true,
    });
    expect(computeStepZoomWindow(-4.5, 3.2, 1000)).toEqual({
      minStep: 0,
      maxStep: 4,
      canRequery: true,
    });
  });

  it("disables server requery when inclusive span exceeds range_cap", () => {
    expect(computeStepZoomWindow(0, 50, 20)).toEqual({
      minStep: 0,
      maxStep: 50,
      canRequery: false,
    });
  });
});

describe("smoothPoints parity with Python line_plot_smoothing", () => {
  const sample = [
    { x: 0, y: 1 },
    { x: 1, y: 3 },
    { x: 2, y: 5 },
  ];

  it("matches zero-debiased exponential smoothing", () => {
    const out = smoothPoints(sample, 0.25, "exponential");
    expect(out.map((p) => p.y)).toEqual([1, 7 / 3, 27 / 7]);
  });

  it("matches centered running average", () => {
    const out = smoothPoints(sample, 2, "average");
    expect(out.map((p) => p.y)).toEqual([1, 2, 4]);
  });

  it("matches time-weighted exponential smoothing", () => {
    const pts = [
      { x: 0, y: 1 },
      { x: 1, y: 3 },
      { x: 1500, y: 5 },
    ];
    const alpha = Math.sqrt(0.25 * 0.999);
    const a = alpha ** ((1 / 1500) * 1500);
    const b = alpha ** ((1499 / 1500) * 1500);
    const expected = [1, (a + 3) / (a + 1), ((a + 3) * b + 5) / ((a + 1) * b + 1)];
    const out = smoothPoints(pts, 0.25, "exponentialTimeWeighted", 1500);
    out.forEach((p, idx) => {
      expect(p.y).toBeCloseTo(expected[idx]!, 10);
    });
  });

  it("matches reflected-boundary gaussian smoothing", () => {
    const sigma = 0.25;
    const a = Math.exp(-1 / (2 * sigma * sigma));
    const b = Math.exp(-4 / (2 * sigma * sigma));
    const first = (b * 3 + a * 1 + 1 + a * 3 + b * 5) / (1 + 2 * a + 2 * b);
    const out = smoothPoints(sample, sigma, "gaussian");
    expect(out[0]!.y).toBeCloseTo(first, 10);
    expect(out[1]!.y).toBeCloseTo(3, 10);
    expect(out[2]!.y).toBeCloseTo(6 - first, 10);
  });
});

describe("active-run Live Refresh planning and delta merge", () => {
  it("queries only active run_ids on ungrouped charts and preserves finished runs on merge", () => {
    const initial = makePayload();
    const plan = buildLiveRefreshPlan(initial, { minStep: 10, maxStep: 50 });
    expect(plan).not.toBeNull();
    expect(plan!.refreshedRunIds).toEqual(["run-live"]);
    expect(plan!.arguments).toMatchObject({
      entity_name: "team",
      project_name: "proj",
      run_ids: ["run-live"],
      min_step: 10,
      max_step: 50,
    });

    const partialIncoming = makePayload({
      series: [
        {
          seriesId: '["run-live","loss","primary"]',
          metadataId: '["run-live","loss"]',
          role: "primary",
          kind: "line",
          metricName: "loss",
          historySeries: [
            { x: 0, y: 4 },
            { x: 1, y: 3 },
            { x: 2, y: 1.5 },
          ],
        },
      ],
      runsById: {
        "run-live": {
          displayName: "Candidate",
          state: "finished",
          active: false,
          url: "https://wandb.ai/team/proj/runs/run-live",
        },
      },
      hasActiveRuns: false,
      arguments: {
        ...initial.arguments,
        run_ids: ["run-live"],
      },
    });

    const merged = mergeRefreshedPayload(initial, partialIncoming, plan!.refreshedRunIds);
    expect(merged.arguments.run_ids).toEqual(["run-done", "run-live"]);
    expect(merged.series.map((s) => s.seriesId)).toEqual([
      '["run-done","loss","primary"]',
      '["run-live","loss","primary"]',
    ]);
    expect(merged.series[0]!.historySeries).toHaveLength(2);
    expect(merged.series[1]!.historySeries).toHaveLength(3);
    expect(merged.hasActiveRuns).toBe(false);
  });
});

describe("reducer, client-side smoothing, non-destructive errors, and model context", () => {
  it("keeps current chart intact when zoom or refresh fails", () => {
    const payload = makePayload({
      effectiveConfig: {
        metrics: ["loss"],
        xAxis: "_step",
        smoothingWeight: 0.5,
        smoothingType: "exponential",
      },
      series: [
        {
          seriesId: '["run-done","loss","primary"]',
          metadataId: '["run-done","loss"]',
          role: "primary",
          kind: "line",
          metricName: "loss",
          historySeries: [
            { x: 0, y: 1 },
            { x: 1, y: 7 / 3 },
          ],
          rawHistorySeries: [
            { x: 0, y: 1 },
            { x: 1, y: 3 },
          ],
        },
      ],
    });

    let state = reduceChartState(createInitialState(), {
      type: "loaded",
      payload,
      now: 1_700_000_000_000,
    });
    expect(state.liveRefreshEnabled).toBe(true);

    // Slide smoothing back to 0 -> uses rawHistorySeries with zero server calls
    state = reduceChartState(state, { type: "setSmoothing", weight: 0 });
    const projected = projectDisplaySeries(state);
    expect(projected[0]!.effectivePoints).toEqual([
      { x: 0, y: 1 },
      { x: 1, y: 3 },
    ]);

    // Simulate failed zoom requery -> chart stays on screen with warning banner
    state = reduceChartState(state, {
      type: "operationFailed",
      kind: "warning",
      message: "Zoom requery failed",
    });
    expect(state.current).not.toBeNull();
    expect(state.inlineBanner?.message).toBe("Zoom requery failed");

    // Expand to fullscreen and verify model context payload
    state = reduceChartState(state, {
      type: "hostContextChanged",
      displayMode: "fullscreen",
      availableDisplayModes: ["inline", "fullscreen"],
    });
    state = reduceChartState(state, { type: "toggleLogY" });

    const ctx = formatModelContextUpdate(state);
    expect(ctx.content[0]?.text).toContain("[MCP Line Plot Context]");
    expect(ctx.content[0]?.text).toContain("mode=fullscreen");
    expect(ctx.structuredContent).toMatchObject({
      entity_name: "team",
      project_name: "proj",
      display_mode: "fullscreen",
      y_scale: "log",
      smoothing_weight: 0,
    });
  });
});
