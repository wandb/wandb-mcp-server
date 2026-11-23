import type { GraphPayload, Point, SeriesEntry, SmoothingType } from "./schema";
import { smoothPoints } from "./smoothing";

export type DisplayMode = "inline" | "fullscreen" | "pip";

export interface StepZoomWindow {
  minStep: number;
  maxStep: number;
  canRequery: boolean;
}

export interface LocalXZoom {
  minX: number;
  maxX: number;
}

export interface InlineBanner {
  kind: "warning" | "error" | "info";
  message: string;
}

export interface ChartState {
  status: "idle" | "loading" | "ready" | "error";
  pendingTitle: string | null;
  overview: GraphPayload | null;
  current: GraphPayload | null;
  fatalError: string | null;
  inlineBanner: InlineBanner | null;
  isBusy: boolean;
  liveRefreshEnabled: boolean;
  lastRefreshedAt: number | null;
  stepZoom: { minStep: number; maxStep: number } | null;
  localZoom: LocalXZoom | null;
  smoothingWeight: number;
  yLogScale: boolean;
  hiddenMetadataIds: ReadonlySet<string>;
  isolatedMetadataId: string | null;
  displayMode: DisplayMode;
  availableDisplayModes: readonly DisplayMode[];
}

export type ChartAction =
  | { type: "toolInput"; title: string | null }
  | { type: "loaded"; payload: GraphPayload; now: number; preserveView?: boolean }
  | { type: "loadFailed"; message: string }
  | { type: "requeryStarted" }
  | {
      type: "zoomLoaded";
      payload: GraphPayload;
      stepZoom: { minStep: number; maxStep: number };
      now: number;
    }
  | {
      type: "refreshed";
      payload: GraphPayload;
      refreshedRunIds: string[] | null;
      now: number;
    }
  | { type: "operationFailed"; message: string; kind?: "warning" | "error" }
  | { type: "dismissBanner" }
  | { type: "setLocalZoom"; zoom: LocalXZoom | null }
  | { type: "resetZoom" }
  | { type: "setLiveRefresh"; enabled: boolean }
  | { type: "setSmoothing"; weight: number }
  | { type: "toggleLogY" }
  | { type: "toggleSeries"; metadataId: string }
  | { type: "isolateSeries"; metadataId: string }
  | {
      type: "hostContextChanged";
      displayMode?: DisplayMode;
      availableDisplayModes?: readonly DisplayMode[];
    };

export function createInitialState(): ChartState {
  return {
    status: "idle",
    pendingTitle: null,
    overview: null,
    current: null,
    fatalError: null,
    inlineBanner: null,
    isBusy: false,
    liveRefreshEnabled: false,
    lastRefreshedAt: null,
    stepZoom: null,
    localZoom: null,
    smoothingWeight: 0,
    yLogScale: false,
    hiddenMetadataIds: new Set(),
    isolatedMetadataId: null,
    displayMode: "inline",
    availableDisplayModes: ["inline"],
  };
}

/**
 * Outward whole-step rounding for `_step` drag-zoom so partial fractional drag bounds
 * always include both boundary steps (`floor(min)..ceil(max)`, clamped >= 0).
 */
export function computeStepZoomWindow(
  minX: number,
  maxX: number,
  rangeCap: number,
): StepZoomWindow {
  const lo = Math.min(minX, maxX);
  const hi = Math.max(minX, maxX);
  const minStep = Math.max(0, Math.floor(lo));
  const maxStep = Math.max(minStep, Math.ceil(hi));
  const span = maxStep - minStep + 1;
  return {
    minStep,
    maxStep,
    canRequery: maxStep > minStep && Number.isFinite(span) && span <= rangeCap,
  };
}

function parseMetadataOwner(metadataId: string): string | null {
  try {
    const parsed: unknown = JSON.parse(metadataId);
    if (Array.isArray(parsed) && typeof parsed[0] === "string") {
      return parsed[0];
    }
  } catch {
    // ignore malformed metadataId
  }
  return null;
}

export function getActiveRunIds(payload: GraphPayload): string[] {
  const active: string[] = [];
  for (const runId of payload.arguments.run_ids) {
    if (payload.runsById[runId]?.active) {
      active.push(runId);
    }
  }
  for (const [runId, info] of Object.entries(payload.runsById)) {
    if (info.active && !active.includes(runId)) {
      active.push(runId);
    }
  }
  return active;
}

export interface RefreshPlan {
  arguments: Record<string, unknown>;
  refreshedRunIds: string[] | null;
}

/**
 * Build arguments for the 10-second active-run Live Refresh loop.
 * For ungrouped charts, requests ONLY active (`running`/`pending`/`preempting`) run IDs
 * so finished runs are served from browser memory without redundant W&B reads.
 */
export function buildLiveRefreshPlan(
  current: GraphPayload,
  stepZoom: { minStep: number; maxStep: number } | null,
): RefreshPlan | null {
  const activeRunIds = getActiveRunIds(current);
  if (activeRunIds.length === 0) {
    return null;
  }
  const subsetRefresh =
    !current.coverage.grouped && activeRunIds.length < current.arguments.run_ids.length;
  const targetRunIds = subsetRefresh ? activeRunIds : current.arguments.run_ids;
  if (targetRunIds.length === 0) {
    return null;
  }
  const args: Record<string, unknown> = {
    entity_name: current.arguments.entity_name,
    project_name: current.arguments.project_name,
    run_ids: targetRunIds,
    config: current.arguments.config,
    samples: current.arguments.samples,
  };
  if (stepZoom && current.axis.xKey === "_step") {
    args.min_step = stepZoom.minStep;
    args.max_step = stepZoom.maxStep;
  } else {
    if (typeof current.arguments.min_step === "number") {
      args.min_step = current.arguments.min_step;
    }
    if (typeof current.arguments.max_step === "number") {
      args.max_step = current.arguments.max_step;
    }
  }
  return {
    arguments: args,
    refreshedRunIds: subsetRefresh ? activeRunIds : null,
  };
}

/**
 * Merge an active-run subset refresh into the cached chart payload while preserving
 * finished runs' series, styles, metadata, and canonical run ordering.
 */
export function mergeRefreshedPayload(
  previous: GraphPayload,
  incoming: GraphPayload,
  refreshedRunIds: readonly string[] | null,
): GraphPayload {
  if (!refreshedRunIds || refreshedRunIds.length === 0 || previous.coverage.grouped) {
    return incoming;
  }
  const refreshedSet = new Set(refreshedRunIds);
  const runsById = { ...previous.runsById, ...incoming.runsById };
  const hasActiveRuns = Object.values(runsById).some((r) => r.active);

  const orderedOwners = [...previous.arguments.run_ids];
  for (const rid of incoming.arguments.run_ids) {
    if (!orderedOwners.includes(rid)) {
      orderedOwners.push(rid);
    }
  }

  const byOwnerPrev = new Map<string, SeriesEntry[]>();
  for (const s of previous.series) {
    const owner = parseMetadataOwner(s.metadataId) ?? "";
    const list = byOwnerPrev.get(owner) ?? [];
    list.push(s);
    byOwnerPrev.set(owner, list);
  }

  const byOwnerNext = new Map<string, SeriesEntry[]>();
  for (const s of incoming.series) {
    const owner = parseMetadataOwner(s.metadataId) ?? "";
    const list = byOwnerNext.get(owner) ?? [];
    list.push(s);
    byOwnerNext.set(owner, list);
  }

  const mergedSeries: SeriesEntry[] = [];
  for (const owner of orderedOwners) {
    const chosen = refreshedSet.has(owner) ? byOwnerNext.get(owner) : byOwnerPrev.get(owner);
    if (chosen) {
      mergedSeries.push(...chosen);
    }
  }

  const prevErrors = previous.coverage.perRunErrors.filter((e) => !refreshedSet.has(e.runId));
  const mergedErrors = [...prevErrors, ...incoming.coverage.perRunErrors];

  return {
    ...previous,
    series: mergedSeries,
    seriesStylesById: { ...previous.seriesStylesById, ...incoming.seriesStylesById },
    seriesMetadataById: { ...previous.seriesMetadataById, ...incoming.seriesMetadataById },
    runsById,
    hasActiveRuns,
    coverage: {
      ...previous.coverage,
      sampled: previous.coverage.sampled || incoming.coverage.sampled,
      truncated: previous.coverage.truncated || incoming.coverage.truncated,
      perRunErrors: mergedErrors,
    },
    arguments: {
      ...previous.arguments,
      min_step: incoming.arguments.min_step,
      max_step: incoming.arguments.max_step,
    },
  };
}

export function reduceChartState(state: ChartState, action: ChartAction): ChartState {
  switch (action.type) {
    case "toolInput":
      if (state.current) {
        return { ...state, isBusy: true, inlineBanner: null };
      }
      return {
        ...state,
        status: "loading",
        pendingTitle: action.title,
        fatalError: null,
        inlineBanner: null,
      };

    case "loaded": {
      const { payload, now, preserveView } = action;
      return {
        ...state,
        status: "ready",
        pendingTitle: null,
        overview: payload,
        current: payload,
        fatalError: null,
        inlineBanner: null,
        isBusy: false,
        liveRefreshEnabled: preserveView ? state.liveRefreshEnabled : payload.hasActiveRuns,
        lastRefreshedAt: now,
        stepZoom:
          typeof payload.arguments.min_step === "number" &&
          typeof payload.arguments.max_step === "number"
            ? { minStep: payload.arguments.min_step, maxStep: payload.arguments.max_step }
            : null,
        localZoom: null,
        smoothingWeight: preserveView
          ? state.smoothingWeight
          : payload.effectiveConfig.smoothingWeight,
        yLogScale: preserveView ? state.yLogScale : payload.axis.yScale === "log",
        hiddenMetadataIds: preserveView ? state.hiddenMetadataIds : new Set(),
        isolatedMetadataId: preserveView ? state.isolatedMetadataId : null,
      };
    }

    case "loadFailed":
      if (state.current) {
        return {
          ...state,
          isBusy: false,
          inlineBanner: { kind: "error", message: action.message },
        };
      }
      return {
        ...state,
        status: "error",
        isBusy: false,
        fatalError: action.message,
      };

    case "requeryStarted":
      return {
        ...state,
        isBusy: true,
        inlineBanner: null,
      };

    case "zoomLoaded":
      return {
        ...state,
        status: "ready",
        current: action.payload,
        isBusy: false,
        inlineBanner: null,
        lastRefreshedAt: action.now,
        stepZoom: action.stepZoom,
        localZoom: null,
      };

    case "refreshed": {
      if (!state.current) {
        return state;
      }
      const merged = mergeRefreshedPayload(
        state.current,
        action.payload,
        action.refreshedRunIds,
      );
      const mergedOverview =
        !state.stepZoom && state.overview
          ? mergeRefreshedPayload(state.overview, action.payload, action.refreshedRunIds)
          : state.overview;
      return {
        ...state,
        current: merged,
        overview: mergedOverview,
        isBusy: false,
        inlineBanner: null,
        lastRefreshedAt: action.now,
        liveRefreshEnabled: merged.hasActiveRuns ? state.liveRefreshEnabled : false,
      };
    }

    case "operationFailed":
      return {
        ...state,
        isBusy: false,
        inlineBanner: {
          kind: action.kind ?? "warning",
          message: action.message,
        },
      };

    case "dismissBanner":
      return {
        ...state,
        inlineBanner: null,
      };

    case "setLocalZoom":
      return {
        ...state,
        localZoom: action.zoom,
      };

    case "resetZoom":
      return {
        ...state,
        current: state.overview ?? state.current,
        stepZoom: null,
        localZoom: null,
        inlineBanner: null,
      };

    case "setLiveRefresh":
      return {
        ...state,
        liveRefreshEnabled: action.enabled,
      };

    case "setSmoothing":
      return {
        ...state,
        smoothingWeight: Math.max(0, action.weight),
      };

    case "toggleLogY":
      return {
        ...state,
        yLogScale: !state.yLogScale,
      };

    case "toggleSeries": {
      const nextHidden = new Set(state.hiddenMetadataIds);
      if (state.isolatedMetadataId) {
        return {
          ...state,
          isolatedMetadataId: null,
          hiddenMetadataIds: nextHidden,
        };
      }
      if (nextHidden.has(action.metadataId)) {
        nextHidden.delete(action.metadataId);
      } else {
        nextHidden.add(action.metadataId);
      }
      return {
        ...state,
        hiddenMetadataIds: nextHidden,
      };
    }

    case "isolateSeries": {
      const unIsolating = state.isolatedMetadataId === action.metadataId;
      return {
        ...state,
        isolatedMetadataId: unIsolating ? null : action.metadataId,
        hiddenMetadataIds: new Set(),
      };
    }

    case "hostContextChanged":
      return {
        ...state,
        displayMode: action.displayMode ?? state.displayMode,
        availableDisplayModes: action.availableDisplayModes ?? state.availableDisplayModes,
      };
  }
}

export interface ProjectedSeriesEntry extends SeriesEntry {
  visible: boolean;
  effectivePoints: Point[];
}

export function isMetadataVisible(state: ChartState, metadataId: string): boolean {
  if (state.isolatedMetadataId !== null) {
    return state.isolatedMetadataId === metadataId;
  }
  return !state.hiddenMetadataIds.has(metadataId);
}

/**
 * Recomputes smoothed curves client-side from raw points whenever the user adjusts
 * the smoothing slider, without round-tripping to the MCP server.
 */
export function projectDisplaySeries(state: ChartState): ProjectedSeriesEntry[] {
  const payload = state.current;
  if (!payload) {
    return [];
  }
  const originalByMetadata = new Map<string, Point[]>();
  for (const s of payload.series) {
    if (s.role === "original") {
      originalByMetadata.set(s.metadataId, s.historySeries);
    }
  }

  let xMax = 100;
  for (const s of payload.series) {
    const raw = s.rawHistorySeries ?? originalByMetadata.get(s.metadataId) ?? s.historySeries;
    for (const p of raw) {
      if (p.x > xMax) {
        xMax = p.x;
      }
    }
  }

  const backendWeight = payload.effectiveConfig.smoothingWeight;
  const backendKind = payload.effectiveConfig.smoothingType;
  const effectiveKind: SmoothingType =
    backendKind === "none" && state.smoothingWeight > 0 ? "exponential" : backendKind;
  const weightUnchanged = Math.abs(state.smoothingWeight - backendWeight) < 1e-9;

  return payload.series.map((entry) => {
    const baseVisible = isMetadataVisible(state, entry.metadataId);
    if (entry.role === "primary") {
      const rawPoints =
        entry.rawHistorySeries ??
        originalByMetadata.get(entry.metadataId) ??
        entry.historySeries;
      const effectivePoints = weightUnchanged
        ? entry.historySeries
        : smoothPoints(rawPoints, state.smoothingWeight, effectiveKind, xMax);
      return {
        ...entry,
        visible: baseVisible,
        effectivePoints,
      };
    }
    if (entry.role === "original") {
      const showOriginal =
        baseVisible &&
        state.smoothingWeight > 0 &&
        Boolean(payload.effectiveConfig.showOriginalAfterSmoothing);
      return {
        ...entry,
        visible: showOriginal,
        effectivePoints: entry.historySeries,
      };
    }
    return {
      ...entry,
      visible: baseVisible,
      effectivePoints: entry.historySeries,
    };
  });
}

export function selectPreferredExpandMode(
  available: readonly DisplayMode[],
): "fullscreen" | "pip" | null {
  if (available.includes("fullscreen")) {
    return "fullscreen";
  }
  if (available.includes("pip")) {
    return "pip";
  }
  return null;
}

export interface ModelContextPayload {
  content: Array<{ type: "text"; text: string }>;
  structuredContent?: Record<string, unknown>;
}

/**
 * Formats the active chart state for `ui/update-model-context` so chat follow-ups
 * know which entity/project, runs, metrics, zoom window, and scale the user is viewing
 * without any server-side session state.
 */
export function formatModelContextUpdate(state: ChartState): ModelContextPayload {
  const payload = state.current;
  if (!payload) {
    return { content: [] };
  }
  const activeRunIds = getActiveRunIds(payload);
  const allMetadataIds = Object.keys(payload.seriesMetadataById);
  const visibleMetadataIds = allMetadataIds.filter((id) => isMetadataVisible(state, id));
  const visibleRange = state.stepZoom
    ? { xKey: "_step", min: state.stepZoom.minStep, max: state.stepZoom.maxStep }
    : state.localZoom
      ? { xKey: payload.axis.xKey, min: state.localZoom.minX, max: state.localZoom.maxX }
      : null;

  const structured: Record<string, unknown> = {
    schema: "wandb.line-plot-context/v1",
    entity_name: payload.arguments.entity_name,
    project_name: payload.arguments.project_name,
    run_ids: payload.arguments.run_ids,
    active_run_ids: activeRunIds,
    metrics: payload.effectiveConfig.metrics,
    display_mode: state.displayMode,
    visible_range: visibleRange,
    y_scale: state.yLogScale ? "log" : "linear",
    smoothing_weight: Number(state.smoothingWeight.toFixed(4)),
    visible_series_count: visibleMetadataIds.length,
    total_series_count: allMetadataIds.length,
  };

  const rangeText = visibleRange
    ? `${visibleRange.xKey} ${visibleRange.min}..${visibleRange.max}`
    : "full overview";
  const text = [
    "[MCP Line Plot Context]",
    `project: ${payload.arguments.entity_name}/${payload.arguments.project_name}`,
    `runs (${payload.arguments.run_ids.length}): ${payload.arguments.run_ids.join(", ") || "none"}${
      activeRunIds.length ? ` (active: ${activeRunIds.join(", ")})` : ""
    }`,
    `metrics: ${payload.effectiveConfig.metrics.join(", ")}`,
    `view: mode=${state.displayMode}, range=${rangeText}, yScale=${
      state.yLogScale ? "log" : "linear"
    }, smoothing=${Number(state.smoothingWeight.toFixed(3))}, visibleSeries=${
      visibleMetadataIds.length
    }/${allMetadataIds.length}`,
  ].join("\n");

  return {
    content: [{ type: "text", text }],
    structuredContent: structured,
  };
}
