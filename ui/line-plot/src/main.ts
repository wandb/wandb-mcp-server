import "./style.css";
import { LinePlotRenderer } from "./chart";
import { LinePlotHostBridge } from "./host";
import { sliderBoundsForSmoothingType } from "./smoothing";
import {
  buildLiveRefreshPlan,
  computeStepZoomWindow,
  createInitialState,
  formatModelContextUpdate,
  getActiveRunIds,
  isMetadataVisible,
  projectDisplaySeries,
  reduceChartState,
  selectPreferredExpandMode,
  type ChartAction,
  type ChartState,
} from "./state";

const LIVE_REFRESH_INTERVAL_MS = 10_000;

function mustGet<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing DOM element #${id}`);
  }
  return el as T;
}

function formatClockTime(epochMs: number): string {
  const d = new Date(epochMs);
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function bootstrap(): void {
  const rootEl = mustGet<HTMLDivElement>("app-root");
  const titleEl = mustGet<HTMLHeadingElement>("lp-title");
  const activeBadgeEl = mustGet<HTMLSpanElement>("lp-active-badge");
  const subtitleEl = mustGet<HTMLParagraphElement>("lp-subtitle");
  const expandBtn = mustGet<HTMLButtonElement>("lp-expand-btn");
  const openWandbBtn = mustGet<HTMLButtonElement>("lp-open-wandb-btn");
  const toolbarEl = mustGet<HTMLElement>("lp-toolbar");
  const smoothSlider = mustGet<HTMLInputElement>("lp-smooth-slider");
  const smoothValueEl = mustGet<HTMLOutputElement>("lp-smooth-value");
  const logYBtn = mustGet<HTMLButtonElement>("lp-logy-btn");
  const resetZoomBtn = mustGet<HTMLButtonElement>("lp-reset-zoom-btn");
  const mainEl = mustGet<HTMLElement>("lp-main");
  const legendEl = mustGet<HTMLDivElement>("lp-legend");
  const stageEl = mustGet<HTMLDivElement>("lp-stage");
  const emptyEl = mustGet<HTMLDivElement>("lp-empty");
  const plotEl = mustGet<HTMLDivElement>("lp-plot");
  const tooltipEl = mustGet<HTMLDivElement>("lp-tooltip");
  const bannerEl = mustGet<HTMLDivElement>("lp-banner");
  const bannerTextEl = mustGet<HTMLSpanElement>("lp-banner-text");
  const bannerDismissBtn = mustGet<HTMLButtonElement>("lp-banner-dismiss");
  const footerEl = mustGet<HTMLElement>("lp-footer");
  const metaEl = mustGet<HTMLDivElement>("lp-meta");
  const liveLabelEl = mustGet<HTMLLabelElement>("lp-live-label");
  const liveCheckbox = mustGet<HTMLInputElement>("lp-live-checkbox");
  const refreshedAtEl = mustGet<HTMLSpanElement>("lp-refreshed-at");
  const refreshedTimeEl = mustGet<HTMLTimeElement>("lp-refreshed-time");

  let state: ChartState = createInitialState();
  let requestSeq = 0;
  let refreshTimer: number | null = null;

  const renderer = new LinePlotRenderer(plotEl, stageEl, tooltipEl, {
    onZoomRange: (minX, maxX) => {
      void handleZoomRange(minX, maxX);
    },
    onResetZoom: () => {
      if (state.stepZoom || state.localZoom) {
        requestSeq += 1;
        dispatch({ type: "resetZoom" }, true);
      }
    },
  });

  const host = new LinePlotHostBridge({
    onToolInput: (title) => {
      requestSeq += 1;
      dispatch({ type: "toolInput", title });
    },
    onPayloadLoaded: (payload) => {
      requestSeq += 1;
      dispatch({ type: "loaded", payload, now: Date.now() }, true);
    },
    onToolFailed: (message) => {
      requestSeq += 1;
      dispatch({ type: "loadFailed", message });
    },
    onHostContext: (displayMode, availableDisplayModes, _theme) => {
      dispatch(
        {
          type: "hostContextChanged",
          displayMode,
          availableDisplayModes,
        },
        true,
      );
      renderer.refreshTheme();
    },
    onTeardown: () => {
      stopRefreshLoop();
      renderer.dispose();
    },
  });

  function stopRefreshLoop(): void {
    if (refreshTimer !== null) {
      window.clearInterval(refreshTimer);
      refreshTimer = null;
    }
  }

  function syncRefreshLoop(): void {
    const shouldPoll =
      state.status === "ready" &&
      Boolean(state.current?.hasActiveRuns) &&
      state.liveRefreshEnabled &&
      host.canCallServerTools();

    if (!shouldPoll) {
      stopRefreshLoop();
      return;
    }
    if (refreshTimer === null) {
      refreshTimer = window.setInterval(() => {
        void runLiveRefreshTick();
      }, LIVE_REFRESH_INTERVAL_MS);
    }
  }

  async function runLiveRefreshTick(): Promise<void> {
    if (state.isBusy || !state.current || !state.liveRefreshEnabled) {
      return;
    }
    const plan = buildLiveRefreshPlan(state.current, state.stepZoom);
    if (!plan) {
      return;
    }
    const seq = ++requestSeq;
    dispatch({ type: "requeryStarted" });
    const outcome = await host.callLinePlotTool(plan.arguments);
    if (seq !== requestSeq) {
      return;
    }
    if (outcome.ok) {
      dispatch(
        {
          type: "refreshed",
          payload: outcome.payload,
          refreshedRunIds: plan.refreshedRunIds,
          now: Date.now(),
        },
        true,
      );
    } else {
      dispatch({
        type: "operationFailed",
        kind: "warning",
        message: `Live refresh paused on error: ${outcome.message}`,
      });
    }
  }

  async function handleZoomRange(minX: number, maxX: number): Promise<void> {
    const payload = state.current;
    if (!payload) {
      return;
    }
    if (payload.axis.xKey === "_step" && host.canCallServerTools()) {
      const win = computeStepZoomWindow(minX, maxX, payload.range_cap);
      if (win.canRequery) {
        const seq = ++requestSeq;
        dispatch({ type: "requeryStarted" });
        const outcome = await host.callLinePlotTool({
          entity_name: payload.arguments.entity_name,
          project_name: payload.arguments.project_name,
          run_ids: payload.arguments.run_ids,
          config: payload.arguments.config,
          samples: payload.arguments.samples,
          min_step: win.minStep,
          max_step: win.maxStep,
        });
        if (seq !== requestSeq) {
          return;
        }
        if (outcome.ok) {
          dispatch(
            {
              type: "zoomLoaded",
              payload: outcome.payload,
              stepZoom: { minStep: win.minStep, maxStep: win.maxStep },
              now: Date.now(),
            },
            true,
          );
        } else {
          dispatch({
            type: "operationFailed",
            kind: "warning",
            message: `Zoom requery failed (${outcome.message}). Keeping current chart.`,
          });
        }
        return;
      }
      if (win.maxStep > win.minStep) {
        dispatch({
          type: "setLocalZoom",
          zoom: { minX: win.minStep, maxX: win.maxStep },
        }, true);
      }
      return;
    }
    dispatch(
      {
        type: "setLocalZoom",
        zoom: { minX: Math.min(minX, maxX), maxX: Math.max(minX, maxX) },
      },
      true,
    );
  }

  function dispatch(action: ChartAction, immediateContext = false): void {
    state = reduceChartState(state, action);
    renderUI();
    syncRefreshLoop();
    if (state.current) {
      host.scheduleModelContextSync(formatModelContextUpdate(state), immediateContext);
    }
  }

  function renderLegend(payload: NonNullable<ChartState["current"]>): void {
    if (!payload.legend.show) {
      legendEl.hidden = true;
      legendEl.replaceChildren();
      return;
    }

    const seenMetadata = new Set<string>();
    const items: HTMLElement[] = [];

    for (const entry of payload.series) {
      if (entry.role !== "primary" || seenMetadata.has(entry.metadataId)) {
        continue;
      }
      seenMetadata.add(entry.metadataId);
      const meta = payload.seriesMetadataById[entry.metadataId];
      const style = payload.seriesStylesById[entry.seriesId];
      if (!meta) {
        continue;
      }

      const visible = isMetadataVisible(state, entry.metadataId);
      const isolated = state.isolatedMetadataId === entry.metadataId;
      const runState = meta.runState;
      const isActiveRun = Boolean(
        meta.runName && payload.runsById[meta.runName]?.active,
      );

      const chip = document.createElement("div");
      chip.className = "lp-legend-item";
      chip.dataset.visible = String(visible);
      chip.dataset.isolated = String(isolated);

      const toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.className = "lp-legend-toggle";
      toggleBtn.setAttribute("aria-pressed", String(visible));
      toggleBtn.title =
        "Click to show/hide · Cmd/Ctrl-click (or double-click) to isolate series";

      const swatch = document.createElement("span");
      swatch.className = "lp-swatch";
      swatch.style.background = style?.color ?? "#00a99d";

      const labelSpan = document.createElement("span");
      labelSpan.className = "lp-legend-label";
      labelSpan.textContent = meta.title;

      toggleBtn.append(swatch, labelSpan);

      if (isActiveRun) {
        const dot = document.createElement("span");
        dot.className = "lp-run-dot";
        dot.title = `Run ${runState ?? "running"}`;
        toggleBtn.append(dot);
      }

      toggleBtn.addEventListener("click", (ev) => {
        if (ev.metaKey || ev.ctrlKey || ev.altKey) {
          dispatch({ type: "isolateSeries", metadataId: entry.metadataId }, true);
        } else {
          dispatch({ type: "toggleSeries", metadataId: entry.metadataId }, true);
        }
      });

      toggleBtn.addEventListener("dblclick", (ev) => {
        ev.preventDefault();
        dispatch({ type: "isolateSeries", metadataId: entry.metadataId }, true);
      });

      const linkBtn = document.createElement("button");
      linkBtn.type = "button";
      linkBtn.className = "lp-legend-link";
      linkBtn.textContent = "↗";
      linkBtn.title = `Open ${meta.displayName} in W&B`;
      linkBtn.setAttribute("aria-label", `Open ${meta.displayName} in W&B`);
      linkBtn.addEventListener("click", () => {
        void host.openExternalUrl(meta.legendHref).then((res) => {
          if (!res.ok) {
            dispatch({ type: "operationFailed", kind: "warning", message: res.message });
          }
        });
      });

      chip.append(toggleBtn, linkBtn);
      items.push(chip);
    }

    legendEl.hidden = items.length === 0;
    legendEl.replaceChildren(...items);
  }

  function renderUI(): void {
    rootEl.dataset.displayMode = state.displayMode;
    rootEl.dataset.busy = String(state.isBusy);

    // Expand / inline button
    const preferredExpand = selectPreferredExpandMode(state.availableDisplayModes);
    if (state.displayMode !== "inline") {
      expandBtn.hidden = false;
      expandBtn.textContent = "Inline view";
      expandBtn.setAttribute("aria-label", "Return chart to inline view");
    } else if (preferredExpand !== null && state.current) {
      expandBtn.hidden = false;
      expandBtn.textContent =
        preferredExpand === "pip" ? "Pop out chart" : "Expand chart";
      expandBtn.setAttribute("aria-label", expandBtn.textContent);
    } else {
      expandBtn.hidden = true;
    }

    // Banner
    if (state.inlineBanner) {
      bannerEl.hidden = false;
      bannerEl.dataset.kind = state.inlineBanner.kind;
      bannerEl.setAttribute(
        "role",
        state.inlineBanner.kind === "error" ? "alert" : "status",
      );
      bannerTextEl.textContent = state.inlineBanner.message;
    } else {
      bannerEl.hidden = true;
      bannerTextEl.textContent = "";
    }

    if (state.status === "idle" || state.status === "loading") {
      titleEl.textContent = state.pendingTitle
        ? `Loading ${state.pendingTitle}…`
        : "Loading W&B metrics…";
      subtitleEl.textContent = "Fetching sampled run history…";
      activeBadgeEl.hidden = true;
      openWandbBtn.hidden = true;
      toolbarEl.hidden = true;
      legendEl.hidden = true;
      footerEl.hidden = true;
      plotEl.hidden = true;
      emptyEl.hidden = false;
      emptyEl.dataset.error = "false";
      emptyEl.textContent = "Loading run metrics from W&B…";
      renderer.destroy();
      return;
    }

    if (state.status === "error" || !state.current) {
      titleEl.textContent = "Unable to render line plot";
      subtitleEl.textContent = "W&B MCP Chart";
      activeBadgeEl.hidden = true;
      openWandbBtn.hidden = true;
      toolbarEl.hidden = true;
      legendEl.hidden = true;
      footerEl.hidden = true;
      plotEl.hidden = true;
      emptyEl.hidden = false;
      emptyEl.dataset.error = "true";
      emptyEl.textContent = state.fatalError ?? "Line plot request failed.";
      renderer.destroy();
      return;
    }

    const payload = state.current;
    titleEl.textContent = payload.title;
    const runCount = payload.arguments.run_ids.length;
    const groupSuffix = payload.coverage.grouped ? " · grouped" : "";
    const busySuffix = state.isBusy ? " · updating…" : "";
    subtitleEl.textContent = `${payload.arguments.entity_name}/${payload.arguments.project_name} · ${runCount} ${
      runCount === 1 ? "run" : "runs"
    }${groupSuffix}${busySuffix}`;

    const activeIds = getActiveRunIds(payload);
    if (activeIds.length > 0) {
      activeBadgeEl.hidden = false;
      activeBadgeEl.textContent = `${activeIds.length} active`;
    } else {
      activeBadgeEl.hidden = true;
    }

    openWandbBtn.hidden = false;
    toolbarEl.hidden = payload.series.length === 0;
    footerEl.hidden = false;
    mainEl.dataset.legend = payload.legend.position;

    // Controls sync
    const bounds = sliderBoundsForSmoothingType(payload.effectiveConfig.smoothingType);
    smoothSlider.min = String(bounds.min);
    smoothSlider.max = String(bounds.max);
    smoothSlider.step = String(bounds.step);
    smoothSlider.value = String(state.smoothingWeight);
    smoothValueEl.textContent = Number(state.smoothingWeight.toFixed(2)).toString();

    logYBtn.setAttribute("aria-pressed", String(state.yLogScale));
    resetZoomBtn.hidden = !(state.stepZoom || state.localZoom);

    renderLegend(payload);

    const projected = projectDisplaySeries(state);
    const hasDrawablePoints = projected.some(
      (s) => s.visible && s.effectivePoints.length > 0,
    );

    if (!hasDrawablePoints) {
      renderer.destroy();
      plotEl.hidden = true;
      emptyEl.hidden = false;
      emptyEl.dataset.error = "false";
      if (payload.series.length === 0 && payload.coverage.autoDiscoveredRuns) {
        emptyEl.textContent = `No runs in ${payload.arguments.entity_name}/${payload.arguments.project_name} have logged ${payload.effectiveConfig.metrics.join(", ")} yet.`;
      } else if (payload.series.length > 0 && projected.every((s) => !s.visible)) {
        emptyEl.textContent = "All series are hidden. Click a series in the legend above to show it.";
      } else {
        emptyEl.textContent = "No finite numeric data points in this window.";
      }
    } else {
      emptyEl.hidden = true;
      plotEl.hidden = false;
      renderer.render(state, projected);
    }

    // Footer metadata & Live Refresh controls
    const badges: string[] = [];
    if (state.stepZoom) {
      badges.push(`Steps ${state.stepZoom.minStep}–${state.stepZoom.maxStep}`);
    } else if (state.localZoom) {
      badges.push(
        `${payload.axis.xKey} ${Number(state.localZoom.minX.toPrecision(4))}–${Number(
          state.localZoom.maxX.toPrecision(4),
        )}`,
      );
    }
    if (payload.coverage.sampled) {
      badges.push("Sampled");
    }
    if (payload.coverage.truncated) {
      badges.push("Truncated");
    }
    if (payload.coverage.perRunErrors.length > 0) {
      badges.push(`${payload.coverage.perRunErrors.length} run error(s)`);
    }
    metaEl.textContent = badges.join(" · ") || `x: ${payload.axis.xKey}`;

    if (payload.hasActiveRuns && host.canCallServerTools()) {
      liveLabelEl.hidden = false;
      liveCheckbox.checked = state.liveRefreshEnabled;
    } else {
      liveLabelEl.hidden = true;
    }

    if (state.lastRefreshedAt !== null) {
      refreshedAtEl.hidden = false;
      refreshedTimeEl.dateTime = new Date(state.lastRefreshedAt).toISOString();
      refreshedTimeEl.textContent = formatClockTime(state.lastRefreshedAt);
    } else {
      refreshedAtEl.hidden = true;
    }
  }

  // Event listeners
  expandBtn.addEventListener("click", () => {
    const targetMode =
      state.displayMode === "inline"
        ? (selectPreferredExpandMode(state.availableDisplayModes) ?? "fullscreen")
        : "inline";
    void host.requestDisplayMode(targetMode).then((res) => {
      if (res.ok) {
        dispatch({ type: "hostContextChanged", displayMode: res.mode }, true);
        renderer.resizeToStage();
      } else {
        dispatch({ type: "operationFailed", kind: "warning", message: res.message });
      }
    });
  });

  openWandbBtn.addEventListener("click", () => {
    const targetUrl = state.current?.workspaceUrl ?? state.current?.url;
    if (!targetUrl) {
      return;
    }
    void host.openExternalUrl(targetUrl).then((res) => {
      if (!res.ok) {
        dispatch({ type: "operationFailed", kind: "warning", message: res.message });
      }
    });
  });

  smoothSlider.addEventListener("input", () => {
    const parsed = Number.parseFloat(smoothSlider.value);
    if (Number.isFinite(parsed)) {
      dispatch({ type: "setSmoothing", weight: parsed });
    }
  });

  logYBtn.addEventListener("click", () => {
    dispatch({ type: "toggleLogY" }, true);
  });

  resetZoomBtn.addEventListener("click", () => {
    requestSeq += 1;
    dispatch({ type: "resetZoom" }, true);
  });

  bannerDismissBtn.addEventListener("click", () => {
    dispatch({ type: "dismissBanner" });
  });

  liveCheckbox.addEventListener("change", () => {
    dispatch({ type: "setLiveRefresh", enabled: liveCheckbox.checked });
  });

  renderUI();
  void host.connect().catch((err) => {
    const message =
      err instanceof Error && err.message
        ? err.message
        : "Failed to initialize MCP App connection.";
    dispatch({ type: "loadFailed", message });
  });
}

bootstrap();
