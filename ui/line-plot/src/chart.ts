import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { GraphPayload, SeriesStyle } from "./schema";
import type { ChartState, ProjectedSeriesEntry } from "./state";

export interface ChartInteractionHandlers {
  onZoomRange: (minX: number, maxX: number) => void;
  onResetZoom: () => void;
}

const DASH_PATTERNS: Record<SeriesStyle["mark"], number[] | undefined> = {
  solid: undefined,
  dashed: [7, 5],
  dotted: [2, 4],
  dotdash: [8, 4, 2, 4],
  dotdotdash: [8, 4, 2, 3, 2, 4],
  points: undefined,
};

function formatNumber(val: number): string {
  if (!Number.isFinite(val)) {
    return "—";
  }
  const abs = Math.abs(val);
  if (abs === 0) {
    return "0";
  }
  if (abs >= 1e5 || abs < 1e-3) {
    return val.toExponential(3);
  }
  return Number(val.toPrecision(5)).toString();
}

function formatXValue(val: number, isDatetime: boolean): string {
  if (!Number.isFinite(val)) {
    return "—";
  }
  if (isDatetime) {
    const d = new Date(val);
    return Number.isNaN(d.getTime()) ? String(val) : d.toISOString().replace("T", " ").slice(0, 19);
  }
  return Number.isInteger(val) ? String(val) : formatNumber(val);
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readThemePalette(): { axisText: string; gridStroke: string } {
  const dark =
    document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme &&
      window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  return dark
    ? { axisText: "#a9b4c4", gridStroke: "rgba(148, 163, 184, 0.14)" }
    : { axisText: "#475569", gridStroke: "rgba(15, 23, 42, 0.08)" };
}

export class LinePlotRenderer {
  private plot: uPlot | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private latestState: ChartState | null = null;
  private latestProjected: ProjectedSeriesEntry[] = [];

  constructor(
    private readonly plotContainer: HTMLElement,
    private readonly stageEl: HTMLElement,
    private readonly tooltipEl: HTMLElement,
    private readonly handlers: ChartInteractionHandlers,
  ) {
    this.plotContainer.addEventListener("dblclick", () => {
      this.handlers.onResetZoom();
    });
    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver(() => {
        this.resizeToStage();
      });
      this.resizeObserver.observe(this.stageEl);
    }
  }

  destroy(): void {
    if (this.plot) {
      this.plot.destroy();
      this.plot = null;
    }
    this.tooltipEl.hidden = true;
  }

  dispose(): void {
    this.destroy();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
  }

  resizeToStage(): void {
    if (!this.plot) {
      return;
    }
    const rect = this.stageEl.getBoundingClientRect();
    const width = Math.max(240, Math.floor(rect.width - 8));
    const expanded = document.getElementById("app-root")?.dataset.displayMode === "fullscreen";
    const height = expanded
      ? Math.max(300, Math.floor(window.innerHeight - 190))
      : Math.max(230, Math.min(360, Math.floor(rect.height || 268)));
    this.plot.setSize({ width, height });
  }

  render(state: ChartState, projected: ProjectedSeriesEntry[]): boolean {
    this.latestState = state;
    this.latestProjected = projected;
    const payload = state.current;
    if (!payload) {
      this.destroy();
      return false;
    }

    const visibleEntries = projected.filter(
      (s) => s.visible && s.effectivePoints.length > 0,
    );
    if (visibleEntries.length === 0) {
      this.destroy();
      return false;
    }

    const useLogX = payload.axis.xScale === "log";
    const useLogY = state.yLogScale;

    const xSet = new Set<number>();
    for (const s of projected) {
      for (const p of s.effectivePoints) {
        if (useLogX && p.x <= 0) {
          continue;
        }
        if (state.localZoom && (p.x < state.localZoom.minX || p.x > state.localZoom.maxX)) {
          continue;
        }
        xSet.add(p.x);
      }
    }
    const xs = Array.from(xSet).sort((a, b) => a - b);
    if (xs.length === 0) {
      this.destroy();
      return false;
    }

    const xIndex = new Map<number, number>();
    xs.forEach((x, idx) => xIndex.set(x, idx));

    const aligned: Array<number[] | Array<number | null>> = [xs];
    const seriesOpts: uPlot.Series[] = [
      {
        label: payload.axis.xTitle || payload.axis.xKey,
      },
    ];
    const bands: uPlot.Band[] = [];
    const tooltipMeta: Array<{
      seriesIndex: number;
      label: string;
      color: string;
      role: string;
    }> = [];

    for (const entry of projected) {
      const style: SeriesStyle = payload.seriesStylesById[entry.seriesId] ?? {
        color: "#00a99d",
        lineWidth: 1.6,
        mark: "solid",
        colorKey: entry.seriesId,
      };
      const meta = payload.seriesMetadataById[entry.metadataId];
      const label = meta?.title ?? entry.metricName;

      if (entry.kind === "area") {
        const hiArr = new Array<number | null>(xs.length).fill(null);
        const loArr = new Array<number | null>(xs.length).fill(null);
        for (const p of entry.effectivePoints) {
          const idx = xIndex.get(p.x);
          if (idx === undefined) {
            continue;
          }
          const hi = p.y;
          const lo = p.y0 ?? p.y;
          hiArr[idx] = useLogY && hi <= 0 ? null : hi;
          loArr[idx] = useLogY && lo <= 0 ? null : lo;
        }
        const hiSeriesIdx = aligned.length;
        aligned.push(hiArr);
        seriesOpts.push({
          label: `${label} (upper)`,
          show: entry.visible,
          stroke: "transparent",
          width: 0,
          points: { show: false },
          spanGaps: true,
        });
        const loSeriesIdx = aligned.length;
        aligned.push(loArr);
        seriesOpts.push({
          label: `${label} (lower)`,
          show: entry.visible,
          stroke: "transparent",
          width: 0,
          points: { show: false },
          spanGaps: true,
        });
        if (entry.visible) {
          bands.push({
            series: [hiSeriesIdx, loSeriesIdx],
            fill: style.color,
          });
        }
        continue;
      }

      const yArr = new Array<number | null>(xs.length).fill(null);
      for (const p of entry.effectivePoints) {
        const idx = xIndex.get(p.x);
        if (idx === undefined) {
          continue;
        }
        yArr[idx] = useLogY && p.y <= 0 ? null : p.y;
      }

      const sIdx = aligned.length;
      aligned.push(yArr);
      const isPointsOnly = style.mark === "points";
      seriesOpts.push({
        label,
        show: entry.visible,
        stroke: isPointsOnly ? "transparent" : style.color,
        width: isPointsOnly ? 0 : style.lineWidth,
        dash: DASH_PATTERNS[style.mark],
        spanGaps: true,
        points: {
          show: isPointsOnly || xs.length <= 2,
          size: isPointsOnly ? 5 : 4,
          fill: style.color,
          stroke: style.color,
        },
      });

      if (entry.visible && entry.role === "primary") {
        tooltipMeta.push({
          seriesIndex: sIdx,
          label,
          color: style.color,
          role: entry.role,
        });
      }
    }

    const palette = readThemePalette();
    const rect = this.stageEl.getBoundingClientRect();
    const expanded = document.getElementById("app-root")?.dataset.displayMode === "fullscreen";
    const width = Math.max(240, Math.floor((rect.width || 620) - 8));
    const height = expanded ? Math.max(300, Math.floor(window.innerHeight - 190)) : 264;
    const finePointer =
      typeof window.matchMedia === "function"
        ? window.matchMedia("(any-pointer: fine)").matches ||
          !window.matchMedia("(any-pointer: coarse)").matches
        : true;

    const xDomainLo = state.localZoom ? state.localZoom.minX : payload.xDomain[0];
    const xDomainHi = state.localZoom ? state.localZoom.maxX : payload.xDomain[1];
    const yDomainLo = payload.yDomain[0];
    const yDomainHi = payload.yDomain[1];

    this.destroy();

    const opts: uPlot.Options = {
      width,
      height,
      ms: 1,
      legend: { show: false },
      padding: [12, 16, 4, 4],
      cursor: {
        drag: {
          x: finePointer,
          y: false,
          setScale: false,
          dist: 6,
        },
        points: {
          size: 6,
        },
      },
      scales: {
        x: {
          time: payload.axis.xIsDatetime,
          distr: useLogX ? 3 : 1,
          ...(xDomainLo !== null || xDomainHi !== null
            ? {
                range: (_u: uPlot, dataMin: number, dataMax: number): [number, number] => [
                  xDomainLo ?? dataMin,
                  xDomainHi ?? dataMax,
                ],
              }
            : {}),
        },
        y: {
          distr: useLogY ? 3 : 1,
          ...(yDomainLo !== null || yDomainHi !== null
            ? {
                range: (_u: uPlot, dataMin: number, dataMax: number): [number, number] => [
                  useLogY && yDomainLo !== null && yDomainLo <= 0 ? dataMin : (yDomainLo ?? dataMin),
                  yDomainHi ?? dataMax,
                ],
              }
            : {}),
        },
      },
      axes: [
        {
          stroke: palette.axisText,
          label: payload.axis.xTitle || payload.axis.xKey,
          labelSize: 22,
          labelFont: "500 11px ui-sans-serif, system-ui, -apple-system, sans-serif",
          font: "11px ui-monospace, SFMono-Regular, Menlo, monospace",
          grid: { stroke: palette.gridStroke, width: 1 },
          ticks: { stroke: palette.gridStroke, width: 1, size: 4 },
        },
        {
          stroke: palette.axisText,
          label: payload.axis.yTitle || undefined,
          labelSize: payload.axis.yTitle ? 24 : 0,
          labelFont: "500 11px ui-sans-serif, system-ui, -apple-system, sans-serif",
          font: "11px ui-monospace, SFMono-Regular, Menlo, monospace",
          size: 58,
          grid: { stroke: palette.gridStroke, width: 1 },
          ticks: { stroke: palette.gridStroke, width: 1, size: 4 },
          values: (_u, splits) => splits.map((v) => (v == null ? "" : formatNumber(v))),
        },
      ],
      series: seriesOpts,
      bands,
      hooks: {
        setSelect: [
          (u) => {
            const sel = u.select;
            if (sel.width < 6) {
              return;
            }
            const minVal = u.posToVal(sel.left, "x");
            const maxVal = u.posToVal(sel.left + sel.width, "x");
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
            if (Number.isFinite(minVal) && Number.isFinite(maxVal) && maxVal > minVal) {
              this.handlers.onZoomRange(minVal, maxVal);
            }
          },
        ],
        setCursor: [
          (u) => {
            this.updateTooltip(u, payload, tooltipMeta);
          },
        ],
      },
    };

    this.plot = new uPlot(opts, aligned as uPlot.AlignedData, this.plotContainer);
    return true;
  }

  refreshTheme(): void {
    if (this.latestState) {
      this.render(this.latestState, this.latestProjected);
    }
  }

  private updateTooltip(
    u: uPlot,
    payload: GraphPayload,
    tooltipMeta: ReadonlyArray<{ seriesIndex: number; label: string; color: string }>,
  ): void {
    const idx = u.cursor.idx;
    const left = u.cursor.left ?? -1;
    const top = u.cursor.top ?? -1;
    if (idx == null || left < 0 || top < 0 || tooltipMeta.length === 0) {
      this.tooltipEl.hidden = true;
      return;
    }
    const xVal = u.data[0]?.[idx];
    if (xVal == null) {
      this.tooltipEl.hidden = true;
      return;
    }

    const rows: string[] = [];
    for (const item of tooltipMeta.slice(0, 8)) {
      const yVal = u.data[item.seriesIndex]?.[idx];
      if (yVal == null || !Number.isFinite(yVal)) {
        continue;
      }
      rows.push(
        `<div class="lp-tip-row"><span class="lp-swatch" style="background:${escapeHtml(
          item.color,
        )}"></span><span class="lp-tip-name">${escapeHtml(
          item.label,
        )}</span><span class="lp-tip-val">${escapeHtml(formatNumber(yVal))}</span></div>`,
      );
    }
    if (rows.length === 0) {
      this.tooltipEl.hidden = true;
      return;
    }

    const more =
      tooltipMeta.length > 8
        ? `<div class="lp-tip-more">+${tooltipMeta.length - 8} more series</div>`
        : "";
    this.tooltipEl.innerHTML = `<div class="lp-tip-x">${escapeHtml(
      payload.axis.xTitle || payload.axis.xKey,
    )}: <strong>${escapeHtml(
      formatXValue(xVal, payload.axis.xIsDatetime),
    )}</strong></div>${rows.join("")}${more}`;
    this.tooltipEl.hidden = false;

    const stageRect = this.stageEl.getBoundingClientRect();
    const tipWidth = this.tooltipEl.offsetWidth || 190;
    const tipHeight = this.tooltipEl.offsetHeight || 80;
    const offsetLeft = u.bbox.left / uPlot.pxRatio + left;
    const offsetTop = u.bbox.top / uPlot.pxRatio + top;
    const clampedX =
      offsetLeft + 14 + tipWidth > stageRect.width - 8
        ? Math.max(8, offsetLeft - tipWidth - 14)
        : offsetLeft + 14;
    const clampedY = Math.min(
      Math.max(8, offsetTop - 12),
      Math.max(8, stageRect.height - tipHeight - 8),
    );
    this.tooltipEl.style.transform = `translate(${Math.round(clampedX)}px, ${Math.round(clampedY)}px)`;
  }
}
