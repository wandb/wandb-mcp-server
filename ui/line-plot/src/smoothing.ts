import type { Point, SmoothingType } from "./schema";

export interface SmoothingSliderBounds {
  min: number;
  max: number;
  step: number;
}

export function sliderBoundsForSmoothingType(kind: SmoothingType): SmoothingSliderBounds {
  if (kind === "gaussian") {
    return { min: 0, max: 20, step: 0.25 };
  }
  if (kind === "average") {
    return { min: 0, max: 100, step: 1 };
  }
  return { min: 0, max: 0.99, step: 0.01 };
}

/**
 * Exact 1-to-1 port of `line_plot_smoothing.py` (Core `lines.ts` / `math.ts`) so
 * interactive slider changes recompute identical curves with zero server round-trips.
 */
export function smoothPoints(
  points: readonly Point[],
  weight: number,
  kind: SmoothingType,
  xMax = 100,
): Point[] {
  if (points.length === 0 || weight <= 0 || kind === "none") {
    return points.map((p) => ({ ...p }));
  }

  const values = points.map((p) => p.y);
  if (kind === "exponential" || kind === "exponentialTimeWeighted") {
    const firstVal = values[0];
    if (values.every((v) => v === firstVal)) {
      return points.map((p) => ({ ...p }));
    }
    const alpha =
      kind === "exponentialTimeWeighted"
        ? Math.sqrt(weight) * Math.sqrt(0.999)
        : Math.min(Math.sqrt(weight), 0.999);
    let last = 0;
    let debias = 0;
    const out: Point[] = [];
    for (let i = 0; i < points.length; i++) {
      const value = values[i]!;
      let y: number;
      if (kind === "exponential") {
        last = last * alpha + (1 - alpha) * value;
        y = last / (1 - alpha ** (i + 1));
      } else {
        const prevIdx = Math.max(0, i - 1);
        const delta = points[i]!.x - points[prevIdx]!.x;
        const span = xMax > 0 ? xMax : 100;
        const adjusted = alpha ** ((delta / span) * 1500);
        last = last * adjusted + value;
        debias = debias * adjusted + 1;
        y = last / debias;
      }
      out.push({ x: points[i]!.x, y });
    }
    return out;
  }

  if (kind === "average") {
    const prefix = [0];
    for (const v of values) {
      prefix.push(prefix[prefix.length - 1]! + v);
    }
    const out: Point[] = [];
    for (let i = 0; i < values.length; i++) {
      const lo = Math.max(0, Math.ceil(i - weight / 2));
      const hi = Math.min(values.length, Math.ceil(i + weight / 2));
      const count = Math.max(1, hi - lo);
      out.push({ x: points[i]!.x, y: (prefix[hi]! - prefix[lo]!) / count });
    }
    return out;
  }

  // gaussian with half-sample symmetric boundary reflection (matching Python `% period`)
  const radius = Math.min(Math.ceil(4 * weight + 0.5), 500);
  const kernel: number[] = [];
  let total = 0;
  for (let j = -radius; j <= radius; j++) {
    const k = Math.exp(-(j * j) / (2 * weight * weight));
    kernel.push(k);
    total += k;
  }
  for (let j = 0; j < kernel.length; j++) {
    kernel[j] = kernel[j]! / total;
  }

  const n = values.length;
  const period = 2 * n;
  const out: Point[] = [];
  for (let i = 0; i < n; i++) {
    let y = 0;
    for (let j = 0; j < kernel.length; j++) {
      let index = (i + j - radius) % period;
      if (index < 0) {
        index += period;
      }
      if (index >= n) {
        index = period - index - 1;
      }
      y += values[index]! * kernel[j]!;
    }
    out.push({ x: points[i]!.x, y });
  }
  return out;
}
