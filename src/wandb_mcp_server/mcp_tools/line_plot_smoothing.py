"""Bounded ports of Core sampled-line smoothing."""

import math

from wandb_mcp_server.admission import raise_if_tool_deadline_exceeded


def smooth(points: list[dict], weight: float, kind: str, x_max: float = 100) -> list[dict]:
    # Core util/plotHelpers/lines.ts:23–59 and util/math.ts:45–262 (frontends/app/src).
    if not points or not weight or kind == "none":
        return [dict(p) for p in points]
    values = [p["y"] for p in points]
    if kind in {"exponential", "exponentialTimeWeighted"}:
        if len(set(values)) == 1:
            return [dict(p) for p in points]
        result = []
        last = debias = 0.0
        alpha = min(math.sqrt(weight), 0.999)
        if kind == "exponentialTimeWeighted":
            alpha = math.sqrt(weight) * math.sqrt(0.999)
        for i, value in enumerate(values):
            if i % 256 == 0:
                raise_if_tool_deadline_exceeded()
            if kind == "exponential":
                last = last * alpha + (1 - alpha) * value
                y = last / (1 - alpha ** (i + 1))
            else:
                delta = points[i]["x"] - points[max(0, i - 1)]["x"]
                adjusted = alpha ** (delta / (x_max if x_max > 0 else 100) * 1500)
                last = last * adjusted + value
                debias = debias * adjusted + 1
                y = last / debias
            result.append(y)
    elif kind == "average":
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + value)
        result = []
        for i in range(len(values)):
            if i % 256 == 0:
                raise_if_tool_deadline_exceeded()
            lo = max(0, math.ceil(i - weight / 2))
            hi = min(len(values), math.ceil(i + weight / 2))
            result.append((prefix[hi] - prefix[lo]) / (hi - lo))
    else:
        radius = math.ceil(4 * weight + 0.5)
        kernel = [math.exp(-(j * j) / (2 * weight * weight)) for j in range(-radius, radius + 1)]
        total = sum(kernel)
        kernel = [k / total for k in kernel]
        result = []
        period = 2 * len(values)
        for i in range(len(values)):
            if i % 64 == 0:
                raise_if_tool_deadline_exceeded()
            y = 0.0
            for j, k in enumerate(kernel):
                index = (i + j - radius) % period
                if index >= len(values):
                    index = period - index - 1
                y += values[index] * k
            result.append(y)
    return [{**point, "y": y} for point, y in zip(points, result)]
