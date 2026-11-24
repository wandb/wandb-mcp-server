import { isAllowedHttpsUrl } from "./url-guard";

const EXAMPLE_LIMIT = 5;
const SPAN_REFERENCE_RE = /\[\[span:([A-Za-z0-9_-]+(?:,[A-Za-z0-9_-]+)*)\]\]/g;

export type FailureSeverityId = "info" | "minor" | "major";

export const FAILURE_SEVERITY_LABELS: Record<FailureSeverityId, string> = {
  info: "Low",
  minor: "Medium",
  major: "High",
};

export function normalizeFailureSeverity(raw: unknown): FailureSeverityId | null {
  if (typeof raw !== "string") {
    return null;
  }
  const lowered = raw.trim().toLowerCase();
  if (lowered === "info" || lowered === "minor" || lowered === "major") {
    return lowered;
  }
  return null;
}

export function formatFailureSeverity(raw: FailureSeverityId | string | null | undefined): string | null {
  const sev = normalizeFailureSeverity(raw);
  return sev ? FAILURE_SEVERITY_LABELS[sev] : null;
}

export type ReasonSegment =
  | { kind: "text"; text: string }
  | { kind: "spans"; spanIds: string[] };

export function parseFailureReasonSegments(reason: string): ReasonSegment[] {
  const segments: ReasonSegment[] = [];
  let offset = 0;
  for (const match of reason.matchAll(SPAN_REFERENCE_RE)) {
    const idx = match.index ?? 0;
    if (idx > offset) {
      segments.push({ kind: "text", text: reason.slice(offset, idx) });
    }
    const ids = (match[1] ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length > 0) {
      segments.push({ kind: "spans", spanIds: ids });
    }
    offset = idx + match[0].length;
  }
  if (offset < reason.length) {
    segments.push({ kind: "text", text: reason.slice(offset) });
  }
  return segments;
}

export function formatPlainFailureReason(reason: string | null | undefined): string {
  if (!reason) {
    return "";
  }
  return reason.replace(SPAN_REFERENCE_RE, (_full, group: string) => {
    const ids = group
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length === 0) {
      return "";
    }
    return `span${ids.length === 1 ? "" : "s"} ${ids.join(", ")}`;
  });
}

export function extractForgeOrigin(forgeUrl: string | null | undefined): string {
  if (forgeUrl && isAllowedHttpsUrl(forgeUrl)) {
    try {
      const parsed = new URL(forgeUrl);
      return `${parsed.protocol}//${parsed.host}`;
    } catch {
      // fall through
    }
  }
  return "https://forge.coreweave.com";
}

export function rangeParamForWindow(window: {
  startAt: string;
  endAt: string;
  windowDays?: number | null;
}): string {
  if (window.windowDays === 1) return "24h";
  if (window.windowDays === 7) return "7d";
  if (window.windowDays === 30) return "30d";
  const startMs = Date.parse(window.startAt);
  const endMs = Date.parse(window.endAt);
  if (Number.isFinite(startMs) && Number.isFinite(endMs) && endMs > startMs) {
    return `exclusive:${Math.trunc(startMs)}:${Math.trunc(endMs)}`;
  }
  return "7d";
}

export function buildInsightsSelectionUrl(opts: {
  forgeUrl?: string | null;
  entity: string;
  project: string;
  insightType: "failure" | "intent";
  categoryId: string;
  clusterId?: string | null;
  window: { startAt: string; endAt: string; windowDays?: number | null };
}): string {
  const origin = extractForgeOrigin(opts.forgeUrl);
  const path = [
    "agent-lens",
    encodeURIComponent(opts.entity),
    encodeURIComponent(opts.project),
    "insights",
    opts.insightType,
  ].join("/");
  const filterVal = opts.clusterId ? `cluster:${opts.clusterId}` : `category:${opts.categoryId}`;
  const filterJson = JSON.stringify([
    {
      id: opts.insightType,
      value: { operator: "eq", value: [filterVal] },
    },
  ]);
  const params = new URLSearchParams({
    range: rangeParamForWindow(opts.window),
    filter: filterJson,
  });
  return `${origin}/${path}?${params.toString()}`;
}

export function buildConversationTurnUrl(opts: {
  forgeUrl?: string | null;
  entity: string;
  project: string;
  conversationId: string;
  traceId?: string | null;
  spanId?: string | null;
}): string {
  const origin = extractForgeOrigin(opts.forgeUrl);
  const segments = [
    "agent-lens",
    encodeURIComponent(opts.entity),
    encodeURIComponent(opts.project),
    "conversations",
    encodeURIComponent(opts.conversationId),
  ];
  if (opts.traceId) {
    if (opts.spanId) {
      segments.push("spans", "turns", encodeURIComponent(opts.traceId), "spans", encodeURIComponent(opts.spanId));
    } else {
      segments.push("thread", "turns", encodeURIComponent(opts.traceId));
    }
  }
  return `${origin}/${segments.join("/")}`;
}

export interface FailureClusterTurnBrief {
  conversation_id: string;
  trace_id: string;
  url: string;
  failure_evidence_span_ids: string[];
  failure_reason?: string;
}

export interface FailureClusterMarkdownInput {
  categoryId: string;
  categoryLabel: string;
  clusterId?: string | null;
  clusterLabel?: string | null;
  description?: string | null;
  severity?: string | null;
  count?: number | null;
  conversationCount?: number | null;
  window: { startAt: string; endAt: string };
  project: { entity: string; project: string };
  url?: string | null;
  turns: FailureClusterTurnBrief[];
}

function isUtcMidnight(timestamp: string): boolean {
  const parsed = new Date(timestamp);
  return !Number.isNaN(parsed.getTime()) && parsed.getTime() % 86_400_000 === 0;
}

function formatBoundary(timestamp: string, withTime: boolean): string {
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return timestamp;
  }
  const iso = parsed.toISOString();
  return withTime ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : iso.slice(0, 10);
}

function formatWindowLabel(window: { startAt: string; endAt: string }): string {
  const withTime = !(isUtcMidnight(window.startAt) && isUtcMidnight(window.endAt));
  return `${formatBoundary(window.startAt, withTime)} to ${formatBoundary(window.endAt, withTime)}`;
}

function formatElapsedLabel(window: { startAt: string; endAt: string }): string {
  const ms = new Date(window.endAt).getTime() - new Date(window.startAt).getTime();
  if (!Number.isFinite(ms) || ms <= 0) {
    return "";
  }
  const units: Array<[string, number]> = [
    ["day", 86_400_000],
    ["hour", 3_600_000],
    ["minute", 60_000],
    ["second", 1_000],
  ];
  for (const [unit, size] of units) {
    if (ms < size) continue;
    const value = Math.round(ms / size);
    return ` over ${value} ${unit}${value === 1 ? "" : "s"}`;
  }
  return "";
}

function formatFailureCount(count: number, categoryLabel: string): string {
  const kind = categoryLabel
    .toLowerCase()
    .replace(/\s*failures?$/, "")
    .trim();
  return `${count.toLocaleString("en-US")} ${kind ? `${kind} ` : ""}failure${count === 1 ? "" : "s"}`;
}

function buildSummaryLine(input: FailureClusterMarkdownInput): string {
  const detected =
    input.count === undefined || input.count === null
      ? "Agent Lens grouped these agent failures as similar"
      : `Agent Lens detected ${formatFailureCount(input.count, input.categoryLabel)}${formatElapsedLabel(input.window)}`;
  const desc = input.description?.trim();
  return desc ? `${detected}:\n"${desc}"` : `${detected}.`;
}

function buildFactList(input: FailureClusterMarkdownInput): string {
  const impactParts: string[] = [];
  if (input.count !== undefined && input.count !== null) {
    impactParts.push(`${input.count.toLocaleString("en-US")} turns`);
  }
  if (input.conversationCount !== undefined && input.conversationCount !== null) {
    impactParts.push(`${input.conversationCount.toLocaleString("en-US")} conversations`);
  }

  const sevLabel = formatFailureSeverity(input.severity) ?? input.severity ?? null;
  const facts: Array<[string, string]> = [
    ["Window", formatWindowLabel(input.window)],
    ["Project", `${input.project.entity}/${input.project.project}`],
    ["Category", input.categoryLabel],
  ];
  if (input.clusterLabel) {
    facts.push(["Cluster", input.clusterLabel]);
  }
  if (sevLabel) {
    facts.push(["Severity", sevLabel]);
  }
  if (impactParts.length > 0) {
    facts.push(["Impact", impactParts.join(" in ")]);
  }
  if (input.clusterId) {
    facts.push(["Topic ID", input.clusterId]);
  }
  return facts.map(([label, value]) => `- ${label}: ${value}`).join("\n");
}

function buildExamplesSection(turns: FailureClusterTurnBrief[]): string {
  if (turns.length === 0) {
    return "None loaded. Expand a cluster or category in Agent Lens Insights to load example turns.";
  }
  return turns
    .slice(0, EXAMPLE_LIMIT)
    .map((turn) => {
      const ids = [
        `- URL: ${turn.url}`,
        `- Conversation ID: ${turn.conversation_id}`,
        `- Trace ID: ${turn.trace_id}`,
        `- Span IDs: [${turn.failure_evidence_span_ids.join(", ")}]`,
      ].join("\n");
      const reason = formatPlainFailureReason(turn.failure_reason)?.trim();
      return reason ? `"${reason}"\n\n${ids}` : ids;
    })
    .join("\n\n");
}

/**
 * Build the Agent Lens failure-cluster markdown brief for coding-agent model context sync.
 */
export function failureClusterToMarkdown(input: FailureClusterMarkdownInput): string {
  const heading = `# ${input.clusterLabel || input.categoryLabel}`;
  const sections: Array<string | undefined> = [
    heading,
    buildSummaryLine(input),
    input.url ? `View similar failures online at ${input.url}` : undefined,
    buildFactList(input),
    "## Examples",
    buildExamplesSection(input.turns),
  ];
  return sections.filter((s): s is string => Boolean(s)).join("\n\n");
}
