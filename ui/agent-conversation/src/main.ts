import "./style.css";
import {
  formatFailureSeverity,
  formatPlainFailureReason,
  parseFailureReasonSegments,
} from "../../shared/failure-cluster-markdown";
import { ConversationHostBridge } from "./host";
import {
  buildSpanWaterfall,
  parseConversationTags,
  parseFailureAttributions,
  parseTagCatalog,
  sameTraceId,
  type AgentFailureFinding,
  type NormalizedMessage,
  type NormalizedTurn,
  type SpanWaterfallNode,
} from "./schema";
import {
  buildActiveConversationForgeLink,
  buildConversationModelContext,
  canPerformWrites,
  createInitialConversationState,
  getActiveFailureFinding,
  getEvidenceSpanIdsSet,
  getVisibleTurns,
  reduceConversationState,
  type ConversationAction,
  type ConversationState,
} from "./state";

const rootEl = document.getElementById("app-root") as HTMLElement;
const titleEl = document.getElementById("ac-title") as HTMLElement;
const copyConvBtn = document.getElementById("ac-copy-conv-btn") as HTMLButtonElement;
const versionPillEl = document.getElementById("ac-version-pill") as HTMLElement;
const errorPillEl = document.getElementById("ac-error-pill") as HTMLElement;
const metaStripEl = document.getElementById("ac-meta-strip") as HTMLElement;
const tagsRowEl = document.getElementById("ac-tags-row") as HTMLElement;
const expandBtn = document.getElementById("ac-expand-btn") as HTMLButtonElement;
const openForgeBtn = document.getElementById("ac-open-forge-btn") as HTMLButtonElement;
const toolbarEl = document.getElementById("ac-toolbar") as HTMLElement;
const tabConvBtn = document.getElementById("ac-tab-conversation") as HTMLButtonElement;
const tabSpansBtn = document.getElementById("ac-tab-spans") as HTMLButtonElement;
const failuresBtn = document.getElementById("ac-failures-btn") as HTMLButtonElement;
const errorsOnlyBtn = document.getElementById("ac-errors-only-btn") as HTMLButtonElement;
const tagMenuBtn = document.getElementById("ac-tag-menu-btn") as HTMLButtonElement;
const failuresPanelEl = document.getElementById("ac-failures-panel") as HTMLElement;
const tagPopoverEl = document.getElementById("ac-tag-popover") as HTMLElement;
const bannerEl = document.getElementById("ac-banner") as HTMLElement;
const bannerTextEl = document.getElementById("ac-banner-text") as HTMLElement;
const bannerDismissBtn = document.getElementById("ac-banner-dismiss") as HTMLButtonElement;
const emptyEl = document.getElementById("ac-empty") as HTMLElement;
const convViewEl = document.getElementById("ac-conversation-view") as HTMLElement;
const spansViewEl = document.getElementById("ac-spans-view") as HTMLElement;
const spansListEl = document.getElementById("ac-spans-list") as HTMLElement;
const spanDrawerEl = document.getElementById("ac-span-drawer") as HTMLElement;

let state: ConversationState = createInitialConversationState();

function makeIconSvg(kind: "turn" | "llm" | "tool" | "copy"): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", "13");
  svg.setAttribute("height", "13");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.4");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");

  const path = document.createElementNS(ns, "path");
  if (kind === "turn") {
    path.setAttribute("d", "M8 1.8 13.5 5v6L8 14.2 2.5 11V5L8 1.8Z M2.6 5.1 8 8.2l5.4-3.1 M8 8.2v6");
  } else if (kind === "llm") {
    path.setAttribute("d", "M2.5 3.5h11v7.5H6l-3.5 2.5V3.5Z");
  } else if (kind === "tool") {
    path.setAttribute("d", "M8 2 13 4.9v6.2L8 14 3 11.1V4.9L8 2Zm0 4a2 2 0 1 0 0 4 2 2 0 0 0 0-4Z");
  } else {
    path.setAttribute("d", "M5.5 5.5h7v7h-7v-7Zm-2 4h-1v-7h7v1");
  }
  svg.appendChild(path);
  return svg;
}

function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) {
    return "—";
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  return `${Math.round(ms)}ms`;
}

function formatCompactCount(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) {
    return "0";
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}K`;
  }
  return n.toLocaleString("en-US");
}

function formatTurnTimestamp(iso: string | null | undefined): string | null {
  if (!iso || typeof iso !== "string") {
    return null;
  }
  const match = iso.match(/T(\d{2}:\d{2}:\d{2})/);
  if (match) {
    return `${match[1]} UTC`;
  }
  return null;
}

async function copyText(value: string, label: string): Promise<void> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      dispatch({ type: "banner", kind: "info", message: `Copied ${label}.` }, false);
      return;
    }
  } catch {
    // clipboard blocked inside sandboxed iframe
  }
  dispatch({ type: "banner", kind: "info", message: `${label}: ${value.slice(0, 120)}` }, false);
}

const bridge = new ConversationHostBridge({
  onToolInput(title, selectedFailure) {
    dispatch({ type: "toolInput", title, selectedFailure }, false, false);
  },
  onPayloadLoaded(payload) {
    dispatch({ type: "loaded", payload }, false, false);
    void loadInitialConversationTags();
    void loadInitialAgentFailures();
  },
  onToolFailed(message) {
    dispatch({ type: "loadFailed", message }, false, false);
  },
  onHostContext(displayMode, availableDisplayModes) {
    dispatch({ type: "hostContextChanged", displayMode, availableDisplayModes }, false, false);
  },
  onTeardown() {
    state = createInitialConversationState();
    render();
  },
});

function dispatch(
  action: ConversationAction,
  syncContext = true,
  immediateContext = false,
): void {
  state = reduceConversationState(state, action);
  render();
  if (syncContext) {
    bridge.scheduleModelContextSync(buildConversationModelContext(state), immediateContext);
  }
}

async function loadInitialConversationTags(): Promise<void> {
  const payload = state.payload;
  if (!payload || !payload.agentLensAvailable || !payload.conversationId || !bridge.canCallServerTools()) {
    return;
  }
  const outcome = await bridge.invokeJsonTool("get_agent_lens_conversation_tags_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    conversation_ids: [payload.conversationId],
  });
  if (outcome.ok) {
    dispatch({ type: "tagsLoaded", tags: parseConversationTags(outcome.data) }, false);
  }
}

async function loadInitialAgentFailures(): Promise<void> {
  const payload = state.payload;
  if (!payload || !payload.agentLensAvailable || !bridge.canCallServerTools()) {
    return;
  }
  const traceIds = Array.from(
    new Set(
      [
        ...payload.turns.map((t) => t.traceId),
        payload.traceId,
        payload.selectedFailure?.traceId ?? null,
      ].filter((id): id is string => Boolean(id)),
    ),
  );
  if (traceIds.length === 0) {
    return;
  }
  const outcome = await bridge.invokeJsonTool("get_agent_lens_failure_attributions_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    trace_ids: traceIds.slice(0, 20),
  });
  if (outcome.ok) {
    const failures = parseFailureAttributions(outcome.data);
    dispatch({ type: "failuresLoaded", failures }, false, false);
  }
}

async function ensureSpanWaterfallLoaded(): Promise<void> {
  const payload = state.payload;
  if (!payload || state.spansStatus === "loading" || state.spansStatus === "ready") {
    return;
  }
  if (!payload.conversationId && !payload.traceId) {
    dispatch(
      {
        type: "spansFailed",
        message: "Span tree requires a conversation_id or trace_id on the trace.",
      },
      false,
    );
    return;
  }
  if (!bridge.canCallServerTools()) {
    dispatch(
      {
        type: "spansFailed",
        message: "This MCP host does not support lazy span tree loading.",
      },
      false,
    );
    return;
  }
  dispatch({ type: "spansLoading" }, false);
  const outcome = await bridge.invokeJsonTool("query_weave_agent_spans_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    ...(payload.conversationId
      ? { conversation_id: payload.conversationId }
      : { trace_id: payload.traceId }),
    include_details: true,
    include_costs: true,
    sort_by: "started_at",
    sort_direction: "asc",
    limit: 200,
  });
  if (!outcome.ok) {
    dispatch({ type: "spansFailed", message: outcome.message }, false);
    return;
  }
  dispatch({ type: "spansLoaded", spans: buildSpanWaterfall(outcome.data) }, true, true);
}

function focusFailureAndEvidenceSpan(finding: AgentFailureFinding, spanId?: string | null): void {
  dispatch({ type: "focusFailure", failureId: finding.failureId, spanId }, true, true);
  if ((spanId ?? finding.evidenceSpanIds[0]) && (state.payload?.conversationId || state.payload?.traceId)) {
    void ensureSpanWaterfallLoaded();
  }
}

async function submitReaction(turn: NormalizedTurn, emoji: "👍" | "👎"): Promise<void> {
  const payload = state.payload;
  if (!payload || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = turn.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "Turn is missing a trace_id for feedback." }, false);
    return;
  }
  dispatch({ type: "actionStarted" }, false);
  const outcome = await bridge.invokeJsonTool("manage_weave_agent_feedback_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "add_reaction",
    conversation_id: payload.conversationId ?? undefined,
    trace_id: traceId,
    emoji,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message }, false);
    return;
  }
  const createdId =
    outcome.data && typeof outcome.data === "object" && typeof (outcome.data as Record<string, unknown>).id === "string"
      ? String((outcome.data as Record<string, unknown>).id)
      : null;
  dispatch(
    {
      type: "feedbackAdded",
      turnIndex: turn.turnIndex,
      item: {
        id: createdId,
        feedbackType: "wandb.agent_user_feedback",
        note: null,
        emoji,
        traceId,
        spanId: turn.spanId,
        createdAt: new Date().toISOString(),
        createdBy: null,
      },
    },
    false,
  );
}

async function submitNote(turn: NormalizedTurn, rawNote: string): Promise<void> {
  const note = rawNote.trim();
  const payload = state.payload;
  if (!note || !payload || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = turn.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "Turn is missing a trace_id for feedback." }, false);
    return;
  }
  dispatch({ type: "actionStarted" }, false);
  const outcome = await bridge.invokeJsonTool("manage_weave_agent_feedback_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "add_note",
    conversation_id: payload.conversationId ?? undefined,
    trace_id: traceId,
    note,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message }, false);
    return;
  }
  const createdId =
    outcome.data && typeof outcome.data === "object" && typeof (outcome.data as Record<string, unknown>).id === "string"
      ? String((outcome.data as Record<string, unknown>).id)
      : null;
  dispatch(
    {
      type: "feedbackAdded",
      turnIndex: turn.turnIndex,
      item: {
        id: createdId,
        feedbackType: "wandb.note.1",
        note,
        emoji: null,
        traceId,
        spanId: turn.spanId,
        createdAt: new Date().toISOString(),
        createdBy: null,
      },
    },
    true,
    true,
  );
}

async function openTagPopoverAndLoadCatalog(): Promise<void> {
  const nextOpen = !state.tagPopoverOpen;
  dispatch({ type: "toggleTagPopover", open: nextOpen }, false);
  const payload = state.payload;
  if (!nextOpen || !payload || !payload.agentLensAvailable || state.tagCatalog.length > 0) {
    return;
  }
  const outcome = await bridge.invokeJsonTool("list_agent_lens_tags_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
  });
  if (outcome.ok) {
    dispatch({ type: "tagCatalogLoaded", catalog: parseTagCatalog(outcome.data) }, false);
  }
}

async function assignConversationTag(opts: { tagId?: string; slug?: string }): Promise<void> {
  const payload = state.payload;
  if (!payload || !payload.conversationId || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = state.selectedTarget?.traceId ?? payload.turns[0]?.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "A turn trace_id is required to assign an Agent Lens tag." }, false);
    return;
  }
  const slug = opts.slug?.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (!opts.tagId && !slug) {
    dispatch({ type: "banner", kind: "warning", message: "Enter a kebab-case tag slug (e.g. needs-eval)." }, false);
    return;
  }
  dispatch({ type: "actionStarted" }, false);
  const args: Record<string, unknown> = {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "assign_tag",
    conversation_id: payload.conversationId,
    trace_id: traceId,
  };
  if (opts.tagId) {
    args.tag_id = opts.tagId;
  } else {
    args.name = slug;
  }
  const outcome = await bridge.invokeJsonTool("manage_agent_lens_tags_tool", args);
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message }, false);
    return;
  }
  const catalogMatch = opts.tagId ? state.tagCatalog.find((c) => c.id === opts.tagId) : null;
  dispatch(
    {
      type: "tagAdded",
      tag: {
        tagId: opts.tagId ?? null,
        name: catalogMatch?.name ?? slug ?? "tagged",
        color: catalogMatch?.color ?? null,
        traceId,
        source: "human",
      },
    },
    true,
    true,
  );
}

async function removeConversationTag(tagId: string, traceId: string | null): Promise<void> {
  const payload = state.payload;
  const resolvedTrace = traceId ?? state.selectedTarget?.traceId ?? payload?.turns[0]?.traceId ?? payload?.traceId;
  if (!payload || !payload.conversationId || !resolvedTrace) {
    return;
  }
  dispatch({ type: "actionStarted" }, false);
  const outcome = await bridge.invokeJsonTool("manage_agent_lens_tags_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "remove_tag",
    conversation_id: payload.conversationId,
    trace_id: resolvedTrace,
    tag_id: tagId,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message }, false);
    return;
  }
  dispatch({ type: "tagRemoved", tagId }, false);
}

function renderFailureReasonWithLinks(finding: AgentFailureFinding): HTMLElement {
  const p = document.createElement("p");
  p.className = "ac-failure-reason";
  const segments = parseFailureReasonSegments(finding.failureReason);
  for (const seg of segments) {
    if (seg.kind === "text") {
      p.appendChild(document.createTextNode(seg.text));
    } else {
      seg.spanIds.forEach((spanId, idx) => {
        if (idx > 0) {
          p.appendChild(document.createTextNode(", "));
        }
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ac-inline-span-link";
        btn.textContent = spanId.slice(0, 16);
        btn.title = `Focus span ${spanId}`;
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          focusFailureAndEvidenceSpan(finding, spanId);
        });
        p.appendChild(btn);
      });
    }
  }
  return p;
}

function renderFailuresPanel(): void {
  failuresPanelEl.replaceChildren();
  if (state.agentFailures.length === 0 || !state.failuresPanelOpen) {
    failuresPanelEl.hidden = true;
    return;
  }
  failuresPanelEl.hidden = false;

  const activeFinding = getActiveFailureFinding(state);
  const pinnedFailureId = state.payload?.selectedFailure?.failureId ?? null;

  for (const finding of state.agentFailures) {
    const row = document.createElement("div");
    row.className = "ac-failure-row";
    const isActive = activeFinding?.failureId === finding.failureId;
    row.dataset.active = isActive ? "true" : "false";
    row.addEventListener("click", () => {
      focusFailureAndEvidenceSpan(finding);
    });

    const head = document.createElement("div");
    head.className = "ac-failure-head";

    const badges = document.createElement("div");
    badges.className = "ac-turn-badges";

    const matchedTurn = state.payload?.turns.find((t) => sameTraceId(t.traceId, finding.traceId));
    const turnChip = document.createElement("span");
    turnChip.className = "ac-turn-num";
    turnChip.textContent = matchedTurn ? `Turn #${matchedTurn.turnIndex}` : finding.traceId.slice(0, 10);
    badges.appendChild(turnChip);

    const sevLabel = formatFailureSeverity(finding.failureSeverity);
    if (sevLabel && finding.failureSeverity) {
      const sev = document.createElement("span");
      sev.className = "ac-sev";
      sev.dataset.sev = finding.failureSeverity;
      sev.textContent = sevLabel;
      badges.appendChild(sev);
    }

    const clusterOrCat = document.createElement("span");
    clusterOrCat.className = "ac-failure-cluster";
    clusterOrCat.textContent = finding.clusterTitle ?? finding.categoryLabel;
    badges.appendChild(clusterOrCat);

    if (pinnedFailureId === finding.failureId) {
      const pinChip = document.createElement("span");
      pinChip.className = "ac-subtle-badge";
      pinChip.textContent = "Pinned";
      badges.appendChild(pinChip);
    }

    head.appendChild(badges);

    if (finding.evidenceSpanIds.length > 0 && state.payload?.conversationId) {
      const jumpBtn = document.createElement("button");
      jumpBtn.type = "button";
      jumpBtn.className = "ac-mini-btn ac-focus-evidence-btn";
      jumpBtn.textContent = `Focus evidence (${finding.evidenceSpanIds.length}) →`;
      jumpBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        focusFailureAndEvidenceSpan(finding, finding.evidenceSpanIds[0]);
      });
      head.appendChild(jumpBtn);
    }

    row.append(head, renderFailureReasonWithLinks(finding));
    failuresPanelEl.appendChild(row);
  }
}

function createToolRow(
  turn: NormalizedTurn,
  toolName: string,
  durationMs: number | null,
  argsText: string | null,
  resultText: string | null,
  errorText: string | null,
  spanId: string | null,
): HTMLElement {
  const details = document.createElement("details");
  details.className = "ac-tool-row";
  details.addEventListener("click", (ev) => {
    ev.stopPropagation();
    dispatch(
      { type: "selectTurn", turnIndex: turn.turnIndex, toolName, spanId },
      true,
      true,
    );
  });

  const summary = document.createElement("summary");
  summary.className = "ac-tool-summary";

  const left = document.createElement("span");
  left.className = "ac-tool-left";
  const iconWrap = document.createElement("span");
  iconWrap.className = "ac-node-icon";
  iconWrap.appendChild(makeIconSvg("tool"));
  const name = document.createElement("span");
  name.className = "ac-tool-name";
  name.textContent = `tools/call ${toolName}`;
  left.append(iconWrap, name);

  const right = document.createElement("span");
  right.className = "ac-tool-right";
  if (errorText) {
    const errDot = document.createElement("span");
    errDot.className = "ac-error-badge";
    errDot.textContent = "Error";
    right.appendChild(errDot);
  }
  if (durationMs !== null) {
    const d = document.createElement("span");
    d.className = "ac-mono-dim";
    d.textContent = formatDuration(durationMs);
    right.appendChild(d);
  }

  summary.append(left, right);
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "ac-tool-payload";
  if (argsText) {
    const pre = document.createElement("pre");
    pre.className = "ac-code-pre";
    pre.textContent = argsText;
    body.appendChild(pre);
  }
  if (resultText) {
    const pre = document.createElement("pre");
    pre.className = "ac-code-pre";
    pre.textContent = resultText;
    body.appendChild(pre);
  }
  if (errorText) {
    const errEl = document.createElement("div");
    errEl.className = "ac-error-callout";
    errEl.textContent = errorText;
    body.appendChild(errEl);
  }
  details.appendChild(body);
  return details;
}

function renderMessageEl(turn: NormalizedTurn, msg: NormalizedMessage): HTMLElement {
  if (msg.role === "tool" || (msg.toolName && !msg.content)) {
    return createToolRow(
      turn,
      msg.toolName ?? "tool",
      msg.durationMs,
      msg.arguments,
      msg.toolResult,
      msg.error,
      msg.spanId,
    );
  }

  const wrap = document.createElement("div");
  wrap.className = "ac-msg";
  wrap.dataset.role = msg.role;

  if (msg.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "ac-user-bubble";
    bubble.textContent = msg.content ?? "";
    wrap.appendChild(bubble);
    return wrap;
  }

  const roleHeader = document.createElement("div");
  roleHeader.className = "ac-agent-head";
  const agentName = document.createElement("span");
  agentName.className = "ac-agent-name";
  agentName.textContent = turn.agentName ?? (msg.role === "assistant" ? "Agent" : msg.role);
  roleHeader.appendChild(agentName);
  if (msg.model) {
    const m = document.createElement("span");
    m.className = "ac-mono-dim";
    m.textContent = msg.model;
    roleHeader.appendChild(m);
  }
  wrap.appendChild(roleHeader);

  for (const tc of msg.toolCalls) {
    wrap.appendChild(
      createToolRow(turn, tc.name, null, tc.arguments, tc.result, tc.error, tc.spanId),
    );
  }

  if (msg.arguments && msg.toolName) {
    wrap.appendChild(
      createToolRow(turn, msg.toolName, msg.durationMs, msg.arguments, msg.toolResult, msg.error, msg.spanId),
    );
  }

  if (msg.content) {
    const p = document.createElement("p");
    p.className = "ac-msg-body";
    p.textContent = msg.content;
    wrap.appendChild(p);
  }

  if (msg.error) {
    const errDiv = document.createElement("div");
    errDiv.className = "ac-error-callout";
    errDiv.textContent = msg.error;
    wrap.appendChild(errDiv);
  }

  return wrap;
}

function renderTurnCard(turn: NormalizedTurn, writable: boolean): HTMLElement {
  const card = document.createElement("article");
  card.className = "ac-turn-card";
  const isSelected =
    state.selectedTarget?.kind === "turn" && state.selectedTarget.turnIndex === turn.turnIndex;
  const turnFailures = turn.traceId
    ? state.agentFailures.filter((f) => sameTraceId(f.traceId, turn.traceId))
    : [];

  card.dataset.selected = isSelected ? "true" : "false";
  card.dataset.error = turn.isError || turnFailures.length > 0 ? "true" : "false";
  card.tabIndex = 0;
  card.addEventListener("click", () => {
    dispatch({ type: "selectTurn", turnIndex: turn.turnIndex }, true, true);
  });
  card.addEventListener("keydown", (ev) => {
    if (ev.target !== ev.currentTarget) {
      return;
    }
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      dispatch({ type: "selectTurn", turnIndex: turn.turnIndex }, true, true);
    }
  });

  const sepBar = document.createElement("div");
  sepBar.className = "ac-turn-separator";

  const leftBadges = document.createElement("div");
  leftBadges.className = "ac-turn-badges";

  const turnIcon = document.createElement("span");
  turnIcon.className = "ac-node-icon";
  turnIcon.appendChild(makeIconSvg("turn"));
  leftBadges.appendChild(turnIcon);

  const numEl = document.createElement("span");
  numEl.className = "ac-turn-num";
  numEl.textContent = `Turn ${turn.turnIndex}`;
  leftBadges.appendChild(numEl);

  const tsText = formatTurnTimestamp(turn.startedAt);
  if (tsText) {
    const tsEl = document.createElement("span");
    tsEl.className = "ac-mono-dim ac-turn-ts";
    tsEl.textContent = tsText;
    leftBadges.appendChild(tsEl);
  }

  if (turn.durationMs !== null) {
    const durEl = document.createElement("span");
    durEl.className = "ac-mono-dim";
    durEl.textContent = formatDuration(turn.durationMs);
    leftBadges.appendChild(durEl);
  }

  if (turn.costUsd !== null && turn.costUsd > 0) {
    const costEl = document.createElement("span");
    costEl.className = "ac-mono-dim";
    costEl.textContent = `$${turn.costUsd.toFixed(4)}`;
    leftBadges.appendChild(costEl);
  }

  if (turn.isError) {
    const errBadge = document.createElement("span");
    errBadge.className = "ac-error-badge";
    errBadge.textContent = "Error";
    leftBadges.appendChild(errBadge);
  }

  for (const tf of turnFailures) {
    const fBadge = document.createElement("span");
    fBadge.className = "ac-sev";
    fBadge.dataset.sev = tf.failureSeverity ?? "minor";
    const sev = formatFailureSeverity(tf.failureSeverity);
    fBadge.textContent = sev ? `${tf.categoryLabel} · ${sev}` : tf.categoryLabel;
    leftBadges.appendChild(fBadge);
  }

  const rightActions = document.createElement("div");
  rightActions.className = "ac-triage-actions";
  rightActions.addEventListener("click", (ev) => ev.stopPropagation());

  if (writable) {
    const upBtn = document.createElement("button");
    upBtn.type = "button";
    upBtn.className = "ac-mini-btn ac-react-up";
    upBtn.textContent = "👍";
    upBtn.disabled = state.isSubmittingAction;
    upBtn.ariaLabel = `Thumbs up Turn ${turn.turnIndex}`;
    upBtn.addEventListener("click", () => void submitReaction(turn, "👍"));

    const downBtn = document.createElement("button");
    downBtn.type = "button";
    downBtn.className = "ac-mini-btn ac-react-down";
    downBtn.textContent = "👎";
    downBtn.disabled = state.isSubmittingAction;
    downBtn.ariaLabel = `Thumbs down Turn ${turn.turnIndex}`;
    downBtn.addEventListener("click", () => void submitReaction(turn, "👎"));

    const noteBtn = document.createElement("button");
    noteBtn.type = "button";
    noteBtn.className = "ac-mini-btn ac-add-note-btn";
    noteBtn.textContent = "Note";
    noteBtn.disabled = state.isSubmittingAction;
    noteBtn.addEventListener("click", () => {
      const next = state.noteComposerTurnIndex === turn.turnIndex ? null : turn.turnIndex;
      dispatch({ type: "openNoteComposer", turnIndex: next }, false);
    });

    rightActions.append(upBtn, downBtn, noteBtn);
  }

  if (turn.traceId) {
    const copyTraceBtn = document.createElement("button");
    copyTraceBtn.type = "button";
    copyTraceBtn.className = "ac-mini-btn ac-mono-dim";
    copyTraceBtn.textContent = turn.traceId.slice(0, 8);
    copyTraceBtn.title = `Copy trace_id ${turn.traceId}`;
    copyTraceBtn.addEventListener("click", () => void copyText(turn.traceId!, "trace_id"));
    rightActions.appendChild(copyTraceBtn);
  }

  sepBar.append(leftBadges, rightActions);
  card.appendChild(sepBar);

  const msgsContainer = document.createElement("div");
  msgsContainer.className = "ac-messages";
  for (const msg of turn.messages) {
    msgsContainer.appendChild(renderMessageEl(turn, msg));
  }
  card.appendChild(msgsContainer);

  if (turn.errorMessage) {
    const errBox = document.createElement("div");
    errBox.className = "ac-error-callout";
    errBox.textContent = turn.errorMessage;
    card.appendChild(errBox);
  }

  if (turn.feedback.length > 0) {
    const fbList = document.createElement("div");
    fbList.className = "ac-feedback-list";
    for (const fb of turn.feedback) {
      const pill = document.createElement("span");
      pill.className = "ac-fb-pill";
      pill.textContent = fb.emoji ? fb.emoji : `Note: ${fb.note ?? ""}`;
      fbList.appendChild(pill);
    }
    card.appendChild(fbList);
  }

  if (writable && state.noteComposerTurnIndex === turn.turnIndex) {
    const composer = document.createElement("div");
    composer.className = "ac-note-composer";
    composer.addEventListener("click", (ev) => ev.stopPropagation());
    const input = document.createElement("input");
    input.type = "text";
    input.className = "ac-input ac-note-input";
    input.placeholder = "Add note or root-cause correction on this turn…";
    input.maxLength = 1000;

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "ac-btn ac-save-note-btn";
    saveBtn.textContent = "Save note";

    saveBtn.addEventListener("click", () => void submitNote(turn, input.value));
    input.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Enter") {
        ev.preventDefault();
        void submitNote(turn, input.value);
      }
    });
    composer.append(input, saveBtn);
    card.appendChild(composer);
  }

  return card;
}

function renderSpanDrawer(span: SpanWaterfallNode | null): void {
  spanDrawerEl.replaceChildren();
  if (!span) {
    spanDrawerEl.hidden = true;
    return;
  }
  spanDrawerEl.hidden = false;

  const head = document.createElement("div");
  head.className = "ac-drawer-head";
  const title = document.createElement("h2");
  title.className = "ac-drawer-title";
  title.textContent = span.name;

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "ac-mini-btn";
  closeBtn.textContent = "✕";
  closeBtn.ariaLabel = "Close span details";
  closeBtn.addEventListener("click", () => dispatch({ type: "selectSpan", spanId: null }, false));
  head.append(title, closeBtn);
  spanDrawerEl.appendChild(head);

  const meta = document.createElement("div");
  meta.className = "ac-drawer-meta";
  const items = [
    span.statusCode,
    formatDuration(span.durationMs),
    `+${Math.round(span.offsetMs)}ms`,
  ];
  if (span.toolName) {
    items.push(`tool: ${span.toolName}`);
  }
  if (span.model) {
    items.push(span.model);
  }
  if (span.inputTokens !== null || span.outputTokens !== null) {
    items.push(`${formatCompactCount(span.inputTokens)} in / ${formatCompactCount(span.outputTokens)} out`);
  }
  meta.textContent = items.join(" • ");
  spanDrawerEl.appendChild(meta);

  const citedFailures = state.agentFailures.filter((f) =>
    f.evidenceSpanIds.includes(span.spanId),
  );
  for (const cf of citedFailures) {
    const evBox = document.createElement("div");
    evBox.className = "ac-evidence-callout";
    evBox.textContent = `Evidence for ${cf.clusterTitle ?? cf.categoryLabel}: ${formatPlainFailureReason(cf.failureReason)}`;
    spanDrawerEl.appendChild(evBox);
  }

  if (span.errorPreview) {
    const errEl = document.createElement("div");
    errEl.className = "ac-error-callout";
    errEl.textContent = span.errorPreview;
    spanDrawerEl.appendChild(errEl);
  }

  if (span.attributesJson) {
    const pre = document.createElement("pre");
    pre.className = "ac-code-pre";
    pre.textContent = span.attributesJson;
    spanDrawerEl.appendChild(pre);
  }
}

function formatSpanTreeLabel(span: SpanWaterfallNode): string {
  if (span.kind === "llm") {
    return span.model ? `chat ${span.model}` : `chat ${span.name}`;
  }
  if (span.kind === "tool") {
    return span.toolName ? `tools/call ${span.toolName}` : `tools/call ${span.name}`;
  }
  return span.name;
}

function renderSpansView(): void {
  spansListEl.replaceChildren();
  if (state.spansStatus === "loading") {
    const loading = document.createElement("div");
    loading.className = "ac-empty";
    loading.textContent = "Loading spans…";
    spansListEl.appendChild(loading);
    spanDrawerEl.hidden = true;
    return;
  }
  if (state.spans.length === 0) {
    const empty = document.createElement("div");
    empty.className = "ac-empty";
    empty.textContent = "No spans recorded for this conversation.";
    spansListEl.appendChild(empty);
    spanDrawerEl.hidden = true;
    return;
  }

  const evidenceSet = getEvidenceSpanIdsSet(state);

  for (const span of state.spans) {
    const isEvidence = evidenceSet.has(span.spanId);
    const row = document.createElement("div");
    row.className = "ac-wf-row";
    row.dataset.selected = state.selectedSpanId === span.spanId ? "true" : "false";
    row.dataset.evidence = isEvidence ? "true" : "false";
    row.dataset.depth = String(span.depth);
    row.tabIndex = 0;
    row.addEventListener("click", () => dispatch({ type: "selectSpan", spanId: span.spanId }, true, true));
    row.addEventListener("keydown", (ev) => {
      if (ev.target !== ev.currentTarget) {
        return;
      }
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        dispatch({ type: "selectSpan", spanId: span.spanId }, true, true);
      }
    });

    const left = document.createElement("div");
    left.className = "ac-wf-left";
    if (span.depth > 0) {
      left.style.paddingLeft = `${span.depth * 18}px`;
      const branch = document.createElement("span");
      branch.className = "ac-tree-branch";
      branch.setAttribute("aria-hidden", "true");
      left.appendChild(branch);
    }

    const iconBox = document.createElement("span");
    iconBox.className = "ac-node-icon";
    iconBox.appendChild(
      makeIconSvg(span.kind === "llm" ? "llm" : span.kind === "tool" ? "tool" : "turn"),
    );
    left.appendChild(iconBox);

    const nameSpan = document.createElement("span");
    nameSpan.className = "ac-wf-name";
    nameSpan.textContent = formatSpanTreeLabel(span);
    left.appendChild(nameSpan);

    const right = document.createElement("div");
    right.className = "ac-wf-right";

    if (isEvidence) {
      const evPill = document.createElement("span");
      evPill.className = "ac-evidence-pill";
      evPill.textContent = "Evidence";
      right.appendChild(evPill);
    }

    if (span.isError) {
      const errPill = document.createElement("span");
      errPill.className = "ac-error-badge";
      errPill.textContent = "Error";
      right.appendChild(errPill);
    }

    if (span.inputTokens !== null || span.outputTokens !== null) {
      const tok = (span.inputTokens ?? 0) + (span.outputTokens ?? 0);
      if (tok > 0) {
        const tokSpan = document.createElement("span");
        tokSpan.className = "ac-mono-dim";
        tokSpan.textContent = `${formatCompactCount(tok)} tok`;
        right.appendChild(tokSpan);
      }
    }

    const dur = document.createElement("span");
    dur.className = "ac-mono-dim ac-wf-dur";
    dur.textContent = formatDuration(span.durationMs);
    right.appendChild(dur);

    row.append(left, right);
    spansListEl.appendChild(row);
  }

  const selectedSpan = state.spans.find((s) => s.spanId === state.selectedSpanId) ?? null;
  renderSpanDrawer(selectedSpan);
}

function renderTagPopover(writable: boolean): void {
  tagPopoverEl.replaceChildren();
  const payload = state.payload;
  if (!writable || !payload || !payload.agentLensAvailable || !payload.conversationId || !state.tagPopoverOpen) {
    tagPopoverEl.hidden = true;
    return;
  }
  tagPopoverEl.hidden = false;

  if (state.tagCatalog.length > 0) {
    const catRow = document.createElement("div");
    catRow.className = "ac-popover-row";
    for (const entry of state.tagCatalog.slice(0, 12)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ac-mini-btn ac-catalog-tag-btn";
      btn.textContent = `+ ${entry.name}`;
      btn.addEventListener("click", () => void assignConversationTag({ tagId: entry.id }));
      catRow.appendChild(btn);
    }
    tagPopoverEl.appendChild(catRow);
  }

  const form = document.createElement("div");
  form.className = "ac-popover-row";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "ac-input";
  input.id = "ac-tag-slug-input";
  input.placeholder = "tag-slug (e.g. needs-eval)";
  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "ac-btn ac-assign-slug-btn";
  submit.textContent = "Assign tag";
  submit.addEventListener("click", () => void assignConversationTag({ slug: input.value }));
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      void assignConversationTag({ slug: input.value });
    }
  });
  form.append(input, submit);
  tagPopoverEl.appendChild(form);
}

function render(): void {
  rootEl.dataset.displayMode = state.displayMode;

  if (state.banner) {
    bannerEl.hidden = false;
    bannerEl.dataset.kind = state.banner.kind;
    bannerTextEl.textContent = state.banner.message;
  } else {
    bannerEl.hidden = true;
  }

  const canFullscreen = state.availableDisplayModes.includes("fullscreen");
  expandBtn.hidden = !canFullscreen || !state.payload;
  expandBtn.textContent = state.displayMode === "fullscreen" ? "Collapse" : "Expand";

  if (state.status === "error") {
    toolbarEl.hidden = true;
    failuresPanelEl.hidden = true;
    convViewEl.hidden = true;
    spansViewEl.hidden = true;
    copyConvBtn.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = state.fatalError ?? "Unable to load conversation.";
    titleEl.textContent = "Unavailable";
    return;
  }

  if (!state.payload) {
    toolbarEl.hidden = true;
    failuresPanelEl.hidden = true;
    convViewEl.hidden = true;
    spansViewEl.hidden = true;
    copyConvBtn.hidden = true;
    emptyEl.hidden = false;
    titleEl.textContent = state.pendingTitle ?? "Loading conversation…";
    emptyEl.textContent = "Waiting for conversation…";
    return;
  }

  const payload = state.payload;
  const writable = canPerformWrites(state, bridge.canCallServerTools());

  titleEl.textContent =
    payload.agentName ?? `${payload.entityName}/${payload.projectName}`;

  if (payload.conversationId) {
    copyConvBtn.hidden = false;
    copyConvBtn.replaceChildren();
    const idTxt = document.createElement("span");
    idTxt.textContent = `conv:${payload.conversationId.slice(0, 18)}`;
    copyConvBtn.append(idTxt, makeIconSvg("copy"));
    copyConvBtn.title = `Copy conversation_id ${payload.conversationId}`;
    copyConvBtn.onclick = () => void copyText(payload.conversationId!, "conversation_id");
  } else {
    copyConvBtn.hidden = true;
  }

  if (payload.agentVersion) {
    versionPillEl.hidden = false;
    versionPillEl.textContent = payload.agentVersion;
  } else {
    versionPillEl.hidden = true;
  }

  const errorCount = Math.max(payload.summary.errorCount, state.agentFailures.length);
  if (errorCount > 0) {
    errorPillEl.hidden = false;
    errorPillEl.textContent =
      state.agentFailures.length > 0
        ? `${state.agentFailures.length} ${state.agentFailures.length === 1 ? "Failure" : "Failures"}`
        : `${payload.summary.errorCount} ${payload.summary.errorCount === 1 ? "Error" : "Errors"}`;
  } else {
    errorPillEl.hidden = true;
  }

  metaStripEl.replaceChildren();
  const metaParts = [
    `In: ${formatCompactCount(payload.summary.totalInputTokens)}`,
    `Out: ${formatCompactCount(payload.summary.totalOutputTokens)}`,
    `${payload.summary.turnCount}/${payload.summary.totalTurns} turns`,
    formatDuration(payload.summary.totalDurationMs),
    payload.summary.totalCostUsd !== null ? `$${payload.summary.totalCostUsd.toFixed(4)}` : null,
    payload.accessMode === "read-only" ? "read-only" : null,
  ].filter((x): x is string => Boolean(x));
  metaStripEl.textContent = metaParts.join(" • ");

  tagsRowEl.replaceChildren();
  if (state.tags.length > 0) {
    tagsRowEl.hidden = false;
    for (const tag of state.tags) {
      const chip = document.createElement("span");
      chip.className = "ac-tag-chip";
      const txt = document.createElement("span");
      txt.textContent = `#${tag.name}`;
      chip.appendChild(txt);
      if (writable && tag.tagId) {
        const rm = document.createElement("button");
        rm.type = "button";
        rm.className = "ac-tag-remove";
        rm.textContent = "✕";
        rm.ariaLabel = `Remove tag ${tag.name}`;
        rm.addEventListener("click", () => void removeConversationTag(tag.tagId!, tag.traceId));
        chip.appendChild(rm);
      }
      tagsRowEl.appendChild(chip);
    }
  } else {
    tagsRowEl.hidden = true;
  }

  openForgeBtn.hidden = false;
  toolbarEl.hidden = false;

  tabConvBtn.setAttribute("aria-selected", state.activeTab === "conversation" ? "true" : "false");
  tabSpansBtn.setAttribute("aria-selected", state.activeTab === "spans" ? "true" : "false");
  tabSpansBtn.hidden = !(payload.conversationId || payload.traceId);

  failuresBtn.hidden = state.agentFailures.length === 0;
  failuresBtn.textContent = `Agent failures (${state.agentFailures.length})`;
  failuresBtn.setAttribute("aria-expanded", state.failuresPanelOpen ? "true" : "false");

  errorsOnlyBtn.hidden = payload.summary.errorCount === 0 && state.agentFailures.length === 0;
  errorsOnlyBtn.setAttribute("aria-pressed", state.errorsOnly ? "true" : "false");

  tagMenuBtn.hidden = !(writable && payload.agentLensAvailable && payload.conversationId);
  tagMenuBtn.setAttribute("aria-expanded", state.tagPopoverOpen ? "true" : "false");

  renderFailuresPanel();
  renderTagPopover(writable);

  if (state.activeTab === "spans") {
    emptyEl.hidden = true;
    convViewEl.hidden = true;
    spansViewEl.hidden = false;
    renderSpansView();
    return;
  }

  spansViewEl.hidden = true;
  const visibleTurns = getVisibleTurns(state);
  if (visibleTurns.length === 0) {
    convViewEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = state.errorsOnly
      ? "No turns with errors match the filter."
      : "No turns recorded in this conversation.";
    return;
  }

  emptyEl.hidden = true;
  convViewEl.hidden = false;
  convViewEl.replaceChildren();
  for (const turn of visibleTurns) {
    convViewEl.appendChild(renderTurnCard(turn, writable));
  }
}

tabConvBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "conversation" }, true));
tabSpansBtn.addEventListener("click", () => {
  dispatch({ type: "setTab", tab: "spans" }, true);
  void ensureSpanWaterfallLoaded();
});
failuresBtn.addEventListener("click", () => dispatch({ type: "toggleFailuresPanel" }, false));
errorsOnlyBtn.addEventListener("click", () => dispatch({ type: "toggleErrorsOnly" }, false));
tagMenuBtn.addEventListener("click", () => void openTagPopoverAndLoadCatalog());
bannerDismissBtn.addEventListener("click", () => dispatch({ type: "dismissBanner" }, false));

expandBtn.addEventListener("click", async () => {
  const target = state.displayMode === "fullscreen" ? "inline" : "fullscreen";
  const next = await bridge.requestDisplayMode(target);
  if (next.ok) {
    dispatch({ type: "hostContextChanged", displayMode: next.mode }, false);
  } else {
    dispatch({ type: "banner", kind: "info", message: next.message }, false);
  }
});

openForgeBtn.addEventListener("click", async () => {
  const url = buildActiveConversationForgeLink(state);
  if (!url) {
    return;
  }
  const res = await bridge.openExternalUrl(url);
  if (!res.ok) {
    dispatch({ type: "banner", kind: "warning", message: res.message }, false);
  }
});

render();
void bridge.connect();
