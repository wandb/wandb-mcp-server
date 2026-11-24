import "./style.css";
import { InsightsHostBridge } from "./host";
import {
  enrichTurnsWithAttributions,
  parseEmbeddedConversation,
  parseExampleTurnsResponse,
  type ExampleTurnItem,
  type InsightsCategory,
  type InsightsCluster,
} from "./schema";
import {
  buildInsightsModelContext,
  createInitialInsightsState,
  reduceInsightsState,
  type InsightsAction,
  type InsightsSelectedContext,
  type InsightsState,
  type SignatureKind,
} from "./state";

const rootEl = document.getElementById("app-root") as HTMLElement;
const titleEl = document.getElementById("ali-title") as HTMLElement;
const agentPillEl = document.getElementById("ali-agent-pill") as HTMLElement;
const busyPillEl = document.getElementById("ali-busy-pill") as HTMLElement;
const kpiStripEl = document.getElementById("ali-kpi-strip") as HTMLElement;
const windowGroupEl = document.getElementById("ali-window-group") as HTMLElement;
const expandBtn = document.getElementById("ali-expand-btn") as HTMLButtonElement;
const openForgeBtn = document.getElementById("ali-open-forge-btn") as HTMLButtonElement;
const toolbarEl = document.getElementById("ali-toolbar") as HTMLElement;
const tabFailuresBtn = document.getElementById("ali-tab-failures") as HTMLButtonElement;
const tabIntentsBtn = document.getElementById("ali-tab-intents") as HTMLButtonElement;
const tabClustersBtn = document.getElementById("ali-tab-clusters") as HTMLButtonElement;
const clusterFilterGroupEl = document.getElementById("ali-cluster-filter-group") as HTMLElement;
const bannerEl = document.getElementById("ali-banner") as HTMLElement;
const bannerTextEl = document.getElementById("ali-banner-text") as HTMLElement;
const bannerDismissBtn = document.getElementById("ali-banner-dismiss") as HTMLButtonElement;
const emptyEl = document.getElementById("ali-empty") as HTMLElement;
const workspaceEl = document.getElementById("ali-workspace") as HTMLElement;
const barsListEl = document.getElementById("ali-bars-list") as HTMLElement;
const convDrawerEl = document.getElementById("ali-conv-drawer") as HTMLElement;

let state: InsightsState = createInitialInsightsState();

function formatPct(share: number): string {
  const pct = Math.max(0, Math.min(100, share * 100));
  return pct >= 10 ? `${pct.toFixed(0)}%` : `${pct.toFixed(1)}%`;
}

function formatDuration(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) {
    return "";
  }
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

const bridge = new InsightsHostBridge({
  onToolInput(title) {
    dispatch({ type: "toolInput", title });
  },
  onPayloadLoaded(payload) {
    dispatch({ type: "loaded", payload }, true);
  },
  onToolFailed(message) {
    dispatch({ type: "loadFailed", message });
  },
  onHostContext(displayMode, availableDisplayModes) {
    dispatch({ type: "hostContextChanged", displayMode, availableDisplayModes });
  },
  onTeardown() {
    state = createInitialInsightsState();
    render();
  },
});

function dispatch(action: InsightsAction, immediateContext = false): void {
  state = reduceInsightsState(state, action);
  render();
  bridge.scheduleModelContextSync(buildInsightsModelContext(state), immediateContext);
}

async function switchWindowPreset(days: 1 | 7 | 14 | 30): Promise<void> {
  const payload = state.payload;
  if (!payload || state.isWindowUpdating) {
    return;
  }
  dispatch({ type: "windowUpdateStarted" });
  const args: Record<string, unknown> = {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    window_days: days,
  };
  if (payload.window.agentName) {
    args.agent_name = payload.window.agentName;
  }
  const outcome = await bridge.callInsightsOverview(args);
  if (outcome.ok) {
    dispatch({ type: "loaded", payload: outcome.payload, preserveTab: true }, true);
  } else {
    dispatch({ type: "loadFailed", message: outcome.message });
  }
}

async function expandAndLoadExamples(opts: {
  rowKey: string;
  signatureType: SignatureKind;
  categoryId: string | null;
  topicId: string | null;
  summary: string;
}): Promise<void> {
  const payload = state.payload;
  if (!payload) {
    return;
  }
  const ctx: InsightsSelectedContext = {
    signatureType: opts.signatureType,
    categoryId: opts.categoryId,
    topicId: opts.topicId,
    conversationId: null,
    traceId: null,
    failureSummary: opts.summary,
  };
  const wasExpanded = state.expandedRowKey === opts.rowKey;
  dispatch({ type: "toggleExpandRow", rowKey: opts.rowKey, context: ctx });
  if (wasExpanded || state.exampleTurnsByKey[opts.rowKey]?.status === "ready") {
    return;
  }
  if (!opts.categoryId) {
    dispatch({
      type: "exampleTurnsFailed",
      rowKey: opts.rowKey,
      message: "Cluster does not list a parent category_id for example-turn drilldown.",
    });
    return;
  }
  if (!bridge.canCallServerTools()) {
    dispatch({
      type: "exampleTurnsFailed",
      rowKey: opts.rowKey,
      message: "This MCP host does not support interactive turn drilldowns.",
    });
    return;
  }

  dispatch({ type: "exampleTurnsLoading", rowKey: opts.rowKey });
  const args: Record<string, unknown> = {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    signature_type: opts.signatureType,
    category_id: opts.categoryId,
    start_at: payload.window.startAt,
    end_at: payload.window.endAt,
    limit: 8,
  };
  if (opts.topicId) {
    args.topic_ids = [opts.topicId];
  }
  if (payload.window.agentName) {
    args.filters = [{ field: "agent_name", values: [payload.window.agentName] }];
  }

  const turnsOutcome = await bridge.invokeJsonTool("list_agent_lens_category_example_turns_tool", args);
  if (!turnsOutcome.ok) {
    dispatch({ type: "exampleTurnsFailed", rowKey: opts.rowKey, message: turnsOutcome.message });
    return;
  }

  let items = parseExampleTurnsResponse(turnsOutcome.data);
  if (opts.signatureType === "failure" && items.length > 0) {
    const traceIds = items
      .map((t) => t.traceId)
      .filter((tid): tid is string => Boolean(tid))
      .slice(0, 10);
    if (traceIds.length > 0) {
      const attrOutcome = await bridge.invokeJsonTool("get_agent_lens_failure_attributions_tool", {
        entity_name: payload.entityName,
        project_name: payload.projectName,
        trace_ids: traceIds,
      });
      if (attrOutcome.ok) {
        items = enrichTurnsWithAttributions(items, attrOutcome.data);
      }
    }
  }

  dispatch({ type: "exampleTurnsLoaded", rowKey: opts.rowKey, items });
}

async function inspectTurnConversation(
  turn: ExampleTurnItem,
  signatureType: SignatureKind,
  categoryId: string | null,
  topicId: string | null,
): Promise<void> {
  const payload = state.payload;
  if (!payload || !bridge.canCallServerTools()) {
    return;
  }
  const ctx: InsightsSelectedContext = {
    signatureType,
    categoryId,
    topicId: turn.topicId ?? topicId,
    conversationId: turn.conversationId,
    traceId: turn.traceId,
    failureSummary: turn.attributionSummary ?? turn.failureReason ?? categoryId,
  };
  dispatch({ type: "drawerLoading", context: ctx });

  const outcome = turn.conversationId
    ? await bridge.invokeJsonTool("get_weave_agent_conversation_tool", {
        entity_name: payload.entityName,
        project_name: payload.projectName,
        conversation_id: turn.conversationId,
        include_feedback: true,
      })
    : turn.traceId
      ? await bridge.invokeJsonTool("get_weave_agent_trace_tool", {
          entity_name: payload.entityName,
          project_name: payload.projectName,
          trace_id: turn.traceId,
          include_feedback: true,
        })
      : { ok: false as const, message: "Example turn has neither conversation_id nor trace_id." };

  if (!outcome.ok) {
    dispatch({ type: "drawerFailed", message: outcome.message });
    return;
  }
  const parsed = parseEmbeddedConversation(outcome.data);
  if (!parsed) {
    dispatch({ type: "drawerFailed", message: "Unable to parse conversation trajectory." });
    return;
  }
  dispatch({ type: "drawerLoaded", data: parsed });
}

function createKpiItem(label: string, value: string): HTMLElement {
  const pill = document.createElement("span");
  pill.className = "ali-kpi";
  const lbl = document.createElement("span");
  lbl.textContent = `${label}:`;
  const strong = document.createElement("strong");
  strong.textContent = value;
  pill.append(lbl, strong);
  return pill;
}

function renderExamplesPanel(
  rowKey: string,
  signatureType: SignatureKind,
  categoryId: string | null,
  topicId: string | null,
): HTMLElement {
  const panel = document.createElement("div");
  panel.className = "ali-examples-panel";

  const entry = state.exampleTurnsByKey[rowKey];
  if (!entry || entry.status === "loading") {
    const msg = document.createElement("span");
    msg.className = "ali-key";
    msg.textContent = "Loading example turns…";
    panel.appendChild(msg);
    return panel;
  }
  if (entry.status === "error") {
    const err = document.createElement("span");
    err.className = "ali-key";
    err.textContent = entry.error ?? "Unable to load example turns.";
    panel.appendChild(err);
    return panel;
  }
  if (entry.items.length === 0) {
    const empty = document.createElement("span");
    empty.className = "ali-key";
    empty.textContent = "No matching example turns in this window.";
    panel.appendChild(empty);
    return panel;
  }

  for (const turn of entry.items) {
    const row = document.createElement("div");
    row.className = "ali-example-row";
    const isSelected =
      (turn.traceId && state.selectedContext?.traceId === turn.traceId) ||
      (turn.conversationId && state.selectedContext?.conversationId === turn.conversationId);
    row.dataset.selected = isSelected ? "true" : "false";

    row.addEventListener("click", () => {
      dispatch({
        type: "selectExampleTurn",
        context: {
          signatureType,
          categoryId,
          topicId: turn.topicId ?? topicId,
          conversationId: turn.conversationId,
          traceId: turn.traceId,
          failureSummary: turn.attributionSummary ?? turn.failureReason ?? categoryId,
        },
      });
    });

    const body = document.createElement("div");
    body.className = "ali-example-body";

    const reason = document.createElement("p");
    reason.className = "ali-example-reason";
    reason.textContent =
      turn.attributionSummary ??
      turn.failureReason ??
      (turn.conversationId ? `Conversation ${turn.conversationId}` : `Turn ${turn.traceId}`);
    body.appendChild(reason);

    const meta = document.createElement("div");
    meta.className = "ali-example-meta";
    if (turn.attribution) {
      const attrChip = document.createElement("span");
      attrChip.className = "ali-attr-chip";
      attrChip.textContent = turn.attribution;
      meta.appendChild(attrChip);
    }
    const metaBits = [
      turn.conversationId ? `conv:${turn.conversationId.slice(0, 10)}` : null,
      turn.traceId ? `trace:${turn.traceId.slice(0, 10)}` : null,
      formatDuration(turn.durationMs) || null,
      turn.totalTokens !== null ? `${turn.totalTokens} tok` : null,
    ].filter((x): x is string => Boolean(x));
    const metaSpan = document.createElement("span");
    metaSpan.textContent = metaBits.join(" · ");
    meta.appendChild(metaSpan);
    body.appendChild(meta);

    const inspectBtn = document.createElement("button");
    inspectBtn.type = "button";
    inspectBtn.className = "ali-inspect-btn";
    inspectBtn.textContent = "Inspect conversation";
    inspectBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      void inspectTurnConversation(turn, signatureType, categoryId, topicId);
    });

    row.append(body, inspectBtn);
    panel.appendChild(row);
  }

  return panel;
}

function renderCategoryBar(cat: InsightsCategory, kind: SignatureKind): HTMLElement {
  const rowKey = `cat:${kind}:${cat.categoryId}`;
  const expanded = state.expandedRowKey === rowKey;

  const card = document.createElement("div");
  card.className = "ali-bar-card";
  card.dataset.expanded = expanded ? "true" : "false";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "ali-bar-trigger";
  trigger.setAttribute("aria-expanded", expanded ? "true" : "false");
  trigger.addEventListener("click", () => {
    void expandAndLoadExamples({
      rowKey,
      signatureType: kind,
      categoryId: cat.categoryId,
      topicId: null,
      summary: `${cat.label} (${cat.count} turns, ${formatPct(cat.share)})`,
    });
  });

  const top = document.createElement("div");
  top.className = "ali-bar-top";

  const left = document.createElement("div");
  left.className = "ali-bar-title-group";
  const title = document.createElement("span");
  title.className = "ali-bar-title";
  title.textContent = cat.label;
  left.appendChild(title);

  if (cat.severity) {
    const sev = document.createElement("span");
    sev.className = "ali-sev";
    sev.dataset.sev = cat.severity;
    sev.textContent = cat.severity;
    left.appendChild(sev);
  }

  if (cat.clusterBreakdowns.length > 0) {
    const topicsChip = document.createElement("span");
    topicsChip.className = "ali-pill";
    topicsChip.textContent = `${cat.clusterBreakdowns.length} ${cat.clusterBreakdowns.length === 1 ? "cluster" : "clusters"}`;
    left.appendChild(topicsChip);
  }

  const metrics = document.createElement("span");
  metrics.className = "ali-bar-metrics";
  metrics.textContent = `${cat.count.toLocaleString("en-US")} turns · ${formatPct(cat.share)}`;
  top.append(left, metrics);

  const track = document.createElement("div");
  track.className = "ali-bar-track";
  const fill = document.createElement("div");
  fill.className = "ali-bar-fill";
  fill.dataset.kind = kind;
  fill.style.width = `${Math.max(2, Math.min(100, cat.share * 100))}%`;
  track.appendChild(fill);

  trigger.append(top, track);
  card.appendChild(trigger);

  if (expanded) {
    card.appendChild(renderExamplesPanel(rowKey, kind, cat.categoryId, null));
  }
  return card;
}

function renderClusterBar(cluster: InsightsCluster, kind: SignatureKind): HTMLElement {
  const rowKey = `cluster:${kind}:${cluster.id}`;
  const expanded = state.expandedRowKey === rowKey;

  const card = document.createElement("div");
  card.className = "ali-bar-card";
  card.dataset.expanded = expanded ? "true" : "false";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "ali-bar-trigger";
  trigger.setAttribute("aria-expanded", expanded ? "true" : "false");
  trigger.addEventListener("click", () => {
    void expandAndLoadExamples({
      rowKey,
      signatureType: kind,
      categoryId: cluster.categoryId,
      topicId: cluster.id,
      summary: `${cluster.title} (${formatPct(cluster.share)})`,
    });
  });

  const top = document.createElement("div");
  top.className = "ali-bar-top";

  const left = document.createElement("div");
  left.className = "ali-bar-title-group";
  const title = document.createElement("span");
  title.className = "ali-bar-title";
  title.textContent = cluster.title;
  left.appendChild(title);

  if (cluster.categoryId) {
    const catPill = document.createElement("span");
    catPill.className = "ali-pill";
    catPill.textContent = cluster.categoryId;
    left.appendChild(catPill);
  }
  if (cluster.severity) {
    const sev = document.createElement("span");
    sev.className = "ali-sev";
    sev.dataset.sev = cluster.severity;
    sev.textContent = cluster.severity;
    left.appendChild(sev);
  }
  if (cluster.attribution) {
    const attr = document.createElement("span");
    attr.className = "ali-attr-chip";
    attr.textContent = cluster.attribution;
    left.appendChild(attr);
  }

  const metrics = document.createElement("span");
  metrics.className = "ali-bar-metrics";
  const countPart = cluster.count !== null ? `${cluster.count.toLocaleString("en-US")} turns · ` : "";
  metrics.textContent = `${countPart}${formatPct(cluster.share)}`;
  top.append(left, metrics);

  const track = document.createElement("div");
  track.className = "ali-bar-track";
  const fill = document.createElement("div");
  fill.className = "ali-bar-fill";
  fill.dataset.kind = kind;
  fill.style.width = `${Math.max(2, Math.min(100, cluster.share * 100))}%`;
  track.appendChild(fill);

  trigger.append(top, track);
  if (cluster.summary) {
    const sum = document.createElement("p");
    sum.className = "ali-bar-summary";
    sum.textContent = cluster.summary;
    trigger.appendChild(sum);
  }
  card.appendChild(trigger);

  if (expanded) {
    card.appendChild(renderExamplesPanel(rowKey, kind, cluster.categoryId, cluster.id));
  }
  return card;
}

function renderDrawer(): void {
  convDrawerEl.replaceChildren();
  if (state.drawerStatus === "closed") {
    convDrawerEl.hidden = true;
    workspaceEl.dataset.drawerOpen = "false";
    return;
  }
  convDrawerEl.hidden = false;
  workspaceEl.dataset.drawerOpen = "true";

  const head = document.createElement("div");
  head.className = "ali-drawer-head";
  const title = document.createElement("h2");
  title.className = "ali-drawer-title";
  title.textContent =
    state.drawerConversation?.conversationId
      ? `Conversation ${state.drawerConversation.conversationId.slice(0, 14)}`
      : state.drawerConversation?.traceId
        ? `Turn ${state.drawerConversation.traceId.slice(0, 14)}`
        : "Conversation trajectory";

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "ali-btn";
  closeBtn.textContent = "Close ✕";
  closeBtn.addEventListener("click", () => dispatch({ type: "closeDrawer" }));
  head.append(title, closeBtn);
  convDrawerEl.appendChild(head);

  if (state.drawerStatus === "loading") {
    const loading = document.createElement("div");
    loading.className = "ali-empty";
    loading.textContent = "Loading full turn trajectory…";
    convDrawerEl.appendChild(loading);
    return;
  }
  if (state.drawerStatus === "error" || !state.drawerConversation) {
    const err = document.createElement("div");
    err.className = "ali-empty";
    err.textContent = state.drawerError ?? "Unable to load conversation.";
    convDrawerEl.appendChild(err);
    return;
  }

  for (const turn of state.drawerConversation.turns) {
    const box = document.createElement("div");
    box.className = "ali-drawer-turn";
    box.dataset.error = turn.isError ? "true" : "false";

    const tHead = document.createElement("div");
    tHead.className = "ali-example-meta";
    tHead.textContent = [
      `Turn #${turn.turnIndex}`,
      turn.isError ? "ERROR" : null,
      formatDuration(turn.durationMs) || null,
      turn.totalTokens !== null ? `${turn.totalTokens} tok` : null,
    ]
      .filter((x): x is string => Boolean(x))
      .join(" · ");
    box.appendChild(tHead);

    if (turn.errorMessage) {
      const errP = document.createElement("div");
      errP.className = "ali-drawer-msg";
      errP.textContent = `⚠ ${turn.errorMessage}`;
      box.appendChild(errP);
    }

    for (const msg of turn.messages) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "ali-drawer-msg";
      const roleSpan = document.createElement("span");
      roleSpan.className = "ali-drawer-role";
      roleSpan.textContent = msg.toolName
        ? `${msg.role.toUpperCase()} · ${msg.toolName}`
        : msg.role.toUpperCase();
      msgDiv.appendChild(roleSpan);

      if (msg.content) {
        const txt = document.createElement("div");
        txt.textContent = msg.content;
        msgDiv.appendChild(txt);
      }
      if (msg.arguments) {
        const pre = document.createElement("pre");
        pre.className = "ali-drawer-pre";
        pre.textContent = msg.arguments;
        msgDiv.appendChild(pre);
      }
      if (msg.toolResult) {
        const pre = document.createElement("pre");
        pre.className = "ali-drawer-pre";
        pre.textContent = msg.toolResult;
        msgDiv.appendChild(pre);
      }
      if (msg.error) {
        const pre = document.createElement("pre");
        pre.className = "ali-drawer-pre";
        pre.textContent = `Error: ${msg.error}`;
        msgDiv.appendChild(pre);
      }
      box.appendChild(msgDiv);
    }
    convDrawerEl.appendChild(box);
  }
}

function render(): void {
  rootEl.dataset.displayMode = state.displayMode;
  busyPillEl.hidden = !state.isWindowUpdating;

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
    workspaceEl.hidden = true;
    windowGroupEl.hidden = true;
    kpiStripEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = state.fatalError ?? "Unable to load Agent Lens insights.";
    titleEl.textContent = "Agent Lens insights unavailable";
    return;
  }

  if (!state.payload) {
    toolbarEl.hidden = true;
    workspaceEl.hidden = true;
    windowGroupEl.hidden = true;
    kpiStripEl.hidden = true;
    emptyEl.hidden = false;
    titleEl.textContent = state.pendingTitle ?? "Loading Agent Lens insights…";
    emptyEl.textContent = "Waiting for Agent Lens insights…";
    return;
  }

  const payload = state.payload;
  titleEl.textContent = `${payload.entityName}/${payload.projectName}`;
  if (payload.window.agentName) {
    agentPillEl.hidden = false;
    agentPillEl.textContent = `agent:${payload.window.agentName}`;
  } else {
    agentPillEl.hidden = true;
  }

  kpiStripEl.hidden = false;
  kpiStripEl.replaceChildren();
  kpiStripEl.appendChild(
    createKpiItem("Failures", `${payload.failure.turnCount.toLocaleString("en-US")} turns`),
  );
  kpiStripEl.appendChild(
    createKpiItem("Intents", `${payload.intent.turnCount.toLocaleString("en-US")} turns`),
  );
  if (payload.coverage.coveragePct !== null) {
    kpiStripEl.appendChild(createKpiItem("Classified", `${payload.coverage.coveragePct.toFixed(1)}%`));
  } else if (payload.coverage.latestWeek) {
    kpiStripEl.appendChild(
      createKpiItem(
        "Coverage",
        payload.coverage.firstWeek
          ? `${payload.coverage.firstWeek} → ${payload.coverage.latestWeek}`
          : payload.coverage.latestWeek,
      ),
    );
  }
  const latestClusterRun = payload.clusteringStatus.find((c) => c.completedAt);
  if (latestClusterRun?.completedAt) {
    kpiStripEl.appendChild(
      createKpiItem("Clustered", latestClusterRun.completedAt.slice(0, 16).replace("T", " ")),
    );
  }

  windowGroupEl.hidden = false;
  for (const btn of windowGroupEl.querySelectorAll<HTMLButtonElement>(".ali-preset-btn")) {
    const days = Number(btn.dataset.days);
    btn.dataset.active = payload.window.windowDays === days ? "true" : "false";
    btn.disabled = state.isWindowUpdating;
  }

  openForgeBtn.hidden = false;
  toolbarEl.hidden = false;

  tabFailuresBtn.textContent = `Failures (${payload.failure.categories.length})`;
  tabIntentsBtn.textContent = `Intents (${payload.intent.categories.length})`;
  const totalClusters = payload.failure.clusters.length + payload.intent.clusters.length;
  tabClustersBtn.textContent = `Topic Clusters (${totalClusters})`;

  tabFailuresBtn.setAttribute("aria-selected", state.activeTab === "failures" ? "true" : "false");
  tabIntentsBtn.setAttribute("aria-selected", state.activeTab === "intents" ? "true" : "false");
  tabClustersBtn.setAttribute("aria-selected", state.activeTab === "clusters" ? "true" : "false");

  clusterFilterGroupEl.hidden = state.activeTab !== "clusters";
  for (const chip of clusterFilterGroupEl.querySelectorAll<HTMLButtonElement>(".ali-filter-chip")) {
    chip.setAttribute("aria-pressed", chip.dataset.filter === state.clusterFilter ? "true" : "false");
  }

  barsListEl.replaceChildren();
  if (state.activeTab === "failures") {
    for (const cat of payload.failure.categories) {
      barsListEl.appendChild(renderCategoryBar(cat, "failure"));
    }
  } else if (state.activeTab === "intents") {
    for (const cat of payload.intent.categories) {
      barsListEl.appendChild(renderCategoryBar(cat, "intent"));
    }
  } else {
    const clusterList =
      state.clusterFilter === "failure" ? payload.failure.clusters : payload.intent.clusters;
    for (const cl of clusterList) {
      barsListEl.appendChild(renderClusterBar(cl, state.clusterFilter));
    }
  }

  if (barsListEl.childElementCount === 0) {
    emptyEl.hidden = false;
    emptyEl.textContent =
      state.activeTab === "failures"
        ? "Zero classified failure categories in this time window."
        : state.activeTab === "intents"
          ? "Zero classified intent categories in this time window."
          : `Zero ${state.clusterFilter} topic clusters in this time window.`;
    workspaceEl.hidden = state.drawerStatus === "closed";
  } else {
    emptyEl.hidden = true;
    workspaceEl.hidden = false;
  }

  renderDrawer();
}

for (const btn of windowGroupEl.querySelectorAll<HTMLButtonElement>(".ali-preset-btn")) {
  btn.addEventListener("click", () => {
    const d = Number(btn.dataset.days);
    if (d === 1 || d === 7 || d === 14 || d === 30) {
      void switchWindowPreset(d);
    }
  });
}

tabFailuresBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "failures" }));
tabIntentsBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "intents" }));
tabClustersBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "clusters" }));

for (const chip of clusterFilterGroupEl.querySelectorAll<HTMLButtonElement>(".ali-filter-chip")) {
  chip.addEventListener("click", () => {
    const filter = chip.dataset.filter === "intent" ? "intent" : "failure";
    dispatch({ type: "setClusterFilter", filter });
  });
}

bannerDismissBtn.addEventListener("click", () => dispatch({ type: "dismissBanner" }));

expandBtn.addEventListener("click", async () => {
  const target = state.displayMode === "fullscreen" ? "inline" : "fullscreen";
  const res = await bridge.requestDisplayMode(target);
  if (res.ok) {
    dispatch({ type: "hostContextChanged", displayMode: res.mode });
  } else {
    dispatch({ type: "banner", kind: "warning", message: res.message });
  }
});

openForgeBtn.addEventListener("click", async () => {
  if (!state.payload) {
    return;
  }
  const res = await bridge.openExternalUrl(state.payload.forgeUrl);
  if (!res.ok) {
    dispatch({ type: "banner", kind: "info", message: res.message });
  }
});

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && state.drawerStatus !== "closed") {
    dispatch({ type: "closeDrawer" });
  }
});

render();
void bridge.connect();
