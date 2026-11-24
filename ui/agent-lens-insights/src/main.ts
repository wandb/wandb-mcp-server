import "./style.css";
import {
  formatFailureSeverity,
  formatPlainFailureReason,
} from "../../shared/failure-cluster-markdown";
import { InsightsHostBridge } from "./host";
import {
  enrichTurnsWithAttributions,
  parseEmbeddedConversation,
  parseExampleTurnsResponse,
  type ClusterBreakdownItem,
  type ExampleTurnItem,
  type InsightsCategory,
} from "./schema";
import {
  buildActiveForgeLink,
  buildInsightsModelContext,
  buildSelectionMarkdownBrief,
  createInitialInsightsState,
  reduceInsightsState,
  type InsightsAction,
  type InsightsState,
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
const bannerEl = document.getElementById("ali-banner") as HTMLElement;
const bannerTextEl = document.getElementById("ali-banner-text") as HTMLElement;
const bannerDismissBtn = document.getElementById("ali-banner-dismiss") as HTMLButtonElement;
const emptyEl = document.getElementById("ali-empty") as HTMLElement;
const workspaceEl = document.getElementById("ali-workspace") as HTMLElement;
const topClustersSectionEl = document.getElementById("ali-top-clusters-section") as HTMLElement;
const topClustersEl = document.getElementById("ali-top-clusters") as HTMLElement;
const explorerHeadingEl = document.getElementById("ali-explorer-heading") as HTMLElement;
const paneGroupEl = document.getElementById("ali-pane-group") as HTMLElement;
const barsListEl = document.getElementById("ali-bars-list") as HTMLElement;
const detailPaneEl = document.getElementById("ali-detail-pane") as HTMLElement;
const convDrawerEl = document.getElementById("ali-conv-drawer") as HTMLElement;

let state: InsightsState = createInitialInsightsState();

function formatPct(share: number | null | undefined): string {
  if (share === null || share === undefined || !Number.isFinite(share)) {
    return "0%";
  }
  const pct = share * 100;
  if (pct > 0 && pct < 1) {
    return `${pct.toFixed(1)}%`;
  }
  return `${pct >= 10 ? Math.round(pct) : Number(pct.toFixed(1))}%`;
}

function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms <= 0) {
    return "0ms";
  }
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatUsd(cost: number | null | undefined): string {
  if (cost === null || cost === undefined || !Number.isFinite(cost) || cost <= 0) {
    return "$0.000";
  }
  return cost >= 0.01 ? `$${cost.toFixed(3)}` : `$${cost.toFixed(4)}`;
}

const bridge = new InsightsHostBridge({
  onToolInput(title) {
    dispatch({ type: "toolInput", title });
  },
  onPayloadLoaded(payload) {
    dispatch({ type: "loaded", payload });
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

function dispatch(action: InsightsAction, syncContext = false, immediateContext = false): void {
  state = reduceInsightsState(state, action);
  render();
  if (syncContext) {
    bridge.scheduleModelContextSync(buildInsightsModelContext(state), immediateContext);
  }
}

async function copyBriefToClipboard(): Promise<void> {
  const brief = buildSelectionMarkdownBrief(state);
  if (!brief) {
    return;
  }
  bridge.scheduleModelContextSync(buildInsightsModelContext(state), true);
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(brief);
      dispatch({
        type: "banner",
        kind: "info",
        message: "Copied markdown brief and synced selection to coding agent context.",
      });
      return;
    }
  } catch {
    // sandboxed iframe fallback
  }
  dispatch({
    type: "banner",
    kind: "info",
    message: "Synced failure cluster markdown brief to coding agent context.",
  });
}

async function switchWindowPreset(days: 1 | 7 | 30): Promise<void> {
  const payload = state.payload;
  if (!payload) {
    return;
  }
  if (!bridge.canCallServerTools()) {
    dispatch({
      type: "banner",
      kind: "warning",
      message: "This MCP host does not support live window switching.",
    });
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
  if (!outcome.ok) {
    dispatch({ type: "loadFailed", message: outcome.message });
    return;
  }
  dispatch({ type: "loaded", payload: outcome.payload });
}

async function ensureExamplesLoaded(opts: {
  examplesKey: string;
  signatureType: "failure" | "intent";
  categoryId: string;
  topicId: string | null;
}): Promise<void> {
  const payload = state.payload;
  if (!payload) {
    return;
  }
  const cached = state.examplesCache[opts.examplesKey];
  if (cached && (cached.status === "ready" || cached.status === "loading")) {
    return;
  }
  if (!bridge.canCallServerTools()) {
    dispatch({
      type: "examplesFailed",
      examplesKey: opts.examplesKey,
      message: "This MCP host does not support interactive example-turn queries.",
    });
    return;
  }

  dispatch({ type: "examplesLoading", examplesKey: opts.examplesKey });
  const args: Record<string, unknown> = {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    signature_type: opts.signatureType,
    categories: [opts.categoryId],
    start_at: payload.window.startAt,
    end_at: payload.window.endAt,
    limit: 6,
  };
  if (opts.topicId) {
    args.topic_ids = [opts.topicId];
  }
  if (payload.window.agentName) {
    args.agent_name = payload.window.agentName;
  }

  const outcome = await bridge.invokeJsonTool("list_agent_lens_category_example_turns_tool", args);
  if (!outcome.ok) {
    dispatch({
      type: "examplesFailed",
      examplesKey: opts.examplesKey,
      message: outcome.message,
    });
    return;
  }

  let items = parseExampleTurnsResponse(outcome.data);
  if (opts.signatureType === "failure" && items.length > 0) {
    const missingReasonTraceIds = items
      .filter((t) => !t.failureReason && t.traceId)
      .map((t) => t.traceId);
    if (missingReasonTraceIds.length > 0) {
      const attrOutcome = await bridge.invokeJsonTool("get_agent_lens_failure_attributions_tool", {
        entity_name: payload.entityName,
        project_name: payload.projectName,
        trace_ids: missingReasonTraceIds.slice(0, 10),
      });
      if (attrOutcome.ok) {
        items = enrichTurnsWithAttributions(items, attrOutcome.data);
      }
    }
  }

  dispatch({ type: "examplesLoaded", examplesKey: opts.examplesKey, items }, true);
}

async function openEmbeddedConversationDrawer(
  signatureType: "failure" | "intent",
  category: InsightsCategory,
  cluster: ClusterBreakdownItem | null,
  examplesKey: string,
  turn: ExampleTurnItem,
): Promise<void> {
  const payload = state.payload;
  if (!payload) {
    return;
  }
  dispatch(
    {
      type: "selectTurn",
      signatureType,
      category,
      cluster,
      examplesKey,
      turn,
    },
    true,
    true,
  );
  if (!bridge.canCallServerTools()) {
    dispatch({
      type: "drawerFailed",
      message: "This MCP host does not support inline conversation queries.",
    });
    return;
  }

  dispatch({ type: "drawerLoading" });
  const selectedFailure =
    turn.traceId && turn.failureId
      ? { trace_id: turn.traceId, failure_id: turn.failureId }
      : undefined;
  const outcome = turn.conversationId
    ? await bridge.invokeJsonTool("get_weave_agent_conversation_tool", {
        entity_name: payload.entityName,
        project_name: payload.projectName,
        conversation_id: turn.conversationId,
        limit: 10,
        include_feedback: true,
        include_model_tool_calls: true,
        ...(selectedFailure ? { selected_failure: selectedFailure } : {}),
      })
    : await bridge.invokeJsonTool("get_weave_agent_trace_tool", {
        entity_name: payload.entityName,
        project_name: payload.projectName,
        trace_id: turn.traceId,
        include_feedback: true,
        include_model_tool_calls: true,
        ...(selectedFailure ? { selected_failure: selectedFailure } : {}),
      });

  if (!outcome.ok) {
    dispatch({ type: "drawerFailed", message: outcome.message });
    return;
  }

  const parsed = parseEmbeddedConversation(outcome.data);
  if (!parsed) {
    dispatch({
      type: "drawerFailed",
      message: "Conversation returned no renderable turns.",
    });
    return;
  }
  dispatch({ type: "drawerLoaded", data: parsed });
}

function renderSeverityPill(severity: string | null): HTMLElement | null {
  const label = formatFailureSeverity(severity);
  if (!label || !severity) {
    return null;
  }
  const sev = document.createElement("span");
  sev.className = "ali-sev";
  sev.dataset.sev = severity;
  sev.textContent = label;
  return sev;
}

function renderImpactSummary(
  impact: {
    count: number;
    share: number;
    conversationCount: number | null;
    failedCount: number | null;
    frustratedCount: number | null;
    failureRate: number | null;
    frustrationRate: number | null;
    averageLatencyMs: number | null;
    averageCostPerTurnUsd: number | null;
  },
  kind: "failure" | "intent",
): HTMLElement {
  const grid = document.createElement("div");
  grid.className = "ali-impact-grid";

  const metrics: Array<[string, string]> = [
    ["Turns", impact.count.toLocaleString("en-US")],
    [
      "Conversations",
      impact.conversationCount !== null ? impact.conversationCount.toLocaleString("en-US") : "—",
    ],
    ["Share of turns", formatPct(impact.share)],
    [
      "Frustrated turns",
      impact.frustrationRate !== null
        ? formatPct(impact.frustrationRate)
        : impact.frustratedCount !== null && impact.count > 0
          ? formatPct(impact.frustratedCount / impact.count)
          : "0%",
    ],
  ];
  if (kind === "intent") {
    metrics.push([
      "Failed turns",
      impact.failureRate !== null
        ? formatPct(impact.failureRate)
        : impact.failedCount !== null && impact.count > 0
          ? formatPct(impact.failedCount / impact.count)
          : "0%",
    ]);
  }
  metrics.push(
    ["Avg. time / turn", formatDuration(impact.averageLatencyMs)],
    ["Avg. cost / turn", formatUsd(impact.averageCostPerTurnUsd)],
  );

  for (const [label, value] of metrics) {
    const cell = document.createElement("div");
    cell.className = "ali-impact-cell";
    const lEl = document.createElement("span");
    lEl.className = "ali-impact-label";
    lEl.textContent = label;
    const vEl = document.createElement("strong");
    vEl.className = "ali-impact-val";
    vEl.textContent = value;
    cell.append(lEl, vEl);
    grid.appendChild(cell);
  }
  return grid;
}

function renderExamplesPanel(
  examplesKey: string,
  kind: "failure" | "intent",
  category: InsightsCategory,
  cluster: ClusterBreakdownItem | null,
): HTMLElement {
  const panel = document.createElement("div");
  panel.className = "ali-examples-panel";

  const subHead = document.createElement("div");
  subHead.className = "ali-examples-subhead";
  subHead.textContent = "Example turns";
  panel.appendChild(subHead);

  const entry = state.examplesCache[examplesKey];
  if (!entry || entry.status === "loading") {
    const loading = document.createElement("div");
    loading.className = "ali-muted-note";
    loading.textContent = "Loading example turns…";
    panel.appendChild(loading);
    return panel;
  }

  if (entry.status === "error") {
    const errRow = document.createElement("div");
    errRow.className = "ali-row-between";
    const msg = document.createElement("span");
    msg.className = "ali-muted-note";
    msg.textContent = entry.errorMessage ?? "Failed to load example turns.";
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "ali-btn";
    retryBtn.textContent = "Retry";
    retryBtn.addEventListener("click", () => {
      void ensureExamplesLoaded({
        examplesKey,
        signatureType: kind,
        categoryId: category.categoryId,
        topicId: cluster?.id ?? null,
      });
    });
    errRow.append(msg, retryBtn);
    panel.appendChild(errRow);
    return panel;
  }

  if (entry.items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "ali-muted-note";
    empty.textContent = "No example turns returned for this selection.";
    panel.appendChild(empty);
    return panel;
  }

  for (const turn of entry.items) {
    const row = document.createElement("div");
    row.className = "ali-example-row";
    const isSelected = state.selectedContext?.traceId === turn.traceId;
    row.dataset.selected = isSelected ? "true" : "false";
    row.addEventListener("click", () => {
      dispatch(
        {
          type: "selectTurn",
          signatureType: kind,
          category,
          cluster,
          examplesKey,
          turn,
        },
        true,
        true,
      );
    });

    const body = document.createElement("div");
    body.className = "ali-example-body";

    const meta = document.createElement("div");
    meta.className = "ali-example-meta";

    const turnChip = document.createElement("span");
    turnChip.className = "ali-turn-num";
    turnChip.textContent = `Turn ${turn.turnNumber}`;
    meta.appendChild(turnChip);

    const sevEl = renderSeverityPill(turn.failureSeverity);
    if (sevEl) {
      meta.appendChild(sevEl);
    }

    if (turn.clusterTitle && !cluster) {
      const clChip = document.createElement("span");
      clChip.className = "ali-subtle-badge";
      clChip.textContent = turn.clusterTitle;
      meta.appendChild(clChip);
    }

    if (turn.failureEvidenceSpanIds.length > 0) {
      const evChip = document.createElement("span");
      evChip.className = "ali-subtle-badge";
      evChip.textContent = `${turn.failureEvidenceSpanIds.length} evidence ${turn.failureEvidenceSpanIds.length === 1 ? "span" : "spans"}`;
      meta.appendChild(evChip);
    }

    const convSpan = document.createElement("span");
    convSpan.className = "ali-mono-dim";
    convSpan.textContent = turn.conversationId.slice(0, 20);
    meta.appendChild(convSpan);

    const reason = document.createElement("p");
    reason.className = "ali-example-reason";
    reason.textContent =
      formatPlainFailureReason(turn.failureReason) ||
      `Example ${kind} turn in ${category.label}`;

    body.append(meta, reason);

    const inspectBtn = document.createElement("button");
    inspectBtn.type = "button";
    inspectBtn.className = "ali-inspect-btn";
    inspectBtn.textContent = "View details";
    inspectBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      void openEmbeddedConversationDrawer(kind, category, cluster, examplesKey, turn);
    });

    row.append(body, inspectBtn);
    panel.appendChild(row);
  }

  return panel;
}

function selectCluster(
  category: InsightsCategory,
  cluster: ClusterBreakdownItem,
  kind: "failure" | "intent",
): void {
  const categoryKey = `cat:${kind}:${category.categoryId}`;
  const examplesKey = `cluster:${kind}:${category.categoryId}:${cluster.id}`;
  if (state.expandedCategoryKey !== categoryKey) {
    state = reduceInsightsState(state, {
      type: "toggleCategory",
      categoryKey,
      signatureType: kind,
      category,
      directExamplesKey: null,
    });
  }
  const willExpand = state.expandedExamplesKey !== examplesKey;
  dispatch(
    {
      type: "toggleClusterExamples",
      examplesKey,
      signatureType: kind,
      category,
      cluster,
    },
    true,
    true,
  );
  if (willExpand) {
    void ensureExamplesLoaded({
      examplesKey,
      signatureType: kind,
      categoryId: category.categoryId,
      topicId: cluster.id,
    });
  }
}

function renderTopClustersStrip(categories: InsightsCategory[], kind: "failure" | "intent"): void {
  const flat: Array<{ category: InsightsCategory; cluster: ClusterBreakdownItem }> = [];
  for (const cat of categories) {
    for (const cl of cat.clusterBreakdowns) {
      flat.push({ category: cat, cluster: cl });
    }
  }
  flat.sort((a, b) => b.cluster.count - a.cluster.count);

  if (flat.length === 0) {
    topClustersSectionEl.hidden = true;
    topClustersEl.replaceChildren();
    return;
  }

  topClustersSectionEl.hidden = false;
  topClustersEl.replaceChildren();
  for (const { category, cluster } of flat.slice(0, 3)) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "ali-top-cluster-card";
    const isActive = state.selectedContext?.topicId === cluster.id;
    card.dataset.active = isActive ? "true" : "false";
    card.addEventListener("click", () => selectCluster(category, cluster, kind));

    const head = document.createElement("div");
    head.className = "ali-top-cluster-head";
    const title = document.createElement("span");
    title.className = "ali-top-cluster-title";
    title.textContent = cluster.title;
    head.appendChild(title);
    const sev = renderSeverityPill(cluster.severity);
    if (sev) {
      head.appendChild(sev);
    }

    const desc = document.createElement("p");
    desc.className = "ali-top-cluster-desc";
    desc.textContent = cluster.description ?? cluster.label ?? category.label;

    const foot = document.createElement("div");
    foot.className = "ali-top-cluster-foot";
    const catBadge = document.createElement("span");
    catBadge.className = "ali-subtle-badge";
    catBadge.textContent = category.label;
    const stats = document.createElement("span");
    stats.className = "ali-mono-dim";
    stats.textContent = `${cluster.count.toLocaleString("en-US")} traces · ${formatPct(cluster.share)}`;
    foot.append(catBadge, stats);

    card.append(head, desc, foot);
    topClustersEl.appendChild(card);
  }
}

function renderNavigatorCategory(cat: InsightsCategory, kind: "failure" | "intent"): HTMLElement {
  const categoryKey = `cat:${kind}:${cat.categoryId}`;
  const directExamplesKey = `cat-examples:${kind}:${cat.categoryId}`;
  const expanded = state.expandedCategoryKey === categoryKey;
  const hasClusters = cat.clusterBreakdowns.length > 0;
  const isCategorySelected =
    state.selectedContext?.categoryId === cat.categoryId && !state.selectedContext?.topicId;

  const group = document.createElement("div");
  group.className = "ali-bar-card";
  group.dataset.expanded = expanded ? "true" : "false";
  group.dataset.selected = isCategorySelected ? "true" : "false";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "ali-bar-trigger";
  trigger.setAttribute("aria-expanded", expanded ? "true" : "false");
  trigger.addEventListener("click", () => {
    const willExpand = state.expandedCategoryKey !== categoryKey;
    dispatch(
      {
        type: "toggleCategory",
        categoryKey,
        signatureType: kind,
        category: cat,
        directExamplesKey: hasClusters ? null : directExamplesKey,
      },
      true,
      true,
    );
    if (willExpand && !hasClusters) {
      void ensureExamplesLoaded({
        examplesKey: directExamplesKey,
        signatureType: kind,
        categoryId: cat.categoryId,
        topicId: null,
      });
    }
  });

  const chevron = document.createElement("span");
  chevron.className = "ali-chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = expanded ? "▾" : "▸";

  const label = document.createElement("span");
  label.className = "ali-bar-title";
  label.textContent = cat.label;

  const share = document.createElement("span");
  share.className = "ali-bar-metrics";
  share.textContent = `${formatPct(cat.share)} of turns`;

  trigger.append(chevron, label, share);
  group.appendChild(trigger);

  if (expanded && hasClusters) {
    const subList = document.createElement("div");
    subList.className = "ali-nav-clusters";
    for (const cl of cat.clusterBreakdowns) {
      const examplesKey = `cluster:${kind}:${cat.categoryId}:${cl.id}`;
      const isClusterSelected = state.expandedExamplesKey === examplesKey;
      const itemWrap = document.createElement("div");
      itemWrap.className = "ali-cluster-card";
      itemWrap.dataset.expanded = isClusterSelected ? "true" : "false";

      const clBtn = document.createElement("button");
      clBtn.type = "button";
      clBtn.className = "ali-cluster-trigger";
      clBtn.setAttribute("aria-expanded", isClusterSelected ? "true" : "false");
      clBtn.addEventListener("click", () => selectCluster(cat, cl, kind));

      const clTop = document.createElement("div");
      clTop.className = "ali-nav-cluster-row";
      const clTitle = document.createElement("span");
      clTitle.className = "ali-cluster-title";
      clTitle.textContent = cl.title;
      clTop.appendChild(clTitle);

      const clMeta = document.createElement("div");
      clMeta.className = "ali-nav-cluster-meta";
      const sev = renderSeverityPill(cl.severity);
      if (sev) {
        clMeta.appendChild(sev);
      }
      const pct = document.createElement("span");
      pct.className = "ali-mono-dim";
      pct.textContent = `${cl.count.toLocaleString("en-US")} · ${formatPct(cl.share)}`;
      clMeta.appendChild(pct);

      clBtn.append(clTop, clMeta);
      itemWrap.appendChild(clBtn);
      subList.appendChild(itemWrap);
    }
    group.appendChild(subList);
  }

  return group;
}

function renderDetailPane(categories: InsightsCategory[], kind: "failure" | "intent"): void {
  detailPaneEl.replaceChildren();
  if (categories.length === 0) {
    return;
  }

  const activeCategory =
    categories.find((c) => c.categoryId === state.selectedContext?.categoryId) ?? categories[0]!;
  const activeCluster = state.selectedContext?.topicId
    ? (activeCategory.clusterBreakdowns.find((cl) => cl.id === state.selectedContext?.topicId) ?? null)
    : null;

  const header = document.createElement("div");
  header.className = "ali-detail-header";

  const titleWrap = document.createElement("div");
  titleWrap.className = "ali-detail-title-wrap";

  const titleRow = document.createElement("div");
  titleRow.className = "ali-detail-title-row";
  const h3 = document.createElement("h3");
  h3.className = "ali-detail-title";
  h3.textContent = activeCluster ? activeCluster.title : activeCategory.label;
  titleRow.appendChild(h3);

  if (activeCluster) {
    const sev = renderSeverityPill(activeCluster.severity);
    if (sev) {
      titleRow.appendChild(sev);
    }
  }
  titleWrap.appendChild(titleRow);

  const subDesc = document.createElement("p");
  subDesc.className = "ali-detail-desc";
  if (activeCluster) {
    subDesc.textContent =
      activeCluster.description ??
      `${activeCategory.label} · ${activeCluster.count.toLocaleString("en-US")} turns (${formatPct(activeCluster.share)} of all turns)`;
  } else {
    subDesc.textContent = `${activeCategory.count.toLocaleString("en-US")} turns — ${formatPct(activeCategory.share)} of all turns${activeCategory.clusterBreakdowns.length > 0 ? ` across ${activeCategory.clusterBreakdowns.length} clusters` : ""}`;
  }
  titleWrap.appendChild(subDesc);

  const actions = document.createElement("div");
  actions.className = "ali-detail-actions";
  if (kind === "failure") {
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "ali-btn ali-copy-brief-btn";
    copyBtn.textContent = "Copy as markdown";
    copyBtn.addEventListener("click", () => void copyBriefToClipboard());
    actions.appendChild(copyBtn);
  }

  header.append(titleWrap, actions);
  detailPaneEl.appendChild(header);

  detailPaneEl.appendChild(renderImpactSummary(activeCluster ?? activeCategory, kind));

  if (activeCluster) {
    const examplesKey = `cluster:${kind}:${activeCategory.categoryId}:${activeCluster.id}`;
    detailPaneEl.appendChild(renderExamplesPanel(examplesKey, kind, activeCategory, activeCluster));
    return;
  }

  if (activeCategory.clusterBreakdowns.length > 0) {
    const clSection = document.createElement("div");
    clSection.className = "ali-detail-clusters";
    const subHead = document.createElement("div");
    subHead.className = "ali-examples-subhead";
    subHead.textContent = `Clusters in ${activeCategory.label} (${activeCategory.clusterBreakdowns.length})`;
    clSection.appendChild(subHead);

    for (const cl of activeCategory.clusterBreakdowns) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "ali-detail-cluster-row";
      row.addEventListener("click", () => selectCluster(activeCategory, cl, kind));

      const left = document.createElement("div");
      left.className = "ali-detail-cluster-main";
      const top = document.createElement("div");
      top.className = "ali-detail-title-row";
      const name = document.createElement("span");
      name.className = "ali-detail-cluster-name";
      name.textContent = cl.title;
      top.appendChild(name);
      const sev = renderSeverityPill(cl.severity);
      if (sev) {
        top.appendChild(sev);
      }
      left.appendChild(top);
      if (cl.description) {
        const d = document.createElement("p");
        d.className = "ali-detail-cluster-desc";
        d.textContent = cl.description;
        left.appendChild(d);
      }

      const right = document.createElement("div");
      right.className = "ali-detail-cluster-stats";
      right.textContent = `${cl.count.toLocaleString("en-US")} turns · ${formatPct(cl.share)}`;

      row.append(left, right);
      clSection.appendChild(row);
    }
    detailPaneEl.appendChild(clSection);
  } else {
    const directExamplesKey = `cat-examples:${kind}:${activeCategory.categoryId}`;
    detailPaneEl.appendChild(renderExamplesPanel(directExamplesKey, kind, activeCategory, null));
  }
}

function renderDrawer(): void {
  convDrawerEl.replaceChildren();
  if (state.drawerStatus === "closed") {
    convDrawerEl.hidden = true;
    paneGroupEl.dataset.drawerOpen = "false";
    return;
  }
  convDrawerEl.hidden = false;
  paneGroupEl.dataset.drawerOpen = "true";

  const head = document.createElement("div");
  head.className = "ali-drawer-head";
  const title = document.createElement("h2");
  title.className = "ali-drawer-title";
  title.textContent = state.drawerConversation?.conversationId
    ? state.drawerConversation.conversationId.slice(0, 22)
    : state.drawerConversation?.traceId
      ? state.drawerConversation.traceId.slice(0, 22)
      : "Conversation";

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "ali-btn";
  closeBtn.textContent = "Close";
  closeBtn.addEventListener("click", () => dispatch({ type: "closeDrawer" }));
  head.append(title, closeBtn);
  convDrawerEl.appendChild(head);

  if (state.drawerStatus === "loading") {
    const loading = document.createElement("div");
    loading.className = "ali-muted-note";
    loading.textContent = "Loading conversation…";
    convDrawerEl.appendChild(loading);
    return;
  }
  if (state.drawerStatus === "error" || !state.drawerConversation) {
    const err = document.createElement("div");
    err.className = "ali-muted-note";
    err.textContent = state.drawerError ?? "Unable to load conversation.";
    convDrawerEl.appendChild(err);
    return;
  }

  for (const turn of state.drawerConversation.turns) {
    const box = document.createElement("div");
    box.className = "ali-drawer-turn";
    box.dataset.error = turn.isError ? "true" : "false";

    const tHead = document.createElement("div");
    tHead.className = "ali-drawer-turn-head";
    tHead.textContent = [
      `Turn ${turn.turnIndex}`,
      turn.isError ? "Error" : null,
      turn.durationMs !== null ? formatDuration(turn.durationMs) : null,
      turn.totalTokens !== null ? `${turn.totalTokens.toLocaleString("en-US")} tok` : null,
    ]
      .filter((x): x is string => Boolean(x))
      .join(" • ");
    box.appendChild(tHead);

    if (turn.errorMessage) {
      const errP = document.createElement("div");
      errP.className = "ali-drawer-error";
      errP.textContent = turn.errorMessage;
      box.appendChild(errP);
    }

    for (const msg of turn.messages) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "ali-drawer-msg";
      const roleSpan = document.createElement("span");
      roleSpan.className = "ali-drawer-role";
      roleSpan.textContent = msg.toolName
        ? `Tool · ${msg.toolName}`
        : msg.role === "user"
          ? "User"
          : "Agent";
      msgDiv.appendChild(roleSpan);

      if (msg.content) {
        const txt = document.createElement("div");
        txt.className = "ali-drawer-prose";
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

function appendMetaItem(container: HTMLElement, label: string, value: string): void {
  if (container.childElementCount > 0) {
    const dot = document.createElement("span");
    dot.className = "ali-sep";
    dot.textContent = "•";
    container.appendChild(dot);
  }
  const item = document.createElement("span");
  item.className = "ali-meta-item";
  const l = document.createElement("span");
  l.className = "ali-meta-label";
  l.textContent = `${label}: `;
  const v = document.createElement("strong");
  v.className = "ali-meta-val";
  v.textContent = value;
  item.append(l, v);
  container.appendChild(item);
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
    titleEl.textContent = "Unavailable";
    return;
  }

  if (!state.payload) {
    toolbarEl.hidden = true;
    workspaceEl.hidden = true;
    windowGroupEl.hidden = true;
    kpiStripEl.hidden = true;
    emptyEl.hidden = false;
    titleEl.textContent = state.pendingTitle ?? "Loading…";
    emptyEl.textContent = "Waiting for Agent Lens insights…";
    return;
  }

  const payload = state.payload;
  titleEl.textContent = `${payload.entityName}/${payload.projectName}`;
  if (payload.window.agentName) {
    agentPillEl.hidden = false;
    agentPillEl.textContent = payload.window.agentName;
  } else {
    agentPillEl.hidden = true;
  }

  const totalTurns = payload.failure.turnCount || payload.intent.turnCount;
  const failingTurns =
    payload.failure.failingTurnCount ??
    payload.failure.categories.reduce((acc, c) => acc + c.count, 0);
  const failingShare = totalTurns > 0 ? failingTurns / totalTurns : 0;

  kpiStripEl.hidden = false;
  kpiStripEl.replaceChildren();
  appendMetaItem(kpiStripEl, "Turns", totalTurns.toLocaleString("en-US"));
  appendMetaItem(
    kpiStripEl,
    "Failing turns",
    `${failingTurns.toLocaleString("en-US")} (${formatPct(failingShare)})`,
  );
  const latestClusterRun =
    payload.clusteringStatus.find((c) => c.signatureType === state.activeTab && c.completedAt) ??
    payload.clusteringStatus.find((c) => c.completedAt);
  if (latestClusterRun?.completedAt) {
    appendMetaItem(
      kpiStripEl,
      "Last clustered",
      `${latestClusterRun.completedAt.slice(0, 16).replace("T", " ")} UTC`,
    );
  }
  if (payload.coverage.latestWeek) {
    appendMetaItem(
      kpiStripEl,
      "Coverage",
      payload.coverage.firstWeek && payload.coverage.firstWeek !== payload.coverage.latestWeek
        ? `${payload.coverage.firstWeek} – ${payload.coverage.latestWeek}`
        : payload.coverage.latestWeek,
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

  tabFailuresBtn.textContent = "Agent failures";
  tabIntentsBtn.textContent = "User intents";
  tabFailuresBtn.setAttribute("aria-selected", state.activeTab === "failure" ? "true" : "false");
  tabIntentsBtn.setAttribute("aria-selected", state.activeTab === "intent" ? "true" : "false");

  const activeCategories =
    state.activeTab === "failure" ? payload.failure.categories : payload.intent.categories;
  explorerHeadingEl.textContent =
    state.activeTab === "failure"
      ? "Failure categories and clusters"
      : "User intent categories and clusters";

  if (activeCategories.length === 0) {
    workspaceEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent =
      state.activeTab === "failure"
        ? "No agent failures recorded in this time window."
        : "No classified user intents recorded in this time window.";
    return;
  }

  emptyEl.hidden = true;
  workspaceEl.hidden = false;

  renderTopClustersStrip(activeCategories, state.activeTab);

  barsListEl.replaceChildren();
  for (const cat of activeCategories) {
    barsListEl.appendChild(renderNavigatorCategory(cat, state.activeTab));
  }

  renderDetailPane(activeCategories, state.activeTab);
  renderDrawer();
}

for (const btn of windowGroupEl.querySelectorAll<HTMLButtonElement>(".ali-preset-btn")) {
  btn.addEventListener("click", () => {
    const d = Number(btn.dataset.days);
    if (d === 1 || d === 7 || d === 30) {
      void switchWindowPreset(d);
    }
  });
}

tabFailuresBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "failure" }, true));
tabIntentsBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "intent" }, true));

expandBtn.addEventListener("click", async () => {
  const target = state.displayMode === "fullscreen" ? "inline" : "fullscreen";
  const next = await bridge.requestDisplayMode(target);
  if (next.ok) {
    dispatch({ type: "hostContextChanged", displayMode: next.mode });
  } else {
    dispatch({ type: "banner", kind: "info", message: next.message });
  }
});

openForgeBtn.addEventListener("click", async () => {
  const targetUrl = buildActiveForgeLink(state);
  if (!targetUrl) {
    return;
  }
  const res = await bridge.openExternalUrl(targetUrl);
  if (!res.ok) {
    dispatch({ type: "banner", kind: "warning", message: res.message });
  }
});

bannerDismissBtn.addEventListener("click", () => dispatch({ type: "dismissBanner" }));

render();
void bridge.connect();
