import {
  buildConversationTurnUrl,
  buildInsightsSelectionUrl,
  failureClusterToMarkdown,
  type FailureSeverityId,
} from "../../shared/failure-cluster-markdown";
import type {
  ClusterBreakdownItem,
  EmbeddedConversationData,
  ExampleTurnItem,
  InsightsCategory,
  InsightsPayload,
} from "./schema";

export type DisplayMode = "inline" | "fullscreen" | "pip";
export type InsightsTab = "failure" | "intent";

export interface InlineBanner {
  kind: "info" | "warning" | "error";
  message: string;
}

export interface ExampleCacheEntry {
  status: "loading" | "ready" | "error";
  items: ExampleTurnItem[];
  errorMessage: string | null;
}

export interface SelectedInsightsContext {
  signatureType: "failure" | "intent";
  categoryId: string | null;
  categoryLabel: string | null;
  topicId: string | null;
  clusterLabel: string | null;
  description: string | null;
  severity: FailureSeverityId | null;
  count: number | null;
  conversationCount: number | null;
  conversationId: string | null;
  traceId: string | null;
  failureId: string | null;
  failureSummary: string | null;
  evidenceSpanIds: string[];
  exampleCacheKey: string | null;
}

export interface ModelContextPayload {
  content: Array<{ type: "text"; text: string }>;
  structuredContent?: Record<string, unknown>;
}

export interface InsightsState {
  status: "idle" | "loading" | "ready" | "error";
  pendingTitle: string | null;
  payload: InsightsPayload | null;
  fatalError: string | null;
  banner: InlineBanner | null;
  isWindowUpdating: boolean;
  activeTab: InsightsTab;
  expandedCategoryKey: string | null;
  expandedExamplesKey: string | null;
  examplesCache: Record<string, ExampleCacheEntry>;
  selectedContext: SelectedInsightsContext | null;
  drawerStatus: "closed" | "loading" | "ready" | "error";
  drawerConversation: EmbeddedConversationData | null;
  drawerError: string | null;
  displayMode: DisplayMode;
  availableDisplayModes: readonly DisplayMode[];
}

export type InsightsAction =
  | { type: "toolInput"; title: string | null }
  | { type: "windowUpdateStarted" }
  | { type: "loaded"; payload: InsightsPayload }
  | { type: "loadFailed"; message: string }
  | { type: "setTab"; tab: InsightsTab }
  | {
      type: "toggleCategory";
      categoryKey: string;
      signatureType: "failure" | "intent";
      category: InsightsCategory;
      directExamplesKey?: string | null;
    }
  | {
      type: "toggleClusterExamples";
      examplesKey: string;
      signatureType: "failure" | "intent";
      category: InsightsCategory;
      cluster: ClusterBreakdownItem | null;
    }
  | { type: "examplesLoading"; examplesKey: string }
  | { type: "examplesLoaded"; examplesKey: string; items: ExampleTurnItem[] }
  | { type: "examplesFailed"; examplesKey: string; message: string }
  | {
      type: "selectTurn";
      signatureType: "failure" | "intent";
      category: InsightsCategory;
      cluster: ClusterBreakdownItem | null;
      examplesKey: string;
      turn: ExampleTurnItem;
    }
  | { type: "drawerLoading" }
  | { type: "drawerLoaded"; data: EmbeddedConversationData }
  | { type: "drawerFailed"; message: string }
  | { type: "closeDrawer" }
  | { type: "banner"; kind: "info" | "warning" | "error"; message: string }
  | { type: "dismissBanner" }
  | {
      type: "hostContextChanged";
      displayMode?: DisplayMode;
      availableDisplayModes?: readonly DisplayMode[];
    };

export function createInitialInsightsState(): InsightsState {
  return {
    status: "idle",
    pendingTitle: null,
    payload: null,
    fatalError: null,
    banner: null,
    isWindowUpdating: false,
    activeTab: "failure",
    expandedCategoryKey: null,
    expandedExamplesKey: null,
    examplesCache: {},
    selectedContext: null,
    drawerStatus: "closed",
    drawerConversation: null,
    drawerError: null,
    displayMode: "inline",
    availableDisplayModes: ["inline"],
  };
}

function buildContextFromCategoryOrCluster(
  signatureType: "failure" | "intent",
  category: InsightsCategory,
  cluster: ClusterBreakdownItem | null,
  exampleCacheKey: string | null,
): SelectedInsightsContext {
  if (cluster) {
    return {
      signatureType,
      categoryId: category.categoryId,
      categoryLabel: category.label,
      topicId: cluster.id,
      clusterLabel: cluster.title,
      description: cluster.description ?? cluster.label,
      severity: cluster.severity,
      count: cluster.count,
      conversationCount: cluster.conversationCount,
      conversationId: null,
      traceId: null,
      failureId: null,
      failureSummary: cluster.description ?? cluster.label ?? cluster.title,
      evidenceSpanIds: [],
      exampleCacheKey,
    };
  }
  return {
    signatureType,
    categoryId: category.categoryId,
    categoryLabel: category.label,
    topicId: null,
    clusterLabel: null,
    description: null,
    severity: null,
    count: category.count,
    conversationCount: category.conversationCount,
    conversationId: null,
    traceId: null,
    failureId: null,
    failureSummary: null,
    evidenceSpanIds: [],
    exampleCacheKey,
  };
}

function defaultSelection(payload: InsightsPayload, tab: InsightsTab): SelectedInsightsContext | null {
  const section = tab === "failure" ? payload.failure : payload.intent;
  const topCat = section.categories[0];
  if (!topCat) {
    return null;
  }
  return buildContextFromCategoryOrCluster(tab, topCat, null, null);
}

export function reduceInsightsState(state: InsightsState, action: InsightsAction): InsightsState {
  switch (action.type) {
    case "toolInput":
      return {
        ...state,
        status: state.payload ? state.status : "loading",
        pendingTitle: action.title,
        fatalError: null,
      };

    case "windowUpdateStarted":
      return {
        ...state,
        isWindowUpdating: true,
        banner: null,
      };

    case "loaded": {
      const nextTab: InsightsTab =
        action.payload.failure.categories.length > 0
          ? "failure"
          : action.payload.intent.categories.length > 0
            ? "intent"
            : state.activeTab;
      const warnText = action.payload.warnings.length > 0 ? action.payload.warnings[0]! : null;
      return {
        ...state,
        status: "ready",
        pendingTitle: null,
        payload: action.payload,
        fatalError: null,
        isWindowUpdating: false,
        activeTab: nextTab,
        expandedCategoryKey: null,
        expandedExamplesKey: null,
        examplesCache: {},
        selectedContext: defaultSelection(action.payload, nextTab),
        drawerStatus: "closed",
        drawerConversation: null,
        drawerError: null,
        banner: warnText ? { kind: "warning", message: warnText } : null,
      };
    }

    case "loadFailed":
      if (state.payload) {
        return {
          ...state,
          isWindowUpdating: false,
          banner: { kind: "error", message: action.message },
        };
      }
      return {
        ...state,
        status: "error",
        isWindowUpdating: false,
        fatalError: action.message,
      };

    case "setTab":
      return {
        ...state,
        activeTab: action.tab,
        expandedCategoryKey: null,
        expandedExamplesKey: null,
        selectedContext: state.payload ? defaultSelection(state.payload, action.tab) : state.selectedContext,
      };

    case "toggleCategory": {
      const collapsing = state.expandedCategoryKey === action.categoryKey;
      return {
        ...state,
        expandedCategoryKey: collapsing ? null : action.categoryKey,
        expandedExamplesKey: collapsing ? null : (action.directExamplesKey ?? null),
        selectedContext: buildContextFromCategoryOrCluster(
          action.signatureType,
          action.category,
          null,
          collapsing ? null : (action.directExamplesKey ?? null),
        ),
      };
    }

    case "toggleClusterExamples": {
      const collapsing = state.expandedExamplesKey === action.examplesKey;
      return {
        ...state,
        expandedExamplesKey: collapsing ? null : action.examplesKey,
        selectedContext: buildContextFromCategoryOrCluster(
          action.signatureType,
          action.category,
          collapsing ? null : action.cluster,
          collapsing ? null : action.examplesKey,
        ),
      };
    }

    case "examplesLoading":
      return {
        ...state,
        examplesCache: {
          ...state.examplesCache,
          [action.examplesKey]: { status: "loading", items: [], errorMessage: null },
        },
      };

    case "examplesLoaded":
      return {
        ...state,
        examplesCache: {
          ...state.examplesCache,
          [action.examplesKey]: { status: "ready", items: action.items, errorMessage: null },
        },
      };

    case "examplesFailed":
      return {
        ...state,
        examplesCache: {
          ...state.examplesCache,
          [action.examplesKey]: { status: "error", items: [], errorMessage: action.message },
        },
      };

    case "selectTurn":
      return {
        ...state,
        selectedContext: {
          ...buildContextFromCategoryOrCluster(
            action.signatureType,
            action.category,
            action.cluster,
            action.examplesKey,
          ),
          conversationId: action.turn.conversationId,
          traceId: action.turn.traceId,
          failureId: action.turn.failureId,
          severity: action.turn.failureSeverity ?? action.cluster?.severity ?? null,
          failureSummary: action.turn.failureReason,
          evidenceSpanIds: action.turn.failureEvidenceSpanIds,
        },
      };

    case "drawerLoading":
      return {
        ...state,
        drawerStatus: "loading",
        drawerError: null,
      };

    case "drawerLoaded":
      return {
        ...state,
        drawerStatus: "ready",
        drawerConversation: action.data,
        drawerError: null,
      };

    case "drawerFailed":
      return {
        ...state,
        drawerStatus: "error",
        drawerError: action.message,
      };

    case "closeDrawer":
      return {
        ...state,
        drawerStatus: "closed",
        drawerConversation: null,
        drawerError: null,
      };

    case "banner":
      return {
        ...state,
        isWindowUpdating: false,
        banner: { kind: action.kind, message: action.message },
      };

    case "dismissBanner":
      return {
        ...state,
        banner: null,
      };

    case "hostContextChanged":
      return {
        ...state,
        displayMode: action.displayMode ?? state.displayMode,
        availableDisplayModes: action.availableDisplayModes ?? state.availableDisplayModes,
      };
  }
}

export function buildActiveForgeLink(state: InsightsState): string | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const sel = state.selectedContext;
  if (sel?.conversationId && sel.traceId) {
    return buildConversationTurnUrl({
      forgeUrl: payload.forgeUrl,
      entity: payload.entityName,
      project: payload.projectName,
      conversationId: sel.conversationId,
      traceId: sel.traceId,
    });
  }
  if (sel?.categoryId) {
    return buildInsightsSelectionUrl({
      forgeUrl: payload.forgeUrl,
      entity: payload.entityName,
      project: payload.projectName,
      insightType: sel.signatureType,
      categoryId: sel.categoryId,
      clusterId: sel.topicId,
      window: payload.window,
    });
  }
  return payload.forgeUrl;
}

export function buildSelectionMarkdownBrief(state: InsightsState): string | null {
  const payload = state.payload;
  const sel = state.selectedContext;
  if (!payload || !sel || !sel.categoryId || !sel.categoryLabel) {
    return null;
  }
  const cachedTurns = sel.exampleCacheKey
    ? (state.examplesCache[sel.exampleCacheKey]?.items ?? [])
    : [];
  const orderedTurns = sel.traceId
    ? [
        ...cachedTurns.filter((t) => t.traceId === sel.traceId),
        ...cachedTurns.filter((t) => t.traceId !== sel.traceId),
      ]
    : cachedTurns;

  const selectionUrl = buildInsightsSelectionUrl({
    forgeUrl: payload.forgeUrl,
    entity: payload.entityName,
    project: payload.projectName,
    insightType: sel.signatureType,
    categoryId: sel.categoryId,
    clusterId: sel.topicId,
    window: payload.window,
  });

  if (sel.signatureType === "intent") {
    return [
      `# ${sel.clusterLabel ?? sel.categoryLabel}`,
      `Agent Lens grouped ${(sel.count ?? 0).toLocaleString("en-US")} user intent turns in ${payload.entityName}/${payload.projectName}.`,
      `View similar intents online at ${selectionUrl}`,
    ].join("\n\n");
  }

  return failureClusterToMarkdown({
    categoryId: sel.categoryId,
    categoryLabel: sel.categoryLabel,
    clusterId: sel.topicId,
    clusterLabel: sel.clusterLabel,
    description: sel.description,
    severity: sel.severity,
    count: sel.count,
    conversationCount: sel.conversationCount,
    window: { startAt: payload.window.startAt, endAt: payload.window.endAt },
    project: { entity: payload.entityName, project: payload.projectName },
    url: selectionUrl,
    turns: orderedTurns.map((t) => ({
      conversation_id: t.conversationId,
      trace_id: t.traceId,
      url: buildConversationTurnUrl({
        forgeUrl: payload.forgeUrl,
        entity: payload.entityName,
        project: payload.projectName,
        conversationId: t.conversationId,
        traceId: t.traceId,
      }),
      failure_evidence_span_ids: t.failureEvidenceSpanIds,
      failure_reason: t.failureReason ?? undefined,
    })),
  });
}

export function buildInsightsModelContext(state: InsightsState): ModelContextPayload | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const sel = state.selectedContext;
  const selectedFailure =
    sel?.traceId && sel.failureId
      ? { trace_id: sel.traceId, failure_id: sel.failureId }
      : null;

  const structured: Record<string, unknown> = {
    context_type: "wandb_agent_lens_insights",
    entity_name: payload.entityName,
    project_name: payload.projectName,
    window_start_at: payload.window.startAt,
    window_end_at: payload.window.endAt,
    window_days: payload.window.windowDays,
    signature_type: sel?.signatureType ?? state.activeTab,
    category_id: sel?.categoryId ?? null,
    category_label: sel?.categoryLabel ?? null,
    topic_id: sel?.topicId ?? null,
    cluster_title: sel?.clusterLabel ?? null,
    severity: sel?.severity ?? null,
    conversation_id: sel?.conversationId ?? null,
    trace_id: sel?.traceId ?? null,
    failure_id: sel?.failureId ?? null,
    selected_failure: selectedFailure,
    failure_evidence_span_ids: sel?.evidenceSpanIds ?? [],
    failure_summary: sel?.failureSummary ?? null,
    active_tab: state.activeTab,
    display_mode: state.displayMode,
  };

  const markdownBrief = buildSelectionMarkdownBrief(state);
  const textBlock =
    markdownBrief ??
    `[MCP Agent Lens Insights Context] project=${payload.entityName}/${payload.projectName} signature_type=${String(structured.signature_type)}`;

  return {
    content: [{ type: "text", text: textBlock }],
    structuredContent: structured,
  };
}
