import type {
  EmbeddedConversationData,
  ExampleTurnItem,
  InsightsPayload,
} from "./schema";

export type DisplayMode = "inline" | "fullscreen" | "pip";
export type InsightsTab = "failures" | "intents" | "clusters";
export type SignatureKind = "failure" | "intent";

export interface InlineBanner {
  kind: "info" | "warning" | "error";
  message: string;
}

export interface InsightsSelectedContext {
  signatureType: SignatureKind;
  categoryId: string | null;
  topicId: string | null;
  conversationId: string | null;
  traceId: string | null;
  failureSummary: string | null;
}

export interface ExampleTurnsEntry {
  status: "loading" | "ready" | "error";
  items: ExampleTurnItem[];
  error?: string;
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
  clusterFilter: SignatureKind;
  expandedRowKey: string | null;
  exampleTurnsByKey: Record<string, ExampleTurnsEntry>;
  selectedContext: InsightsSelectedContext | null;
  drawerStatus: "closed" | "loading" | "ready" | "error";
  drawerConversation: EmbeddedConversationData | null;
  drawerError: string | null;
  displayMode: DisplayMode;
  availableDisplayModes: readonly DisplayMode[];
}

export type InsightsAction =
  | { type: "toolInput"; title: string | null }
  | { type: "loaded"; payload: InsightsPayload; preserveTab?: boolean }
  | { type: "loadFailed"; message: string }
  | { type: "windowUpdateStarted" }
  | { type: "setTab"; tab: InsightsTab }
  | { type: "setClusterFilter"; filter: SignatureKind }
  | {
      type: "toggleExpandRow";
      rowKey: string;
      context: InsightsSelectedContext;
    }
  | { type: "exampleTurnsLoading"; rowKey: string }
  | { type: "exampleTurnsLoaded"; rowKey: string; items: ExampleTurnItem[] }
  | { type: "exampleTurnsFailed"; rowKey: string; message: string }
  | { type: "selectExampleTurn"; context: InsightsSelectedContext }
  | { type: "drawerLoading"; context: InsightsSelectedContext }
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
    activeTab: "failures",
    clusterFilter: "failure",
    expandedRowKey: null,
    exampleTurnsByKey: {},
    selectedContext: null,
    drawerStatus: "closed",
    drawerConversation: null,
    drawerError: null,
    displayMode: "inline",
    availableDisplayModes: ["inline"],
  };
}

function defaultContextForPayload(payload: InsightsPayload): InsightsSelectedContext | null {
  const topFailureCat = payload.failure.categories[0];
  if (topFailureCat) {
    return {
      signatureType: "failure",
      categoryId: topFailureCat.categoryId,
      topicId: topFailureCat.clusterBreakdowns[0]?.id ?? null,
      conversationId: null,
      traceId: null,
      failureSummary: `${topFailureCat.label} (${topFailureCat.count} turns)`,
    };
  }
  const topIntentCat = payload.intent.categories[0];
  if (topIntentCat) {
    return {
      signatureType: "intent",
      categoryId: topIntentCat.categoryId,
      topicId: topIntentCat.clusterBreakdowns[0]?.id ?? null,
      conversationId: null,
      traceId: null,
      failureSummary: `${topIntentCat.label} (${topIntentCat.count} turns)`,
    };
  }
  return null;
}

export function reduceInsightsState(
  state: InsightsState,
  action: InsightsAction,
): InsightsState {
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
      const autoTab: InsightsTab =
        action.preserveTab
          ? state.activeTab
          : action.payload.failure.turnCount > 0 || action.payload.intent.turnCount === 0
            ? "failures"
            : "intents";
      const warningBanner: InlineBanner | null =
        action.payload.warnings.length > 0
          ? { kind: "warning", message: action.payload.warnings[0]! }
          : null;
      return {
        ...state,
        status: "ready",
        pendingTitle: null,
        payload: action.payload,
        fatalError: null,
        banner: warningBanner,
        isWindowUpdating: false,
        activeTab: autoTab,
        expandedRowKey: null,
        exampleTurnsByKey: {},
        selectedContext: defaultContextForPayload(action.payload),
        drawerStatus: "closed",
        drawerConversation: null,
        drawerError: null,
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
      };

    case "setClusterFilter":
      return {
        ...state,
        clusterFilter: action.filter,
      };

    case "toggleExpandRow": {
      const closing = state.expandedRowKey === action.rowKey;
      return {
        ...state,
        expandedRowKey: closing ? null : action.rowKey,
        selectedContext: action.context,
      };
    }

    case "exampleTurnsLoading":
      return {
        ...state,
        exampleTurnsByKey: {
          ...state.exampleTurnsByKey,
          [action.rowKey]: { status: "loading", items: [] },
        },
      };

    case "exampleTurnsLoaded":
      return {
        ...state,
        exampleTurnsByKey: {
          ...state.exampleTurnsByKey,
          [action.rowKey]: { status: "ready", items: action.items },
        },
      };

    case "exampleTurnsFailed":
      return {
        ...state,
        exampleTurnsByKey: {
          ...state.exampleTurnsByKey,
          [action.rowKey]: { status: "error", items: [], error: action.message },
        },
      };

    case "selectExampleTurn":
      return {
        ...state,
        selectedContext: action.context,
      };

    case "drawerLoading":
      return {
        ...state,
        selectedContext: action.context,
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

export function buildInsightsModelContext(state: InsightsState): ModelContextPayload | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const sel = state.selectedContext;
  const structured: Record<string, unknown> = {
    context_type: "wandb_agent_lens_insights",
    entity_name: payload.entityName,
    project_name: payload.projectName,
    window_start_at: payload.window.startAt,
    window_end_at: payload.window.endAt,
    window_days: payload.window.windowDays,
    signature_type: sel?.signatureType ?? "failure",
    category_id: sel?.categoryId ?? null,
    topic_id: sel?.topicId ?? null,
    conversation_id: sel?.conversationId ?? null,
    trace_id: sel?.traceId ?? null,
    failure_summary: sel?.failureSummary ?? null,
    active_tab: state.activeTab,
    display_mode: state.displayMode,
  };

  const parts = [
    "[MCP Agent Lens Insights Context]",
    `project=${payload.entityName}/${payload.projectName}`,
    `signature_type=${String(structured.signature_type)}`,
  ];
  if (sel?.categoryId) {
    parts.push(`category_id=${sel.categoryId}`);
  }
  if (sel?.topicId) {
    parts.push(`topic_id=${sel.topicId}`);
  }
  if (sel?.conversationId) {
    parts.push(`conversation_id=${sel.conversationId}`);
  }
  if (sel?.traceId) {
    parts.push(`trace_id=${sel.traceId}`);
  }
  if (sel?.failureSummary) {
    parts.push(`summary="${sel.failureSummary.replace(/\s+/g, " ").slice(0, 180)}"`);
  }

  return {
    content: [{ type: "text", text: parts.join(" ") }],
    structuredContent: structured,
  };
}
