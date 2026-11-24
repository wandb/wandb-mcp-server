import type {
  AppliedConversationTag,
  ConversationPayload,
  NormalizedFeedback,
  NormalizedTurn,
  SpanWaterfallNode,
  TagCatalogEntry,
} from "./schema";

export type DisplayMode = "inline" | "fullscreen" | "pip";
export type ActiveTab = "conversation" | "spans";

export interface InlineBanner {
  kind: "info" | "warning" | "error";
  message: string;
}

export interface SelectedContextTarget {
  kind: "turn" | "span";
  turnIndex: number | null;
  traceId: string | null;
  spanId: string | null;
  spanName: string | null;
  toolName: string | null;
  errorPreview: string | null;
}

export interface ModelContextPayload {
  content: Array<{ type: "text"; text: string }>;
  structuredContent?: Record<string, unknown>;
}

export interface ConversationState {
  status: "idle" | "loading" | "ready" | "error";
  pendingTitle: string | null;
  payload: ConversationPayload | null;
  fatalError: string | null;
  banner: InlineBanner | null;
  activeTab: ActiveTab;
  errorsOnly: boolean;
  selectedTarget: SelectedContextTarget | null;
  spansStatus: "idle" | "loading" | "ready" | "error";
  spans: SpanWaterfallNode[];
  selectedSpanId: string | null;
  tags: AppliedConversationTag[];
  tagCatalog: TagCatalogEntry[];
  tagPopoverOpen: boolean;
  noteComposerTurnIndex: number | null;
  isSubmittingAction: boolean;
  displayMode: DisplayMode;
  availableDisplayModes: readonly DisplayMode[];
}

export type ConversationAction =
  | { type: "toolInput"; title: string | null }
  | { type: "loaded"; payload: ConversationPayload }
  | { type: "loadFailed"; message: string }
  | { type: "setTab"; tab: ActiveTab }
  | { type: "toggleErrorsOnly" }
  | { type: "selectTurn"; turnIndex: number; toolName?: string | null; spanId?: string | null }
  | { type: "spansLoading" }
  | { type: "spansLoaded"; spans: SpanWaterfallNode[] }
  | { type: "spansFailed"; message: string }
  | { type: "selectSpan"; spanId: string | null }
  | { type: "tagsLoaded"; tags: AppliedConversationTag[] }
  | { type: "tagCatalogLoaded"; catalog: TagCatalogEntry[] }
  | { type: "toggleTagPopover"; open?: boolean }
  | { type: "openNoteComposer"; turnIndex: number | null }
  | { type: "actionStarted" }
  | { type: "feedbackAdded"; turnIndex: number; item: NormalizedFeedback }
  | { type: "tagAdded"; tag: AppliedConversationTag }
  | { type: "tagRemoved"; tagId: string }
  | { type: "banner"; kind: "info" | "warning" | "error"; message: string }
  | { type: "dismissBanner" }
  | {
      type: "hostContextChanged";
      displayMode?: DisplayMode;
      availableDisplayModes?: readonly DisplayMode[];
    };

export function createInitialConversationState(): ConversationState {
  return {
    status: "idle",
    pendingTitle: null,
    payload: null,
    fatalError: null,
    banner: null,
    activeTab: "conversation",
    errorsOnly: false,
    selectedTarget: null,
    spansStatus: "idle",
    spans: [],
    selectedSpanId: null,
    tags: [],
    tagCatalog: [],
    tagPopoverOpen: false,
    noteComposerTurnIndex: null,
    isSubmittingAction: false,
    displayMode: "inline",
    availableDisplayModes: ["inline"],
  };
}

function defaultSelectionForPayload(payload: ConversationPayload): SelectedContextTarget | null {
  if (payload.turns.length === 0) {
    return null;
  }
  const firstError = payload.turns.find((t) => t.isError) ?? payload.turns[0]!;
  return buildTargetFromTurn(firstError);
}

export function buildTargetFromTurn(
  turn: NormalizedTurn,
  overrideToolName?: string | null,
  overrideSpanId?: string | null,
): SelectedContextTarget {
  let detectedTool = overrideToolName ?? null;
  let detectedSpan = overrideSpanId ?? turn.spanId ?? null;
  if (!detectedTool) {
    for (const msg of turn.messages) {
      if (msg.toolName) {
        detectedTool = msg.toolName;
        detectedSpan = detectedSpan ?? msg.spanId;
        break;
      }
      if (msg.toolCalls.length > 0) {
        const firstErrCall = msg.toolCalls.find((tc) => tc.error || tc.statusCode === "ERROR") ?? msg.toolCalls[0]!;
        detectedTool = firstErrCall.name;
        detectedSpan = detectedSpan ?? firstErrCall.spanId;
        break;
      }
    }
  }
  return {
    kind: "turn",
    turnIndex: turn.turnIndex,
    traceId: turn.traceId,
    spanId: detectedSpan,
    spanName: turn.agentName ? `turn.${turn.agentName}` : `turn_${turn.turnIndex}`,
    toolName: detectedTool,
    errorPreview: turn.errorMessage ? turn.errorMessage.slice(0, 280) : null,
  };
}

export function canPerformWrites(
  state: ConversationState,
  hostSupportsServerTools: boolean,
): boolean {
  return Boolean(
    state.payload &&
      state.payload.accessMode === "read-write" &&
      hostSupportsServerTools,
  );
}

export function getVisibleTurns(state: ConversationState): NormalizedTurn[] {
  if (!state.payload) {
    return [];
  }
  if (!state.errorsOnly) {
    return state.payload.turns;
  }
  return state.payload.turns.filter((t) => t.isError);
}

export function reduceConversationState(
  state: ConversationState,
  action: ConversationAction,
): ConversationState {
  switch (action.type) {
    case "toolInput":
      return {
        ...state,
        status: state.payload ? state.status : "loading",
        pendingTitle: action.title,
        fatalError: null,
      };

    case "loaded":
      return {
        ...state,
        status: "ready",
        pendingTitle: null,
        payload: action.payload,
        fatalError: null,
        banner: null,
        activeTab: "conversation",
        errorsOnly: false,
        selectedTarget: defaultSelectionForPayload(action.payload),
        spansStatus: "idle",
        spans: [],
        selectedSpanId: null,
        tags: [],
        tagPopoverOpen: false,
        noteComposerTurnIndex: null,
        isSubmittingAction: false,
      };

    case "loadFailed":
      if (state.payload) {
        return {
          ...state,
          banner: { kind: "error", message: action.message },
          isSubmittingAction: false,
        };
      }
      return {
        ...state,
        status: "error",
        fatalError: action.message,
        isSubmittingAction: false,
      };

    case "setTab":
      return {
        ...state,
        activeTab: action.tab,
      };

    case "toggleErrorsOnly":
      return {
        ...state,
        errorsOnly: !state.errorsOnly,
      };

    case "selectTurn": {
      const turn = state.payload?.turns.find((t) => t.turnIndex === action.turnIndex);
      if (!turn) {
        return state;
      }
      return {
        ...state,
        selectedTarget: buildTargetFromTurn(turn, action.toolName, action.spanId),
      };
    }

    case "spansLoading":
      return {
        ...state,
        spansStatus: "loading",
      };

    case "spansLoaded": {
      const firstErrSpan = action.spans.find((s) => s.isError) ?? action.spans[0] ?? null;
      return {
        ...state,
        spansStatus: "ready",
        spans: action.spans,
        selectedSpanId: firstErrSpan ? firstErrSpan.spanId : null,
      };
    }

    case "spansFailed":
      return {
        ...state,
        spansStatus: "error",
        banner: { kind: "error", message: action.message },
      };

    case "selectSpan": {
      if (!action.spanId) {
        return { ...state, selectedSpanId: null };
      }
      const span = state.spans.find((s) => s.spanId === action.spanId);
      if (!span) {
        return state;
      }
      return {
        ...state,
        selectedSpanId: span.spanId,
        selectedTarget: {
          kind: "span",
          turnIndex: null,
          traceId: span.traceId ?? state.payload?.traceId ?? null,
          spanId: span.spanId,
          spanName: span.name,
          toolName: span.toolName,
          errorPreview: span.errorPreview,
        },
      };
    }

    case "tagsLoaded":
      return {
        ...state,
        tags: action.tags,
      };

    case "tagCatalogLoaded":
      return {
        ...state,
        tagCatalog: action.catalog,
      };

    case "toggleTagPopover":
      return {
        ...state,
        tagPopoverOpen: action.open ?? !state.tagPopoverOpen,
      };

    case "openNoteComposer":
      return {
        ...state,
        noteComposerTurnIndex: action.turnIndex,
      };

    case "actionStarted":
      return {
        ...state,
        isSubmittingAction: true,
      };

    case "feedbackAdded": {
      if (!state.payload) {
        return { ...state, isSubmittingAction: false };
      }
      const nextTurns = state.payload.turns.map((t) =>
        t.turnIndex === action.turnIndex
          ? { ...t, feedback: [...t.feedback, action.item] }
          : t,
      );
      return {
        ...state,
        isSubmittingAction: false,
        noteComposerTurnIndex: null,
        payload: {
          ...state.payload,
          turns: nextTurns,
          feedback: [...state.payload.feedback, action.item],
        },
        banner: {
          kind: "info",
          message: action.item.emoji
            ? `Recorded ${action.item.emoji} reaction on Turn ${action.turnIndex}.`
            : `Saved note on Turn ${action.turnIndex}.`,
        },
      };
    }

    case "tagAdded":
      return {
        ...state,
        isSubmittingAction: false,
        tagPopoverOpen: false,
        tags: [...state.tags.filter((t) => t.name !== action.tag.name), action.tag],
        banner: { kind: "info", message: `Applied tag "${action.tag.name}".` },
      };

    case "tagRemoved":
      return {
        ...state,
        isSubmittingAction: false,
        tags: state.tags.filter((t) => t.tagId !== action.tagId),
        banner: { kind: "info", message: "Removed conversation tag." },
      };

    case "banner":
      return {
        ...state,
        isSubmittingAction: false,
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

export function buildConversationModelContext(state: ConversationState): ModelContextPayload | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const sel = state.selectedTarget;
  const structured: Record<string, unknown> = {
    context_type: "wandb_agent_conversation",
    entity_name: payload.entityName,
    project_name: payload.projectName,
    mode: payload.mode,
    conversation_id: payload.conversationId,
    trace_id: sel?.traceId ?? payload.traceId,
    span_id: sel?.spanId ?? null,
    span_name: sel?.spanName ?? null,
    tool_name: sel?.toolName ?? null,
    turn_index: sel?.turnIndex ?? null,
    error_preview: sel?.errorPreview ?? null,
    active_tab: state.activeTab,
    display_mode: state.displayMode,
  };

  const parts = [
    "[MCP Agent Conversation Context]",
    `project=${payload.entityName}/${payload.projectName}`,
  ];
  if (payload.conversationId) {
    parts.push(`conversation_id=${payload.conversationId}`);
  }
  if (structured.trace_id) {
    parts.push(`trace_id=${String(structured.trace_id)}`);
  }
  if (sel?.spanId) {
    parts.push(`span_id=${sel.spanId}`);
  }
  if (sel?.spanName) {
    parts.push(`span_name=${sel.spanName}`);
  }
  if (sel?.toolName) {
    parts.push(`tool_name=${sel.toolName}`);
  }
  if (sel?.errorPreview) {
    parts.push(`error="${sel.errorPreview.replace(/\s+/g, " ").slice(0, 180)}"`);
  }

  return {
    content: [{ type: "text", text: parts.join(" ") }],
    structuredContent: structured,
  };
}
