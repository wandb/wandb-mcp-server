import {
  buildConversationTurnUrl,
  formatPlainFailureReason,
} from "../../shared/failure-cluster-markdown";
import {
  canonicalTraceKey,
  sameTraceId,
  type AgentFailureFinding,
  type AppliedConversationTag,
  type ConversationPayload,
  type NormalizedFeedback,
  type NormalizedTurn,
  type SelectedFailurePin,
  type SpanWaterfallNode,
  type TagCatalogEntry,
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
  pendingSelectedFailure: SelectedFailurePin | null;
  payload: ConversationPayload | null;
  fatalError: string | null;
  banner: InlineBanner | null;
  activeTab: ActiveTab;
  errorsOnly: boolean;
  failuresPanelOpen: boolean;
  agentFailures: AgentFailureFinding[];
  activeFailureId: string | null;
  pendingFocusSpanId: string | null;
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
  | { type: "toolInput"; title: string | null; selectedFailure?: SelectedFailurePin | null }
  | { type: "loaded"; payload: ConversationPayload }
  | { type: "loadFailed"; message: string }
  | { type: "setTab"; tab: ActiveTab }
  | { type: "toggleErrorsOnly" }
  | { type: "toggleFailuresPanel"; open?: boolean }
  | { type: "failuresLoaded"; failures: AgentFailureFinding[] }
  | { type: "focusFailure"; failureId: string; spanId?: string | null }
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
    pendingSelectedFailure: null,
    payload: null,
    fatalError: null,
    banner: null,
    activeTab: "conversation",
    errorsOnly: false,
    failuresPanelOpen: true,
    agentFailures: [],
    activeFailureId: null,
    pendingFocusSpanId: null,
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

function defaultSelectionForPayload(
  payload: ConversationPayload,
  pinned: SelectedFailurePin | null,
): SelectedContextTarget | null {
  if (payload.turns.length === 0) {
    return null;
  }
  const pinnedTurn = pinned
    ? payload.turns.find((t) => sameTraceId(t.traceId, pinned.traceId))
    : undefined;
  const targetTurn =
    pinnedTurn ?? payload.turns.find((t) => t.isError) ?? payload.turns[0]!;
  return buildTargetFromTurn(targetTurn);
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
        const firstErrCall =
          msg.toolCalls.find((tc) => tc.error || tc.statusCode === "ERROR") ?? msg.toolCalls[0]!;
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

function buildTargetFromSpan(
  span: SpanWaterfallNode,
  payload: ConversationPayload | null,
): SelectedContextTarget {
  const turn = payload?.turns.find((t) => sameTraceId(t.traceId, span.traceId));
  return {
    kind: "span",
    turnIndex: turn?.turnIndex ?? null,
    traceId: span.traceId ?? payload?.traceId ?? null,
    spanId: span.spanId,
    spanName: span.name,
    toolName: span.toolName,
    errorPreview: span.errorPreview,
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
  const failureTraces = new Set(
    state.agentFailures.map((f) => canonicalTraceKey(f.traceId)).filter((k): k is string => k !== null),
  );
  return state.payload.turns.filter((t) => {
    const tKey = canonicalTraceKey(t.traceId);
    return t.isError || (tKey !== null && failureTraces.has(tKey));
  });
}

export function getActiveFailureFinding(state: ConversationState): AgentFailureFinding | null {
  if (state.agentFailures.length === 0) {
    return null;
  }
  if (state.activeFailureId) {
    const byId = state.agentFailures.find((f) => f.failureId === state.activeFailureId);
    if (byId) {
      return byId;
    }
  }
  const selSpan = state.selectedTarget?.spanId;
  if (selSpan) {
    const byEvidence = state.agentFailures.find((f) => f.evidenceSpanIds.includes(selSpan));
    if (byEvidence) {
      return byEvidence;
    }
  }
  const selTrace = state.selectedTarget?.traceId;
  if (selTrace) {
    const byTrace = state.agentFailures.find((f) => sameTraceId(f.traceId, selTrace));
    if (byTrace) {
      return byTrace;
    }
  }
  return null;
}

export function getEvidenceSpanIdsSet(state: ConversationState): Set<string> {
  const ids = new Set<string>();
  const active = getActiveFailureFinding(state);
  if (active) {
    for (const sid of active.evidenceSpanIds) {
      ids.add(sid);
    }
    return ids;
  }
  for (const f of state.agentFailures) {
    for (const sid of f.evidenceSpanIds) {
      ids.add(sid);
    }
  }
  return ids;
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
        pendingSelectedFailure: action.selectedFailure ?? state.pendingSelectedFailure,
        fatalError: null,
      };

    case "loaded": {
      const pinned = action.payload.selectedFailure ?? state.pendingSelectedFailure;
      return {
        ...state,
        status: "ready",
        pendingTitle: null,
        pendingSelectedFailure: null,
        payload: pinned && !action.payload.selectedFailure
          ? { ...action.payload, selectedFailure: pinned }
          : action.payload,
        fatalError: null,
        banner: null,
        activeTab: "conversation",
        errorsOnly: false,
        failuresPanelOpen: true,
        agentFailures: [],
        activeFailureId: pinned?.failureId ?? null,
        pendingFocusSpanId: null,
        selectedTarget: defaultSelectionForPayload(action.payload, pinned),
        spansStatus: "idle",
        spans: [],
        selectedSpanId: null,
        tags: [],
        tagPopoverOpen: false,
        noteComposerTurnIndex: null,
        isSubmittingAction: false,
      };
    }

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

    case "toggleFailuresPanel":
      return {
        ...state,
        failuresPanelOpen: action.open ?? !state.failuresPanelOpen,
      };

    case "failuresLoaded": {
      const pinned = state.payload?.selectedFailure;
      const matchedPin = pinned
        ? (action.failures.find((f) => f.failureId === pinned.failureId) ??
          action.failures.find((f) => sameTraceId(f.traceId, pinned.traceId)) ??
          null)
        : null;
      const defaultFinding = matchedPin ?? action.failures[0] ?? null;
      let nextTarget = state.selectedTarget;
      if (matchedPin && state.payload) {
        const turn = state.payload.turns.find((t) => sameTraceId(t.traceId, matchedPin.traceId));
        if (turn) {
          nextTarget = buildTargetFromTurn(turn, null, matchedPin.evidenceSpanIds[0] ?? null);
        }
      }
      return {
        ...state,
        agentFailures: action.failures,
        activeFailureId: defaultFinding?.failureId ?? state.activeFailureId,
        selectedTarget: nextTarget,
      };
    }

    case "focusFailure": {
      const finding = state.agentFailures.find((f) => f.failureId === action.failureId);
      if (!finding) {
        return state;
      }
      const targetSpanId = action.spanId ?? finding.evidenceSpanIds[0] ?? null;
      if (targetSpanId && state.spansStatus === "ready") {
        const span = state.spans.find((s) => s.spanId === targetSpanId);
        if (span) {
          return {
            ...state,
            activeFailureId: finding.failureId,
            activeTab: "spans",
            selectedSpanId: span.spanId,
            pendingFocusSpanId: null,
            selectedTarget: buildTargetFromSpan(span, state.payload),
          };
        }
      }
      const turn = state.payload?.turns.find((t) => sameTraceId(t.traceId, finding.traceId));
      return {
        ...state,
        activeFailureId: finding.failureId,
        activeTab:
          targetSpanId && (state.payload?.conversationId || state.payload?.traceId)
            ? "spans"
            : state.activeTab,
        pendingFocusSpanId: targetSpanId,
        selectedTarget: turn
          ? buildTargetFromTurn(turn, null, targetSpanId)
          : state.selectedTarget,
      };
    }

    case "selectTurn": {
      const turn = state.payload?.turns.find((t) => t.turnIndex === action.turnIndex);
      if (!turn) {
        return state;
      }
      const matchingFailure = turn.traceId
        ? state.agentFailures.find((f) => sameTraceId(f.traceId, turn.traceId))
        : null;
      return {
        ...state,
        activeFailureId: matchingFailure?.failureId ?? state.activeFailureId,
        selectedTarget: buildTargetFromTurn(turn, action.toolName, action.spanId),
      };
    }

    case "spansLoading":
      return {
        ...state,
        spansStatus: "loading",
      };

    case "spansLoaded": {
      const activeFailure = getActiveFailureFinding(state);
      const preferredSpanId =
        state.pendingFocusSpanId ?? activeFailure?.evidenceSpanIds[0] ?? null;
      const chosenSpan =
        (preferredSpanId
          ? action.spans.find((s) => s.spanId === preferredSpanId)
          : undefined) ??
        action.spans.find((s) => s.isError) ??
        action.spans[0] ??
        null;
      return {
        ...state,
        spansStatus: "ready",
        spans: action.spans,
        pendingFocusSpanId: null,
        selectedSpanId: chosenSpan ? chosenSpan.spanId : null,
        selectedTarget: chosenSpan
          ? buildTargetFromSpan(chosenSpan, state.payload)
          : state.selectedTarget,
      };
    }

    case "spansFailed":
      return {
        ...state,
        spansStatus: "error",
        pendingFocusSpanId: null,
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
      const citedFailure = state.agentFailures.find((f) =>
        f.evidenceSpanIds.includes(span.spanId),
      );
      return {
        ...state,
        activeFailureId: citedFailure?.failureId ?? state.activeFailureId,
        selectedSpanId: span.spanId,
        selectedTarget: buildTargetFromSpan(span, state.payload),
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

export function buildActiveConversationForgeLink(state: ConversationState): string | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const traceId = state.selectedTarget?.traceId ?? payload.traceId;
  if (payload.conversationId && traceId) {
    return buildConversationTurnUrl({
      forgeUrl: payload.forgeUrl,
      entity: payload.entityName,
      project: payload.projectName,
      conversationId: payload.conversationId,
      traceId,
      spanId: state.activeTab === "spans" ? state.selectedSpanId : null,
    });
  }
  return payload.forgeUrl;
}

export function buildConversationModelContext(state: ConversationState): ModelContextPayload | null {
  const payload = state.payload;
  if (!payload) {
    return null;
  }
  const sel = state.selectedTarget;
  const resolvedTraceId = sel?.traceId ?? payload.traceId;
  const citedBySpan = sel?.spanId
    ? state.agentFailures.find((f) => f.evidenceSpanIds.includes(sel.spanId!))
    : undefined;
  const activeFailure = citedBySpan ?? getActiveFailureFinding(state);
  const isEvidenceSpan = Boolean(
    sel?.spanId && activeFailure?.evidenceSpanIds.includes(sel.spanId),
  );
  const plainReason = activeFailure ? formatPlainFailureReason(activeFailure.failureReason) : null;
  const deepUrl = buildActiveConversationForgeLink(state);

  const selectedTurn = sel?.turnIndex
    ? payload.turns.find((t) => t.turnIndex === sel.turnIndex)
    : payload.turns.find((t) => sameTraceId(t.traceId, resolvedTraceId));
  const humanNotes = (selectedTurn?.feedback ?? payload.feedback)
    .map((fb) => fb.note?.trim() ?? "")
    .filter((n) => n.length > 0);
  const tagNames = state.tags.map((t) => t.name);

  const structured: Record<string, unknown> = {
    context_type: "wandb_agent_conversation",
    entity_name: payload.entityName,
    project_name: payload.projectName,
    mode: payload.mode,
    conversation_id: payload.conversationId,
    trace_id: resolvedTraceId,
    span_id: sel?.spanId ?? null,
    span_name: sel?.spanName ?? null,
    tool_name: sel?.toolName ?? null,
    turn_index: sel?.turnIndex ?? null,
    error_preview: sel?.errorPreview ?? null,
    failure_id: activeFailure?.failureId ?? null,
    failure_category: activeFailure?.failureCategory ?? null,
    failure_severity: activeFailure?.failureSeverity ?? null,
    failure_cluster_title: activeFailure?.clusterTitle ?? null,
    failure_reason: plainReason,
    is_evidence_span: isEvidenceSpan,
    human_notes: humanNotes,
    tags: tagNames,
    url: deepUrl,
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
  if (resolvedTraceId) {
    parts.push(`trace_id=${resolvedTraceId}`);
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
  if (isEvidenceSpan) {
    parts.push("evidence_span=true");
  }
  if (plainReason && (isEvidenceSpan || sameTraceId(activeFailure?.traceId, resolvedTraceId))) {
    parts.push(`failure_reason="${plainReason.replace(/\s+/g, " ").slice(0, 220)}"`);
  }
  if (humanNotes.length > 0) {
    parts.push(`human_note="${humanNotes[humanNotes.length - 1]!.replace(/\s+/g, " ").slice(0, 200)}"`);
  }
  if (tagNames.length > 0) {
    parts.push(`tags=${tagNames.join(",")}`);
  }
  if (sel?.errorPreview) {
    parts.push(`error="${sel.errorPreview.replace(/\s+/g, " ").slice(0, 180)}"`);
  }
  if (deepUrl) {
    parts.push(`url=${deepUrl}`);
  }

  return {
    content: [{ type: "text", text: parts.join(" ") }],
    structuredContent: structured,
  };
}
