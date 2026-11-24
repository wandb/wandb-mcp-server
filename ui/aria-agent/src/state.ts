import type {
  AccessMode,
  AriaAgentQuestion,
  AriaParallelBatchPayload,
  AriaPermissionRequest,
  AriaScope,
  AriaThreadPayload,
  AriaThreadsListPayload,
  AriaThreadSummary,
  AriaTurnPayload,
} from "./schema";

export type DisplayMode = "inline" | "fullscreen" | "pip";
export type ParallelLayout = "grid" | "split";

export type SessionPhase =
  | "queued"
  | "working"
  | "responding"
  | "awaiting_permission"
  | "awaiting_user_input"
  | "completed"
  | "errored"
  | "cancelled";

export interface SyncedTurnEntry {
  turn: AriaTurnPayload;
  syncedAtMs: number;
  optimistic?: boolean;
}

export interface AriaAgentSession {
  sessionId: string;
  label: string;
  scope: AriaScope;
  forgeUrl: string | null;
  turns: SyncedTurnEntry[];
}

export interface AriaStudioState {
  sessions: AriaAgentSession[];
  selectedSessionId: string | null;
  parallelLayout: ParallelLayout;
  autoPoll: boolean;
  pollInFlight: boolean;
  spawnDrawerOpen: boolean;
  threadsDrawerOpen: boolean;
  recentThreads: AriaThreadSummary[];
  accessMode: AccessMode;
  forgeHomeUrl: string;
  displayMode: DisplayMode;
  availableDisplayModes: DisplayMode[];
  bannerMessage: string | null;
  bannerTone: "info" | "warning" | "error";
  expandedStepTurnIds: string[];
}

export interface FleetSummary {
  total: number;
  active: number;
  awaitingPermission: number;
  awaitingInput: number;
  completed: number;
  errored: number;
  allTerminal: boolean;
}

export interface ModelContextPayload {
  text: string;
  structuredContent: Record<string, unknown>;
}

const DEFAULT_FORGE_HOME_URL = "https://forge.coreweave.com/home/aria";

export function createInitialState(): AriaStudioState {
  return {
    sessions: [],
    selectedSessionId: null,
    parallelLayout: "grid",
    autoPoll: true,
    pollInFlight: false,
    spawnDrawerOpen: false,
    threadsDrawerOpen: false,
    recentThreads: [],
    accessMode: "read-write",
    forgeHomeUrl: DEFAULT_FORGE_HOME_URL,
    displayMode: "inline",
    availableDisplayModes: ["inline"],
    bannerMessage: null,
    bannerTone: "info",
    expandedStepTurnIds: [],
  };
}

export function latestSessionEntry(session: AriaAgentSession): SyncedTurnEntry | null {
  return session.turns.length > 0 ? session.turns[session.turns.length - 1]! : null;
}

export function deriveTurnPhase(turn: AriaTurnPayload): SessionPhase {
  if (turn.ok === false || turn.state === "errored") {
    return "errored";
  }
  if (turn.state === "cancelled") {
    return "cancelled";
  }
  const explicitPhase = turn.progress?.phase;
  if (turn.state === "completed") {
    if (
      turn.requires_permission ||
      explicitPhase === "awaiting_permission" ||
      (Array.isArray(turn.permission_requests) && turn.permission_requests.length > 0)
    ) {
      return "awaiting_permission";
    }
    if (
      turn.requires_user_input ||
      explicitPhase === "awaiting_user_input" ||
      (Array.isArray(turn.agent_questions) && turn.agent_questions.length > 0)
    ) {
      return "awaiting_user_input";
    }
    return "completed";
  }
  if (turn.state === "queued") {
    return "queued";
  }
  if (explicitPhase === "responding" || Boolean(turn.latest_response?.trim())) {
    return "responding";
  }
  return "working";
}

export function deriveSessionPhase(session: AriaAgentSession): SessionPhase {
  const latest = latestSessionEntry(session);
  if (!latest) {
    return "queued";
  }
  return deriveTurnPhase(latest.turn);
}

export function phaseDisplayMeta(phase: SessionPhase): {
  label: string;
  tone: "active" | "warning" | "success" | "danger" | "muted";
} {
  switch (phase) {
    case "queued":
      return { label: "Queued", tone: "active" };
    case "working":
      return { label: "Working", tone: "active" };
    case "responding":
      return { label: "Responding", tone: "active" };
    case "awaiting_permission":
      return { label: "Needs network access", tone: "warning" };
    case "awaiting_user_input":
      return { label: "Needs answer", tone: "warning" };
    case "completed":
      return { label: "Completed", tone: "success" };
    case "errored":
      return { label: "Errored", tone: "danger" };
    case "cancelled":
      return { label: "Cancelled", tone: "muted" };
  }
}

export function extractRequestedDomains(turn: AriaTurnPayload): {
  domains: string[];
  reasons: string[];
} {
  const fromProgress = turn.progress?.requested_network_domains ?? [];
  const domains: string[] = [];
  const reasons: string[] = [];
  const seen = new Set<string>();
  for (const d of fromProgress) {
    const clean = d.trim().toLowerCase();
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      domains.push(clean);
    }
  }
  const reqs: AriaPermissionRequest[] = turn.permission_requests ?? [];
  for (const req of reqs) {
    if (req.reason && req.reason.trim() && !reasons.includes(req.reason.trim())) {
      reasons.push(req.reason.trim());
    }
    for (const d of req.domains ?? []) {
      const clean = d.trim().toLowerCase();
      if (clean && !seen.has(clean)) {
        seen.add(clean);
        domains.push(clean);
      }
    }
  }
  return { domains, reasons };
}

export function extractAgentQuestions(turn: AriaTurnPayload): AriaAgentQuestion[] {
  return Array.isArray(turn.agent_questions) ? turn.agent_questions : [];
}

export function computeLiveElapsedSeconds(
  entry: SyncedTurnEntry,
  nowMs: number,
): number | null {
  const { turn, syncedAtMs } = entry;
  const base = turn.progress?.elapsed_seconds;
  if (typeof base === "number" && Number.isFinite(base)) {
    if (turn.is_terminal) {
      return Math.max(0, base);
    }
    const deltaSec = Math.max(0, (nowMs - syncedAtMs) / 1000);
    return Math.max(0, base + deltaSec);
  }
  if (turn.created_at) {
    const createdMs = Date.parse(turn.created_at);
    if (Number.isFinite(createdMs)) {
      const endMs =
        turn.is_terminal && turn.updated_at && Number.isFinite(Date.parse(turn.updated_at))
          ? Date.parse(turn.updated_at)
          : nowMs;
      return Math.max(0, (endMs - createdMs) / 1000);
    }
  }
  if (!turn.is_terminal) {
    return Math.max(0, (nowMs - syncedAtMs) / 1000);
  }
  return null;
}

export function formatElapsedClock(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) {
    return "";
  }
  const whole = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  if (mins === 0) {
    return `${secs}s`;
  }
  const hours = Math.floor(mins / 60);
  if (hours > 0) {
    const remMins = mins % 60;
    return `${hours}h ${remMins}m`;
  }
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

function deriveLabelFromPromptOrTurn(
  turn: AriaTurnPayload,
  fallbackIndex: number,
): string {
  if (turn.title && turn.title.trim()) {
    return turn.title.trim();
  }
  if (turn.user_prompt && turn.user_prompt.trim()) {
    const firstLine = turn.user_prompt.trim().split(/\r?\n/)[0]!.trim();
    return firstLine.length > 56 ? `${firstLine.slice(0, 53).trimEnd()}…` : firstLine;
  }
  if (turn.scope.project) {
    return turn.scope.entity
      ? `${turn.scope.entity}/${turn.scope.project}`
      : turn.scope.project;
  }
  return fallbackIndex > 1 ? `ARIA Agent #${fallbackIndex}` : "ARIA Session";
}

function findMatchingSessionIndex(
  sessions: AriaAgentSession[],
  turn: AriaTurnPayload,
): number {
  if (turn.thread_id) {
    const byThread = sessions.findIndex((s) => s.sessionId === turn.thread_id);
    if (byThread !== -1) {
      return byThread;
    }
  }
  const lineageIds = new Set(
    [turn.turn_id, turn.parent_turn_id, turn.resumed_from_turn_id].filter(
      (v): v is string => typeof v === "string" && v.length > 0,
    ),
  );
  for (let i = 0; i < sessions.length; i += 1) {
    const s = sessions[i]!;
    if (lineageIds.has(s.sessionId)) {
      return i;
    }
    if (s.turns.some((t) => lineageIds.has(t.turn.turn_id))) {
      return i;
    }
  }
  // Match optimistic pending session if one exists for the same parent or root
  const optimisticIdx = sessions.findIndex((s) =>
    s.turns.some(
      (t) =>
        t.optimistic &&
        ((turn.parent_turn_id && t.turn.parent_turn_id === turn.parent_turn_id) ||
          (!turn.parent_turn_id && !t.turn.parent_turn_id)),
    ),
  );
  return optimisticIdx;
}

function sortSessionTurns(entries: SyncedTurnEntry[]): SyncedTurnEntry[] {
  if (entries.length <= 1) {
    return entries;
  }
  const byId = new Map(entries.map((e) => [e.turn.turn_id, e]));
  const childOf = new Map<string, SyncedTurnEntry[]>();
  const roots: SyncedTurnEntry[] = [];

  for (const entry of entries) {
    const parentId = entry.turn.parent_turn_id;
    if (parentId && byId.has(parentId)) {
      const list = childOf.get(parentId) ?? [];
      list.push(entry);
      childOf.set(parentId, list);
    } else {
      roots.push(entry);
    }
  }

  const byCreated = (a: SyncedTurnEntry, b: SyncedTurnEntry): number => {
    const ta = a.turn.created_at ? Date.parse(a.turn.created_at) : a.syncedAtMs;
    const tb = b.turn.created_at ? Date.parse(b.turn.created_at) : b.syncedAtMs;
    return ta - tb;
  };
  roots.sort(byCreated);

  const ordered: SyncedTurnEntry[] = [];
  const visited = new Set<string>();
  const visit = (node: SyncedTurnEntry): void => {
    if (visited.has(node.turn.turn_id)) {
      return;
    }
    visited.add(node.turn.turn_id);
    ordered.push(node);
    const kids = childOf.get(node.turn.turn_id);
    if (kids) {
      kids.sort(byCreated);
      for (const k of kids) {
        visit(k);
      }
    }
  };
  for (const root of roots) {
    visit(root);
  }
  for (const entry of entries) {
    if (!visited.has(entry.turn.turn_id)) {
      ordered.push(entry);
    }
  }
  return ordered;
}

export function upsertTurnIntoState(
  state: AriaStudioState,
  turn: AriaTurnPayload,
  nowMs: number,
  options?: { selectUpserted?: boolean },
): AriaStudioState {
  const prevCount = state.sessions.length;
  const nextSessions = state.sessions.map((s) => ({ ...s, turns: [...s.turns] }));
  const matchIdx = findMatchingSessionIndex(nextSessions, turn);

  let upsertedSessionId: string;
  if (matchIdx === -1) {
    const sessionId = turn.thread_id || turn.turn_id;
    upsertedSessionId = sessionId;
    nextSessions.push({
      sessionId,
      label: deriveLabelFromPromptOrTurn(turn, nextSessions.length + 1),
      scope: turn.scope,
      forgeUrl: turn.forge_url ?? null,
      turns: [{ turn, syncedAtMs: nowMs }],
    });
  } else {
    const existing = nextSessions[matchIdx]!;
    const cleanedTurns = existing.turns.filter(
      (entry) =>
        !(
          entry.optimistic &&
          ((turn.parent_turn_id && entry.turn.parent_turn_id === turn.parent_turn_id) ||
            (!turn.parent_turn_id && !entry.turn.parent_turn_id))
        ),
    );
    const existingTurnIdx = cleanedTurns.findIndex((e) => e.turn.turn_id === turn.turn_id);
    if (existingTurnIdx === -1) {
      cleanedTurns.push({ turn, syncedAtMs: nowMs });
    } else {
      const prior = cleanedTurns[existingTurnIdx]!;
      const mergedTurn: AriaTurnPayload = {
        ...prior.turn,
        ...turn,
        user_prompt: turn.user_prompt ?? prior.turn.user_prompt,
        title: turn.title ?? prior.turn.title,
      };
      cleanedTurns[existingTurnIdx] = { turn: mergedTurn, syncedAtMs: nowMs };
    }
    const nextSessionId = turn.thread_id || existing.sessionId;
    upsertedSessionId = nextSessionId;
    nextSessions[matchIdx] = {
      sessionId: nextSessionId,
      label:
        turn.title?.trim() ||
        (existing.label.startsWith("ARIA ") || existing.label === "Starting ARIA turn…"
          ? deriveLabelFromPromptOrTurn(turn, matchIdx + 1)
          : existing.label),
      scope: {
        entity: turn.scope.entity ?? existing.scope.entity,
        project: turn.scope.project ?? existing.scope.project,
      },
      forgeUrl: turn.forge_url ?? existing.forgeUrl,
      turns: sortSessionTurns(cleanedTurns),
    };
  }

  const transitionedToParallel = prevCount < 2 && nextSessions.length >= 2;
  const nextSelected =
    options?.selectUpserted || !state.selectedSessionId
      ? upsertedSessionId
      : nextSessions.some((s) => s.sessionId === state.selectedSessionId)
        ? state.selectedSessionId
        : upsertedSessionId;

  return {
    ...state,
    sessions: nextSessions,
    selectedSessionId: nextSelected,
    parallelLayout: transitionedToParallel ? "grid" : state.parallelLayout,
    accessMode: turn.access_mode ?? state.accessMode,
    forgeHomeUrl: turn.forge_home_url || state.forgeHomeUrl,
    bannerMessage: null,
  };
}

export function applyParallelBatchToState(
  state: AriaStudioState,
  batch: AriaParallelBatchPayload,
  nowMs: number,
): AriaStudioState {
  let next = state;
  batch.parallel_turns.forEach((member, idx) => {
    const enriched: AriaTurnPayload = {
      ...member,
      access_mode: member.access_mode ?? batch.access_mode,
      forge_home_url: member.forge_home_url ?? batch.forge_home_url,
    };
    next = upsertTurnIntoState(next, enriched, nowMs, {
      selectUpserted: idx === 0 && !next.selectedSessionId,
    });
  });
  return next;
}

export function applyThreadPayloadToState(
  state: AriaStudioState,
  payload: AriaThreadPayload,
  nowMs: number,
): AriaStudioState {
  const { thread, turns } = payload;
  const chronological = [...turns].reverse();
  let next = state;
  if (chronological.length === 0) {
    const placeholder: AriaTurnPayload = {
      ok: true,
      turn_id: `${thread.thread_id}-empty`,
      thread_id: thread.thread_id,
      title: thread.title ?? null,
      user_prompt: thread.prompt_preview ?? null,
      scope: thread.scope,
      state: "completed",
      is_terminal: true,
      forge_url: thread.forge_url ?? null,
      access_mode: payload.access_mode,
      forge_home_url: payload.forge_home_url,
    };
    return upsertTurnIntoState(next, placeholder, nowMs, { selectUpserted: true });
  }

  chronological.forEach((summary, index) => {
    const turnPayload: AriaTurnPayload = {
      ok: true,
      turn_id: summary.turn_id,
      thread_id: thread.thread_id,
      parent_turn_id: summary.parent_turn_id ?? null,
      title: index === 0 ? (thread.title ?? null) : null,
      user_prompt: index === 0 ? (thread.prompt_preview ?? null) : null,
      scope: thread.scope,
      state: summary.state,
      is_terminal: summary.is_terminal,
      created_at: summary.created_at ?? null,
      updated_at: summary.updated_at ?? null,
      forge_url: thread.forge_url ?? null,
      access_mode: payload.access_mode,
      forge_home_url: payload.forge_home_url,
    };
    next = upsertTurnIntoState(next, turnPayload, nowMs, {
      selectUpserted: index === chronological.length - 1,
    });
  });
  return next;
}

export function applyThreadsListToState(
  state: AriaStudioState,
  payload: AriaThreadsListPayload,
): AriaStudioState {
  return {
    ...state,
    recentThreads: payload.threads,
    threadsDrawerOpen: state.sessions.length === 0 ? true : state.threadsDrawerOpen,
    accessMode: payload.access_mode ?? state.accessMode,
    forgeHomeUrl: payload.forge_home_url || state.forgeHomeUrl,
  };
}

export function applyOptimisticSendInput(
  state: AriaStudioState,
  args: Record<string, unknown>,
  nowMs: number,
): AriaStudioState {
  const message = typeof args.message === "string" ? args.message.trim() : "";
  if (!message) {
    return state;
  }
  const parentTurnId =
    typeof args.parent_turn_id === "string" && args.parent_turn_id.trim()
      ? args.parent_turn_id.trim()
      : null;
  const title =
    typeof args.title === "string" && args.title.trim() ? args.title.trim() : null;
  const entity =
    typeof args.entity === "string" && args.entity.trim() ? args.entity.trim() : null;
  const project =
    typeof args.project === "string" && args.project.trim() ? args.project.trim() : null;
  const allowedDomains = Array.isArray(args.allowed_network_domains)
    ? args.allowed_network_domains.filter((d): d is string => typeof d === "string")
    : undefined;

  const tempId = `pending-${nowMs}-${Math.random().toString(36).slice(2, 7)}`;
  const optimisticTurn: AriaTurnPayload = {
    ok: true,
    turn_id: tempId,
    thread_id: null,
    parent_turn_id: parentTurnId,
    title,
    user_prompt: message,
    scope: { entity, project },
    state: "queued",
    is_terminal: false,
    created_at: new Date(nowMs).toISOString(),
    active_allowed_network_domains: allowedDomains,
    progress: {
      phase: "queued",
      status_text: "Submitting ARIA turn and waiting for initial status…",
      message_count: 0,
      tool_call_record_count: 0,
    },
  };

  const nextSessions = state.sessions.map((s) => ({ ...s, turns: [...s.turns] }));
  if (parentTurnId) {
    const idx = nextSessions.findIndex((s) =>
      s.turns.some((t) => t.turn.turn_id === parentTurnId),
    );
    if (idx !== -1) {
      const target = nextSessions[idx]!;
      target.turns.push({ turn: optimisticTurn, syncedAtMs: nowMs, optimistic: true });
      return {
        ...state,
        sessions: nextSessions,
        selectedSessionId: target.sessionId,
      };
    }
  }

  const sessionId = tempId;
  nextSessions.push({
    sessionId,
    label: deriveLabelFromPromptOrTurn(optimisticTurn, nextSessions.length + 1),
    scope: { entity, project },
    forgeUrl: null,
    turns: [{ turn: optimisticTurn, syncedAtMs: nowMs, optimistic: true }],
  });
  return {
    ...state,
    sessions: nextSessions,
    selectedSessionId: sessionId,
    parallelLayout:
      state.sessions.length < 2 && nextSessions.length >= 2 ? "grid" : state.parallelLayout,
  };
}

export function summarizeFleet(sessions: AriaAgentSession[]): FleetSummary {
  let active = 0;
  let awaitingPermission = 0;
  let awaitingInput = 0;
  let completed = 0;
  let errored = 0;

  for (const session of sessions) {
    const phase = deriveSessionPhase(session);
    if (phase === "queued" || phase === "working" || phase === "responding") {
      active += 1;
    } else if (phase === "awaiting_permission") {
      awaitingPermission += 1;
    } else if (phase === "awaiting_user_input") {
      awaitingInput += 1;
    } else if (phase === "completed") {
      completed += 1;
    } else {
      errored += 1;
    }
  }

  return {
    total: sessions.length,
    active,
    awaitingPermission,
    awaitingInput,
    completed,
    errored,
    allTerminal: active === 0,
  };
}

export function getPollableTurnIds(state: AriaStudioState): string[] {
  const ids: string[] = [];
  for (const session of state.sessions) {
    const latest = latestSessionEntry(session);
    if (!latest || latest.optimistic) {
      continue;
    }
    if (!latest.turn.is_terminal) {
      ids.push(latest.turn.turn_id);
    }
  }
  return ids;
}

export function getHydratableCompletedTurnIds(state: AriaStudioState): string[] {
  const ids: string[] = [];
  for (const session of state.sessions) {
    const latest = latestSessionEntry(session);
    if (!latest || latest.optimistic) {
      continue;
    }
    // Turns loaded via aria_get_thread metadata lack latest_response & progress
    if (
      latest.turn.state === "completed" &&
      latest.turn.latest_response === undefined &&
      latest.turn.progress === undefined &&
      !latest.turn.turn_id.endsWith("-empty")
    ) {
      ids.push(latest.turn.turn_id);
    }
  }
  return ids;
}

export function buildModelContextPayload(state: AriaStudioState): ModelContextPayload | null {
  if (state.sessions.length === 0) {
    return null;
  }
  const summary = summarizeFleet(state.sessions);
  const isParallel = state.sessions.length >= 2;
  const sessionItems = state.sessions.map((session) => {
    const latest = latestSessionEntry(session);
    const turn = latest?.turn ?? null;
    const phase = deriveSessionPhase(session);
    const { domains } = turn ? extractRequestedDomains(turn) : { domains: [] };
    const snippet = turn?.latest_response
      ? turn.latest_response.replace(/\s+/g, " ").trim().slice(0, 280)
      : null;
    return {
      sessionId: session.sessionId,
      label: session.label,
      scope: session.scope,
      turnId: turn?.turn_id ?? null,
      threadId: turn?.thread_id ?? null,
      state: turn?.state ?? "queued",
      phase,
      isTerminal: Boolean(turn?.is_terminal),
      requestedNetworkDomains: domains,
      latestResponsePreview: snippet,
    };
  });

  const lines: string[] = [];
  if (isParallel) {
    lines.push(
      `ARIA Parallel Sub-Agents (${summary.total}): ${summary.completed} completed, ${summary.active} active, ${summary.awaitingPermission} awaiting permission, ${summary.awaitingInput} awaiting input, ${summary.errored} errored.`,
    );
  } else {
    const s = sessionItems[0]!;
    lines.push(
      `ARIA Session "${s.label}" (turn_id=${s.turnId ?? "pending"}, phase=${s.phase}, terminal=${s.isTerminal}).`,
    );
  }
  for (const item of sessionItems) {
    const domSuffix =
      item.requestedNetworkDomains.length > 0
        ? ` [needs domains: ${item.requestedNetworkDomains.join(", ")}]`
        : "";
    const respSuffix = item.latestResponsePreview ? ` — ${item.latestResponsePreview}` : "";
    lines.push(`- ${item.label} (${item.turnId ?? "pending"}): ${item.phase}${domSuffix}${respSuffix}`);
  }

  return {
    text: lines.join("\n"),
    structuredContent: {
      schema: "wandb.aria_agent_context/v1",
      mode: isParallel ? "parallel" : "single",
      selectedSessionId: state.selectedSessionId,
      summary,
      sessions: sessionItems,
    },
  };
}
