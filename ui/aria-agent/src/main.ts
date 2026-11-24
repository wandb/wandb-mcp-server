import "./style.css";
import { AriaHostBridge } from "./host";
import { renderMarkdownFragment } from "./markdown";
import type { AriaToolStep, AriaTurnPayload, ParsedAriaToolEnvelope } from "./schema";
import {
  applyOptimisticSendInput,
  applyParallelBatchToState,
  applyThreadPayloadToState,
  applyThreadsListToState,
  buildModelContextPayload,
  computeLiveElapsedSeconds,
  createInitialState,
  deriveSessionPhase,
  deriveTurnPhase,
  extractAgentQuestions,
  extractRequestedDomains,
  formatElapsedClock,
  getHydratableCompletedTurnIds,
  getPollableTurnIds,
  latestSessionEntry,
  phaseDisplayMeta,
  summarizeFleet,
  upsertTurnIntoState,
  type AriaAgentSession,
  type AriaStudioState,
  type SyncedTurnEntry,
} from "./state";

const appRoot = document.getElementById("app-root") as HTMLDivElement;
const titleEl = document.getElementById("ar-title") as HTMLHeadingElement;
const scopePillEl = document.getElementById("ar-scope-pill") as HTMLSpanElement;
const phasePillEl = document.getElementById("ar-phase-pill") as HTMLSpanElement;
const subtitleEl = document.getElementById("ar-subtitle") as HTMLParagraphElement;

const autoPollBtn = document.getElementById("ar-autopoll-btn") as HTMLButtonElement;
const pollNowBtn = document.getElementById("ar-poll-now-btn") as HTMLButtonElement;
const spawnToggleBtn = document.getElementById("ar-spawn-toggle-btn") as HTMLButtonElement;
const threadsToggleBtn = document.getElementById("ar-threads-toggle-btn") as HTMLButtonElement;
const expandBtn = document.getElementById("ar-expand-btn") as HTMLButtonElement;
const openForgeBtn = document.getElementById("ar-open-forge-btn") as HTMLButtonElement;

const fleetBarEl = document.getElementById("ar-fleet-bar") as HTMLElement;
const fleetCountsEl = document.getElementById("ar-fleet-counts") as HTMLDivElement;
const fleetMeterEl = document.getElementById("ar-fleet-meter") as HTMLDivElement;
const layoutGridBtn = document.getElementById("ar-layout-grid-btn") as HTMLButtonElement;
const layoutSplitBtn = document.getElementById("ar-layout-split-btn") as HTMLButtonElement;

const spawnDrawerEl = document.getElementById("ar-spawn-drawer") as HTMLElement;
const spawnCloseBtn = document.getElementById("ar-spawn-close-btn") as HTMLButtonElement;
const spawnEntityInput = document.getElementById("ar-spawn-entity") as HTMLInputElement;
const spawnProjectInput = document.getElementById("ar-spawn-project") as HTMLInputElement;
const spawnTitleInput = document.getElementById("ar-spawn-title") as HTMLInputElement;
const spawnPromptsInput = document.getElementById("ar-spawn-prompts") as HTMLTextAreaElement;
const spawnCountHintEl = document.getElementById("ar-spawn-count-hint") as HTMLSpanElement;
const spawnSubmitBtn = document.getElementById("ar-spawn-submit-btn") as HTMLButtonElement;

const threadsDrawerEl = document.getElementById("ar-threads-drawer") as HTMLElement;
const threadsCloseBtn = document.getElementById("ar-threads-close-btn") as HTMLButtonElement;
const threadsListEl = document.getElementById("ar-threads-list") as HTMLDivElement;

const bannerEl = document.getElementById("ar-banner") as HTMLDivElement;
const bannerTextEl = document.getElementById("ar-banner-text") as HTMLSpanElement;
const bannerDismissBtn = document.getElementById("ar-banner-dismiss") as HTMLButtonElement;

const emptyEl = document.getElementById("ar-empty") as HTMLDivElement;
const gridViewEl = document.getElementById("ar-grid-view") as HTMLDivElement;
const workspaceEl = document.getElementById("ar-workspace") as HTMLDivElement;
const railEl = document.getElementById("ar-rail") as HTMLElement;
const turnStreamEl = document.getElementById("ar-turn-stream") as HTMLDivElement;
const composerEl = document.getElementById("ar-composer") as HTMLDivElement;
const replyInput = document.getElementById("ar-reply-input") as HTMLTextAreaElement;
const replySendBtn = document.getElementById("ar-reply-send-btn") as HTMLButtonElement;

let state: AriaStudioState = createInitialState();
let pollTimer: number | null = null;
let clockInterval: number | null = null;
let actionBusy = false;

function showBanner(message: string, tone: "info" | "warning" | "error" = "info"): void {
  state = { ...state, bannerMessage: message, bannerTone: tone };
  renderBanner();
}

function clearPollTimer(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function parseSpawnTasks(raw: string): string[] {
  const blocks = raw
    .split(/\n\s*---\s*\n/g)
    .map((b) => b.trim())
    .filter(Boolean);
  return blocks;
}

function updateSpawnHint(): void {
  const tasks = parseSpawnTasks(spawnPromptsInput.value);
  const count = Math.max(1, tasks.length);
  spawnCountHintEl.textContent =
    count === 1
      ? "1 async ARIA sub-agent ready"
      : `${count} parallel async ARIA sub-agents ready`;
  spawnSubmitBtn.textContent =
    count === 1 ? "Launch async ARIA" : `Launch ${count} parallel ARIA agents`;
}

function selectedSession(): AriaAgentSession | null {
  if (state.sessions.length === 0) {
    return null;
  }
  return (
    state.sessions.find((s) => s.sessionId === state.selectedSessionId) ??
    state.sessions[0]!
  );
}

function handleOpenLink(url: string): void {
  void bridge.openExternalUrl(url).then((res) => {
    if (!res.ok) {
      showBanner(res.message, "info");
    }
  });
}

function handleEnvelope(envelope: ParsedAriaToolEnvelope, immediateSync = false): void {
  const now = Date.now();
  if (envelope.kind === "error") {
    showBanner(envelope.message, envelope.retryable ? "warning" : "error");
    renderAll();
    return;
  }
  if (envelope.kind === "parallel_turns") {
    state = applyParallelBatchToState(state, envelope.data, now);
  } else if (envelope.kind === "turn") {
    state = upsertTurnIntoState(state, envelope.data, now);
  } else if (envelope.kind === "thread") {
    state = applyThreadPayloadToState(state, envelope.data, now);
  } else if (envelope.kind === "threads_list") {
    state = applyThreadsListToState(state, envelope.data);
  }

  // Prefill spawn scope from active session when empty
  const active = selectedSession();
  if (active) {
    if (!spawnEntityInput.value && active.scope.entity) {
      spawnEntityInput.value = active.scope.entity;
    }
    if (!spawnProjectInput.value && active.scope.project) {
      spawnProjectInput.value = active.scope.project;
    }
  }

  renderAll();
  bridge.scheduleModelContextSync(buildModelContextPayload(state), immediateSync);
  scheduleNextPollOrHydration();
}

async function pollSpecificTurnIds(turnIds: string[]): Promise<void> {
  if (turnIds.length === 0 || state.pollInFlight || !bridge.canCallServerTools()) {
    return;
  }
  clearPollTimer();
  state = { ...state, pollInFlight: true };
  renderHeader();

  const primary = turnIds[0]!;
  const rest = turnIds.slice(1, 16);
  const args: Record<string, unknown> = {
    turn_id: primary,
    wait_seconds: 0,
  };
  if (rest.length > 0) {
    args.parallel_turn_ids = rest;
  }

  const envelope = await bridge.callAriaTool("aria_get_turn", args);
  state = { ...state, pollInFlight: false };
  handleEnvelope(envelope);
}

function scheduleNextPollOrHydration(): void {
  clearPollTimer();
  if (!bridge.canCallServerTools() || state.pollInFlight) {
    return;
  }

  const hydratable = getHydratableCompletedTurnIds(state);
  if (hydratable.length > 0) {
    pollTimer = window.setTimeout(() => {
      void pollSpecificTurnIds(hydratable);
    }, 80);
    return;
  }

  if (!state.autoPoll) {
    return;
  }

  const pollable = getPollableTurnIds(state);
  if (pollable.length === 0) {
    return;
  }

  let delaySec = 2.4;
  for (const session of state.sessions) {
    const latest = latestSessionEntry(session);
    const hint = latest?.turn.poll_after_seconds;
    if (typeof hint === "number" && hint > 0) {
      delaySec = Math.max(2, Math.min(6, hint));
      break;
    }
  }

  pollTimer = window.setTimeout(() => {
    void pollSpecificTurnIds(pollable);
  }, Math.round(delaySec * 1000));
}

async function sendContinuationMessage(
  turn: AriaTurnPayload,
  message: string,
  allowedDomains?: string[],
): Promise<void> {
  if (actionBusy || state.accessMode === "read-only") {
    return;
  }
  const trimmed = message.trim();
  if (!trimmed) {
    return;
  }
  actionBusy = true;
  const sendArgs: Record<string, unknown> = {
    parent_turn_id: turn.turn_id,
    message: trimmed,
    wait_seconds: 0,
  };
  if (allowedDomains && allowedDomains.length > 0) {
    sendArgs.allowed_network_domains = allowedDomains;
  }

  state = applyOptimisticSendInput(state, sendArgs, Date.now());
  renderAll();

  const envelope = await bridge.callAriaTool("aria_send_message", sendArgs);
  actionBusy = false;
  handleEnvelope(envelope, true);
}

async function launchParallelSubAgents(): Promise<void> {
  if (actionBusy || state.accessMode === "read-only") {
    return;
  }
  const prompts = parseSpawnTasks(spawnPromptsInput.value);
  if (prompts.length === 0) {
    showBanner("Enter at least one prompt to launch an ARIA sub-agent.", "warning");
    return;
  }
  actionBusy = true;
  spawnSubmitBtn.disabled = true;

  const entity = spawnEntityInput.value.trim() || undefined;
  const project = spawnProjectInput.value.trim() || undefined;
  const titlePrefix = spawnTitleInput.value.trim();

  const calls = prompts.map((promptText, idx) => {
    const computedTitle =
      prompts.length > 1
        ? titlePrefix
          ? `${titlePrefix} #${idx + 1}`
          : undefined
        : titlePrefix || undefined;
    const args: Record<string, unknown> = {
      message: promptText,
      wait_seconds: 0,
    };
    if (entity) {
      args.entity = entity;
    }
    if (project) {
      args.project = project;
    }
    if (computedTitle) {
      args.title = computedTitle;
    }
    state = applyOptimisticSendInput(state, args, Date.now() + idx);
    return bridge.callAriaTool("aria_send_message", args);
  });

  state = {
    ...state,
    spawnDrawerOpen: false,
    parallelLayout: state.sessions.length >= 2 ? "grid" : state.parallelLayout,
  };
  spawnPromptsInput.value = "";
  updateSpawnHint();
  renderAll();

  const results = await Promise.all(calls);
  actionBusy = false;
  spawnSubmitBtn.disabled = false;
  for (const env of results) {
    handleEnvelope(env, true);
  }
}

function renderBanner(): void {
  if (!state.bannerMessage) {
    bannerEl.hidden = true;
    return;
  }
  bannerEl.hidden = false;
  bannerEl.dataset.tone = state.bannerTone;
  bannerTextEl.textContent = state.bannerMessage;
}

function buildPhasePillContent(
  pill: HTMLElement,
  session: AriaAgentSession,
  nowMs: number,
): void {
  const phase = deriveSessionPhase(session);
  const meta = phaseDisplayMeta(phase);
  pill.dataset.tone = meta.tone;
  pill.replaceChildren();

  const dot = document.createElement("span");
  dot.className = "ar-live-dot";
  pill.appendChild(dot);

  const label = document.createElement("span");
  label.textContent = meta.label;
  pill.appendChild(label);

  const latest = latestSessionEntry(session);
  if (latest) {
    const elapsed = computeLiveElapsedSeconds(latest, nowMs);
    const formatted = formatElapsedClock(elapsed);
    if (formatted) {
      const clock = document.createElement("span");
      clock.dataset.elapsedTurnId = latest.turn.turn_id;
      clock.textContent = `· ${formatted}`;
      pill.appendChild(clock);
    }
  }
}

function renderHeader(): void {
  appRoot.dataset.displayMode = state.displayMode;
  const nowMs = Date.now();
  const isParallel = state.sessions.length >= 2;
  const active = selectedSession();
  const pollable = getPollableTurnIds(state);
  const canTools = bridge.canCallServerTools();
  const isReadWrite = state.accessMode === "read-write";

  if (state.sessions.length === 0) {
    titleEl.textContent =
      state.recentThreads.length > 0 ? "ARIA Conversations" : "Waiting for ARIA…";
    scopePillEl.hidden = true;
    phasePillEl.hidden = true;
    subtitleEl.hidden = true;
  } else if (isParallel) {
    const summary = summarizeFleet(state.sessions);
    titleEl.textContent = `Parallel ARIA Sub-Agents (${summary.total})`;
    const sharedEntity = active?.scope.entity;
    const sharedProject = active?.scope.project;
    if (sharedEntity || sharedProject) {
      scopePillEl.hidden = false;
      scopePillEl.textContent = [sharedEntity, sharedProject].filter(Boolean).join("/");
    } else {
      scopePillEl.hidden = true;
    }
    phasePillEl.hidden = false;
    phasePillEl.dataset.tone =
      summary.active > 0
        ? "active"
        : summary.awaitingPermission + summary.awaitingInput > 0
          ? "warning"
          : summary.errored > 0 && summary.completed === 0
            ? "danger"
            : "success";
    phasePillEl.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "ar-live-dot";
    phasePillEl.appendChild(dot);
    const text = document.createElement("span");
    text.textContent =
      summary.active > 0
        ? `${summary.active} running · ${summary.completed}/${summary.total} done`
        : summary.awaitingPermission > 0
          ? `${summary.awaitingPermission} awaiting access`
          : `${summary.completed}/${summary.total} completed`;
    phasePillEl.appendChild(text);
    subtitleEl.hidden = true;
  } else if (active) {
    titleEl.textContent = active.label;
    if (active.scope.entity || active.scope.project) {
      scopePillEl.hidden = false;
      scopePillEl.textContent = [active.scope.entity, active.scope.project]
        .filter(Boolean)
        .join("/");
    } else {
      scopePillEl.hidden = true;
    }
    phasePillEl.hidden = false;
    buildPhasePillContent(phasePillEl, active, nowMs);
    subtitleEl.hidden = true;
  }

  autoPollBtn.hidden = !canTools || state.sessions.length === 0;
  autoPollBtn.setAttribute("aria-pressed", state.autoPoll ? "true" : "false");
  autoPollBtn.textContent = state.autoPoll ? "Auto-poll: On" : "Auto-poll: Paused";

  pollNowBtn.hidden = !canTools || state.sessions.length === 0;
  pollNowBtn.disabled = state.pollInFlight;
  if (state.pollInFlight) {
    pollNowBtn.textContent = "Polling…";
  } else if (pollable.length > 1) {
    pollNowBtn.textContent = `Poll active (${pollable.length})`;
  } else {
    pollNowBtn.textContent = "Poll now";
  }

  spawnToggleBtn.hidden = !canTools || !isReadWrite;
  threadsToggleBtn.hidden = !canTools;

  const canFullscreen = state.availableDisplayModes.includes("fullscreen");
  expandBtn.hidden = !canFullscreen;
  expandBtn.textContent = state.displayMode === "fullscreen" ? "Inline" : "Expand";

  const targetForgeUrl =
    (!isParallel ? active?.forgeUrl : null) || state.forgeHomeUrl;
  openForgeBtn.hidden = !targetForgeUrl;
}

function renderFleetBar(): void {
  const isParallel = state.sessions.length >= 2;
  fleetBarEl.hidden = !isParallel;
  if (!isParallel) {
    return;
  }
  const summary = summarizeFleet(state.sessions);
  fleetCountsEl.replaceChildren();

  const parts: string[] = [`${summary.total} ARIA agents`];
  if (summary.active > 0) {
    parts.push(`${summary.active} working`);
  }
  if (summary.awaitingPermission > 0) {
    parts.push(`${summary.awaitingPermission} awaiting network access`);
  }
  if (summary.awaitingInput > 0) {
    parts.push(`${summary.awaitingInput} awaiting input`);
  }
  parts.push(`${summary.completed} completed`);
  if (summary.errored > 0) {
    parts.push(`${summary.errored} errored`);
  }

  const label = document.createElement("strong");
  label.textContent = parts.join(" · ");
  fleetCountsEl.appendChild(label);

  fleetMeterEl.replaceChildren();
  const total = Math.max(1, summary.total);
  const segs: Array<{ count: number; tone: string }> = [
    { count: summary.completed, tone: "success" },
    { count: summary.active, tone: "active" },
    { count: summary.awaitingPermission + summary.awaitingInput, tone: "warning" },
    { count: summary.errored, tone: "danger" },
  ];
  for (const seg of segs) {
    if (seg.count <= 0) {
      continue;
    }
    const bar = document.createElement("span");
    bar.className = "ar-fleet-seg";
    bar.dataset.tone = seg.tone;
    bar.style.width = `${Math.round((seg.count / total) * 100)}%`;
    fleetMeterEl.appendChild(bar);
  }

  layoutGridBtn.setAttribute(
    "aria-selected",
    state.parallelLayout === "grid" ? "true" : "false",
  );
  layoutSplitBtn.setAttribute(
    "aria-selected",
    state.parallelLayout === "split" ? "true" : "false",
  );
}

function renderDrawers(): void {
  spawnDrawerEl.hidden = !state.spawnDrawerOpen || state.accessMode === "read-only";
  threadsDrawerEl.hidden = !state.threadsDrawerOpen;

  if (state.threadsDrawerOpen) {
    threadsListEl.replaceChildren();
    if (state.recentThreads.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ar-muted-note";
      empty.textContent = "No recent ARIA threads loaded.";
      threadsListEl.appendChild(empty);
    } else {
      for (const th of state.recentThreads) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "ar-thread-row";

        const left = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = th.title || th.prompt_preview || th.thread_id;
        left.appendChild(title);
        const scopeStr = [th.scope.entity, th.scope.project].filter(Boolean).join("/");
        if (scopeStr) {
          const scopeSpan = document.createElement("div");
          scopeSpan.className = "ar-muted-note";
          scopeSpan.textContent = scopeStr;
          left.appendChild(scopeSpan);
        }
        row.appendChild(left);

        const openTag = document.createElement("span");
        openTag.className = "ar-muted-note";
        openTag.textContent = "Load thread →";
        row.appendChild(openTag);

        row.addEventListener("click", () => {
          state = { ...state, threadsDrawerOpen: false };
          renderDrawers();
          void bridge
            .callAriaTool("aria_get_thread", { thread_id: th.thread_id, limit: 20 })
            .then((env) => handleEnvelope(env, true));
        });
        threadsListEl.appendChild(row);
      }
    }
  }
}

function buildStepChip(step: AriaToolStep): HTMLSpanElement {
  const pill = document.createElement("span");
  pill.className = "ar-step-pill";
  pill.dataset.status = step.status;
  const glyph =
    step.status === "running"
      ? "●"
      : step.status === "recovered_error"
        ? "⚠"
        : "✓";
  pill.textContent = `${glyph} ${step.name}`;
  return pill;
}

function renderParallelGrid(nowMs: number): void {
  gridViewEl.replaceChildren();
  state.sessions.forEach((session, idx) => {
    const latest = latestSessionEntry(session);
    const turn = latest?.turn ?? null;
    const phase = deriveSessionPhase(session);
    const meta = phaseDisplayMeta(phase);

    const card = document.createElement("article");
    card.className = "ar-agent-card";
    card.dataset.tone = meta.tone;
    card.dataset.selected = session.sessionId === state.selectedSessionId ? "true" : "false";
    card.dataset.sessionId = session.sessionId;

    const top = document.createElement("div");
    top.className = "ar-card-top";

    const titleGroup = document.createElement("div");
    titleGroup.className = "ar-card-title-group";
    const badge = document.createElement("span");
    badge.className = "ar-idx-badge";
    badge.textContent = `#${idx + 1}`;
    titleGroup.appendChild(badge);

    const title = document.createElement("span");
    title.className = "ar-card-title";
    title.textContent = session.label;
    title.title = session.label;
    titleGroup.appendChild(title);
    top.appendChild(titleGroup);

    const pill = document.createElement("span");
    pill.className = "ar-phase-pill";
    buildPhasePillContent(pill, session, nowMs);
    top.appendChild(pill);
    card.appendChild(top);

    // Status line
    const statusLine = document.createElement("div");
    statusLine.className = "ar-card-status-line";
    const latestTool =
      turn?.progress?.latest_tool_activity?.name ??
      (turn?.tool_steps && turn.tool_steps.length > 0
        ? turn.tool_steps[turn.tool_steps.length - 1]!.name
        : null);
    const statusText = document.createElement("span");
    if (!turn?.is_terminal && latestTool) {
      statusText.textContent = `Running ${latestTool}`;
    } else {
      statusText.textContent =
        turn?.progress?.status_text ||
        (turn?.is_terminal ? "Turn finished" : "Waiting for sandbox…");
    }
    statusLine.appendChild(statusText);

    const toolCount = turn?.progress?.tool_call_record_count ?? 0;
    if (toolCount > 0) {
      const counts = document.createElement("span");
      counts.textContent = `${Math.ceil(toolCount / 2)} tools`;
      statusLine.appendChild(counts);
    }
    card.appendChild(statusLine);

    // Preview box
    const preview = document.createElement("div");
    preview.className = "ar-card-preview";
    if (turn?.latest_response && turn.latest_response.trim()) {
      const mdWrap = document.createElement("div");
      mdWrap.className = "ar-md-body";
      mdWrap.appendChild(renderMarkdownFragment(turn.latest_response, handleOpenLink));
      preview.appendChild(mdWrap);
    } else if (phase === "awaiting_permission" && turn) {
      const { domains, reasons } = extractRequestedDomains(turn);
      const p = document.createElement("p");
      p.className = "ar-md-p";
      p.textContent =
        reasons[0] ||
        `ARIA paused to request outbound sandbox access to ${domains.join(", ") || "an external domain"}.`;
      preview.appendChild(p);
    } else if (turn?.user_prompt) {
      const p = document.createElement("p");
      p.className = "ar-md-p ar-muted-note";
      p.textContent = turn.user_prompt;
      preview.appendChild(p);
    } else {
      const p = document.createElement("p");
      p.className = "ar-md-p ar-muted-note";
      p.textContent = "Executing in ARIA workspace…";
      preview.appendChild(p);
    }
    card.appendChild(preview);

    // Footer actions
    const footer = document.createElement("div");
    footer.className = "ar-card-footer";

    if (
      phase === "awaiting_permission" &&
      turn &&
      state.accessMode === "read-write" &&
      bridge.canCallServerTools()
    ) {
      const { domains } = extractRequestedDomains(turn);
      if (domains.length > 0) {
        const grantBtn = document.createElement("button");
        grantBtn.type = "button";
        grantBtn.className = "ar-btn ar-btn-warning";
        grantBtn.textContent = `Grant ${domains.join(", ")}`;
        grantBtn.addEventListener("click", () => {
          void sendContinuationMessage(
            turn,
            `Outbound network access granted for ${domains.join(", ")}. Continue from the saved /workspace state without repeating completed work.`,
            domains,
          );
        });
        footer.appendChild(grantBtn);
      }
    }

    const inspectBtn = document.createElement("button");
    inspectBtn.type = "button";
    inspectBtn.className = "ar-btn";
    inspectBtn.textContent = "Open conversation →";
    inspectBtn.addEventListener("click", () => {
      state = {
        ...state,
        selectedSessionId: session.sessionId,
        parallelLayout: "split",
      };
      renderAll();
    });
    footer.appendChild(inspectBtn);

    card.appendChild(footer);
    gridViewEl.appendChild(card);
  });
}

function renderSplitRail(nowMs: number): void {
  const isSplit = state.sessions.length >= 2 && state.parallelLayout === "split";
  railEl.hidden = !isSplit;
  if (!isSplit) {
    return;
  }
  railEl.replaceChildren();
  state.sessions.forEach((session, idx) => {
    const phase = deriveSessionPhase(session);
    const meta = phaseDisplayMeta(phase);
    const latest = latestSessionEntry(session);
    const elapsed = latest ? formatElapsedClock(computeLiveElapsedSeconds(latest, nowMs)) : "";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ar-rail-item";
    btn.setAttribute(
      "aria-selected",
      session.sessionId === state.selectedSessionId ? "true" : "false",
    );

    const top = document.createElement("div");
    top.className = "ar-rail-item-top";
    const label = document.createElement("span");
    label.className = "ar-rail-item-label";
    label.textContent = `#${idx + 1} ${session.label}`;
    top.appendChild(label);
    btn.appendChild(top);

    const bottom = document.createElement("div");
    bottom.className = "ar-rail-item-meta";
    const status = document.createElement("span");
    status.textContent = meta.label;
    bottom.appendChild(status);
    if (elapsed && latest) {
      const clock = document.createElement("span");
      clock.dataset.elapsedTurnId = latest.turn.turn_id;
      clock.textContent = elapsed;
      bottom.appendChild(clock);
    }
    btn.appendChild(bottom);

    btn.addEventListener("click", () => {
      state = { ...state, selectedSessionId: session.sessionId };
      renderAll();
    });
    railEl.appendChild(btn);
  });
}

function buildTurnGroupNode(
  entry: SyncedTurnEntry,
  turnIndex: number,
  isLatestTurn: boolean,
  nowMs: number,
): HTMLElement {
  const { turn } = entry;
  const phase = deriveTurnPhase(turn);
  const group = document.createElement("section");
  group.className = "ar-turn-group";
  group.dataset.turnId = turn.turn_id;

  // 1. User / IDE Prompt Bubble
  if (turn.user_prompt) {
    const userCard = document.createElement("div");
    userCard.className = "ar-user-card";
    const meta = document.createElement("div");
    meta.className = "ar-bubble-meta";
    const role = document.createElement("span");
    role.className = "ar-role-tag";
    role.textContent = turnIndex === 0 ? "IDE → ARIA" : `Follow-up #${turnIndex + 1}`;
    meta.appendChild(role);

    const activeDomains = turn.active_allowed_network_domains ?? [];
    if (activeDomains.length > 0) {
      const allowWrap = document.createElement("span");
      allowWrap.className = "ar-domain-pill";
      allowWrap.textContent = `Allowed egress: ${activeDomains.join(", ")}`;
      meta.appendChild(allowWrap);
    }
    userCard.appendChild(meta);

    const p = document.createElement("p");
    p.className = "ar-user-text";
    p.textContent = turn.user_prompt;
    userCard.appendChild(p);
    group.appendChild(userCard);
  }

  // 2. Live Activity & Tool Trail Card
  const steps = turn.tool_steps ?? [];
  const toolRecords = turn.progress?.tool_call_record_count ?? 0;
  const msgCount = turn.progress?.message_count ?? 0;
  const isActive = !turn.is_terminal;
  const stepsExpanded = state.expandedStepTurnIds.includes(turn.turn_id);

  if (isActive || steps.length > 0 || toolRecords > 0 || turn.progress?.poll_interrupted) {
    const activity = document.createElement("div");
    activity.className = "ar-activity-card";
    activity.dataset.active = isActive ? "true" : "false";

    const top = document.createElement("div");
    top.className = "ar-activity-top";

    const left = document.createElement("div");
    left.className = "ar-activity-status";
    const dot = document.createElement("span");
    dot.className = "ar-live-dot";
    left.appendChild(dot);

    const statusSpan = document.createElement("span");
    const elapsedStr = formatElapsedClock(computeLiveElapsedSeconds(entry, nowMs));
    if (isActive) {
      statusSpan.textContent =
        turn.progress?.status_text || "ARIA is working in the sandbox…";
    } else {
      const approxTools = steps.length || Math.ceil(toolRecords / 2);
      statusSpan.textContent =
        approxTools > 0
          ? `Executed ${approxTools} sandbox/API step${approxTools === 1 ? "" : "s"}${elapsedStr ? ` in ${elapsedStr}` : ""}`
          : `Turn finished${elapsedStr ? ` in ${elapsedStr}` : ""}`;
    }
    left.appendChild(statusSpan);
    top.appendChild(left);

    const right = document.createElement("div");
    right.className = "ar-activity-stats";
    if (isActive && elapsedStr) {
      const clock = document.createElement("span");
      clock.dataset.elapsedTurnId = turn.turn_id;
      clock.textContent = elapsedStr;
      right.appendChild(clock);
    }
    if (msgCount > 0 || toolRecords > 0) {
      const counts = document.createElement("span");
      counts.textContent = `${msgCount} msgs · ${Math.ceil(toolRecords / 2)} tool calls`;
      right.appendChild(counts);
    }
    if (!isActive && steps.length > 0) {
      const toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.className = "ar-btn";
      toggleBtn.textContent = stepsExpanded ? "Hide steps" : "Show steps";
      toggleBtn.addEventListener("click", () => {
        const nextIds = stepsExpanded
          ? state.expandedStepTurnIds.filter((id) => id !== turn.turn_id)
          : [...state.expandedStepTurnIds, turn.turn_id];
        state = { ...state, expandedStepTurnIds: nextIds };
        renderAll();
      });
      right.appendChild(toggleBtn);
    }
    top.appendChild(right);
    activity.appendChild(top);

    if (isActive) {
      const track = document.createElement("div");
      track.className = "ar-progress-track";
      const bar = document.createElement("div");
      bar.className = "ar-progress-bar";
      track.appendChild(bar);
      activity.appendChild(track);
    }

    if (steps.length > 0 && (isActive || stepsExpanded)) {
      const stepsRow = document.createElement("div");
      stepsRow.className = "ar-steps-row";
      for (const step of steps) {
        stepsRow.appendChild(buildStepChip(step));
      }
      activity.appendChild(stepsRow);
    }

    if (turn.progress?.internal_error_note) {
      const note = document.createElement("div");
      note.className = "ar-muted-note";
      note.textContent = turn.progress.internal_error_note;
      activity.appendChild(note);
    }

    group.appendChild(activity);
  }

  // 3. Permission Gate Card (`network_access`)
  if (phase === "awaiting_permission") {
    const { domains, reasons } = extractRequestedDomains(turn);
    const gate = document.createElement("div");
    gate.className = "ar-gate-card";

    const heading = document.createElement("div");
    heading.className = "ar-gate-title";
    heading.textContent = "ARIA paused to request outbound sandbox network access";
    gate.appendChild(heading);

    if (reasons.length > 0) {
      const reasonP = document.createElement("p");
      reasonP.className = "ar-gate-reason";
      reasonP.textContent = reasons.join(" ");
      gate.appendChild(reasonP);
    }

    if (domains.length > 0) {
      const domRow = document.createElement("div");
      domRow.className = "ar-gate-domains";
      for (const dom of domains) {
        const chip = document.createElement("span");
        chip.className = "ar-domain-pill";
        chip.textContent = dom;
        domRow.appendChild(chip);
      }
      gate.appendChild(domRow);
    }

    if (!isLatestTurn || turn.continued_in_turn_id) {
      const resolved = document.createElement("div");
      resolved.className = "ar-gate-resolved";
      resolved.textContent = "Continued in subsequent turn below.";
      gate.appendChild(resolved);
    } else if (state.accessMode === "read-only" || !bridge.canCallServerTools()) {
      const ro = document.createElement("div");
      ro.className = "ar-gate-resolved";
      ro.textContent =
        state.accessMode === "read-only"
          ? "Server is in read-only mode; network grants require read-write access."
          : "Approve or decline via aria_send_message(parent_turn_id=..., allowed_network_domains=...).";
      gate.appendChild(ro);
    } else {
      const actions = document.createElement("div");
      actions.className = "ar-gate-actions";
      if (domains.length > 0) {
        const grantBtn = document.createElement("button");
        grantBtn.type = "button";
        grantBtn.className = "ar-btn ar-btn-warning";
        grantBtn.dataset.action = "grant-network";
        grantBtn.textContent = `Grant access (${domains.join(", ")}) & continue`;
        grantBtn.addEventListener("click", () => {
          void sendContinuationMessage(
            turn,
            `Outbound network access granted for ${domains.join(", ")}. Continue from the saved /workspace state without repeating completed work.`,
            domains,
          );
        });
        actions.appendChild(grantBtn);
      }
      const offlineBtn = document.createElement("button");
      offlineBtn.type = "button";
      offlineBtn.className = "ar-btn";
      offlineBtn.dataset.action = "continue-offline";
      offlineBtn.textContent = "Continue offline";
      offlineBtn.addEventListener("click", () => {
        void sendContinuationMessage(
          turn,
          "Complete the analysis offline using existing /workspace files and W&B APIs without repeating completed work.",
        );
      });
      actions.appendChild(offlineBtn);
      gate.appendChild(actions);
    }
    group.appendChild(gate);
  }

  // 4. Clarifying Question Card (`agent_questions`)
  if (phase === "awaiting_user_input") {
    const questions = extractAgentQuestions(turn);
    for (const q of questions) {
      const gate = document.createElement("div");
      gate.className = "ar-gate-card";

      const heading = document.createElement("div");
      heading.className = "ar-gate-title";
      heading.textContent = "ARIA asked a clarifying question";
      gate.appendChild(heading);

      const qText = document.createElement("p");
      qText.className = "ar-gate-reason";
      qText.textContent = q.question;
      gate.appendChild(qText);

      if (
        isLatestTurn &&
        state.accessMode === "read-write" &&
        bridge.canCallServerTools() &&
        Array.isArray(q.options) &&
        q.options.length > 0
      ) {
        const optRow = document.createElement("div");
        optRow.className = "ar-question-options";
        for (const opt of q.options) {
          const optBtn = document.createElement("button");
          optBtn.type = "button";
          optBtn.className = "ar-btn";
          optBtn.textContent = opt;
          optBtn.addEventListener("click", () => {
            void sendContinuationMessage(turn, opt);
          });
          optRow.appendChild(optBtn);
        }
        gate.appendChild(optRow);
      }
      group.appendChild(gate);
    }
  }

  // 5. Error Card
  if (phase === "errored") {
    const errCard = document.createElement("div");
    errCard.className = "ar-banner";
    errCard.dataset.tone = "error";
    errCard.textContent =
      turn.error?.message ||
      (turn.error_info ? `ARIA stopped with error: ${turn.error_info}` : "ARIA turn errored.");
    group.appendChild(errCard);
  }

  // 6. ARIA Response Card
  if (turn.latest_response && turn.latest_response.trim()) {
    const assistantCard = document.createElement("div");
    assistantCard.className = "ar-assistant-card";

    const meta = document.createElement("div");
    meta.className = "ar-bubble-meta";
    const leftMeta = document.createElement("div");
    leftMeta.style.display = "flex";
    leftMeta.style.alignItems = "center";
    leftMeta.style.gap = "6px";

    const role = document.createElement("span");
    role.className = "ar-role-tag";
    role.textContent = "ARIA";
    leftMeta.appendChild(role);

    if (!turn.is_terminal) {
      const draftBadge = document.createElement("span");
      draftBadge.className = "ar-draft-badge";
      draftBadge.textContent = "LIVE DRAFT";
      leftMeta.appendChild(draftBadge);
    }
    meta.appendChild(leftMeta);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "ar-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => {
      void navigator.clipboard?.writeText(turn.latest_response ?? "").then(() => {
        copyBtn.textContent = "Copied";
        window.setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1200);
      });
    });
    meta.appendChild(copyBtn);
    assistantCard.appendChild(meta);

    const mdBody = document.createElement("div");
    mdBody.className = "ar-md-body";
    mdBody.appendChild(renderMarkdownFragment(turn.latest_response, handleOpenLink));
    assistantCard.appendChild(mdBody);
    group.appendChild(assistantCard);
  }

  return group;
}

function renderMain(): void {
  const nowMs = Date.now();
  if (state.sessions.length === 0) {
    emptyEl.hidden = false;
    gridViewEl.hidden = true;
    workspaceEl.hidden = true;
    emptyEl.textContent =
      state.recentThreads.length > 0
        ? "Select an ARIA conversation above to inspect its turns."
        : "Waiting for ARIA turn…";
    return;
  }

  emptyEl.hidden = true;
  const isParallelGrid = state.sessions.length >= 2 && state.parallelLayout === "grid";

  if (isParallelGrid) {
    gridViewEl.hidden = false;
    workspaceEl.hidden = true;
    renderParallelGrid(nowMs);
    return;
  }

  gridViewEl.hidden = true;
  workspaceEl.hidden = false;
  renderSplitRail(nowMs);

  const active = selectedSession();
  turnStreamEl.replaceChildren();
  if (!active) {
    composerEl.hidden = true;
    return;
  }

  active.turns.forEach((entry, idx) => {
    const isLatest = idx === active.turns.length - 1;
    turnStreamEl.appendChild(buildTurnGroupNode(entry, idx, isLatest, nowMs));
  });

  const latest = latestSessionEntry(active);
  const canReply =
    state.accessMode === "read-write" &&
    bridge.canCallServerTools() &&
    Boolean(latest && !latest.optimistic && latest.turn.state === "completed");
  composerEl.hidden = !canReply;
  if (canReply && latest) {
    const phase = deriveTurnPhase(latest.turn);
    replyInput.placeholder =
      phase === "awaiting_user_input"
        ? "Type your answer to ARIA… (Enter to send)"
        : "Ask ARIA a follow-up in this conversation… (Enter to send, Shift+Enter for newline)";
  }
}

function renderAll(): void {
  renderHeader();
  renderFleetBar();
  renderDrawers();
  renderBanner();
  renderMain();
}

function tickLiveClocks(): void {
  if (state.sessions.length === 0) {
    return;
  }
  const nowMs = Date.now();
  const entryByTurnId = new Map<string, SyncedTurnEntry>();
  for (const session of state.sessions) {
    for (const entry of session.turns) {
      if (!entry.turn.is_terminal) {
        entryByTurnId.set(entry.turn.turn_id, entry);
      }
    }
  }
  if (entryByTurnId.size === 0) {
    return;
  }
  const nodes = document.querySelectorAll<HTMLElement>("[data-elapsed-turn-id]");
  nodes.forEach((el) => {
    const tid = el.dataset.elapsedTurnId;
    if (!tid) {
      return;
    }
    const entry = entryByTurnId.get(tid);
    if (!entry) {
      return;
    }
    const formatted = formatElapsedClock(computeLiveElapsedSeconds(entry, nowMs));
    if (formatted) {
      el.textContent = el.closest(".ar-phase-pill") ? `· ${formatted}` : formatted;
    }
  });
}

const bridge = new AriaHostBridge({
  onToolInput(args) {
    if (!args) {
      return;
    }
    if (typeof args.message === "string" && args.message.trim()) {
      state = applyOptimisticSendInput(state, args, Date.now());
      renderAll();
    }
  },
  onToolEnvelope(envelope) {
    handleEnvelope(envelope, true);
  },
  onHostContext(displayMode, availableDisplayModes) {
    state = {
      ...state,
      displayMode: displayMode ?? state.displayMode,
      availableDisplayModes: availableDisplayModes ?? state.availableDisplayModes,
    };
    renderHeader();
  },
  onTeardown() {
    clearPollTimer();
    if (clockInterval !== null) {
      window.clearInterval(clockInterval);
      clockInterval = null;
    }
  },
});

// Wire static DOM controls
autoPollBtn.addEventListener("click", () => {
  state = { ...state, autoPoll: !state.autoPoll };
  renderHeader();
  scheduleNextPollOrHydration();
});

pollNowBtn.addEventListener("click", () => {
  const activeIds = getPollableTurnIds(state);
  if (activeIds.length > 0) {
    void pollSpecificTurnIds(activeIds);
    return;
  }
  const current = selectedSession();
  const latest = current ? latestSessionEntry(current) : null;
  if (latest && !latest.optimistic) {
    void pollSpecificTurnIds([latest.turn.turn_id]);
  }
});

spawnToggleBtn.addEventListener("click", () => {
  state = {
    ...state,
    spawnDrawerOpen: !state.spawnDrawerOpen,
    threadsDrawerOpen: false,
  };
  renderDrawers();
});

spawnCloseBtn.addEventListener("click", () => {
  state = { ...state, spawnDrawerOpen: false };
  renderDrawers();
});

spawnPromptsInput.addEventListener("input", updateSpawnHint);
spawnSubmitBtn.addEventListener("click", () => {
  void launchParallelSubAgents();
});

threadsToggleBtn.addEventListener("click", () => {
  const willOpen = !state.threadsDrawerOpen;
  state = {
    ...state,
    threadsDrawerOpen: willOpen,
    spawnDrawerOpen: false,
  };
  renderDrawers();
  if (willOpen && bridge.canCallServerTools()) {
    const active = selectedSession();
    const args: Record<string, unknown> = { limit: 15 };
    if (active?.scope.entity) {
      args.entity = active.scope.entity;
    }
    if (active?.scope.project) {
      args.project = active.scope.project;
    }
    void bridge.callAriaTool("aria_list_threads", args).then((env) => handleEnvelope(env));
  }
});

threadsCloseBtn.addEventListener("click", () => {
  state = { ...state, threadsDrawerOpen: false };
  renderDrawers();
});

expandBtn.addEventListener("click", () => {
  const target = state.displayMode === "fullscreen" ? "inline" : "fullscreen";
  void bridge.requestDisplayMode(target).then((res) => {
    if (res.ok) {
      state = { ...state, displayMode: res.mode };
      renderHeader();
    } else {
      showBanner(res.message, "info");
    }
  });
});

openForgeBtn.addEventListener("click", () => {
  const isParallel = state.sessions.length >= 2;
  const active = selectedSession();
  const target = (!isParallel ? active?.forgeUrl : null) || state.forgeHomeUrl;
  if (target) {
    handleOpenLink(target);
  }
});

layoutGridBtn.addEventListener("click", () => {
  state = { ...state, parallelLayout: "grid" };
  renderAll();
});

layoutSplitBtn.addEventListener("click", () => {
  state = { ...state, parallelLayout: "split" };
  renderAll();
});

bannerDismissBtn.addEventListener("click", () => {
  state = { ...state, bannerMessage: null };
  renderBanner();
});

replySendBtn.addEventListener("click", () => {
  const current = selectedSession();
  const latest = current ? latestSessionEntry(current) : null;
  if (!latest || latest.optimistic) {
    return;
  }
  const text = replyInput.value.trim();
  if (!text) {
    return;
  }
  replyInput.value = "";
  void sendContinuationMessage(latest.turn, text);
});

replyInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    replySendBtn.click();
  }
});

clockInterval = window.setInterval(tickLiveClocks, 1000);
updateSpawnHint();
renderAll();
void bridge.connect();
