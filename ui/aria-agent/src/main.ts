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
  countRecoveredErrors,
  countTurnSteps,
  createInitialState,
  deriveSessionPhase,
  deriveTurnPhase,
  extractAgentQuestions,
  extractRequestedDomains,
  formatElapsedClock,
  formatLiveStepLine,
  getHydratableCompletedTurnIds,
  getPollableTurnIds,
  isLongInlineAnswer,
  latestSessionEntry,
  phaseDisplayMeta,
  summarizeBatch,
  upsertTurnIntoState,
  type AriaAgentSession,
  type AriaCardState,
  type SyncedTurnEntry,
} from "./state";

const appRoot = document.getElementById("app-root") as HTMLDivElement;
const backBtn = document.getElementById("ar-back-btn") as HTMLButtonElement;
const titleEl = document.getElementById("ar-title") as HTMLHeadingElement;
const scopePillEl = document.getElementById("ar-scope-pill") as HTMLSpanElement;
const phasePillEl = document.getElementById("ar-phase-pill") as HTMLSpanElement;

const refreshLinkBtn = document.getElementById("ar-refresh-link") as HTMLButtonElement;
const expandBtn = document.getElementById("ar-expand-btn") as HTMLButtonElement;
const menuWrapEl = document.getElementById("ar-menu-wrap") as HTMLDivElement;
const menuBtn = document.getElementById("ar-menu-btn") as HTMLButtonElement;
const menuPopoverEl = document.getElementById("ar-menu-popover") as HTMLDivElement;
const menuThreadsBtn = document.getElementById("ar-menu-threads") as HTMLButtonElement;
const menuLiveBtn = document.getElementById("ar-menu-live") as HTMLButtonElement;
const menuForgeBtn = document.getElementById("ar-menu-forge") as HTMLButtonElement;

const bannerEl = document.getElementById("ar-banner") as HTMLDivElement;
const bannerTextEl = document.getElementById("ar-banner-text") as HTMLSpanElement;
const bannerDismissBtn = document.getElementById("ar-banner-dismiss") as HTMLButtonElement;

const emptyEl = document.getElementById("ar-empty") as HTMLDivElement;
const runViewEl = document.getElementById("ar-run-view") as HTMLElement;
const earlierWrapEl = document.getElementById("ar-earlier-wrap") as HTMLDivElement;
const latestTurnEl = document.getElementById("ar-latest-turn") as HTMLDivElement;
const readonlyLineEl = document.getElementById("ar-readonly-line") as HTMLDivElement;
const composerEl = document.getElementById("ar-composer") as HTMLDivElement;
const replyInput = document.getElementById("ar-reply-input") as HTMLTextAreaElement;
const replySendBtn = document.getElementById("ar-reply-send-btn") as HTMLButtonElement;

const runListViewEl = document.getElementById("ar-run-list-view") as HTMLElement;
const runListSummaryEl = document.getElementById("ar-run-list-summary") as HTMLDivElement;
const runListRowsEl = document.getElementById("ar-run-list-rows") as HTMLDivElement;

const threadListViewEl = document.getElementById("ar-thread-list-view") as HTMLElement;
const threadListSummaryEl = document.getElementById("ar-thread-list-summary") as HTMLDivElement;
const threadListRowsEl = document.getElementById("ar-thread-list-rows") as HTMLDivElement;

let state: AriaCardState = createInitialState();
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

function closeMenu(restoreFocus = false): void {
  if (!state.menuOpen) {
    return;
  }
  state = { ...state, menuOpen: false };
  renderHeader();
  if (restoreFocus) {
    menuBtn.focus();
  }
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

function handleEnvelope(
  envelope: ParsedAriaToolEnvelope,
  options?: { immediateSync?: boolean; fromBackgroundPoll?: boolean },
): void {
  const now = Date.now();
  if (envelope.kind === "error") {
    if (options?.fromBackgroundPoll) {
      state = { ...state, pollFailed: true };
      renderHeader();
      return;
    }
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

  renderAll();
  bridge.scheduleModelContextSync(buildModelContextPayload(state), Boolean(options?.immediateSync));
  scheduleNextPollOrHydration();
}

async function pollSpecificTurnIds(turnIds: string[], manual = false): Promise<void> {
  if (turnIds.length === 0 || state.pollInFlight || !bridge.canCallServerTools()) {
    return;
  }
  clearPollTimer();
  state = { ...state, pollInFlight: true, ...(manual ? { pollFailed: false } : {}) };
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
  handleEnvelope(envelope, { fromBackgroundPoll: !manual });
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

  if (state.liveUpdatesPaused || state.pollFailed) {
    return;
  }

  const pollable = getPollableTurnIds(state);
  if (pollable.length === 0) {
    return;
  }

  let delaySec = 2.2;
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
  handleEnvelope(envelope, { immediateSync: true });
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

  if (meta.tone === "active") {
    const dot = document.createElement("span");
    dot.className = "ar-live-dot";
    pill.appendChild(dot);
  }

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
  const active = selectedSession();
  const pollable = getPollableTurnIds(state);
  const canTools = bridge.canCallServerTools();

  // Back link when navigating to/from Recent threads
  if (state.view === "threadList" && state.previousView && state.sessions.length > 0) {
    backBtn.hidden = false;
    backBtn.textContent = "← Back";
  } else if (state.view === "run" && state.cameFromThreadList && state.recentThreads.length > 0) {
    backBtn.hidden = false;
    backBtn.textContent = "← Threads";
  } else {
    backBtn.hidden = true;
  }

  if (state.view === "threadList") {
    titleEl.textContent = "Recent conversations";
    titleEl.title = "Recent conversations";
    const sampleScope = active?.scope ?? state.recentThreads[0]?.scope;
    const scopeStr = sampleScope
      ? [sampleScope.entity, sampleScope.project].filter(Boolean).join("/")
      : "";
    scopePillEl.hidden = !scopeStr;
    scopePillEl.textContent = scopeStr;
    phasePillEl.hidden = false;
    phasePillEl.dataset.tone = "muted";
    phasePillEl.replaceChildren();
    phasePillEl.textContent = `${state.recentThreads.length}`;
  } else if (state.view === "runList" && state.sessions.length > 0) {
    const summary = summarizeBatch(state.sessions);
    titleEl.textContent = `${summary.total} ARIA runs`;
    titleEl.title = `${summary.total} ARIA runs`;
    const sharedScope = active
      ? [active.scope.entity, active.scope.project].filter(Boolean).join("/")
      : "";
    scopePillEl.hidden = !sharedScope;
    scopePillEl.textContent = sharedScope;

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
    if (summary.active > 0) {
      const dot = document.createElement("span");
      dot.className = "ar-live-dot";
      phasePillEl.appendChild(dot);
    }
    const text = document.createElement("span");
    text.textContent =
      summary.active > 0
        ? `${summary.active} running · ${summary.completed}/${summary.total} done`
        : summary.awaitingPermission + summary.awaitingInput > 0
          ? `${summary.awaitingPermission + summary.awaitingInput} waiting`
          : `${summary.completed}/${summary.total} done`;
    phasePillEl.appendChild(text);
  } else if (active) {
    titleEl.textContent = active.label;
    titleEl.title = active.label;
    const scopeStr = [active.scope.entity, active.scope.project].filter(Boolean).join("/");
    scopePillEl.hidden = !scopeStr;
    scopePillEl.textContent = scopeStr;
    phasePillEl.hidden = false;
    buildPhasePillContent(phasePillEl, active, nowMs);
  } else {
    titleEl.textContent = "Waiting for ARIA…";
    titleEl.title = "Waiting for ARIA…";
    scopePillEl.hidden = true;
    phasePillEl.hidden = true;
  }

  // Silent polling: Refresh appears ONLY if a poll failed, live updates were paused while active,
  // or the host cannot call server tools on an in-flight turn.
  const needsManualRefresh =
    state.sessions.length > 0 &&
    (state.pollFailed || (pollable.length > 0 && (state.liveUpdatesPaused || !canTools)));
  refreshLinkBtn.hidden = !needsManualRefresh;
  if (needsManualRefresh) {
    if (!canTools) {
      refreshLinkBtn.disabled = true;
      refreshLinkBtn.textContent = "Static snapshot";
    } else if (state.pollInFlight) {
      refreshLinkBtn.disabled = true;
      refreshLinkBtn.textContent = "Refreshing…";
    } else {
      refreshLinkBtn.disabled = false;
      refreshLinkBtn.textContent = state.pollFailed ? "Retry refresh" : "Refresh";
    }
  }

  const canFullscreen = state.availableDisplayModes.includes("fullscreen");
  expandBtn.hidden = !canFullscreen;
  expandBtn.textContent = state.displayMode === "fullscreen" ? "Inline" : "Expand";

  // Overflow menu
  const targetForgeUrl =
    (state.view === "run" ? active?.forgeUrl : null) || state.forgeHomeUrl;
  menuThreadsBtn.hidden = !canTools;
  menuLiveBtn.hidden = !canTools;
  menuLiveBtn.textContent = state.liveUpdatesPaused
    ? "Resume live updates"
    : "Pause live updates";
  menuForgeBtn.hidden = !targetForgeUrl;

  const hasAnyMenuAction = !menuThreadsBtn.hidden || !menuLiveBtn.hidden || !menuForgeBtn.hidden;
  menuWrapEl.hidden = !hasAnyMenuAction;
  menuBtn.setAttribute("aria-expanded", state.menuOpen ? "true" : "false");
  menuPopoverEl.hidden = !state.menuOpen || !hasAnyMenuAction;
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

function buildGateOrErrorNode(
  turn: AriaTurnPayload,
  isLatestTurn: boolean,
): HTMLElement | null {
  const phase = deriveTurnPhase(turn);
  const canAct =
    isLatestTurn &&
    !turn.continued_in_turn_id &&
    state.accessMode === "read-write" &&
    bridge.canCallServerTools();

  if (phase === "awaiting_permission") {
    const { domains, reasons } = extractRequestedDomains(turn);
    const gate = document.createElement("div");
    gate.className = "ar-gate-block";

    const head = document.createElement("div");
    head.className = "ar-gate-head";
    const title = document.createElement("span");
    title.className = "ar-gate-title";
    title.textContent = "Outbound network access requested";
    head.appendChild(title);

    for (const dom of domains) {
      const chip = document.createElement("span");
      chip.className = "ar-domain-pill";
      chip.textContent = dom;
      head.appendChild(chip);
    }
    gate.appendChild(head);

    if (reasons.length > 0) {
      const reasonP = document.createElement("p");
      reasonP.className = "ar-gate-reason";
      reasonP.textContent = reasons.join(" ");
      gate.appendChild(reasonP);
    }

    if (!isLatestTurn || turn.continued_in_turn_id) {
      const note = document.createElement("div");
      note.className = "ar-muted-note";
      note.textContent = "Resolved in follow-up turn.";
      gate.appendChild(note);
    } else if (canAct) {
      const actions = document.createElement("div");
      actions.className = "ar-gate-actions";
      if (domains.length > 0) {
        const grantBtn = document.createElement("button");
        grantBtn.type = "button";
        grantBtn.className = "ar-btn ar-btn-warning";
        grantBtn.dataset.action = "grant-network";
        grantBtn.textContent = `Grant ${domains.join(", ")}`;
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
    } else if (state.accessMode !== "read-only" && !bridge.canCallServerTools()) {
      const note = document.createElement("div");
      note.className = "ar-muted-note";
      note.textContent =
        "Resume via aria_send_message(parent_turn_id=..., allowed_network_domains=...).";
      gate.appendChild(note);
    }
    return gate;
  }

  if (phase === "awaiting_user_input") {
    const questions = extractAgentQuestions(turn);
    if (questions.length === 0) {
      return null;
    }
    const wrap = document.createElement("div");
    wrap.className = "ar-gate-stack";
    for (const q of questions) {
      const gate = document.createElement("div");
      gate.className = "ar-gate-block";

      const title = document.createElement("div");
      title.className = "ar-gate-title";
      title.textContent = "Clarifying question";
      gate.appendChild(title);

      const qText = document.createElement("p");
      qText.className = "ar-gate-reason";
      qText.textContent = q.question;
      gate.appendChild(qText);

      if (canAct && Array.isArray(q.options) && q.options.length > 0) {
        const optRow = document.createElement("div");
        optRow.className = "ar-gate-actions";
        for (const opt of q.options) {
          const optBtn = document.createElement("button");
          optBtn.type = "button";
          optBtn.className = "ar-btn";
          optBtn.dataset.action = "answer-question";
          optBtn.textContent = opt;
          optBtn.addEventListener("click", () => {
            void sendContinuationMessage(turn, opt);
          });
          optRow.appendChild(optBtn);
        }
        gate.appendChild(optRow);
      }
      wrap.appendChild(gate);
    }
    return wrap;
  }

  if (phase === "errored") {
    const errBlock = document.createElement("div");
    errBlock.className = "ar-error-block";
    errBlock.textContent =
      turn.error?.message ||
      (turn.error_info ? `ARIA stopped with error: ${turn.error_info}` : "ARIA turn errored.");
    return errBlock;
  }

  return null;
}

function buildLatestTurnNode(
  session: AriaAgentSession,
  entry: SyncedTurnEntry,
  nowMs: number,
): HTMLElement {
  const { turn } = entry;
  const wrap = document.createElement("div");
  wrap.className = "ar-turn-body";
  wrap.dataset.turnId = turn.turn_id;

  const isRunning = !turn.is_terminal;
  const steps = turn.tool_steps ?? [];
  const stepCount = countTurnSteps(turn);
  const recoveredCount = countRecoveredErrors(turn);
  const stepsExpanded = state.expandedStepTurnIds.includes(turn.turn_id);
  const hasAnswer = Boolean(turn.latest_response && turn.latest_response.trim());

  if (isRunning) {
    // 1. Prompt clamped to two lines
    if (turn.user_prompt) {
      const promptBlock = document.createElement("div");
      promptBlock.className = "ar-running-prompt";
      const pText = document.createElement("p");
      pText.className = "ar-prompt-clamp";
      pText.textContent = turn.user_prompt;
      promptBlock.appendChild(pText);
      wrap.appendChild(promptBlock);
    }

    // 2. One live step line: e.g. "Running shell, 3 steps, 37s"
    const liveWrap = document.createElement("div");
    liveWrap.className = "ar-live-block";

    const liveRow = document.createElement("div");
    liveRow.className = "ar-live-line";

    const liveLeft = document.createElement("div");
    liveLeft.className = "ar-live-left";
    const dot = document.createElement("span");
    dot.className = "ar-live-dot ar-live-dot-active";
    liveLeft.appendChild(dot);

    const lineText = document.createElement("span");
    lineText.className = "ar-live-text";
    lineText.dataset.liveStepTurnId = turn.turn_id;
    lineText.textContent = formatLiveStepLine(entry, nowMs);
    liveLeft.appendChild(lineText);

    if (recoveredCount > 0) {
      const retryBadge = document.createElement("span");
      retryBadge.className = "ar-retry-tag";
      retryBadge.textContent =
        recoveredCount === 1 ? "1 retry recovered" : `${recoveredCount} retries recovered`;
      liveLeft.appendChild(retryBadge);
    }
    liveRow.appendChild(liveLeft);

    if (steps.length > 0) {
      const toggleStepsBtn = document.createElement("button");
      toggleStepsBtn.type = "button";
      toggleStepsBtn.className = "ar-link-btn";
      toggleStepsBtn.dataset.action = "toggle-steps";
      toggleStepsBtn.textContent = stepsExpanded ? "Hide steps" : "Show steps";
      toggleStepsBtn.addEventListener("click", () => {
        const nextIds = stepsExpanded
          ? state.expandedStepTurnIds.filter((id) => id !== turn.turn_id)
          : [...state.expandedStepTurnIds, turn.turn_id];
        state = { ...state, expandedStepTurnIds: nextIds };
        renderAll();
      });
      liveRow.appendChild(toggleStepsBtn);
    }
    liveWrap.appendChild(liveRow);

    const track = document.createElement("div");
    track.className = "ar-progress-track";
    const bar = document.createElement("div");
    bar.className = "ar-progress-bar";
    track.appendChild(bar);
    liveWrap.appendChild(track);

    if (steps.length > 0 && stepsExpanded) {
      const stepsRow = document.createElement("div");
      stepsRow.className = "ar-steps-row";
      for (const step of steps) {
        stepsRow.appendChild(buildStepChip(step));
      }
      liveWrap.appendChild(stepsRow);
    }

    wrap.appendChild(liveWrap);

    // 3. Streaming draft answer if present
    if (hasAnswer) {
      const draftWrap = document.createElement("div");
      draftWrap.className = "ar-draft-answer";
      draftWrap.appendChild(renderMarkdownFragment(turn.latest_response!, handleOpenLink));
      wrap.appendChild(draftWrap);
    }

    return wrap;
  }

  // Completed / terminal turn:
  // Show a quiet 1-line prompt cue on follow-up turns (since earlier turns are collapsed)
  // or when no answer text arrived yet (e.g. paused on a gate).
  if (turn.user_prompt && (session.turns.length > 1 || !hasAnswer)) {
    const cue = document.createElement("p");
    cue.className = hasAnswer ? "ar-followup-cue" : "ar-prompt-clamp";
    cue.textContent = turn.user_prompt;
    wrap.appendChild(cue);
  }

  const gateOrErr = buildGateOrErrorNode(turn, true);
  if (gateOrErr) {
    wrap.appendChild(gateOrErr);
  }

  if (hasAnswer) {
    const isClamped =
      state.displayMode !== "fullscreen" &&
      isLongInlineAnswer(turn.latest_response) &&
      !state.expandedAnswerTurnIds.includes(turn.turn_id);
    const isExpandedLong =
      state.displayMode !== "fullscreen" &&
      isLongInlineAnswer(turn.latest_response) &&
      state.expandedAnswerTurnIds.includes(turn.turn_id);

    const answerOuter = document.createElement("div");
    answerOuter.className = "ar-answer-outer";

    const answerBody = document.createElement("div");
    answerBody.className = "ar-md-body ar-answer-body";
    if (isClamped) {
      answerBody.dataset.clamped = "true";
    }
    answerBody.appendChild(renderMarkdownFragment(turn.latest_response!, handleOpenLink));
    answerOuter.appendChild(answerBody);

    if (isClamped || isExpandedLong) {
      const moreBtn = document.createElement("button");
      moreBtn.type = "button";
      moreBtn.className = "ar-link-btn ar-show-more-btn";
      moreBtn.dataset.action = "toggle-answer";
      moreBtn.textContent = isClamped ? "Show more" : "Show less";
      moreBtn.addEventListener("click", () => {
        const nextIds = isClamped
          ? [...state.expandedAnswerTurnIds, turn.turn_id]
          : state.expandedAnswerTurnIds.filter((id) => id !== turn.turn_id);
        state = { ...state, expandedAnswerTurnIds: nextIds };
        renderAll();
      });
      answerOuter.appendChild(moreBtn);
    }

    wrap.appendChild(answerOuter);
  }

  // One footer line: step count, elapsed time, Show steps, and Copy
  const elapsedStr = formatElapsedClock(computeLiveElapsedSeconds(entry, nowMs));
  const activeDomains = turn.active_allowed_network_domains ?? [];
  const hasFooterContent =
    stepCount > 0 ||
    Boolean(elapsedStr) ||
    steps.length > 0 ||
    hasAnswer ||
    recoveredCount > 0 ||
    activeDomains.length > 0;

  if (hasFooterContent) {
    const footer = document.createElement("div");
    footer.className = "ar-turn-footer";

    const left = document.createElement("div");
    left.className = "ar-footer-left";

    const metaParts: string[] = [];
    if (stepCount > 0) {
      metaParts.push(`${stepCount} ${stepCount === 1 ? "step" : "steps"}`);
    }
    if (elapsedStr) {
      metaParts.push(elapsedStr);
    }
    if (recoveredCount > 0) {
      metaParts.push(
        recoveredCount === 1 ? "1 retry recovered" : `${recoveredCount} retries recovered`,
      );
    }
    if (activeDomains.length > 0) {
      metaParts.push(`allowed ${activeDomains.join(", ")}`);
    }

    if (metaParts.length > 0) {
      const metaSpan = document.createElement("span");
      metaSpan.className = "ar-footer-meta";
      metaSpan.textContent = metaParts.join(" · ");
      left.appendChild(metaSpan);
    }

    if (steps.length > 0) {
      const stepsBtn = document.createElement("button");
      stepsBtn.type = "button";
      stepsBtn.className = "ar-link-btn";
      stepsBtn.dataset.action = "toggle-steps";
      stepsBtn.textContent = stepsExpanded ? "Hide steps" : "Show steps";
      stepsBtn.addEventListener("click", () => {
        const nextIds = stepsExpanded
          ? state.expandedStepTurnIds.filter((id) => id !== turn.turn_id)
          : [...state.expandedStepTurnIds, turn.turn_id];
        state = { ...state, expandedStepTurnIds: nextIds };
        renderAll();
      });
      left.appendChild(stepsBtn);
    }
    footer.appendChild(left);

    if (hasAnswer) {
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "ar-link-btn";
      copyBtn.dataset.action = "copy-answer";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", () => {
        void navigator.clipboard?.writeText(turn.latest_response ?? "").then(() => {
          copyBtn.textContent = "Copied";
          window.setTimeout(() => {
            copyBtn.textContent = "Copy";
          }, 1200);
        });
      });
      footer.appendChild(copyBtn);
    }

    wrap.appendChild(footer);

    if (steps.length > 0 && stepsExpanded) {
      const stepsRow = document.createElement("div");
      stepsRow.className = "ar-steps-row";
      for (const step of steps) {
        stepsRow.appendChild(buildStepChip(step));
      }
      wrap.appendChild(stepsRow);
    }
  }

  return wrap;
}

function buildEarlierTurnsNode(
  session: AriaAgentSession,
  earlierEntries: SyncedTurnEntry[],
  nowMs: number,
): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "ar-earlier-section";
  const isOpen = state.expandedEarlierSessionIds.includes(session.sessionId);
  const count = earlierEntries.length;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "ar-earlier-toggle";
  toggle.dataset.action = "toggle-earlier";
  toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
  toggle.textContent = `${isOpen ? "▾" : "▸"} ${count} earlier ${count === 1 ? "turn" : "turns"}`;
  toggle.addEventListener("click", () => {
    const nextIds = isOpen
      ? state.expandedEarlierSessionIds.filter((id) => id !== session.sessionId)
      : [...state.expandedEarlierSessionIds, session.sessionId];
    state = { ...state, expandedEarlierSessionIds: nextIds };
    renderAll();
  });
  wrap.appendChild(toggle);

  if (isOpen) {
    const list = document.createElement("div");
    list.className = "ar-earlier-list";
    earlierEntries.forEach((entry, idx) => {
      const item = document.createElement("div");
      item.className = "ar-earlier-item";
      item.dataset.turnId = entry.turn.turn_id;

      if (entry.turn.user_prompt) {
        const promptP = document.createElement("p");
        promptP.className = "ar-earlier-prompt";
        promptP.textContent = entry.turn.user_prompt;
        item.appendChild(promptP);
      }

      if (entry.turn.latest_response && entry.turn.latest_response.trim()) {
        const md = document.createElement("div");
        md.className = "ar-md-body ar-earlier-answer";
        md.appendChild(renderMarkdownFragment(entry.turn.latest_response, handleOpenLink));
        item.appendChild(md);
      } else {
        const phase = deriveTurnPhase(entry.turn);
        const note = document.createElement("div");
        note.className = "ar-muted-note";
        if (phase === "awaiting_permission") {
          const { domains } = extractRequestedDomains(entry.turn);
          note.textContent = domains.length
            ? `Paused for network access (${domains.join(", ")}) — continued in turn #${idx + 2}.`
            : `Paused for network access — continued in turn #${idx + 2}.`;
        } else if (phase === "awaiting_user_input") {
          note.textContent = `Asked clarifying question — answered in turn #${idx + 2}.`;
        } else {
          const elapsed = formatElapsedClock(computeLiveElapsedSeconds(entry, nowMs));
          note.textContent = `Completed${elapsed ? ` in ${elapsed}` : ""}.`;
        }
        item.appendChild(note);
      }

      list.appendChild(item);
    });
    wrap.appendChild(list);
  }

  return wrap;
}

function renderRunView(nowMs: number): void {
  const active = selectedSession();
  if (!active || active.turns.length === 0) {
    runViewEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = "Waiting for ARIA…";
    return;
  }

  emptyEl.hidden = true;
  runViewEl.hidden = false;

  const earlier = active.turns.slice(0, -1);
  if (earlier.length > 0) {
    earlierWrapEl.hidden = false;
    earlierWrapEl.replaceChildren(buildEarlierTurnsNode(active, earlier, nowMs));
  } else {
    earlierWrapEl.hidden = true;
    earlierWrapEl.replaceChildren();
  }

  const latest = active.turns[active.turns.length - 1]!;
  latestTurnEl.replaceChildren(buildLatestTurnNode(active, latest, nowMs));

  const latestPhase = deriveTurnPhase(latest.turn);
  if (state.accessMode === "read-only") {
    readonlyLineEl.hidden = false;
    readonlyLineEl.textContent =
      "Read-only mode — replies and network approvals are disabled.";
    composerEl.hidden = true;
  } else {
    readonlyLineEl.hidden = true;
    const canReply =
      bridge.canCallServerTools() &&
      !latest.optimistic &&
      (latestPhase === "completed" || latestPhase === "awaiting_user_input");
    composerEl.hidden = !canReply;
    if (canReply) {
      replyInput.placeholder =
        latestPhase === "awaiting_user_input"
          ? "Type your answer… (Enter to send)"
          : "Ask a follow-up… (Enter to send)";
    }
  }
}

function renderRunListView(nowMs: number): void {
  const summary = summarizeBatch(state.sessions);
  const summaryParts: string[] = [`${summary.total} runs`];
  if (summary.active > 0) {
    summaryParts.push(`${summary.active} running`);
  }
  if (summary.awaitingPermission > 0) {
    summaryParts.push(`${summary.awaitingPermission} needs network access`);
  }
  if (summary.awaitingInput > 0) {
    summaryParts.push(`${summary.awaitingInput} needs answer`);
  }
  summaryParts.push(`${summary.completed} done`);
  if (summary.errored > 0) {
    summaryParts.push(`${summary.errored} errored`);
  }
  runListSummaryEl.textContent = summaryParts.join(" · ");

  runListRowsEl.replaceChildren();
  for (const session of state.sessions) {
    const latest = latestSessionEntry(session);
    const turn = latest?.turn ?? null;
    const phase = deriveSessionPhase(session);
    const meta = phaseDisplayMeta(phase);
    const isExpanded = state.expandedBatchSessionId === session.sessionId;

    const row = document.createElement("div");
    row.className = "ar-list-row";
    row.dataset.sessionId = session.sessionId;
    row.dataset.expanded = isExpanded ? "true" : "false";

    const headBtn = document.createElement("button");
    headBtn.type = "button";
    headBtn.className = "ar-list-row-btn";
    headBtn.setAttribute("aria-expanded", isExpanded ? "true" : "false");

    const left = document.createElement("div");
    left.className = "ar-list-row-left";

    const chevron = document.createElement("span");
    chevron.className = "ar-chevron";
    chevron.textContent = isExpanded ? "▾" : "▸";
    left.appendChild(chevron);

    const title = document.createElement("span");
    title.className = "ar-list-row-title";
    title.textContent = session.label;
    left.appendChild(title);

    if (!isExpanded && turn) {
      const hint = document.createElement("span");
      hint.className = "ar-list-row-hint";
      if (!turn.is_terminal) {
        const runningStep =
          turn.tool_steps?.[turn.tool_steps.length - 1]?.name ??
          turn.progress?.latest_tool_activity?.name;
        hint.textContent = runningStep ? `Running ${runningStep}` : "Working";
      } else if (phase === "awaiting_permission") {
        const { domains } = extractRequestedDomains(turn);
        hint.textContent = domains.join(", ");
      } else if (phase === "errored") {
        hint.textContent = turn.error?.message || turn.error_info || "";
      }
      if (hint.textContent) {
        left.appendChild(hint);
      }
    }
    headBtn.appendChild(left);

    const right = document.createElement("div");
    right.className = "ar-list-row-right";

    const badge = document.createElement("span");
    badge.className = "ar-status-tag";
    badge.dataset.tone = meta.tone;
    badge.textContent = meta.label;
    right.appendChild(badge);

    if (latest) {
      const elapsed = formatElapsedClock(computeLiveElapsedSeconds(latest, nowMs));
      if (elapsed) {
        const clock = document.createElement("span");
        clock.className = "ar-row-clock";
        clock.dataset.elapsedTurnId = latest.turn.turn_id;
        clock.textContent = elapsed;
        right.appendChild(clock);
      }
    }
    headBtn.appendChild(right);

    headBtn.addEventListener("click", () => {
      state = {
        ...state,
        selectedSessionId: session.sessionId,
        expandedBatchSessionId: isExpanded ? null : session.sessionId,
      };
      renderAll();
    });
    row.appendChild(headBtn);

    if (isExpanded && latest) {
      const detail = document.createElement("div");
      detail.className = "ar-list-row-detail";

      const earlier = session.turns.slice(0, -1);
      if (earlier.length > 0) {
        detail.appendChild(buildEarlierTurnsNode(session, earlier, nowMs));
      }
      detail.appendChild(buildLatestTurnNode(session, latest, nowMs));

      const latestPhase = deriveTurnPhase(latest.turn);
      if (state.accessMode === "read-only") {
        const ro = document.createElement("div");
        ro.className = "ar-readonly-line";
        ro.textContent =
          "Read-only mode — replies and network approvals are disabled.";
        detail.appendChild(ro);
      } else if (
        bridge.canCallServerTools() &&
        !latest.optimistic &&
        (latestPhase === "completed" || latestPhase === "awaiting_user_input")
      ) {
        const inlineComposer = document.createElement("div");
        inlineComposer.className = "ar-composer";
        const input = document.createElement("textarea");
        input.className = "ar-reply-input";
        input.rows = 1;
        input.placeholder =
          latestPhase === "awaiting_user_input"
            ? "Type your answer… (Enter to send)"
            : "Ask a follow-up… (Enter to send)";
        input.setAttribute("aria-label", `Follow-up message for ${session.label}`);

        const sendBtn = document.createElement("button");
        sendBtn.type = "button";
        sendBtn.className = "ar-btn ar-btn-primary";
        sendBtn.textContent = "Send";

        const submitInline = () => {
          const text = input.value.trim();
          if (!text) {
            return;
          }
          input.value = "";
          void sendContinuationMessage(latest.turn, text);
        };
        sendBtn.addEventListener("click", submitInline);
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            submitInline();
          }
        });
        inlineComposer.appendChild(input);
        inlineComposer.appendChild(sendBtn);
        detail.appendChild(inlineComposer);
      }

      row.appendChild(detail);
    }

    runListRowsEl.appendChild(row);
  }
}

function renderThreadListView(): void {
  const count = state.recentThreads.length;
  threadListSummaryEl.textContent =
    count === 0
      ? "No recent ARIA conversations found."
      : `${count} recent ${count === 1 ? "conversation" : "conversations"}`;

  threadListRowsEl.replaceChildren();
  for (const th of state.recentThreads) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ar-thread-item";
    btn.dataset.threadId = th.thread_id;

    const left = document.createElement("div");
    left.className = "ar-thread-item-left";
    const title = document.createElement("span");
    title.className = "ar-list-row-title";
    title.textContent = th.title || th.prompt_preview || th.thread_id;
    left.appendChild(title);

    const scopeStr = [th.scope.entity, th.scope.project].filter(Boolean).join("/");
    if (scopeStr) {
      const scope = document.createElement("span");
      scope.className = "ar-list-row-hint";
      scope.textContent = scopeStr;
      left.appendChild(scope);
    }
    btn.appendChild(left);

    const openCue = document.createElement("span");
    openCue.className = "ar-thread-open-cue";
    openCue.textContent = "Open →";
    btn.appendChild(openCue);

    btn.addEventListener("click", () => {
      if (!bridge.canCallServerTools()) {
        return;
      }
      void bridge
        .callAriaTool("aria_get_thread", { thread_id: th.thread_id, limit: 20 })
        .then((env) => handleEnvelope(env, { immediateSync: true }));
    });

    threadListRowsEl.appendChild(btn);
  }
}

function renderMain(): void {
  const nowMs = Date.now();
  emptyEl.hidden = true;
  runViewEl.hidden = true;
  runListViewEl.hidden = true;
  threadListViewEl.hidden = true;

  if (state.view === "threadList") {
    threadListViewEl.hidden = false;
    renderThreadListView();
    return;
  }

  if (state.view === "runList" && state.sessions.length > 0) {
    runListViewEl.hidden = false;
    renderRunListView(nowMs);
    return;
  }

  renderRunView(nowMs);
}

function renderAll(): void {
  renderHeader();
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

  document.querySelectorAll<HTMLElement>("[data-elapsed-turn-id]").forEach((el) => {
    const tid = el.dataset.elapsedTurnId;
    const entry = tid ? entryByTurnId.get(tid) : undefined;
    if (!entry) {
      return;
    }
    const formatted = formatElapsedClock(computeLiveElapsedSeconds(entry, nowMs));
    if (formatted) {
      el.textContent = el.closest(".ar-phase-pill") ? `· ${formatted}` : formatted;
    }
  });

  document.querySelectorAll<HTMLElement>("[data-live-step-turn-id]").forEach((el) => {
    const tid = el.dataset.liveStepTurnId;
    const entry = tid ? entryByTurnId.get(tid) : undefined;
    if (entry) {
      el.textContent = formatLiveStepLine(entry, nowMs);
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
    handleEnvelope(envelope, { immediateSync: true });
  },
  onHostContext(displayMode, availableDisplayModes) {
    state = {
      ...state,
      displayMode: displayMode ?? state.displayMode,
      availableDisplayModes: availableDisplayModes ?? state.availableDisplayModes,
    };
    renderAll();
  },
  onTeardown() {
    clearPollTimer();
    if (clockInterval !== null) {
      window.clearInterval(clockInterval);
      clockInterval = null;
    }
  },
});

// Header & menu controls
backBtn.addEventListener("click", () => {
  if (state.view === "threadList" && state.previousView) {
    state = { ...state, view: state.previousView };
    renderAll();
    return;
  }
  if (state.view === "run" && state.cameFromThreadList && state.recentThreads.length > 0) {
    state = { ...state, view: "threadList", previousView: "run" };
    renderAll();
  }
});

refreshLinkBtn.addEventListener("click", () => {
  const activeIds = getPollableTurnIds(state);
  if (activeIds.length > 0) {
    void pollSpecificTurnIds(activeIds, true);
    return;
  }
  const current = selectedSession();
  const latest = current ? latestSessionEntry(current) : null;
  if (latest && !latest.optimistic) {
    void pollSpecificTurnIds([latest.turn.turn_id], true);
  }
});

expandBtn.addEventListener("click", () => {
  const target = state.displayMode === "fullscreen" ? "inline" : "fullscreen";
  void bridge.requestDisplayMode(target).then((res) => {
    if (res.ok) {
      state = { ...state, displayMode: res.mode };
      renderAll();
    } else {
      showBanner(res.message, "info");
    }
  });
});

menuBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  state = { ...state, menuOpen: !state.menuOpen };
  renderHeader();
});

menuThreadsBtn.addEventListener("click", () => {
  closeMenu();
  if (!bridge.canCallServerTools()) {
    return;
  }
  const active = selectedSession();
  const args: Record<string, unknown> = { limit: 15 };
  if (active?.scope.entity) {
    args.entity = active.scope.entity;
  }
  if (active?.scope.project) {
    args.project = active.scope.project;
  }
  void bridge
    .callAriaTool("aria_list_threads", args)
    .then((env) => handleEnvelope(env, { immediateSync: true }));
});

menuLiveBtn.addEventListener("click", () => {
  const nextPaused = !state.liveUpdatesPaused;
  state = { ...state, liveUpdatesPaused: nextPaused, menuOpen: false };
  renderHeader();
  if (nextPaused) {
    clearPollTimer();
  } else {
    scheduleNextPollOrHydration();
  }
});

menuForgeBtn.addEventListener("click", () => {
  closeMenu();
  const active = selectedSession();
  const target = (state.view === "run" ? active?.forgeUrl : null) || state.forgeHomeUrl;
  if (target) {
    handleOpenLink(target);
  }
});

document.addEventListener("click", (event) => {
  if (state.menuOpen && !menuWrapEl.contains(event.target as Node)) {
    closeMenu();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.menuOpen) {
    event.preventDefault();
    closeMenu(true);
  }
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
renderAll();
void bridge.connect();
