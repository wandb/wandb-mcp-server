import "./style.css";
import { ConversationHostBridge } from "./host";
import {
  buildSpanWaterfall,
  parseConversationTags,
  parseTagCatalog,
  type NormalizedMessage,
  type NormalizedTurn,
  type SpanWaterfallNode,
} from "./schema";
import {
  buildConversationModelContext,
  canPerformWrites,
  createInitialConversationState,
  getVisibleTurns,
  reduceConversationState,
  type ConversationAction,
  type ConversationState,
} from "./state";

const rootEl = document.getElementById("app-root") as HTMLElement;
const modePillEl = document.getElementById("ac-mode-pill") as HTMLElement;
const titleEl = document.getElementById("ac-title") as HTMLElement;
const versionPillEl = document.getElementById("ac-version-pill") as HTMLElement;
const errorPillEl = document.getElementById("ac-error-pill") as HTMLElement;
const metaStripEl = document.getElementById("ac-meta-strip") as HTMLElement;
const tagsRowEl = document.getElementById("ac-tags-row") as HTMLElement;
const expandBtn = document.getElementById("ac-expand-btn") as HTMLButtonElement;
const openForgeBtn = document.getElementById("ac-open-forge-btn") as HTMLButtonElement;
const toolbarEl = document.getElementById("ac-toolbar") as HTMLElement;
const tabConvBtn = document.getElementById("ac-tab-conversation") as HTMLButtonElement;
const tabSpansBtn = document.getElementById("ac-tab-spans") as HTMLButtonElement;
const errorsOnlyBtn = document.getElementById("ac-errors-only-btn") as HTMLButtonElement;
const tagMenuBtn = document.getElementById("ac-tag-menu-btn") as HTMLButtonElement;
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

function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) {
    return "—";
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s`;
  }
  return `${Math.round(ms)}ms`;
}

function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) {
    return "0";
  }
  return n.toLocaleString("en-US");
}

async function copyText(value: string, label: string): Promise<void> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      dispatch({ type: "banner", kind: "info", message: `Copied ${label}.` });
      return;
    }
  } catch {
    // clipboard blocked inside sandboxed iframe; show value cleanly
  }
  dispatch({ type: "banner", kind: "info", message: `${label}: ${value.slice(0, 120)}` });
}

const bridge = new ConversationHostBridge({
  onToolInput(title) {
    dispatch({ type: "toolInput", title });
  },
  onPayloadLoaded(payload) {
    dispatch({ type: "loaded", payload }, true);
    void loadInitialConversationTags();
  },
  onToolFailed(message) {
    dispatch({ type: "loadFailed", message });
  },
  onHostContext(displayMode, availableDisplayModes) {
    dispatch({ type: "hostContextChanged", displayMode, availableDisplayModes });
  },
  onTeardown() {
    state = createInitialConversationState();
    render();
  },
});

function dispatch(action: ConversationAction, immediateContext = false): void {
  state = reduceConversationState(state, action);
  render();
  bridge.scheduleModelContextSync(buildConversationModelContext(state), immediateContext);
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
    dispatch({ type: "tagsLoaded", tags: parseConversationTags(outcome.data) });
  }
}

async function ensureSpanWaterfallLoaded(): Promise<void> {
  const payload = state.payload;
  if (!payload || state.spansStatus === "loading" || state.spansStatus === "ready") {
    return;
  }
  if (!payload.conversationId) {
    dispatch({
      type: "spansFailed",
      message: "Span waterfall requires a conversation_id on the trace.",
    });
    return;
  }
  if (!bridge.canCallServerTools()) {
    dispatch({
      type: "spansFailed",
      message: "This MCP host does not support lazy span tree loading.",
    });
    return;
  }
  dispatch({ type: "spansLoading" });
  const outcome = await bridge.invokeJsonTool("get_weave_agent_conversation_spans_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    conversation_ids: [payload.conversationId],
  });
  if (!outcome.ok) {
    dispatch({ type: "spansFailed", message: outcome.message });
    return;
  }
  dispatch({ type: "spansLoaded", spans: buildSpanWaterfall(outcome.data) });
}

async function submitReaction(turn: NormalizedTurn, emoji: "👍" | "👎"): Promise<void> {
  const payload = state.payload;
  if (!payload || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = turn.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "Turn is missing a trace_id for feedback." });
    return;
  }
  dispatch({ type: "actionStarted" });
  const outcome = await bridge.invokeJsonTool("manage_weave_agent_feedback_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "add_reaction",
    conversation_id: payload.conversationId ?? undefined,
    trace_id: traceId,
    emoji,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message });
    return;
  }
  const createdId =
    outcome.data && typeof outcome.data === "object" && typeof (outcome.data as Record<string, unknown>).id === "string"
      ? String((outcome.data as Record<string, unknown>).id)
      : null;
  dispatch({
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
  });
}

async function submitNote(turn: NormalizedTurn, rawNote: string): Promise<void> {
  const note = rawNote.trim();
  const payload = state.payload;
  if (!note || !payload || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = turn.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "Turn is missing a trace_id for feedback." });
    return;
  }
  dispatch({ type: "actionStarted" });
  const outcome = await bridge.invokeJsonTool("manage_weave_agent_feedback_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "add_note",
    conversation_id: payload.conversationId ?? undefined,
    trace_id: traceId,
    note,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message });
    return;
  }
  const createdId =
    outcome.data && typeof outcome.data === "object" && typeof (outcome.data as Record<string, unknown>).id === "string"
      ? String((outcome.data as Record<string, unknown>).id)
      : null;
  dispatch({
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
  });
}

async function openTagPopoverAndLoadCatalog(): Promise<void> {
  const nextOpen = !state.tagPopoverOpen;
  dispatch({ type: "toggleTagPopover", open: nextOpen });
  const payload = state.payload;
  if (!nextOpen || !payload || !payload.agentLensAvailable || state.tagCatalog.length > 0) {
    return;
  }
  const outcome = await bridge.invokeJsonTool("list_agent_lens_tags_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
  });
  if (outcome.ok) {
    dispatch({ type: "tagCatalogLoaded", catalog: parseTagCatalog(outcome.data) });
  }
}

async function assignConversationTag(opts: { tagId?: string; slug?: string }): Promise<void> {
  const payload = state.payload;
  if (!payload || !payload.conversationId || !canPerformWrites(state, bridge.canCallServerTools())) {
    return;
  }
  const traceId = state.selectedTarget?.traceId ?? payload.turns[0]?.traceId ?? payload.traceId;
  if (!traceId) {
    dispatch({ type: "banner", kind: "warning", message: "A turn trace_id is required to assign an Agent Lens tag." });
    return;
  }
  const slug = opts.slug?.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (!opts.tagId && !slug) {
    dispatch({ type: "banner", kind: "warning", message: "Enter a kebab-case tag slug (e.g. needs-eval)." });
    return;
  }
  dispatch({ type: "actionStarted" });
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
    dispatch({ type: "banner", kind: "error", message: outcome.message });
    return;
  }
  const catalogMatch = opts.tagId ? state.tagCatalog.find((c) => c.id === opts.tagId) : null;
  dispatch({
    type: "tagAdded",
    tag: {
      tagId: opts.tagId ?? null,
      name: catalogMatch?.name ?? slug ?? "tagged",
      color: catalogMatch?.color ?? null,
      traceId,
      source: "human",
    },
  });
}

async function removeConversationTag(tagId: string, traceId: string | null): Promise<void> {
  const payload = state.payload;
  const resolvedTrace = traceId ?? state.selectedTarget?.traceId ?? payload?.turns[0]?.traceId ?? payload?.traceId;
  if (!payload || !payload.conversationId || !resolvedTrace) {
    return;
  }
  dispatch({ type: "actionStarted" });
  const outcome = await bridge.invokeJsonTool("manage_agent_lens_tags_tool", {
    entity_name: payload.entityName,
    project_name: payload.projectName,
    action: "remove_tag",
    conversation_id: payload.conversationId,
    trace_id: resolvedTrace,
    tag_id: tagId,
  });
  if (!outcome.ok) {
    dispatch({ type: "banner", kind: "error", message: outcome.message });
    return;
  }
  dispatch({ type: "tagRemoved", tagId });
}

function createCodeDisclosure(label: string, code: string): HTMLElement {
  const details = document.createElement("details");
  details.className = "ac-code-details";
  details.addEventListener("click", (ev) => ev.stopPropagation());

  const summary = document.createElement("summary");
  const labelSpan = document.createElement("span");
  labelSpan.textContent = label;
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "ac-mini-btn";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    void copyText(code, label);
  });
  summary.append(labelSpan, copyBtn);

  const pre = document.createElement("pre");
  pre.className = "ac-code-pre";
  pre.textContent = code;
  details.append(summary, pre);
  return details;
}

function renderMessageEl(turn: NormalizedTurn, msg: NormalizedMessage): HTMLElement {
  const box = document.createElement("div");
  box.className = "ac-msg";
  box.dataset.role = msg.role;

  const head = document.createElement("div");
  head.className = "ac-msg-head";
  const roleLabel = document.createElement("span");
  const titleParts = [msg.role.toUpperCase()];
  if (msg.toolName) {
    titleParts.push(`· ${msg.toolName}`);
  }
  if (msg.model) {
    titleParts.push(`· ${msg.model}`);
  }
  roleLabel.textContent = titleParts.join(" ");

  const rightMeta = document.createElement("span");
  if (msg.durationMs !== null) {
    rightMeta.textContent = formatDuration(msg.durationMs);
  }
  head.append(roleLabel, rightMeta);
  box.appendChild(head);

  if (msg.content) {
    const p = document.createElement("p");
    p.className = "ac-msg-body";
    p.textContent = msg.content;
    box.appendChild(p);
  }

  if (msg.arguments) {
    box.appendChild(createCodeDisclosure(`Arguments (${msg.toolName ?? "tool"})`, msg.arguments));
  }

  for (const tc of msg.toolCalls) {
    const tcWrap = document.createElement("div");
    tcWrap.addEventListener("click", (ev) => {
      ev.stopPropagation();
      dispatch({ type: "selectTurn", turnIndex: turn.turnIndex, toolName: tc.name, spanId: tc.spanId });
    });
    if (tc.arguments) {
      tcWrap.appendChild(createCodeDisclosure(`Tool call: ${tc.name}`, tc.arguments));
    }
    if (tc.result) {
      tcWrap.appendChild(createCodeDisclosure(`Result: ${tc.name}`, tc.result));
    }
    if (tc.error) {
      const errEl = document.createElement("div");
      errEl.className = "ac-error-callout";
      errEl.textContent = `${tc.name}: ${tc.error}`;
      tcWrap.appendChild(errEl);
    }
    box.appendChild(tcWrap);
  }

  if (msg.toolResult) {
    box.appendChild(createCodeDisclosure(`Tool output (${msg.toolName ?? "result"})`, msg.toolResult));
  }

  if (msg.error) {
    const errDiv = document.createElement("div");
    errDiv.className = "ac-error-callout";
    errDiv.textContent = msg.error;
    box.appendChild(errDiv);
  }

  if (msg.toolName) {
    box.addEventListener("click", (ev) => {
      ev.stopPropagation();
      dispatch({ type: "selectTurn", turnIndex: turn.turnIndex, toolName: msg.toolName, spanId: msg.spanId });
    });
  }

  return box;
}

function renderTurnCard(turn: NormalizedTurn, writable: boolean): HTMLElement {
  const card = document.createElement("article");
  card.className = "ac-turn-card";
  const isSelected =
    state.selectedTarget?.kind === "turn" && state.selectedTarget.turnIndex === turn.turnIndex;
  card.dataset.selected = isSelected ? "true" : "false";
  card.dataset.error = turn.isError ? "true" : "false";
  card.tabIndex = 0;
  card.addEventListener("click", () => {
    dispatch({ type: "selectTurn", turnIndex: turn.turnIndex });
  });
  card.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      dispatch({ type: "selectTurn", turnIndex: turn.turnIndex });
    }
  });

  const head = document.createElement("div");
  head.className = "ac-turn-header";

  const leftBadges = document.createElement("div");
  leftBadges.className = "ac-turn-badges";

  const numEl = document.createElement("span");
  numEl.className = "ac-turn-num";
  numEl.textContent = `Turn #${turn.turnIndex}`;
  leftBadges.appendChild(numEl);

  if (turn.isError) {
    const errBadge = document.createElement("span");
    errBadge.className = "ac-pill ac-pill-error";
    errBadge.textContent = "ERROR";
    leftBadges.appendChild(errBadge);
  }

  if (turn.model) {
    const modelPill = document.createElement("span");
    modelPill.className = "ac-pill";
    modelPill.textContent = turn.model;
    leftBadges.appendChild(modelPill);
  }

  const rightBadges = document.createElement("div");
  rightBadges.className = "ac-turn-badges";
  if (turn.durationMs !== null) {
    const durPill = document.createElement("span");
    durPill.className = "ac-pill";
    durPill.textContent = formatDuration(turn.durationMs);
    rightBadges.appendChild(durPill);
  }
  if (turn.totalTokens !== null) {
    const tokPill = document.createElement("span");
    tokPill.className = "ac-pill";
    tokPill.textContent = `${formatNumber(turn.totalTokens)} tok`;
    rightBadges.appendChild(tokPill);
  }
  if (turn.traceId) {
    const copyTraceBtn = document.createElement("button");
    copyTraceBtn.type = "button";
    copyTraceBtn.className = "ac-mini-btn ac-mono";
    copyTraceBtn.textContent = `${turn.traceId.slice(0, 8)}…`;
    copyTraceBtn.title = `Copy trace_id ${turn.traceId}`;
    copyTraceBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      void copyText(turn.traceId!, "trace_id");
    });
    rightBadges.appendChild(copyTraceBtn);
  }

  head.append(leftBadges, rightBadges);
  card.appendChild(head);

  if (turn.errorMessage) {
    const errBox = document.createElement("div");
    errBox.className = "ac-error-callout";
    errBox.textContent = turn.errorMessage;
    card.appendChild(errBox);
  }

  const msgsContainer = document.createElement("div");
  msgsContainer.className = "ac-messages";
  for (const msg of turn.messages) {
    msgsContainer.appendChild(renderMessageEl(turn, msg));
  }
  card.appendChild(msgsContainer);

  if (turn.feedback.length > 0 || writable) {
    const footer = document.createElement("div");
    footer.className = "ac-turn-footer";
    footer.addEventListener("click", (ev) => ev.stopPropagation());

    const fbList = document.createElement("div");
    fbList.className = "ac-feedback-list";
    for (const fb of turn.feedback) {
      const pill = document.createElement("span");
      pill.className = "ac-fb-pill";
      pill.textContent = fb.emoji ? fb.emoji : `Note: ${fb.note ?? ""}`;
      fbList.appendChild(pill);
    }
    footer.appendChild(fbList);

    if (writable) {
      const actions = document.createElement("div");
      actions.className = "ac-triage-actions";

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
      noteBtn.textContent = "+ Note";
      noteBtn.disabled = state.isSubmittingAction;
      noteBtn.addEventListener("click", () => {
        const next = state.noteComposerTurnIndex === turn.turnIndex ? null : turn.turnIndex;
        dispatch({ type: "openNoteComposer", turnIndex: next });
      });

      actions.append(upBtn, downBtn, noteBtn);
      footer.appendChild(actions);
    }
    card.appendChild(footer);
  }

  if (writable && state.noteComposerTurnIndex === turn.turnIndex) {
    const composer = document.createElement("div");
    composer.className = "ac-note-composer";
    composer.addEventListener("click", (ev) => ev.stopPropagation());
    const input = document.createElement("input");
    input.type = "text";
    input.className = "ac-input ac-note-input";
    input.placeholder = "Add triage note on this turn…";
    input.maxLength = 1000;

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "ac-mini-btn ac-save-note-btn";
    saveBtn.textContent = "Save note";

    saveBtn.addEventListener("click", () => {
      void submitNote(turn, input.value);
    });
    input.addEventListener("keydown", (ev) => {
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
  closeBtn.addEventListener("click", () => dispatch({ type: "selectSpan", spanId: null }));
  head.append(title, closeBtn);
  spanDrawerEl.appendChild(head);

  const meta = document.createElement("div");
  meta.className = "ac-meta-strip";
  const items = [
    `kind=${span.kind}`,
    `status=${span.statusCode}`,
    `+${Math.round(span.offsetMs)}ms`,
    formatDuration(span.durationMs),
  ];
  if (span.toolName) {
    items.push(`tool=${span.toolName}`);
  }
  if (span.model) {
    items.push(`model=${span.model}`);
  }
  if (span.inputTokens !== null || span.outputTokens !== null) {
    items.push(`${formatNumber(span.inputTokens)} in / ${formatNumber(span.outputTokens)} out`);
  }
  meta.textContent = items.join(" · ");
  spanDrawerEl.appendChild(meta);

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

function renderSpansView(): void {
  spansListEl.replaceChildren();
  if (state.spansStatus === "loading") {
    const loading = document.createElement("div");
    loading.className = "ac-empty";
    loading.textContent = "Loading OTel span tree…";
    spansListEl.appendChild(loading);
    spanDrawerEl.hidden = true;
    return;
  }
  if (state.spans.length === 0) {
    const empty = document.createElement("div");
    empty.className = "ac-empty";
    empty.textContent = "No OTel spans returned for this conversation.";
    spansListEl.appendChild(empty);
    spanDrawerEl.hidden = true;
    return;
  }

  for (const span of state.spans) {
    const row = document.createElement("div");
    row.className = "ac-wf-row";
    row.dataset.selected = state.selectedSpanId === span.spanId ? "true" : "false";
    row.tabIndex = 0;
    row.addEventListener("click", () => dispatch({ type: "selectSpan", spanId: span.spanId }));
    row.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        dispatch({ type: "selectSpan", spanId: span.spanId });
      }
    });

    const label = document.createElement("div");
    label.className = "ac-wf-label";
    label.style.paddingLeft = `${span.depth * 14}px`;
    label.textContent = `${span.isError ? "⚠ " : ""}${span.name}`;

    const track = document.createElement("div");
    track.className = "ac-wf-track";
    const bar = document.createElement("div");
    bar.className = "ac-wf-bar";
    bar.dataset.kind = span.kind;
    bar.dataset.error = span.isError ? "true" : "false";
    bar.style.left = `${span.leftPct}%`;
    bar.style.width = `${span.widthPct}%`;

    const dur = document.createElement("span");
    dur.className = "ac-wf-dur";
    dur.textContent = formatDuration(span.durationMs);
    track.append(bar, dur);

    row.append(label, track);
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
  input.placeholder = "tag-slug (e.g. tool-bug)";
  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "ac-btn ac-assign-slug-btn";
  submit.textContent = "Assign tag";
  submit.addEventListener("click", () => {
    void assignConversationTag({ slug: input.value });
  });
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
    convViewEl.hidden = true;
    spansViewEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = state.fatalError ?? "Unable to load agent trajectory.";
    titleEl.textContent = "Trajectory unavailable";
    return;
  }

  if (!state.payload) {
    toolbarEl.hidden = true;
    convViewEl.hidden = true;
    spansViewEl.hidden = true;
    emptyEl.hidden = false;
    titleEl.textContent = state.pendingTitle ?? "Loading agent trajectory…";
    emptyEl.textContent = "Waiting for Weave agent trajectory…";
    return;
  }

  const payload = state.payload;
  const writable = canPerformWrites(state, bridge.canCallServerTools());

  modePillEl.textContent = payload.mode === "trace" ? "SINGLE TURN" : "CONVERSATION";
  titleEl.textContent =
    payload.agentName ?? `${payload.entityName}/${payload.projectName}`;

  if (payload.agentVersion) {
    versionPillEl.hidden = false;
    versionPillEl.textContent = payload.agentVersion;
  } else {
    versionPillEl.hidden = true;
  }

  if (payload.summary.errorCount > 0) {
    errorPillEl.hidden = false;
    errorPillEl.textContent = `${payload.summary.errorCount} ${payload.summary.errorCount === 1 ? "error" : "errors"}`;
  } else {
    errorPillEl.hidden = true;
  }

  metaStripEl.replaceChildren();
  const metaParts = [
    `${payload.entityName}/${payload.projectName}`,
    payload.conversationId ? `conv:${payload.conversationId.slice(0, 12)}` : null,
    `${payload.summary.turnCount}/${payload.summary.totalTurns} turns`,
    `${formatNumber(payload.summary.totalInputTokens)} in · ${formatNumber(payload.summary.totalOutputTokens)} out`,
    formatDuration(payload.summary.totalDurationMs),
    payload.summary.totalCostUsd !== null ? `$${payload.summary.totalCostUsd.toFixed(4)}` : null,
    payload.accessMode === "read-only" ? "read-only" : null,
  ].filter((x): x is string => Boolean(x));
  metaStripEl.textContent = metaParts.join(" · ");

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
  tabSpansBtn.hidden = !payload.conversationId;

  errorsOnlyBtn.hidden = payload.summary.errorCount === 0;
  errorsOnlyBtn.setAttribute("aria-pressed", state.errorsOnly ? "true" : "false");

  tagMenuBtn.hidden = !(writable && payload.agentLensAvailable && payload.conversationId);
  tagMenuBtn.setAttribute("aria-expanded", state.tagPopoverOpen ? "true" : "false");
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
      : "No turns recorded in this trajectory.";
    return;
  }

  emptyEl.hidden = true;
  convViewEl.hidden = false;
  convViewEl.replaceChildren();
  for (const turn of visibleTurns) {
    convViewEl.appendChild(renderTurnCard(turn, writable));
  }
}

tabConvBtn.addEventListener("click", () => dispatch({ type: "setTab", tab: "conversation" }));
tabSpansBtn.addEventListener("click", () => {
  dispatch({ type: "setTab", tab: "spans" });
  void ensureSpanWaterfallLoaded();
});
errorsOnlyBtn.addEventListener("click", () => dispatch({ type: "toggleErrorsOnly" }));
tagMenuBtn.addEventListener("click", () => void openTagPopoverAndLoadCatalog());
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
  if (ev.key === "Escape") {
    if (state.tagPopoverOpen) {
      dispatch({ type: "toggleTagPopover", open: false });
    } else if (state.noteComposerTurnIndex !== null) {
      dispatch({ type: "openNoteComposer", turnIndex: null });
    } else if (state.selectedSpanId !== null && state.activeTab === "spans") {
      dispatch({ type: "selectSpan", spanId: null });
    }
  }
});

render();
void bridge.connect();
