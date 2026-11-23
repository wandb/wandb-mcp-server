import {
  App,
  PostMessageTransport,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
  type McpUiHostContext,
} from "@modelcontextprotocol/ext-apps";
import {
  GraphPayloadSchema,
  ToolErrorPayloadSchema,
  type GraphPayload,
} from "./schema";
import type { DisplayMode, ModelContextPayload } from "./state";

export const SERVER_TOOL_NAME = "render_wandb_line_plot_tool";

export interface HostCallbacks {
  onToolInput: (title: string | null) => void;
  onPayloadLoaded: (payload: GraphPayload) => void;
  onToolFailed: (message: string) => void;
  onHostContext: (
    displayMode: DisplayMode | undefined,
    availableDisplayModes: DisplayMode[] | undefined,
    theme: "light" | "dark" | undefined,
  ) => void;
  onTeardown: () => void;
}

function parseModes(raw: unknown): DisplayMode[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined;
  }
  const valid = raw.filter(
    (m): m is DisplayMode => m === "inline" || m === "fullscreen" || m === "pip",
  );
  return valid.length > 0 ? valid : undefined;
}

function extractInputTitle(args: Record<string, unknown> | undefined): string | null {
  if (!args) {
    return null;
  }
  if (typeof args.metric === "string" && args.metric.trim()) {
    return args.metric.trim();
  }
  const cfg = args.config;
  if (cfg && typeof cfg === "object") {
    const rec = cfg as Record<string, unknown>;
    if (typeof rec.chartTitle === "string" && rec.chartTitle.trim()) {
      return rec.chartTitle.trim();
    }
    if (Array.isArray(rec.metrics) && rec.metrics.length > 0) {
      return rec.metrics.filter((m): m is string => typeof m === "string").join(", ");
    }
  }
  if (typeof args.project_name === "string" && args.project_name.trim()) {
    return args.project_name.trim();
  }
  return null;
}

export function extractGraphResult(result: {
  isError?: boolean;
  structuredContent?: unknown;
  content?: Array<{ type: string; text?: string }>;
}): { ok: true; payload: GraphPayload } | { ok: false; message: string } {
  if (result.isError) {
    const parsedErr = ToolErrorPayloadSchema.safeParse(result.structuredContent);
    if (parsedErr.success) {
      return { ok: false, message: parsedErr.data.message };
    }
    const textBlock = result.content?.find((c) => c.type === "text" && typeof c.text === "string");
    if (textBlock?.text) {
      try {
        const errJson = ToolErrorPayloadSchema.safeParse(JSON.parse(textBlock.text));
        if (errJson.success) {
          return { ok: false, message: errJson.data.message };
        }
      } catch {
        // non-JSON text error
      }
      return { ok: false, message: textBlock.text.slice(0, 240) };
    }
    return { ok: false, message: "W&B line plot request failed." };
  }

  const parsed = GraphPayloadSchema.safeParse(result.structuredContent);
  if (!parsed.success) {
    return { ok: false, message: "Received an unrecognized W&B chart payload." };
  }
  return { ok: true, payload: parsed.data };
}

export class LinePlotHostBridge {
  private readonly app: App;
  private connected = false;
  private contextTimer: number | null = null;
  private lastPublishedSignature = "";

  constructor(private readonly callbacks: HostCallbacks) {
    this.app = new App(
      { name: "wandb-line-plot", version: "0.4.6" },
      {},
      { autoResize: true },
    );

    this.app.addEventListener("toolinput", (params) => {
      this.callbacks.onToolInput(extractInputTitle(params.arguments));
    });

    this.app.addEventListener("toolresult", (params) => {
      const outcome = extractGraphResult(params);
      if (outcome.ok) {
        this.callbacks.onPayloadLoaded(outcome.payload);
      } else {
        this.callbacks.onToolFailed(outcome.message);
      }
    });

    this.app.addEventListener("toolcancelled", (params) => {
      this.callbacks.onToolFailed(params.reason ?? "Chart request was cancelled.");
    });

    this.app.addEventListener("hostcontextchanged", (ctx) => {
      this.applyHostEnvironment(ctx);
    });

    this.app.onteardown = async () => {
      if (this.contextTimer !== null) {
        window.clearTimeout(this.contextTimer);
        this.contextTimer = null;
      }
      this.callbacks.onTeardown();
      if (this.connected && this.app.getHostCapabilities()?.updateModelContext) {
        try {
          await this.app.updateModelContext({ content: [] });
        } catch {
          // ignore teardown errors
        }
      }
      return {};
    };
  }

  private applyHostEnvironment(ctx: McpUiHostContext | undefined): void {
    if (!ctx) {
      return;
    }
    if (ctx.theme === "light" || ctx.theme === "dark") {
      applyDocumentTheme(ctx.theme);
      document.documentElement.dataset.theme = ctx.theme;
    }
    if (ctx.styles?.variables) {
      applyHostStyleVariables(ctx.styles.variables);
    }
    if (ctx.styles?.css?.fonts) {
      applyHostFonts(ctx.styles.css.fonts);
    }
    const mode =
      ctx.displayMode === "inline" ||
      ctx.displayMode === "fullscreen" ||
      ctx.displayMode === "pip"
        ? ctx.displayMode
        : undefined;
    const available = parseModes(ctx.availableDisplayModes);
    const theme = ctx.theme === "light" || ctx.theme === "dark" ? ctx.theme : undefined;
    this.callbacks.onHostContext(mode, available, theme);
  }

  async connect(): Promise<void> {
    if (window.parent === window) {
      return;
    }
    await this.app.connect(new PostMessageTransport(window.parent, window.parent));
    this.connected = true;
    this.applyHostEnvironment(this.app.getHostContext());
  }

  canCallServerTools(): boolean {
    if (!this.connected) {
      return false;
    }
    const caps = this.app.getHostCapabilities();
    return Boolean(caps?.serverTools);
  }

  canOpenLinks(): boolean {
    if (!this.connected) {
      return false;
    }
    return Boolean(this.app.getHostCapabilities()?.openLinks);
  }

  async callLinePlotTool(
    args: Record<string, unknown>,
  ): Promise<{ ok: true; payload: GraphPayload } | { ok: false; message: string }> {
    if (!this.canCallServerTools()) {
      return { ok: false, message: "This MCP host does not support interactive tool calls." };
    }
    try {
      const result = await this.app.callServerTool({
        name: SERVER_TOOL_NAME,
        arguments: args,
      });
      return extractGraphResult(result);
    } catch (err) {
      const detail = err instanceof Error && err.message ? err.message : "Server request failed.";
      return { ok: false, message: detail };
    }
  }

  async requestDisplayMode(
    mode: DisplayMode,
  ): Promise<{ ok: true; mode: DisplayMode } | { ok: false; message: string }> {
    if (!this.connected) {
      return { ok: false, message: "Not connected to an MCP host." };
    }
    try {
      const result = await this.app.requestDisplayMode({ mode });
      if (result.mode !== mode) {
        return {
          ok: false,
          message: `Host kept chart in ${result.mode} mode.`,
        };
      }
      return { ok: true, mode: result.mode };
    } catch {
      return { ok: false, message: "Host declined display mode change." };
    }
  }

  async openExternalUrl(url: string): Promise<{ ok: true } | { ok: false; message: string }> {
    if (!this.connected || !this.canOpenLinks()) {
      return { ok: false, message: `Copy link to open in browser: ${url}` };
    }
    try {
      const res = await this.app.openLink({ url });
      if (res.isError) {
        return { ok: false, message: `Host blocked external link (${url}).` };
      }
      return { ok: true };
    } catch {
      return { ok: false, message: `Unable to open link (${url}).` };
    }
  }

  scheduleModelContextSync(payload: ModelContextPayload, immediate = false): void {
    if (!this.connected || !this.app.getHostCapabilities()?.updateModelContext) {
      return;
    }
    const signature = JSON.stringify(payload.structuredContent ?? payload.content);
    if (signature === this.lastPublishedSignature) {
      return;
    }
    if (this.contextTimer !== null) {
      window.clearTimeout(this.contextTimer);
      this.contextTimer = null;
    }
    const publish = () => {
      this.contextTimer = null;
      this.lastPublishedSignature = signature;
      void this.app.updateModelContext(payload).catch(() => {
        // Non-fatal if host rejects model context updates
      });
    };
    if (immediate) {
      publish();
    } else {
      this.contextTimer = window.setTimeout(publish, 120);
    }
  }
}
