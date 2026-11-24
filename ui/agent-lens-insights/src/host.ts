import {
  App,
  PostMessageTransport,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
  type McpUiHostContext,
} from "@modelcontextprotocol/ext-apps";
import { isAllowedHttpsUrl } from "../../shared/url-guard";
import {
  InsightsPayloadSchema,
  ToolErrorPayloadSchema,
  type InsightsPayload,
} from "./schema";
import type { DisplayMode, ModelContextPayload } from "./state";

export const INSIGHTS_SERVER_TOOL_NAME = "render_agent_lens_insights_tool";

export interface InsightsHostCallbacks {
  onToolInput: (title: string | null) => void;
  onPayloadLoaded: (payload: InsightsPayload) => void;
  onToolFailed: (message: string) => void;
  onHostContext: (
    displayMode: DisplayMode | undefined,
    availableDisplayModes: DisplayMode[] | undefined,
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
  const ent = typeof args.entity_name === "string" ? args.entity_name.trim() : "";
  const proj = typeof args.project_name === "string" ? args.project_name.trim() : "";
  if (ent && proj) {
    return `${ent}/${proj}`;
  }
  return proj || null;
}

function extractTextError(result: {
  structuredContent?: unknown;
  content?: Array<{ type: string; text?: string }>;
}): string {
  const parsedErr = ToolErrorPayloadSchema.safeParse(result.structuredContent);
  if (parsedErr.success) {
    return parsedErr.data.message;
  }
  const textBlock = result.content?.find((c) => c.type === "text" && typeof c.text === "string");
  if (textBlock?.text) {
    try {
      const decoded: unknown = JSON.parse(textBlock.text);
      const errJson = ToolErrorPayloadSchema.safeParse(decoded);
      if (errJson.success) {
        return errJson.data.message;
      }
    } catch {
      // plain text error
    }
    return textBlock.text.slice(0, 260);
  }
  return "Agent Lens insights request failed.";
}

export function extractInsightsResult(result: {
  isError?: boolean;
  structuredContent?: unknown;
  content?: Array<{ type: string; text?: string }>;
}): { ok: true; payload: InsightsPayload } | { ok: false; message: string } {
  if (result.isError) {
    return { ok: false, message: extractTextError(result) };
  }
  const parsed = InsightsPayloadSchema.safeParse(result.structuredContent);
  if (!parsed.success) {
    return { ok: false, message: "Received an unrecognized Agent Lens insights payload." };
  }
  return { ok: true, payload: parsed.data };
}

export class InsightsHostBridge {
  private readonly app: App;
  private connected = false;
  private contextTimer: number | null = null;
  private lastPublishedSignature = "";

  constructor(private readonly callbacks: InsightsHostCallbacks) {
    this.app = new App(
      { name: "wandb-agent-lens-insights", version: "0.4.7" },
      {},
      { autoResize: true },
    );

    this.app.addEventListener("toolinput", (params) => {
      this.callbacks.onToolInput(extractInputTitle(params.arguments));
    });

    this.app.addEventListener("toolresult", (params) => {
      const outcome = extractInsightsResult(params);
      if (outcome.ok) {
        this.callbacks.onPayloadLoaded(outcome.payload);
      } else {
        this.callbacks.onToolFailed(outcome.message);
      }
    });

    this.app.addEventListener("toolcancelled", (params) => {
      this.callbacks.onToolFailed(params.reason ?? "Insights request was cancelled.");
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
    this.callbacks.onHostContext(mode, available);
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
    return Boolean(this.app.getHostCapabilities()?.serverTools);
  }

  canOpenLinks(): boolean {
    if (!this.connected) {
      return false;
    }
    return Boolean(this.app.getHostCapabilities()?.openLinks);
  }

  async callInsightsOverview(
    args: Record<string, unknown>,
  ): Promise<{ ok: true; payload: InsightsPayload } | { ok: false; message: string }> {
    if (!this.canCallServerTools()) {
      return { ok: false, message: "This MCP host does not support interactive window changes." };
    }
    try {
      const result = await this.app.callServerTool({
        name: INSIGHTS_SERVER_TOOL_NAME,
        arguments: args,
      });
      return extractInsightsResult(result);
    } catch (err) {
      const detail = err instanceof Error && err.message ? err.message : "Insights refresh failed.";
      return { ok: false, message: detail };
    }
  }

  async invokeJsonTool(
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<{ ok: true; data: unknown } | { ok: false; message: string }> {
    if (!this.canCallServerTools()) {
      return { ok: false, message: "This MCP host does not support interactive drilldowns." };
    }
    try {
      const result = await this.app.callServerTool({
        name: toolName,
        arguments: args,
      });
      if (result.isError) {
        return { ok: false, message: extractTextError(result) };
      }
      if (result.structuredContent !== undefined) {
        return { ok: true, data: result.structuredContent };
      }
      const textBlock = result.content?.find(
        (c): c is Extract<(typeof result.content)[number], { type: "text" }> => c.type === "text",
      );
      if (textBlock?.text) {
        try {
          const parsed: unknown = JSON.parse(textBlock.text);
          if (
            parsed &&
            typeof parsed === "object" &&
            "error" in (parsed as Record<string, unknown>) &&
            (parsed as Record<string, unknown>).error
          ) {
            return {
              ok: false,
              message: String((parsed as Record<string, unknown>).message ?? "Server tool returned an error."),
            };
          }
          return { ok: true, data: parsed };
        } catch {
          return { ok: true, data: textBlock.text };
        }
      }
      return { ok: true, data: {} };
    } catch (err) {
      const detail = err instanceof Error && err.message ? err.message : "Server tool call failed.";
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
        return { ok: false, message: `Host kept explorer in ${result.mode} mode.` };
      }
      return { ok: true, mode: result.mode };
    } catch {
      return { ok: false, message: "Host declined display mode change." };
    }
  }

  async openExternalUrl(url: string): Promise<{ ok: true } | { ok: false; message: string }> {
    if (!isAllowedHttpsUrl(url)) {
      return { ok: false, message: "Blocked non-HTTPS external link." };
    }
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

  scheduleModelContextSync(payload: ModelContextPayload | null, immediate = false): void {
    if (!payload || !this.connected || !this.app.getHostCapabilities()?.updateModelContext) {
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
        // non-fatal
      });
    };
    if (immediate) {
      publish();
    } else {
      this.contextTimer = window.setTimeout(publish, 100);
    }
  }
}
