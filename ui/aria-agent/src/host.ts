import {
  App,
  PostMessageTransport,
  applyDocumentTheme,
  applyHostFonts,
  applyHostStyleVariables,
  type McpUiHostContext,
} from "@modelcontextprotocol/ext-apps";
import { isAllowedHttpsUrl } from "../../shared/url-guard";
import { parseAriaToolEnvelope, type ParsedAriaToolEnvelope } from "./schema";
import type { DisplayMode, ModelContextPayload } from "./state";

export interface AriaHostCallbacks {
  onToolInput: (args: Record<string, unknown> | undefined) => void;
  onToolEnvelope: (envelope: ParsedAriaToolEnvelope) => void;
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

export class AriaHostBridge {
  private readonly app: App;
  private connected = false;
  private contextTimer: number | null = null;
  private lastPublishedSignature = "";

  constructor(private readonly callbacks: AriaHostCallbacks) {
    this.app = new App(
      { name: "wandb-aria-agent", version: "0.4.7" },
      {},
      { autoResize: true },
    );

    this.app.addEventListener("toolinput", (params) => {
      this.callbacks.onToolInput(params.arguments);
    });

    this.app.addEventListener("toolresult", (params) => {
      this.callbacks.onToolEnvelope(parseAriaToolEnvelope(params));
    });

    this.app.addEventListener("toolcancelled", (params) => {
      this.callbacks.onToolEnvelope({
        kind: "error",
        message: params.reason ?? "ARIA tool call was cancelled.",
        retryable: true,
        turnId: null,
      });
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

  async callAriaTool(
    toolName:
      | "aria_send_message"
      | "aria_get_turn"
      | "aria_get_thread"
      | "aria_list_threads",
    args: Record<string, unknown>,
  ): Promise<ParsedAriaToolEnvelope> {
    if (!this.canCallServerTools()) {
      return {
        kind: "error",
        message: "This MCP host does not support interactive server tool calls.",
        retryable: false,
        turnId: null,
      };
    }
    try {
      const result = await this.app.callServerTool({
        name: toolName,
        arguments: args,
      });
      return parseAriaToolEnvelope(result);
    } catch (err) {
      const detail = err instanceof Error && err.message ? err.message : "ARIA tool call failed.";
      return {
        kind: "error",
        message: detail,
        retryable: true,
        turnId: null,
      };
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
        return { ok: false, message: `Host kept ARIA studio in ${result.mode} mode.` };
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
    const signature = JSON.stringify({
      text: payload.text,
      structuredContent: payload.structuredContent,
    });
    if (signature === this.lastPublishedSignature) {
      return;
    }
    if (this.contextTimer !== null) {
      window.clearTimeout(this.contextTimer);
      this.contextTimer = null;
    }

    const publish = async () => {
      this.contextTimer = null;
      this.lastPublishedSignature = signature;
      try {
        await this.app.updateModelContext({
          content: [{ type: "text", text: payload.text }],
          structuredContent: payload.structuredContent,
        });
      } catch {
        // non-fatal if host rejects context update
      }
    };

    if (immediate) {
      void publish();
    } else {
      this.contextTimer = window.setTimeout(() => {
        void publish();
      }, 180);
    }
  }
}
