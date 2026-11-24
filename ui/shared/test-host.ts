import {
  AppBridge,
  PostMessageTransport,
  type McpUiHostContext,
} from "@modelcontextprotocol/ext-apps/app-bridge";
import type { CallToolResult } from "@modelcontextprotocol/client";

export type ToolHandler = (
  name: string,
  args: Record<string, unknown>,
) => CallToolResult | Promise<CallToolResult>;

export interface McpTestHostApi {
  mount: (appHtml: string, initialContext?: McpUiHostContext) => Promise<void>;
  sendToolInput: (args: Record<string, unknown>) => Promise<void>;
  sendToolResult: (result: CallToolResult) => Promise<void>;
  setHostContext: (ctx: McpUiHostContext) => void;
  teardown: () => Promise<void>;
  setToolHandler: (fn: ToolHandler) => void;
  toolCalls: Array<{ name: string; arguments: Record<string, unknown> }>;
  contextUpdates: Array<{ content?: unknown; structuredContent?: unknown }>;
  openedLinks: string[];
  requestedModes: string[];
}

declare global {
  interface Window {
    __MCP_APP_TEST_HOST__: McpTestHostApi;
  }
}

let bridge: AppBridge | null = null;
let toolHandler: ToolHandler | null = null;

const toolCalls: Array<{ name: string; arguments: Record<string, unknown> }> = [];
const contextUpdates: Array<{ content?: unknown; structuredContent?: unknown }> = [];
const openedLinks: string[] = [];
const requestedModes: string[] = [];

window.__MCP_APP_TEST_HOST__ = {
  toolCalls,
  contextUpdates,
  openedLinks,
  requestedModes,

  setToolHandler(fn) {
    toolHandler = fn;
  },

  async mount(appHtml: string, initialContext?: McpUiHostContext) {
    const container = document.getElementById("host-container")!;
    container.innerHTML = "";

    const iframe = document.createElement("iframe");
    iframe.id = "mcp-app-frame";
    iframe.setAttribute("sandbox", "allow-scripts");
    iframe.style.width = "840px";
    iframe.style.height = "560px";
    iframe.style.border = "1px solid #cbd5e1";

    const hostContext: McpUiHostContext = {
      theme: "dark",
      displayMode: "inline",
      availableDisplayModes: ["inline", "fullscreen"],
      ...initialContext,
    };

    container.appendChild(iframe);

    let resolveInit!: () => void;
    const initialized = new Promise<void>((resolve) => {
      resolveInit = resolve;
    });

    bridge = new AppBridge(
      null,
      { name: "PlaywrightTestHost", version: "1.0.0" },
      {
        openLinks: {},
        serverTools: {},
        logging: {},
        updateModelContext: { text: {}, structuredContent: {} },
      },
      { hostContext },
    );

    bridge.oninitialized = () => {
      resolveInit();
    };

    bridge.oncalltool = async (params) => {
      const args = (params.arguments ?? {}) as Record<string, unknown>;
      toolCalls.push({ name: params.name, arguments: args });
      if (toolHandler) {
        return await toolHandler(params.name, args);
      }
      return {
        content: [{ type: "text", text: `unexpected tool call: ${params.name}` }],
        isError: true,
      };
    };

    bridge.onupdatemodelcontext = async (params) => {
      contextUpdates.push({
        content: params.content,
        structuredContent: params.structuredContent,
      });
      return {};
    };

    bridge.onrequestdisplaymode = async ({ mode }) => {
      requestedModes.push(mode);
      const allowed = hostContext.availableDisplayModes ?? ["inline"];
      const actual = allowed.includes(mode) ? mode : (hostContext.displayMode ?? "inline");
      hostContext.displayMode = actual;
      return { mode: actual };
    };

    bridge.onopenlink = async ({ url }) => {
      openedLinks.push(url);
      return {};
    };

    const transport = new PostMessageTransport(
      iframe.contentWindow!,
      iframe.contentWindow!,
    );
    await bridge.connect(transport);
    iframe.srcdoc = appHtml;
    await initialized;
  },

  async sendToolInput(args) {
    await bridge!.sendToolInput({ arguments: args });
  },

  async sendToolResult(result) {
    await bridge!.sendToolResult(result);
  },

  setHostContext(ctx) {
    bridge!.setHostContext(ctx);
  },

  async teardown() {
    await bridge!.teardownResource({});
  },
};
