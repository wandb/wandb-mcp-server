import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { build as viteBuild } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uiRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(uiRoot, "../..");
const bundledAppPath = path.join(repoRoot, "src/wandb_mcp_server/data/agent-conversation.html");
const sharedHostEntry = path.resolve(uiRoot, "../shared/test-host.ts");

function makeConversationPayload(overrides = {}) {
  return {
    schema: "wandb.agent_conversation/v1",
    mode: "conversation",
    entityName: "acme",
    projectName: "support-agent",
    conversationId: "conv-77",
    traceId: null,
    agentName: "billing-router",
    agentVersion: "v4",
    forgeUrl: "https://forge.coreweave.com/acme/support-agent/agent-lens",
    accessMode: "read-write",
    agentLensAvailable: true,
    compacted: false,
    summary: {
      turnCount: 2,
      totalTurns: 2,
      hasMore: false,
      totalInputTokens: 640,
      totalOutputTokens: 210,
      totalTokens: 850,
      totalDurationMs: 1620,
      totalCostUsd: 0.0051,
      errorCount: 1,
    },
    turns: [
      {
        turnIndex: 1,
        traceId: "trace-turn-1",
        spanId: "span-1",
        agentName: "billing-router",
        agentVersion: "v4",
        statusCode: "OK",
        isError: false,
        errorMessage: null,
        startedAt: "2026-11-23T18:00:00Z",
        durationMs: 420,
        inputTokens: 240,
        outputTokens: 90,
        totalTokens: 330,
        costUsd: 0.002,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "user",
            content: "Can you check invoice #77?",
            toolName: null,
            toolCallId: null,
            arguments: null,
            toolCalls: [],
            toolResult: null,
            spanId: null,
            spanName: null,
            statusCode: null,
            error: null,
            durationMs: null,
            model: null,
          },
          {
            index: 1,
            role: "assistant",
            content: "Let me look up invoice #77 right now.",
            toolName: null,
            toolCallId: null,
            arguments: null,
            toolCalls: [],
            toolResult: null,
            spanId: null,
            spanName: null,
            statusCode: null,
            error: null,
            durationMs: null,
            model: "gpt-5.4",
          },
        ],
        feedback: [],
      },
      {
        turnIndex: 2,
        traceId: "trace-turn-2",
        spanId: "span-2",
        agentName: "billing-router",
        agentVersion: "v4",
        statusCode: "ERROR",
        isError: true,
        errorMessage: "InvoiceAPIError: missing currency parameter",
        startedAt: "2026-11-23T18:00:05Z",
        durationMs: 1200,
        inputTokens: 400,
        outputTokens: 120,
        totalTokens: 520,
        costUsd: 0.0031,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "tool",
            content: null,
            toolName: "lookup_invoice",
            toolCallId: "tc-77",
            arguments: '{"invoice_id":"77"}',
            toolCalls: [],
            toolResult: '{"status":422,"detail":"currency required"}',
            spanId: "span-tool-77",
            spanName: "tool.lookup_invoice",
            statusCode: "ERROR",
            error: "422 Unprocessable Entity: currency required",
            durationMs: 980,
            model: null,
          },
        ],
        feedback: [],
      },
    ],
    feedback: [],
    ...overrides,
  };
}

async function bundleHostJs() {
  const result = await viteBuild({
    configFile: false,
    logLevel: "silent",
    build: {
      write: false,
      target: "es2022",
      minify: false,
      lib: {
        entry: sharedHostEntry,
        formats: ["es"],
        fileName: "host-bundle",
      },
    },
  });
  const output = Array.isArray(result) ? result[0] : result;
  const chunk = output.output.find((o) => o.type === "chunk");
  if (!chunk) {
    throw new Error("Failed to bundle shared test-host.ts");
  }
  return chunk.code;
}

async function main() {
  const rawAppHtml = fs.readFileSync(bundledAppPath, "utf8");
  const strictCspTag = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none';">`;
  const appHtmlWithCsp = rawAppHtml.replace("<head>", `<head>\n    ${strictCspTag}`);

  const hostJs = await bundleHostJs();
  const hostPageHtml = `<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Conversation Inspector Host Harness</title></head>
  <body>
    <div id="host-container"></div>
    <script type="module">${hostJs}</script>
  </body>
</html>`;

  const externalRequests = [];
  const server = http.createServer((req, res) => {
    if (req.url === "/" || req.url === "/index.html") {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(hostPageHtml);
      return;
    }
    res.writeHead(404);
    res.end("not found");
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 920, height: 680 } });
    const pageErrors = [];
    const consoleErrors = [];
    page.on("pageerror", (err) => pageErrors.push(String(err)));
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });
    page.on("request", (req) => {
      const url = req.url();
      if (!url.startsWith(origin) && !url.startsWith("data:") && url !== "about:srcdoc") {
        externalRequests.push(url);
      }
    });

    await page.goto(origin, { waitUntil: "domcontentloaded" });

    await page.evaluate(async (html) => {
      window.__MCP_APP_TEST_HOST__.setToolHandler((name, args) => {
        if (name === "get_agent_lens_conversation_tags_tool") {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  data: [{ tag_id: "30201f95-1221-433a-9ea5-1e513081962f", tag_name: "escalated", trace_id: "trace-turn-2" }],
                }),
              },
            ],
          };
        }
        if (name === "get_weave_agent_conversation_spans_tool") {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  conversations: [
                    {
                      conversation_id: args.conversation_ids?.[0] ?? "conv-77",
                      spans: [
                        {
                          span_id: "span-root",
                          trace_id: "trace-turn-2",
                          name: "turn.billing-router",
                          is_invocation: true,
                          started_at: "2026-11-23T18:00:05.000Z",
                          duration_ms: 1200,
                          status_code: "ERROR",
                        },
                        {
                          span_id: "span-tool-77",
                          parent_id: "span-root",
                          trace_id: "trace-turn-2",
                          name: "tool.lookup_invoice",
                          tool_name: "lookup_invoice",
                          started_at: "2026-11-23T18:00:05.200Z",
                          duration_ms: 980,
                          status_code: "ERROR",
                          error: "422 currency required",
                          inputs: { invoice_id: "77" },
                        },
                      ],
                    },
                  ],
                }),
              },
            ],
          };
        }
        if (name === "manage_weave_agent_feedback_tool") {
          return {
            content: [{ type: "text", text: JSON.stringify({ id: "fb-new-1" }) }],
          };
        }
        if (name === "list_agent_lens_tags_tool") {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  data: [{ id: "11111111-2222-4333-8444-555555555555", name: "needs-eval", color: "#10BFCC" }],
                }),
              },
            ],
          };
        }
        if (name === "manage_agent_lens_tags_tool") {
          return {
            content: [{ type: "text", text: JSON.stringify({ data: { ok: true } }) }],
          };
        }
        return { content: [{ type: "text", text: "{}" }] };
      });

      await window.__MCP_APP_TEST_HOST__.mount(html, {
        theme: "dark",
        displayMode: "inline",
        availableDisplayModes: ["inline", "fullscreen"],
      });
      await window.__MCP_APP_TEST_HOST__.sendToolInput({ conversation_id: "conv-77" });
    }, appHtmlWithCsp);

    const frame = page.frameLocator("#mcp-app-frame");
    await frame.locator("#ac-title").waitFor();
    assert.match(await frame.locator("#ac-title").innerText(), /Conversation conv-77/);

    await page.evaluate(async (payload) => {
      await window.__MCP_APP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "{}" }],
        structuredContent: payload,
      });
    }, makeConversationPayload());

    // Verify turns, error badge, and lazy-loaded Agent Lens tag chip
    await frame.locator(".ac-turn-card").first().waitFor();
    assert.equal(await frame.locator(".ac-turn-card").count(), 2);
    assert.equal(await frame.locator("#ac-error-pill").innerText(), "1 error");
    await frame.locator(".ac-tag-chip").first().waitFor();
    assert.match(await frame.locator(".ac-tag-chip").first().innerText(), /#escalated/);

    // Verify Errors-only filter
    await frame.locator("#ac-errors-only-btn").click();
    assert.equal(await frame.locator(".ac-turn-card").count(), 1);
    await frame.locator("#ac-errors-only-btn").click();
    assert.equal(await frame.locator(".ac-turn-card").count(), 2);

    // Verify inline read-write feedback (reaction + note) and tag assignment
    await frame.locator(".ac-turn-card").nth(1).locator(".ac-react-down").click();
    await frame.locator(".ac-turn-card").nth(1).locator(".ac-fb-pill").first().waitFor();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "manage_weave_agent_feedback_tool" && c.arguments.action === "add_reaction",
      ),
    );

    await frame.locator(".ac-turn-card").nth(1).locator(".ac-add-note-btn").click();
    await frame.locator(".ac-note-input").fill("Pass USD default currency");
    await frame.locator(".ac-save-note-btn").click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "manage_weave_agent_feedback_tool" && c.arguments.action === "add_note",
      ),
    );

    await frame.locator("#ac-tag-menu-btn").click();
    await frame.locator(".ac-catalog-tag-btn").first().waitFor();
    await frame.locator(".ac-catalog-tag-btn").first().click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "manage_agent_lens_tags_tool" && c.arguments.action === "assign_tag",
      ),
    );

    // Verify OTel Spans tab lazy waterfall + Span Drawer + Context Sync
    await frame.locator("#ac-tab-spans").click();
    await frame.locator(".ac-wf-row").nth(1).waitFor();
    assert.equal(await frame.locator(".ac-wf-row").count(), 2);
    await frame.locator(".ac-wf-row").nth(1).click();
    assert.equal(await frame.locator("#ac-span-drawer").isVisible(), true);
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.contextUpdates.some(
        (u) => u?.structuredContent?.span_id === "span-tool-77",
      ),
    );

    // Verify Open in Forge Agent Lens HTTPS deep link + non-HTTPS block
    await frame.locator("#ac-open-forge-btn").click();
    await page.waitForFunction(() => window.__MCP_APP_TEST_HOST__.openedLinks.length === 1);
    assert.deepEqual(await page.evaluate(() => window.__MCP_APP_TEST_HOST__.openedLinks), [
      "https://forge.coreweave.com/acme/support-agent/agent-lens",
    ]);

    // Verify read-only mode hides write controls AND blocks http:// URL
    await page.setViewportSize({ width: 540, height: 640 });
    await page.evaluate(async (roPayload) => {
      await window.__MCP_APP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "{}" }],
        structuredContent: roPayload,
      });
    }, makeConversationPayload({ accessMode: "read-only", forgeUrl: "http://insecure.example/agent-lens" }));

    assert.equal(await frame.locator(".ac-react-down").count(), 0);
    assert.equal(await frame.locator("#ac-tag-menu-btn").isHidden(), true);
    await frame.locator("#ac-open-forge-btn").click();
    assert.equal(await page.evaluate(() => window.__MCP_APP_TEST_HOST__.openedLinks.length), 1);

    await page.evaluate(async () => {
      await window.__MCP_APP_TEST_HOST__.teardown();
    });
    await page.waitForFunction(() => {
      const last = window.__MCP_APP_TEST_HOST__.contextUpdates.at(-1);
      return Array.isArray(last?.content) && last.content.length === 0;
    });

    assert.deepEqual(externalRequests, [], `Unexpected external requests: ${externalRequests}`);
    assert.deepEqual(pageErrors, [], `Unexpected page errors: ${pageErrors}`);
    assert.deepEqual(consoleErrors, [], `Unexpected console errors: ${consoleErrors}`);

    console.log("Playwright Conversation Inspector AppBridge suite passed (0 console/page/CSP errors).");
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
