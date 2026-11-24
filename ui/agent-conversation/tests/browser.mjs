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
const contractFixture = JSON.parse(
  fs.readFileSync(path.resolve(uiRoot, "../shared/fixtures/agent-lens-contract.json"), "utf8"),
);

function makeConversationPayload(overrides = {}) {
  return {
    schema: "wandb.agent_conversation/v1",
    mode: "conversation",
    entityName: "acme",
    projectName: "support-agent",
    conversationId: "conv-stripe-8841",
    traceId: null,
    agentName: "billing-router",
    agentVersion: "v4.2",
    forgeUrl: "https://forge.coreweave.com/agent-lens/acme/support-agent",
    accessMode: "read-write",
    agentLensAvailable: true,
    compacted: false,
    selectedFailure: {
      traceId: "trace-stripe-8841",
      failureId: "77869ce8-2917-5d33-bc74-ab63603e278e",
    },
    summary: {
      turnCount: 2,
      totalTurns: 2,
      hasMore: false,
      totalInputTokens: 610,
      totalOutputTokens: 180,
      totalTokens: 790,
      totalDurationMs: 1530,
      totalCostUsd: 0.0046,
      errorCount: 1,
    },
    turns: [
      {
        turnIndex: 1,
        traceId: "trace-stripe-8840",
        spanId: "span-turn-8840",
        agentName: "billing-router",
        agentVersion: "v4.2",
        statusCode: "OK",
        isError: false,
        errorMessage: null,
        startedAt: "2026-11-23T21:14:00Z",
        durationMs: 410,
        inputTokens: 220,
        outputTokens: 68,
        totalTokens: 288,
        costUsd: 0.0014,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "user",
            content: "Can you check why EU-West invoice #8841 failed to settle?",
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
            content: "Let me look up EU-West invoice #8841 right now.",
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
        traceId: "trace-stripe-8841",
        spanId: "span-turn-8841",
        agentName: "billing-router",
        agentVersion: "v4.2",
        statusCode: "ERROR",
        isError: true,
        errorMessage: "StripeInvalidRequestError: 422 Unprocessable Entity — Missing required param: currency.",
        startedAt: "2026-11-23T21:14:05Z",
        durationMs: 1120,
        inputTokens: 390,
        outputTokens: 112,
        totalTokens: 502,
        costUsd: 0.0032,
        model: "gpt-5.4",
        messages: [
          {
            index: 0,
            role: "tool",
            content: null,
            toolName: "lookup_invoice",
            toolCallId: "call_8841",
            arguments: '{"invoice_id":"8841","ledger":"eu-west"}',
            toolCalls: [],
            toolResult: '{"status":422,"error":{"param":"currency","message":"Missing required param: currency."}}',
            spanId: "span-tool-8841",
            spanName: "tool.lookup_invoice",
            statusCode: "ERROR",
            error: "422 Unprocessable Entity: Missing required param: currency.",
            durationMs: 640,
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
    const page = await browser.newPage({ viewport: { width: 960, height: 720 } });
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

    await page.evaluate(
      async ({ html, fixture }) => {
        window.__MCP_APP_TEST_HOST__.setToolHandler((name) => {
          if (name === "get_agent_lens_conversation_tags_tool") {
            return {
              content: [{ type: "text", text: JSON.stringify(fixture.conversation_tags) }],
            };
          }
          if (name === "get_agent_lens_failure_attributions_tool") {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    data: [fixture.failure_attributions.data[0]],
                  }),
                },
              ],
            };
          }
          if (
            name === "query_weave_agent_spans_tool" ||
            name === "get_weave_agent_conversation_spans_tool"
          ) {
            return {
              content: [{ type: "text", text: JSON.stringify(fixture.weave_conversation_spans) }],
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
        await window.__MCP_APP_TEST_HOST__.sendToolInput({
          conversation_id: "conv-stripe-8841",
          selected_failure: {
            trace_id: "trace-stripe-8841",
            failure_id: "77869ce8-2917-5d33-bc74-ab63603e278e",
          },
        });
      },
      { html: appHtmlWithCsp, fixture: contractFixture },
    );

    const frame = page.frameLocator("#mcp-app-frame");
    await frame.locator("#ac-title").waitFor();
    assert.match(await frame.locator("#ac-title").innerText(), /Conversation conv-stripe-8841/);

    await page.evaluate(async (payload) => {
      await window.__MCP_APP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "{}" }],
        structuredContent: payload,
      });
    }, makeConversationPayload());

    // 1. Product tab names: Thread and Spans + copyable conv_id + UTC turn timestamps
    await frame.locator(".ac-turn-card").first().waitFor();
    assert.deepEqual(await frame.locator(".ac-tab").allInnerTexts(), ["Thread", "Spans"]);
    assert.equal(await frame.locator(".ac-turn-card").count(), 2);
    assert.match(await frame.locator(".ac-copy-conv-btn").innerText(), /conv:conv-stripe-8841/);
    assert.match(await frame.locator(".ac-turn-ts").first().innerText(), /UTC/);

    // 2. Real conversation_tags (`tag` field) + Agent failures list + PINNED finding
    await frame.locator(".ac-tag-chip").first().waitFor();
    assert.match(await frame.locator(".ac-tag-chip").first().innerText(), /#eu-ledger/);

    await frame.locator(".ac-failure-row").first().waitFor();
    assert.equal(await frame.locator("#ac-failures-btn").innerText(), "Agent failures (1)");
    const failureText = await frame.locator(".ac-failure-row").first().innerText();
    assert.match(failureText, /Turn #2/);
    assert.match(failureText, /High/);
    assert.match(failureText, /lookup_invoice omits required currency/);
    assert.match(failureText, /Pinned/i);

    // 3. Inline feedback + free-text human note sync + tag assignment
    await frame.locator(".ac-turn-card").nth(1).locator(".ac-react-down").click();
    await frame.locator(".ac-turn-card").nth(1).locator(".ac-fb-pill").first().waitFor();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "manage_weave_agent_feedback_tool" && c.arguments.action === "add_reaction",
      ),
    );

    await frame.locator(".ac-turn-card").nth(1).locator(".ac-add-note-btn").click();
    await frame.locator(".ac-note-input").click();
    await frame.locator(".ac-note-input").pressSequentially("Missing ISO currency on EU ledger; add to eval batch", { delay: 5 });
    await frame.locator(".ac-save-note-btn").click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.contextUpdates.some(
        (u) =>
          Array.isArray(u?.structuredContent?.human_notes) &&
          u.structuredContent.human_notes.includes("Missing ISO currency on EU ledger; add to eval batch"),
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

    // 4. Click Focus evidence span -> switches to Spans, marks Evidence row, selects span-tool-8841, syncs failure reason
    await frame.locator(".ac-focus-evidence-btn").first().click();
    await frame.locator('.ac-wf-row[data-evidence="true"]').first().waitFor();
    assert.match(await frame.locator('.ac-wf-row[data-evidence="true"]').first().innerText(), /Evidence/i);
    assert.equal(await frame.locator("#ac-span-drawer").isVisible(), true);
    assert.match(await frame.locator("#ac-span-drawer").innerText(), /Evidence for lookup_invoice omits required currency/);

    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.contextUpdates.some(
        (u) =>
          u?.structuredContent?.span_id === "span-tool-8841" &&
          u?.structuredContent?.is_evidence_span === true &&
          typeof u?.structuredContent?.failure_reason === "string",
      ),
    );

    // 5. Open in Forge Agent Lens deep link (/agent-lens/acme/support-agent/conversations/.../spans/turns/.../spans/...)
    await frame.locator("#ac-open-forge-btn").click();
    await page.waitForFunction(() => window.__MCP_APP_TEST_HOST__.openedLinks.length === 1);
    assert.deepEqual(await page.evaluate(() => window.__MCP_APP_TEST_HOST__.openedLinks), [
      "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/spans/turns/trace-stripe-8841/spans/span-tool-8841",
    ]);

    // 6. Read-only mode hides write controls and blocks http:// URL
    await page.setViewportSize({ width: 540, height: 640 });
    await page.evaluate(async (roPayload) => {
      await window.__MCP_APP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "{}" }],
        structuredContent: roPayload,
      });
    }, makeConversationPayload({ accessMode: "read-only", forgeUrl: "http://insecure.example/agent-lens" }));

    assert.equal(await frame.locator(".ac-react-down").count(), 0);
    assert.equal(await frame.locator("#ac-tag-menu-btn").isHidden(), true);

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
