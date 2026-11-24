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
const bundledAppPath = path.join(repoRoot, "src/wandb_mcp_server/data/agent-lens-insights.html");
const sharedHostEntry = path.resolve(uiRoot, "../shared/test-host.ts");

function makeInsightsPayload(windowDays = 7) {
  return {
    schema: "wandb.agent_lens_insights/v1",
    entityName: "acme",
    projectName: "support-agent",
    forgeUrl: "https://forge.coreweave.com/acme/support-agent/agent-lens",
    accessMode: "read-write",
    window: {
      startAt: "2026-11-16T00:00:00Z",
      endAt: "2026-11-23T00:00:00Z",
      windowDays,
      agentName: null,
    },
    coverage: {
      firstWeek: "2026-10-05",
      latestWeek: "2026-11-16",
      classifiedTurnCount: 950,
      totalTurnCount: 1000,
      coveragePct: 95.0,
    },
    clusteringStatus: [
      { signatureType: "failure", status: "ready", completedAt: "2026-11-23T03:00:00Z" },
    ],
    failure: {
      turnCount: windowDays === 14 ? 64 : 32,
      categories: [
        {
          categoryId: "tool_failure",
          label: "Tool Failure",
          count: windowDays === 14 ? 48 : 24,
          share: 0.75,
          severity: "critical",
          clusterBreakdowns: [{ id: "topic-stripe", title: "Stripe 422", count: 16, share: 0.5, severity: "critical" }],
        },
      ],
      clusters: [
        {
          id: "topic-stripe",
          title: "Stripe 422 Missing Currency",
          summary: "lookup_invoice calls omit required currency field",
          categoryId: "tool_failure",
          count: 16,
          share: 0.5,
          severity: "critical",
          attribution: "tool_error",
        },
      ],
    },
    intent: {
      turnCount: 918,
      categories: [
        {
          categoryId: "invoice_status",
          label: "Invoice Status",
          count: 600,
          share: 0.65,
          severity: null,
          clusterBreakdowns: [],
        },
      ],
      clusters: [],
    },
    failureAttributions: [],
    warnings: [],
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
  <head><meta charset="utf-8"><title>Insights Explorer Host Harness</title></head>
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
    const page = await browser.newPage({ viewport: { width: 960, height: 700 } });
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
      async ({ html, payload14d }) => {
        window.__MCP_APP_TEST_HOST__.setToolHandler((name) => {
          if (name === "render_agent_lens_insights_tool") {
            return {
              content: [{ type: "text", text: "{}" }],
              structuredContent: payload14d,
            };
          }
          if (name === "list_agent_lens_category_example_turns_tool") {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    data: [
                      {
                        conversation_id: "conv-stripe-1",
                        trace_id: "trace-stripe-1",
                        failure_reason: "422 currency required",
                        duration_ms: 910,
                        total_tokens: 380,
                      },
                    ],
                    next_cursor: null,
                  }),
                },
              ],
            };
          }
          if (name === "get_agent_lens_failure_attributions_tool") {
            return {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    data: [
                      {
                        trace_id: "trace-stripe-1",
                        root_cause: "tool_error",
                        summary: "lookup_invoice omitted required currency field",
                      },
                    ],
                  }),
                },
              ],
            };
          }
          if (name === "get_weave_agent_conversation_tool") {
            return {
              content: [{ type: "text", text: "{}" }],
              structuredContent: {
                schema: "wandb.agent_conversation/v1",
                conversationId: "conv-stripe-1",
                traceId: "trace-stripe-1",
                agentName: "billing-router",
                turns: [
                  {
                    turnIndex: 1,
                    traceId: "trace-stripe-1",
                    isError: true,
                    errorMessage: "422 currency required",
                    durationMs: 910,
                    totalTokens: 380,
                    messages: [
                      { role: "user", content: "Check Stripe invoice #88" },
                      {
                        role: "tool",
                        toolName: "lookup_invoice",
                        arguments: '{"invoice_id":"88"}',
                        toolResult: '{"status":422}',
                        error: "currency required",
                      },
                    ],
                  },
                ],
              },
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
          entity_name: "acme",
          project_name: "support-agent",
        });
      },
      { html: appHtmlWithCsp, payload14d: makeInsightsPayload(14) },
    );

    const frame = page.frameLocator("#mcp-app-frame");
    await frame.locator("#ali-title").waitFor();
    assert.equal(await frame.locator("#ali-title").innerText(), "acme/support-agent");

    await page.evaluate(async (payload7d) => {
      await window.__MCP_APP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "{}" }],
        structuredContent: payload7d,
      });
    }, makeInsightsPayload(7));

    await frame.locator(".ali-bar-card").first().waitFor();
    assert.match(await frame.locator("#ali-kpi-strip").innerText(), /32 turns/);
    assert.match(await frame.locator("#ali-kpi-strip").innerText(), /95\.0%/);

    // 1. Switch window preset to 14d -> calls render_agent_lens_insights_tool
    await frame.locator('.ali-preset-btn[data-days="14"]').click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "render_agent_lens_insights_tool" && c.arguments.window_days === 14,
      ),
    );
    assert.match(await frame.locator("#ali-kpi-strip").innerText(), /64 turns/);

    // 2. Expand Failure Category -> loads example turns + failure attributions
    await frame.locator(".ali-bar-trigger").first().click();
    await frame.locator(".ali-example-row").first().waitFor();
    assert.match(await frame.locator(".ali-example-row").first().innerText(), /tool_error/);
    assert.match(
      await frame.locator(".ali-example-row").first().innerText(),
      /lookup_invoice omitted required currency field/,
    );

    // 3. Click Inspect conversation -> opens Embedded Conversation Drawer + syncs context
    await frame.locator(".ali-inspect-btn").first().click();
    await frame.locator("#ali-conv-drawer .ali-drawer-turn").first().waitFor();
    assert.match(await frame.locator("#ali-conv-drawer").innerText(), /Check Stripe invoice #88/);
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.contextUpdates.some(
        (u) => u?.structuredContent?.conversation_id === "conv-stripe-1",
      ),
    );

    // 4. Switch to Topic Clusters tab and verify cluster drilldown with topic_ids
    await frame.locator("#ali-tab-clusters").click();
    await frame.locator(".ali-bar-trigger").first().waitFor();
    await frame.locator(".ali-bar-trigger").first().click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) =>
          c.name === "list_agent_lens_category_example_turns_tool" &&
          Array.isArray(c.arguments.topic_ids) &&
          c.arguments.topic_ids[0] === "topic-stripe",
      ),
    );

    // 5. Open in Forge Agent Lens + fullscreen + teardown
    await frame.locator("#ali-expand-btn").click();
    await frame.locator('#app-root[data-display-mode="fullscreen"]').waitFor({ state: "attached" });

    await frame.locator("#ali-open-forge-btn").click();
    await page.waitForFunction(() => window.__MCP_APP_TEST_HOST__.openedLinks.length === 1);
    assert.deepEqual(await page.evaluate(() => window.__MCP_APP_TEST_HOST__.openedLinks), [
      "https://forge.coreweave.com/acme/support-agent/agent-lens",
    ]);

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

    console.log("Playwright Insights Explorer AppBridge suite passed (0 console/page/CSP errors).");
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
