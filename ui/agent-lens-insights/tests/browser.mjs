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
const contractFixture = JSON.parse(
  fs.readFileSync(path.resolve(uiRoot, "../shared/fixtures/agent-lens-contract.json"), "utf8"),
);

function makeInsightsPayload(windowDays = 7) {
  const scale = windowDays === 30 ? 4 : 1;
  const f = contractFixture.failure_breakdowns.data;
  const i = contractFixture.intent_breakdowns.data;
  const toolMisuse = f.categories.find((c) => c.category === "tool_misuse");
  const clusterRow = toolMisuse.cluster_breakdowns[0];

  const totalTurns = f.turn_count * scale;
  const failingTurns = 612 * scale;
  const catCount = toolMisuse.count * scale;
  const clCount = clusterRow.count * scale;

  return {
    schema: "wandb.agent_lens_insights/v1",
    entityName: "acme",
    projectName: "support-agent",
    forgeUrl: "https://forge.coreweave.com/agent-lens/acme/support-agent",
    accessMode: "read-write",
    window: {
      startAt: windowDays === 30 ? contractFixture.window_30d.start_at : contractFixture.window_7d.start_at,
      endAt: contractFixture.window_7d.end_at,
      windowDays,
      agentName: "billing-router",
    },
    coverage: {
      firstWeek: contractFixture.latest_week.data.first_week,
      latestWeek: contractFixture.latest_week.data.latest_week,
    },
    clusteringStatus: [
      {
        signatureType: "failure",
        status: "completed",
        completedAt: "2026-11-24T03:19:48Z",
        windowStart: "2026-11-17T03:00:00Z",
        windowEnd: "2026-11-24T03:00:00Z",
      },
    ],
    failure: {
      turnCount: totalTurns,
      failingTurnCount: failingTurns,
      categories: [
        {
          categoryId: "tool_misuse",
          label: "Tool misuse",
          count: catCount,
          share: catCount / totalTurns,
          conversationCount: toolMisuse.conversation_count * scale,
          failedCount: catCount,
          frustratedCount: toolMisuse.frustrated_count * scale,
          failureRate: 1,
          frustrationRate: toolMisuse.frustrated_count / toolMisuse.count,
          averageLatencyMs: toolMisuse.average_latency_ms,
          averageCostPerTurnUsd: toolMisuse.average_cost_per_turn_usd,
          clusterBreakdowns: [
            {
              id: clusterRow.id,
              title: clusterRow.title,
              label: clusterRow.label,
              description: clusterRow.description,
              severity: clusterRow.severity,
              count: clCount,
              share: clCount / totalTurns,
              conversationCount: clusterRow.conversation_count * scale,
              failedCount: clCount,
              frustratedCount: clusterRow.frustrated_count * scale,
              failureRate: 1,
              frustrationRate: clusterRow.frustrated_count / clusterRow.count,
              averageLatencyMs: clusterRow.average_latency_ms,
              averageCostPerTurnUsd: clusterRow.average_cost_per_turn_usd,
            },
          ],
        },
      ],
      clusters: [],
    },
    intent: {
      turnCount: i.turn_count * scale,
      failingTurnCount: null,
      categories: [
        {
          categoryId: "action_request",
          label: "Task request",
          count: i.categories[0].count * scale,
          share: i.categories[0].count / i.turn_count,
          conversationCount: i.categories[0].conversation_count * scale,
          failedCount: i.categories[0].failed_count * scale,
          frustratedCount: i.categories[0].frustrated_count * scale,
          failureRate: i.categories[0].failed_count / i.categories[0].count,
          frustrationRate: i.categories[0].frustrated_count / i.categories[0].count,
          averageLatencyMs: i.categories[0].average_latency_ms,
          averageCostPerTurnUsd: i.categories[0].average_cost_per_turn_usd,
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
    const page = await browser.newPage({ viewport: { width: 1024, height: 740 } });
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
      async ({ html, payload30d, fixture }) => {
        window.__MCP_APP_TEST_HOST__.setToolHandler((name) => {
          if (name === "render_agent_lens_insights_tool") {
            return {
              content: [{ type: "text", text: "{}" }],
              structuredContent: payload30d,
            };
          }
          if (name === "list_agent_lens_category_example_turns_tool") {
            return {
              content: [{ type: "text", text: JSON.stringify(fixture.example_turns) }],
            };
          }
          if (name === "get_agent_lens_failure_attributions_tool") {
            return {
              content: [{ type: "text", text: JSON.stringify(fixture.failure_attributions) }],
            };
          }
          if (name === "get_weave_agent_conversation_tool") {
            return {
              content: [{ type: "text", text: JSON.stringify(fixture.weave_conversation) }],
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
      { html: appHtmlWithCsp, payload30d: makeInsightsPayload(30), fixture: contractFixture },
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

    // 1. Two tabs only (no Topic Clusters tab) and 24h / 7d / 30d presets (no 14d)
    assert.equal(await frame.locator(".ali-tab").count(), 2);
    assert.deepEqual(await frame.locator(".ali-preset-btn").allInnerTexts(), ["24h", "7d", "30d"]);

    const kpiText = await frame.locator("#ali-kpi-strip").innerText();
    assert.match(kpiText, /Turns:\s*12,840/);
    assert.match(kpiText, /Failing turns:\s*612 \(4\.8%\)/);
    assert.doesNotMatch(kpiText, /Classified/);

    // 2. Category bar shows "% of turns"
    assert.match(await frame.locator(".ali-bar-card").first().innerText(), /2\.8% of turns/);

    // 3. Switch window preset to 30d -> calls render_agent_lens_insights_tool(window_days=30)
    await frame.locator('.ali-preset-btn[data-days="30"]').click();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "render_agent_lens_insights_tool" && c.arguments.window_days === 30,
      ),
    );
    assert.match(await frame.locator("#ali-kpi-strip").innerText(), /51,360/);

    // 4. Expand category -> shows impact grid + nested cluster -> expand cluster -> loads example turns
    await frame.locator(".ali-bar-trigger").first().click();
    await frame.locator(".ali-cluster-trigger").first().waitFor();
    assert.match(await frame.locator(".ali-impact-grid").first().innerText(), /Conversations/);
    assert.match(await frame.locator(".ali-cluster-trigger").first().innerText(), /High/);

    await frame.locator(".ali-cluster-trigger").first().click();
    await frame.locator(".ali-example-row").first().waitFor();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) =>
          c.name === "list_agent_lens_category_example_turns_tool" &&
          c.arguments.category_id === "tool_misuse" &&
          Array.isArray(c.arguments.topic_ids) &&
          c.arguments.topic_ids[0] === "01a1fc1d-f6a4-7e82-9d62-fbdeb71f5d12",
      ),
    );

    const exRowText = await frame.locator(".ali-example-row").first().innerText();
    assert.match(exRowText, /Turn 2/);
    assert.match(exRowText, /High/);
    assert.match(exRowText, /1 evidence span/);
    assert.doesNotMatch(exRowText, /\[\[span:/);
    assert.doesNotMatch(exRowText, /tok\b/);

    // Verify failureClusterToMarkdown brief sent via ui/update-model-context
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.contextUpdates.some(
        (u) =>
          typeof u?.content?.[0]?.text === "string" &&
          u.content[0].text.includes("# lookup_invoice omits required currency") &&
          u.content[0].text.includes("## Examples"),
      ),
    );

    // 5. Inspect conversation passes selected_failure and opens embedded drawer
    await frame.locator(".ali-inspect-btn").first().click();
    await frame.locator("#ali-conv-drawer .ali-drawer-turn").first().waitFor();
    await page.waitForFunction(() =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) =>
          c.name === "get_weave_agent_conversation_tool" &&
          c.arguments.selected_failure?.failure_id === "77869ce8-2917-5d33-bc74-ab63603e278e",
      ),
    );

    // 6. Fullscreen + Open in Forge deep link (/agent-lens/acme/support-agent/conversations/...)
    await frame.locator("#ali-expand-btn").click();
    await frame.locator('#app-root[data-display-mode="fullscreen"]').waitFor({ state: "attached" });

    await frame.locator("#ali-open-forge-btn").click();
    await page.waitForFunction(() => window.__MCP_APP_TEST_HOST__.openedLinks.length === 1);
    assert.deepEqual(await page.evaluate(() => window.__MCP_APP_TEST_HOST__.openedLinks), [
      "https://forge.coreweave.com/agent-lens/acme/support-agent/conversations/conv-stripe-8841/thread/turns/trace-stripe-8841",
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
