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
const bundledAppPath = path.join(repoRoot, "src/wandb_mcp_server/data/line-plot.html");

function buildSampleGraphPayload({
  minStep = null,
  maxStep = null,
  active = true,
  runIds = ["run-done", "run-live"],
} = {}) {
  const series = [];
  if (runIds.includes("run-done")) {
    series.push({
      seriesId: '["run-done","train/loss","primary"]',
      metadataId: '["run-done","train/loss"]',
      role: "primary",
      kind: "line",
      metricName: "train/loss",
      historySeries: [
        { x: minStep ?? 0, y: 1.8 },
        { x: (minStep ?? 0) + 5, y: 1.1 },
        { x: maxStep ?? 10, y: 0.7 },
      ],
    });
  }
  if (runIds.includes("run-live")) {
    series.push({
      seriesId: '["run-live","train/loss","primary"]',
      metadataId: '["run-live","train/loss"]',
      role: "primary",
      kind: "line",
      metricName: "train/loss",
      historySeries: [
        { x: minStep ?? 0, y: 2.0 },
        { x: (minStep ?? 0) + 5, y: 0.9 },
        { x: maxStep ?? 10, y: active ? 0.55 : 0.42 },
      ],
    });
  }

  return {
    schema: "wandb.graph/v1",
    series,
    seriesStylesById: {
      '["run-done","train/loss","primary"]': {
        color: "#00a99d",
        lineWidth: 1.8,
        mark: "solid",
        colorKey: "run-done:train/loss",
      },
      '["run-live","train/loss","primary"]': {
        color: "#e87535",
        lineWidth: 1.8,
        mark: "solid",
        colorKey: "run-live:train/loss",
      },
    },
    seriesMetadataById: {
      '["run-done","train/loss"]': {
        displayName: "baseline-v1",
        title: "baseline-v1 · train/loss",
        runName: "run-done",
        runState: "finished",
        legendHref: "https://wandb.ai/acme/llm-train/runs/run-done",
        entityName: "acme",
        projectName: "llm-train",
      },
      '["run-live","train/loss"]': {
        displayName: "cand-v2",
        title: "cand-v2 · train/loss",
        runName: "run-live",
        runState: active ? "running" : "finished",
        legendHref: "https://wandb.ai/acme/llm-train/runs/run-live",
        entityName: "acme",
        projectName: "llm-train",
      },
    },
    runsById: {
      ...(runIds.includes("run-done")
        ? {
            "run-done": {
              displayName: "baseline-v1",
              state: "finished",
              active: false,
              url: "https://wandb.ai/acme/llm-train/runs/run-done",
            },
          }
        : {}),
      ...(runIds.includes("run-live")
        ? {
            "run-live": {
              displayName: "cand-v2",
              state: active ? "running" : "finished",
              active,
              url: "https://wandb.ai/acme/llm-train/runs/run-live",
            },
          }
        : {}),
    },
    hasActiveRuns: active,
    axis: {
      xKey: "_step",
      xIsDatetime: false,
      xScale: "linear",
      yScale: "linear",
      xTitle: "_step",
      yTitle: "loss",
    },
    xDomain: [null, null],
    yDomain: [null, null],
    title: "train/loss",
    legend: { show: true, position: "north" },
    effectiveConfig: {
      metrics: ["train/loss"],
      xAxis: "_step",
      smoothingWeight: 0,
      smoothingType: "exponential",
    },
    coverage: {
      sampled: true,
      truncated: false,
      ignoredConfig: [],
      perRunErrors: [],
      grouped: false,
      autoDiscoveredRuns: true,
    },
    url: "https://wandb.ai/acme/llm-train",
    workspaceUrl: "https://wandb.ai/acme/llm-train/workspace",
    range_cap: 20_000,
    arguments: {
      entity_name: "acme",
      project_name: "llm-train",
      run_ids: runIds,
      config: { metrics: ["train/loss"] },
      samples: 500,
      min_step: minStep,
      max_step: maxStep,
    },
  };
}

async function bundleTestHostJs() {
  const result = await viteBuild({
    configFile: false,
    logLevel: "silent",
    build: {
      write: false,
      target: "es2022",
      minify: false,
      lib: {
        entry: path.join(__dirname, "host.ts"),
        formats: ["es"],
        fileName: "host-bundle",
      },
    },
  });
  const output = Array.isArray(result) ? result[0] : result;
  const chunk = output.output.find((o) => o.type === "chunk");
  if (!chunk) {
    throw new Error("Failed to bundle tests/host.ts");
  }
  return chunk.code;
}

async function main() {
  const rawAppHtml = fs.readFileSync(bundledAppPath, "utf8");
  // Inject strict MCP App CSP inside srcdoc to verify zero unsafe-eval / zero network calls
  const strictCspTag = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none';">`;
  const appHtmlWithCsp = rawAppHtml.replace("<head>", `<head>\n    ${strictCspTag}`);

  const hostJs = await bundleTestHostJs();
  const hostPageHtml = `<!doctype html>
<html>
  <head><meta charset="utf-8"><title>MCP AppBridge Host Harness</title></head>
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
  const address = server.address();
  const origin = `http://127.0.0.1:${address.port}`;

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
    await page.clock.install({ time: new Date("2026-11-23T19:00:00Z") });
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

    // 1. Mount sandboxed iframe and verify MCP Apps handshake + initial tool input/result
    await page.evaluate(async (html) => {
      await window.__LP_TEST_HOST__.mount(html, {
        theme: "dark",
        displayMode: "inline",
        availableDisplayModes: ["inline", "fullscreen"],
      });
      await window.__LP_TEST_HOST__.sendToolInput({
        project_name: "llm-train",
        metric: "train/loss",
      });
    }, appHtmlWithCsp);

    const frame = page.frameLocator("#mcp-app-frame");
    await frame.locator("#lp-title").waitFor();
    assert.match(await frame.locator("#lp-title").innerText(), /train\/loss/);

    const initialPayload = buildSampleGraphPayload();
    await page.evaluate(async (payload) => {
      await window.__LP_TEST_HOST__.sendToolResult({
        content: [{ type: "text", text: "train/loss summary" }],
        structuredContent: payload,
      });
    }, initialPayload);

    await frame.locator("#lp-plot canvas").first().waitFor();
    assert.equal(await frame.locator("#lp-active-badge").innerText(), "1 active");
    assert.equal(await frame.locator(".lp-legend-item").count(), 2);

    // 2. Verify Expand chart -> fullscreen + stateless ui/update-model-context sync
    await frame.locator("#lp-expand-btn").click();
    await frame
      .locator('#app-root[data-display-mode="fullscreen"]')
      .waitFor({ state: "attached" });
    await page.clock.runFor(200);
    await page.waitForFunction(() =>
      window.__LP_TEST_HOST__.contextUpdates.some(
        (u) => u?.structuredContent?.display_mode === "fullscreen",
      ),
    );

    const modeAndCtx = await page.evaluate(() => ({
      modes: window.__LP_TEST_HOST__.requestedModes,
      latestCtx: window.__LP_TEST_HOST__.contextUpdates.at(-1),
    }));
    assert.deepEqual(modeAndCtx.modes, ["fullscreen"]);
    assert.equal(modeAndCtx.latestCtx?.structuredContent?.display_mode, "fullscreen");
    assert.deepEqual(modeAndCtx.latestCtx?.structuredContent?.run_ids, [
      "run-done",
      "run-live",
    ]);

    // 2b. Verify 10s active-run Live Refresh queries ONLY active run_ids and preserves finished runs
    await page.evaluate((partialFinishedPayload) => {
      window.__LP_TEST_HOST__.setNextToolHandler((args) => ({
        content: [{ type: "text", text: "refreshed active run" }],
        structuredContent: {
          ...partialFinishedPayload,
          arguments: {
            ...partialFinishedPayload.arguments,
            run_ids: args.run_ids,
          },
        },
      }));
    }, buildSampleGraphPayload({ active: false, runIds: ["run-live"] }));

    await page.clock.runFor(10_100);
    // Allow microtask postMessage round-trip to settle
    await page.waitForFunction(() => window.__LP_TEST_HOST__.toolCalls.length === 1);
    const liveCall = await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls[0]);
    assert.equal(liveCall.name, "render_wandb_line_plot_tool");
    assert.deepEqual(liveCall.arguments.run_ids, ["run-live"]);

    // Both baseline-v1 (cached finished) and cand-v2 remain in legend, and active badge hides once finished!
    await frame.locator("#lp-active-badge").waitFor({ state: "hidden" });
    assert.equal(await frame.locator(".lp-legend-item").count(), 2);

    // Advancing another 10s makes ZERO extra server calls because no active runs remain
    await page.clock.runFor(10_500);
    assert.equal(await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls.length), 1);

    // 3. Client-side Smoothing + Log Y + Series Isolate (zero extra server tool calls!)
    await frame.locator("#lp-smooth-slider").fill("0.6");
    await frame.locator("#lp-smooth-slider").dispatchEvent("input");
    await frame.locator("#lp-logy-btn").click();
    assert.equal(await frame.locator("#lp-logy-btn").getAttribute("aria-pressed"), "true");

    await frame.locator(".lp-legend-toggle").first().click({ modifiers: ["Meta"] });
    assert.equal(
      await frame.locator(".lp-legend-item").nth(1).getAttribute("data-visible"),
      "false",
    );
    // Un-isolate back to all visible
    await frame.locator(".lp-legend-toggle").first().click({ modifiers: ["Meta"] });
    assert.equal(
      await frame.locator(".lp-legend-item").nth(1).getAttribute("data-visible"),
      "true",
    );
    assert.equal(await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls.length), 1);

    // 4. Outward `_step` drag-zoom requeries server tool and shows Reset zoom
    await page.evaluate((zoomedPayload) => {
      window.__LP_TEST_HOST__.setNextToolHandler((args) => ({
        content: [{ type: "text", text: "zoomed" }],
        structuredContent: {
          ...zoomedPayload,
          arguments: {
            ...zoomedPayload.arguments,
            min_step: args.min_step,
            max_step: args.max_step,
          },
        },
      }));
    }, buildSampleGraphPayload({ minStep: 2, maxStep: 8 }));

    const canvasBox = await frame.locator("#lp-plot .u-over").boundingBox();
    assert(canvasBox, "Expected uPlot .u-over bounding box");
    await page.mouse.move(canvasBox.x + 60, canvasBox.y + 50);
    await page.mouse.down();
    await page.mouse.move(canvasBox.x + 240, canvasBox.y + 50, { steps: 6 });
    await page.mouse.up();

    await frame.locator("#lp-reset-zoom-btn").waitFor({ state: "visible" });
    const zoomCall = await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls.at(-1));
    assert.equal(zoomCall.name, "render_wandb_line_plot_tool");
    assert.equal(typeof zoomCall.arguments.min_step, "number");
    assert.equal(typeof zoomCall.arguments.max_step, "number");
    assert(zoomCall.arguments.min_step >= 0);
    assert(zoomCall.arguments.max_step > zoomCall.arguments.min_step);

    // Reset zoom restores overview without calling server
    const callsBeforeReset = await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls.length);
    await frame.locator("#lp-reset-zoom-btn").click();
    assert.equal(await frame.locator("#lp-reset-zoom-btn").isHidden(), true);
    assert.equal(
      await page.evaluate(() => window.__LP_TEST_HOST__.toolCalls.length),
      callsBeforeReset,
    );

    // 5. Non-destructive error handling when zoom requery fails: chart stays visible!
    await page.evaluate(() => {
      window.__LP_TEST_HOST__.setNextToolHandler(() => ({
        content: [{ type: "text", text: '{"error":"busy","message":"Upstream busy"}' }],
        structuredContent: { error: "busy", message: "Upstream busy" },
        isError: true,
      }));
    });
    await page.mouse.move(canvasBox.x + 70, canvasBox.y + 50);
    await page.mouse.down();
    await page.mouse.move(canvasBox.x + 210, canvasBox.y + 50, { steps: 5 });
    await page.mouse.up();

    await frame.locator("#lp-banner").waitFor({ state: "visible" });
    assert.match(await frame.locator("#lp-banner-text").innerText(), /Upstream busy/);
    assert.equal(await frame.locator("#lp-plot canvas").first().isVisible(), true);

    // 6. Open in W&B link + graceful teardown clearing model context
    await frame.locator("#lp-open-wandb-btn").click();
    await page.waitForFunction(() => window.__LP_TEST_HOST__.openedLinks.length === 1);
    const opened = await page.evaluate(() => window.__LP_TEST_HOST__.openedLinks);
    assert.deepEqual(opened, ["https://wandb.ai/acme/llm-train/workspace"]);

    await page.evaluate(async () => {
      await window.__LP_TEST_HOST__.teardown();
    });
    await page.waitForFunction(() => {
      const last = window.__LP_TEST_HOST__.contextUpdates.at(-1);
      return Array.isArray(last?.content) && last.content.length === 0;
    });
    const finalCtx = await page.evaluate(() => window.__LP_TEST_HOST__.contextUpdates.at(-1));
    assert.deepEqual(finalCtx?.content, []);

    assert.deepEqual(externalRequests, [], `Unexpected external requests: ${externalRequests}`);
    assert.deepEqual(pageErrors, [], `Unexpected page errors: ${pageErrors}`);
    assert.deepEqual(consoleErrors, [], `Unexpected console errors: ${consoleErrors}`);

    console.log("Playwright MCP AppBridge sandboxed iframe suite passed (0 console/page/CSP errors).");
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
