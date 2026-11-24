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
const bundledAppPath = path.join(repoRoot, "src/wandb_mcp_server/data/aria-agent.html");
const sharedHostEntry = path.resolve(uiRoot, "../shared/test-host.ts");

async function buildHostHarnessBundle() {
  const result = await viteBuild({
    configFile: false,
    logLevel: "silent",
    build: {
      write: false,
      minify: false,
      target: "es2022",
      lib: {
        entry: sharedHostEntry,
        formats: ["es"],
        fileName: "host",
      },
    },
  });
  const output = Array.isArray(result) ? result[0].output : result.output;
  return output[0].code;
}

const appHtml = fs.readFileSync(bundledAppPath, "utf8");
const hostJs = await buildHostHarnessBundle();

const hostPageHtml = `<!doctype html>
<html>
  <head><meta charset="utf-8" /><title>ARIA MCP App Host Test</title></head>
  <body>
    <div id="host-container"></div>
    <script type="module" src="/host.js"></script>
  </body>
</html>`;

const server = http.createServer((req, res) => {
  if (req.url === "/host.js") {
    res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
    res.end(hostJs);
    return;
  }
  res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
  res.end(hostPageHtml);
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const { port } = server.address();

const browser = await chromium.launch({ headless: true });
const consoleErrors = [];

try {
  const page = await browser.newPage({ viewport: { width: 980, height: 740 } });
  page.on("pageerror", (err) => consoleErrors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(`console.error: ${msg.text()}`);
    }
  });

  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle" });

  // -------------------------------------------------------------------------
  // Scenario 1: 1-to-1 IDE <-> ARIA Back-and-Forth (Polling -> Permission -> Reply)
  // -------------------------------------------------------------------------
  await page.evaluate(async (html) => {
    window.__MCP_APP_TEST_HOST__.setToolHandler(async (name, args) => {
      if (name === "aria_get_turn" && args.turn_id === "turn-1") {
        return {
          structuredContent: {
            result: {
              ok: true,
              turn_id: "turn-1",
              thread_id: "thread-solo",
              title: "Chorus Extrapolation Audit",
              user_prompt: "Audit 16-bar chorus drift and cite the looped-attention paper.",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              requires_permission: true,
              permission_requests: [
                {
                  type: "network_access",
                  reason: "Fetch the looped-attention paper on arXiv without repeating workspace evals.",
                  domains: ["arxiv.org"],
                },
              ],
              tool_steps: [
                { call_id: "c1", name: "query_wandb", status: "ok" },
                { call_id: "c2", name: "shell", status: "ok" },
                { call_id: "c3", name: "request_network_access", status: "ok" },
              ],
              progress: {
                phase: "awaiting_permission",
                status_text: "ARIA paused to request outbound network access to arxiv.org.",
                elapsed_seconds: 44,
                message_count: 4,
                tool_call_record_count: 6,
                requested_network_domains: ["arxiv.org"],
              },
              forge_url: "https://forge.coreweave.com/home/aria/threads/thread-solo",
              access_mode: "read-write",
            },
          },
        };
      }
      if (name === "aria_send_message" && args.parent_turn_id === "turn-1") {
        return {
          structuredContent: {
            result: {
              ok: true,
              turn_id: "turn-2",
              thread_id: "thread-solo",
              parent_turn_id: "turn-1",
              user_prompt: String(args.message),
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              active_allowed_network_domains: ["arxiv.org"],
              latest_response:
                "### Extrapolation Summary\n\n| Arm | 16-Bar Acc |\n| --- | --- |\n| Baseline | 0.612 |\n| Gated Loop | **0.941** |\n\nSee [Run comparison](https://wandb.ai/studio-ai/looped-synth-study).",
              tool_steps: [{ call_id: "c4", name: "shell", status: "ok" }],
              progress: {
                phase: "completed",
                status_text: "ARIA completed the turn.",
                elapsed_seconds: 26,
                message_count: 3,
                tool_call_record_count: 2,
              },
              forge_url: "https://forge.coreweave.com/home/aria/threads/thread-solo",
              access_mode: "read-write",
            },
          },
        };
      }
      if (name === "aria_send_message" && args.parent_turn_id === "turn-2") {
        return {
          structuredContent: {
            result: {
              ok: true,
              turn_id: "turn-3",
              thread_id: "thread-solo",
              parent_turn_id: "turn-2",
              user_prompt: String(args.message),
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              latest_response: "Seed-0 and seed-1 stay within ±0.008 on 16-bar phrases.",
              progress: {
                phase: "completed",
                elapsed_seconds: 12,
                message_count: 2,
                tool_call_record_count: 2,
              },
              access_mode: "read-write",
            },
          },
        };
      }
      return { isError: true, content: [{ type: "text", text: `unhandled ${name}` }] };
    });

    await window.__MCP_APP_TEST_HOST__.mount(html, { theme: "dark", displayMode: "inline" });
    await window.__MCP_APP_TEST_HOST__.sendToolInput({
      message: "Audit 16-bar chorus drift and cite the looped-attention paper.",
      entity: "studio-ai",
      project: "looped-synth-study",
    });
  }, appHtml);

  const frame = page.frameLocator("#mcp-app-frame");
  await frame.locator(".ar-user-text").waitFor({ state: "visible" });
  assert.equal(await frame.locator("#ar-fleet-bar").isVisible(), false, "1-to-1 mode must hide parallel fleet bar");

  // Push in-progress polling turn
  await page.evaluate(async () => {
    await window.__MCP_APP_TEST_HOST__.sendToolResult({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "turn-1",
          thread_id: "thread-solo",
          title: "Chorus Extrapolation Audit",
          user_prompt: "Audit 16-bar chorus drift and cite the looped-attention paper.",
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          state: "in_progress",
          is_terminal: false,
          poll_after_seconds: 5,
          tool_steps: [
            { call_id: "c1", name: "query_wandb", status: "ok" },
            { call_id: "c2", name: "shell", status: "running" },
          ],
          progress: {
            phase: "working",
            status_text: "ARIA is running shell.",
            elapsed_seconds: 19,
            message_count: 2,
            tool_call_record_count: 3,
          },
          forge_url: "https://forge.coreweave.com/home/aria/threads/thread-solo",
          access_mode: "read-write",
        },
      },
    });
  });

  await frame.locator(".ar-step-pill[data-status='running']").waitFor({ state: "visible" });
  assert.match(await frame.locator("#ar-title").innerText(), /Chorus Extrapolation Audit/);

  // Poll now -> advances to awaiting_permission (`arxiv.org`)
  await frame.locator("#ar-poll-now-btn").click();
  const grantBtn = frame.locator("button[data-action='grant-network']");
  await grantBtn.waitFor({ state: "visible" });
  assert.match(await grantBtn.innerText(), /arxiv\.org/);

  // Grant access inline -> calls aria_send_message(parent_turn_id='turn-1', allowed_network_domains=['arxiv.org'])
  await grantBtn.click();
  await frame.locator(".ar-md-table").waitFor({ state: "visible" });
  assert.match(await frame.locator(".ar-md-table").innerText(), /0\.941/);

  // Click validated HTTPS link in ARIA markdown response
  await frame.locator(".ar-md-link").first().click();
  await page.waitForFunction(() =>
    window.__MCP_APP_TEST_HOST__.openedLinks.includes(
      "https://wandb.ai/studio-ai/looped-synth-study",
    ),
  );

  // Send 1-to-1 follow-up message with Enter key (no <form> submit!)
  await frame.locator("#ar-reply-input").fill("Check seed-0 vs seed-1 variance.");
  await frame.locator("#ar-reply-input").press("Enter");
  await page.waitForFunction(
    () =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) => c.name === "aria_send_message" && c.arguments.parent_turn_id === "turn-2",
      ),
  );
  await frame.locator(".ar-turn-group[data-turn-id='turn-3']").waitFor({ state: "visible" });

  // -------------------------------------------------------------------------
  // Scenario 2: N=4 Parallel Async ARIA Sub-Agents Fleet + Batch Polling + Read-Only Guard
  // -------------------------------------------------------------------------
  await page.evaluate(async (html) => {
    window.__MCP_APP_TEST_HOST__.toolCalls.length = 0;
    window.__MCP_APP_TEST_HOST__.contextUpdates.length = 0;

    window.__MCP_APP_TEST_HOST__.setToolHandler(async (name, args) => {
      if (name === "aria_get_turn") {
        return {
          structuredContent: {
            result: {
              ok: true,
              turn_id: "arm-b",
              state: "completed",
              is_terminal: true,
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              parallel_turns: [
                {
                  ok: true,
                  turn_id: "arm-b",
                  thread_id: "th-b",
                  title: "Arm B · 1-Layer Control",
                  scope: { entity: "studio-ai", project: "looped-synth-study" },
                  state: "completed",
                  is_terminal: true,
                  latest_response: "Arm B plateaued at **0.604**.",
                },
                {
                  ok: true,
                  turn_id: "arm-d",
                  thread_id: "th-d",
                  title: "Arm D · Weight-Tied Loop",
                  scope: { entity: "studio-ai", project: "looped-synth-study" },
                  state: "completed",
                  is_terminal: true,
                  latest_response: "Arm D reached **0.889**.",
                },
              ],
              access_mode: "read-write",
            },
          },
        };
      }
      return { isError: true, content: [{ type: "text", text: `unexpected ${name}` }] };
    });

    await window.__MCP_APP_TEST_HOST__.mount(html, { theme: "dark", displayMode: "inline" });
    await window.__MCP_APP_TEST_HOST__.sendToolResult({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "arm-a",
          state: "completed",
          is_terminal: true,
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          access_mode: "read-write",
          parallel_turns: [
            {
              ok: true,
              turn_id: "arm-a",
              thread_id: "th-a",
              title: "Arm A · 4-Layer Baseline",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              latest_response: "Baseline 16-bar score **0.742**.",
            },
            {
              ok: true,
              turn_id: "arm-b",
              thread_id: "th-b",
              title: "Arm B · 1-Layer Control",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "in_progress",
              is_terminal: false,
              poll_after_seconds: 6,
              progress: {
                phase: "working",
                status_text: "ARIA is running shell.",
                elapsed_seconds: 51,
                message_count: 3,
                tool_call_record_count: 4,
                latest_tool_activity: { name: "shell", type: "invocation" },
              },
            },
            {
              ok: true,
              turn_id: "arm-c",
              thread_id: "th-c",
              title: "Arm C · Gated Input-Injected",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "completed",
              is_terminal: true,
              latest_response: "Winner: **0.948** across 3 seeds.",
            },
            {
              ok: true,
              turn_id: "arm-d",
              thread_id: "th-d",
              title: "Arm D · Weight-Tied Loop",
              scope: { entity: "studio-ai", project: "looped-synth-study" },
              state: "in_progress",
              is_terminal: false,
              poll_after_seconds: 6,
              progress: {
                phase: "working",
                status_text: "ARIA is running query_wandb.",
                elapsed_seconds: 38,
                message_count: 2,
                tool_call_record_count: 2,
                latest_tool_activity: { name: "query_wandb", type: "invocation" },
              },
            },
          ],
        },
      },
    });
  }, appHtml);

  const parallelFrame = page.frameLocator("#mcp-app-frame");
  await parallelFrame.locator("#ar-fleet-bar").waitFor({ state: "visible" });
  assert.equal(await parallelFrame.locator(".ar-agent-card").count(), 4);
  assert.match(await parallelFrame.locator("#ar-poll-now-btn").innerText(), /Poll active \(2\)/);

  // Batch poll active turns -> single aria_get_turn with parallel_turn_ids=['arm-d']
  await parallelFrame.locator("#ar-poll-now-btn").click();
  await page.waitForFunction(
    () =>
      window.__MCP_APP_TEST_HOST__.toolCalls.some(
        (c) =>
          c.name === "aria_get_turn" &&
          c.arguments.turn_id === "arm-b" &&
          Array.isArray(c.arguments.parallel_turn_ids) &&
          c.arguments.parallel_turn_ids.includes("arm-d"),
      ),
  );
  await parallelFrame.locator(".ar-fleet-counts").waitFor({ state: "visible" });
  assert.match(await parallelFrame.locator(".ar-fleet-counts").innerText(), /4 completed/);

  // Switch into Split focus for Arm C
  await parallelFrame
    .locator(".ar-agent-card[data-session-id='th-c'] button")
    .filter({ hasText: "Open conversation" })
    .click();
  await parallelFrame.locator("#ar-rail").waitFor({ state: "visible" });
  assert.match(await parallelFrame.locator(".ar-assistant-card").innerText(), /0\.948/);

  // Verify model context sync published parallel structuredContent
  const ctxUpdates = await page.evaluate(() => window.__MCP_APP_TEST_HOST__.contextUpdates);
  assert.ok(
    ctxUpdates.some(
      (u) => u.structuredContent && u.structuredContent.mode === "parallel",
    ),
    "Expected parallel model context update",
  );

  // Verify read-only mode hides write controls
  await page.evaluate(async () => {
    await window.__MCP_APP_TEST_HOST__.sendToolResult({
      structuredContent: {
        result: {
          ok: true,
          turn_id: "arm-c",
          thread_id: "th-c",
          scope: { entity: "studio-ai", project: "looped-synth-study" },
          state: "completed",
          is_terminal: true,
          latest_response: "Winner: **0.948** across 3 seeds.",
          access_mode: "read-only",
        },
      },
    });
  });
  assert.equal(await parallelFrame.locator("#ar-spawn-toggle-btn").isVisible(), false);
  assert.equal(await parallelFrame.locator("#ar-composer").isVisible(), false);

  await page.evaluate(() => window.__MCP_APP_TEST_HOST__.teardown());
  assert.deepEqual(consoleErrors, [], `Unexpected console/page errors: ${consoleErrors.join("; ")}`);
  console.log("aria-agent AppBridge browser suite passed (1-to-1 + 4-parallel + read-only).");
} finally {
  await browser.close();
  server.close();
}
