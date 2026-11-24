import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uiWorkspaceRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(uiWorkspaceRoot, "..");

/**
 * Verify single-file inline HTML invariants, stamp SHA-256 bundle integrity,
 * write `src/wandb_mcp_server/data/<outFileName>`, and deterministically regenerate
 * unified `src/wandb_mcp_server/data/THIRD_PARTY_NOTICES.txt` across all MCP Apps.
 */
export function finalizeMcpAppBundle({ appRoot, outFileName }) {
  const distHtmlPath = path.join(appRoot, "dist/index.html");
  const outHtmlPath = path.join(repoRoot, "src/wandb_mcp_server/data", outFileName);
  const outNoticesPath = path.join(repoRoot, "src/wandb_mcp_server/data/THIRD_PARTY_NOTICES.txt");

  let html = fs.readFileSync(distHtmlPath, "utf8").replace(/\r\n/g, "\n");

  const scriptMatch = html.match(/<script[^>]*>([\s\S]*?)<\/script>/i);
  const styleMatch = html.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
  if (!scriptMatch || !styleMatch) {
    throw new Error(`Expected single-file inline <script> and <style> blocks in ${distHtmlPath}.`);
  }

  // Verify zero external network asset URLs in HTML attributes, CSS url(...), or network fetch calls
  const externalAttr = /\b(?:src|href)\s*=\s*["'](?:https?:)?\/\//i;
  const externalCssUrl = /url\(\s*["']?(?:https?:)?\/\//i;
  const fetchToken = "fe" + "tch(";
  if (externalAttr.test(html) || externalCssUrl.test(styleMatch[1]) || html.includes(fetchToken)) {
    throw new Error(`Bundled ${outFileName} contains an external URL or network call reference.`);
  }

  const jsSha = crypto.createHash("sha256").update(scriptMatch[1], "utf8").digest("hex");
  const cssSha = crypto.createHash("sha256").update(styleMatch[1], "utf8").digest("hex");

  html = `${html.trimEnd()}\n<!-- bundle-integrity js:${jsSha} css:${cssSha} -->\n`;
  fs.writeFileSync(outHtmlPath, html, "utf8");

  // Deterministically collect MIT license notices across all bundled runtime packages
  const runtimePackages = [
    "@modelcontextprotocol/ext-apps",
    "@modelcontextprotocol/client",
    "@modelcontextprotocol/core",
    "uplot",
    "zod",
  ];

  const candidateModuleRoots = [
    path.join(uiWorkspaceRoot, "node_modules"),
    path.join(uiWorkspaceRoot, "line-plot/node_modules"),
    path.join(uiWorkspaceRoot, "agent-conversation/node_modules"),
    path.join(uiWorkspaceRoot, "agent-lens-insights/node_modules"),
  ];

  const sections = [
    "THIRD-PARTY SOFTWARE NOTICES — wandb_mcp_server/data/*.html",
    "  - line-plot.html",
    "  - agent-conversation.html",
    "  - agent-lens-insights.html",
    "====================================================================",
    "",
  ];

  for (const pkgName of runtimePackages) {
    const pkgDir = candidateModuleRoots
      .map((root) => path.join(root, pkgName))
      .find((candidate) => fs.existsSync(path.join(candidate, "package.json")));
    if (!pkgDir) {
      continue;
    }
    const pkgJson = JSON.parse(fs.readFileSync(path.join(pkgDir, "package.json"), "utf8"));
    const licenseCandidates = ["LICENSE", "LICENSE.md", "LICENSE.txt"];
    let licenseText = "";
    for (const fname of licenseCandidates) {
      const fpath = path.join(pkgDir, fname);
      if (fs.existsSync(fpath)) {
        licenseText = fs.readFileSync(fpath, "utf8").trim();
        break;
      }
    }
    sections.push(
      "--------------------------------------------------------------------",
      `${pkgJson.name}@${pkgJson.version} (${pkgJson.license || "MIT"})`,
      "--------------------------------------------------------------------",
      licenseText || `Licensed under ${pkgJson.license || "MIT"}.`,
      "",
    );
  }

  fs.writeFileSync(outNoticesPath, `${sections.join("\n").trimEnd()}\n`, "utf8");
  console.log(
    `Wrote ${path.relative(repoRoot, outHtmlPath)} (${Buffer.byteLength(html, "utf8")} bytes) and ${path.relative(repoRoot, outNoticesPath)}`,
  );
}
