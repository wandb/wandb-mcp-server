import path from "node:path";
import { fileURLToPath } from "node:url";
import { finalizeMcpAppBundle } from "../../shared/finalize-bundle.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
finalizeMcpAppBundle({
  appRoot: path.resolve(__dirname, ".."),
  outFileName: "line-plot.html",
});
