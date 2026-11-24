import { isAllowedHttpsUrl } from "../../shared/url-guard";

export type OpenLinkCallback = (url: string) => void;

const INLINE_TOKEN_RE =
  /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[([^\]\n]+)\]\((https:\/\/[^\s)]+)\)|(https:\/\/[^\s<>"'`]+?[a-zA-Z0-9/_#-]))/g;

function appendInlineNodes(
  parent: Node,
  text: string,
  onOpenLink: OpenLinkCallback,
): void {
  let lastIndex = 0;
  INLINE_TOKEN_RE.lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = INLINE_TOKEN_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
    }
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      const code = document.createElement("code");
      code.className = "ar-md-code";
      code.textContent = token.slice(1, -1);
      parent.appendChild(code);
    } else if (token.startsWith("**") && token.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.appendChild(strong);
    } else if (match[2] && match[3] && isAllowedHttpsUrl(match[3])) {
      const url = match[3];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ar-md-link";
      btn.textContent = `${match[2]} ↗`;
      btn.title = url;
      btn.addEventListener("click", () => onOpenLink(url));
      parent.appendChild(btn);
    } else if (match[4] && isAllowedHttpsUrl(match[4])) {
      const url = match[4];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ar-md-link";
      let shortLabel = url;
      try {
        const parsed = new URL(url);
        const pathTail =
          parsed.pathname.length > 34
            ? `${parsed.pathname.slice(0, 31)}…`
            : parsed.pathname;
        shortLabel = `${parsed.hostname}${pathTail}`;
      } catch {
        // keep raw
      }
      btn.textContent = `${shortLabel} ↗`;
      btn.title = url;
      btn.addEventListener("click", () => onOpenLink(url));
      parent.appendChild(btn);
    } else {
      parent.appendChild(document.createTextNode(token));
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parent.appendChild(document.createTextNode(text.slice(lastIndex)));
  }
}

function splitTableCells(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((c) => c.trim());
}

function isTableSeparatorLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.includes("|") && !trimmed.includes("-")) {
    return false;
  }
  const cells = splitTableCells(trimmed);
  return cells.length >= 2 && cells.every((c) => /^:?-{2,}:?$/.test(c));
}

export function renderMarkdownFragment(
  rawMarkdown: string,
  onOpenLink: OpenLinkCallback,
): DocumentFragment {
  const frag = document.createDocumentFragment();
  const lines = rawMarkdown.replace(/\r\n/g, "\n").split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i]!;
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // Fenced code block
    if (trimmed.startsWith("```")) {
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i]!.trim().startsWith("```")) {
        codeLines.push(lines[i]!);
        i += 1;
      }
      if (i < lines.length) {
        i += 1;
      }
      const pre = document.createElement("pre");
      pre.className = "ar-md-pre";
      if (lang) {
        pre.dataset.lang = lang;
      }
      const code = document.createElement("code");
      code.textContent = codeLines.join("\n");
      pre.appendChild(code);
      frag.appendChild(pre);
      continue;
    }

    // GFM Markdown table
    if (
      trimmed.includes("|") &&
      i + 1 < lines.length &&
      isTableSeparatorLine(lines[i + 1]!)
    ) {
      const headers = splitTableCells(trimmed);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i]!.trim().includes("|") && lines[i]!.trim() !== "") {
        rows.push(splitTableCells(lines[i]!));
        i += 1;
      }
      const wrap = document.createElement("div");
      wrap.className = "ar-md-table-wrap";
      const table = document.createElement("table");
      table.className = "ar-md-table";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const h of headers) {
        const th = document.createElement("th");
        appendInlineNodes(th, h, onOpenLink);
        headRow.appendChild(th);
      }
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const r of rows) {
        const tr = document.createElement("tr");
        for (let col = 0; col < headers.length; col += 1) {
          const td = document.createElement("td");
          appendInlineNodes(td, r[col] ?? "", onOpenLink);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      wrap.appendChild(table);
      frag.appendChild(wrap);
      continue;
    }

    // Headings
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(4, headingMatch[1]!.length + 1);
      const heading = document.createElement(`h${level}` as "h2" | "h3" | "h4");
      heading.className = `ar-md-h ar-md-h${level}`;
      appendInlineNodes(heading, headingMatch[2]!, onOpenLink);
      frag.appendChild(heading);
      i += 1;
      continue;
    }

    // Unordered or ordered lists
    if (/^[-*+]\s+/.test(trimmed) || /^\d+\.\s+/.test(trimmed)) {
      const ordered = /^\d+\.\s+/.test(trimmed);
      const list = document.createElement(ordered ? "ol" : "ul");
      list.className = "ar-md-list";
      while (i < lines.length) {
        const itemTrim = lines[i]!.trim();
        const bulletMatch = ordered
          ? itemTrim.match(/^\d+\.\s+(.+)$/)
          : itemTrim.match(/^[-*+]\s+(.+)$/);
        if (!bulletMatch) {
          break;
        }
        const li = document.createElement("li");
        appendInlineNodes(li, bulletMatch[1]!, onOpenLink);
        list.appendChild(li);
        i += 1;
      }
      frag.appendChild(list);
      continue;
    }

    // Blockquote
    if (trimmed.startsWith(">")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i]!.trim().startsWith(">")) {
        quoteLines.push(lines[i]!.trim().replace(/^>\s?/, ""));
        i += 1;
      }
      const bq = document.createElement("blockquote");
      bq.className = "ar-md-quote";
      appendInlineNodes(bq, quoteLines.join(" "), onOpenLink);
      frag.appendChild(bq);
      continue;
    }

    // Standard paragraph
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i]!.trim() !== "" &&
      !lines[i]!.trim().startsWith("```") &&
      !lines[i]!.trim().startsWith("#") &&
      !lines[i]!.trim().startsWith(">") &&
      !/^[-*+]\s+/.test(lines[i]!.trim()) &&
      !/^\d+\.\s+/.test(lines[i]!.trim()) &&
      !(
        lines[i]!.trim().includes("|") &&
        i + 1 < lines.length &&
        isTableSeparatorLine(lines[i + 1]!)
      )
    ) {
      paraLines.push(lines[i]!.trim());
      i += 1;
    }
    if (paraLines.length > 0) {
      const p = document.createElement("p");
      p.className = "ar-md-p";
      appendInlineNodes(p, paraLines.join(" "), onOpenLink);
      frag.appendChild(p);
    } else {
      i += 1;
    }
  }

  return frag;
}
