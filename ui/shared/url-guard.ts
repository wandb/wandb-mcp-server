/**
 * Client-side defense-in-depth scheme validator for external deep links
 * passed to `app.openLink({ url })`. Only explicit `https://` origins are permitted.
 */
export function isAllowedHttpsUrl(raw: string): boolean {
  if (typeof raw !== "string") {
    return false;
  }
  const trimmed = raw.trim();
  if (!trimmed || !/^https:\/\//i.test(trimmed)) {
    return false;
  }
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === "https:" && Boolean(parsed.hostname);
  } catch {
    return false;
  }
}
