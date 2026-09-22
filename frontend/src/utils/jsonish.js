import JSON5 from "json5";

/** Return the first balanced {...} substring, or null. */
function extractBalancedObject(text) {
  let start = -1, depth = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start !== -1) {
        return text.slice(start, i + 1);
      }
    }
  }
  return null;
}

/** Parse JSON-ish content inside free-form text (Python dicts, single quotes, True/False/None). */
export function parseJsonish(text) {
  if (!text || typeof text !== "string") return null;

  // Try strict JSON first
  try {
    return JSON.parse(text);
  } catch {
    // not strict JSON — fall through to the jsonish extraction below
  }

  // Extract the first balanced object
  let candidate = extractBalancedObject(text);
  if (!candidate) return null;

  // Normalize Python tokens → JSON5
  candidate = candidate
    .replace(/\bNone\b/g, "null")
    .replace(/\bTrue\b/g, "true")
    .replace(/\bFalse\b/g, "false");

  try {
    return JSON5.parse(candidate); // handles single quotes, trailing commas
  } catch {
    return null;
  }
}
