export function fmtShape(shape) {
  return shape ? shape.join(" x ") : "pending";
}

export function displayToken(token) {
  if (token === "\n") return "\\n";
  if (token === " ") return "space";
  return escapeHtml(token);
}

export function formatMaybe(value) {
  return value === null || value === undefined ? "none" : Number(value).toFixed(4);
}

export function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
