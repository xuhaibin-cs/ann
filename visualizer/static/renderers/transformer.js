import { displayToken, fmtShape } from "../core/format.js";
import { drawMatrix } from "./canvas.js";

export function renderTransformer(els, snapshot, forward) {
  renderStructure(els, snapshot, forward);
  renderSelectors(els, forward);
  renderTokens(els, forward);
  renderProbabilities(els, forward);
  renderAttention(els, forward);
  renderMask(els, forward);
}

export function renderAttention(els, forward) {
  if (!forward) return;
  const layer = Number(els.layerSelect.value || 0);
  const head = Number(els.headSelect.value || 0);
  const matrix = forward.attentions?.[layer]?.[0]?.[head] || [];
  drawMatrix(els.attentionCanvas, matrix, {
    labels: forward.tokens,
    color: [15, 118, 110],
    empty: "Run Inspect to see attention",
  });
}

function renderStructure(els, snapshot, forward) {
  const c = snapshot.config;
  const shapes = forward.shapes;
  const nodes = [
    ["Token Embedding", `vocab ${c.vocab_size} -> ${c.embed_dim}`],
    ["Position Embedding", `context ${c.context_length}, dim ${c.embed_dim}`],
    ["Transformer Block 0", `heads ${c.num_heads}, ff ${c.ff_dim}, shape ${fmtShape(shapes.block_0)}`],
    ["Final LayerNorm", `features ${c.embed_dim}`],
    ["LM Head", `${c.embed_dim} -> vocab ${c.vocab_size}, logits ${fmtShape(shapes.logits)}`],
  ];
  els.structure.innerHTML = nodes
    .map(([title, detail]) => `<div class="node"><strong>${title}</strong><span>${detail}</span></div>`)
    .join("");
}

function renderSelectors(els, forward) {
  const attentions = forward.attentions || [];
  const oldLayer = els.layerSelect.value || "0";
  const oldHead = els.headSelect.value || "0";
  els.layerSelect.innerHTML = attentions.map((_, i) => `<option value="${i}">L${i}</option>`).join("");
  const layer = Math.min(Number(oldLayer), Math.max(attentions.length - 1, 0));
  els.layerSelect.value = String(layer);
  const headCount = attentions[layer]?.[0]?.length || 0;
  els.headSelect.innerHTML = Array.from({ length: headCount }, (_, i) => `<option value="${i}">H${i}</option>`).join("");
  els.headSelect.value = String(Math.min(Number(oldHead), Math.max(headCount - 1, 0)));
}

function renderTokens(els, forward) {
  els.tokens.innerHTML = forward.tokens
    .map((token, i) => `<span class="token" title="index ${i}">${displayToken(token)}</span>`)
    .join("");
}

function renderProbabilities(els, forward) {
  els.probabilities.innerHTML = forward.topTokens
    .map((item) => {
      const percent = item.probability * 100;
      return `<div class="prob-row">
        <span class="prob-token">${displayToken(item.token)}</span>
        <span class="bar"><span style="width:${percent}%"></span></span>
        <span>${percent.toFixed(1)}%</span>
      </div>`;
    })
    .join("");
}

function renderMask(els, forward) {
  drawMatrix(els.maskCanvas, forward.causalMask || [], {
    color: [180, 83, 9],
    empty: "No mask",
    compact: true,
  });
}
