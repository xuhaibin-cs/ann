import { fmtShape, formatMaybe } from "../core/format.js";

export function renderANN(els, snapshot) {
  const ann = snapshot.ann;
  if (!ann) return;
  els.annSummary.textContent = `step ${ann.step} | loss ${ann.loss.toFixed(4)} | params ${ann.parameterCount}`;
  els.annLegend.innerHTML = `
    <span><i class="legend-swatch class-zero"></i>class 0</span>
    <span><i class="legend-swatch boundary"></i>p=0.5</span>
    <span><i class="legend-swatch class-one"></i>class 1</span>
  `;
  renderANNBoundary(els.annCanvas, ann);
  els.annActivations.innerHTML = ann.activations
    .map(
      (item) => `<div class="ann-row">
        <strong>${item.name} ${fmtShape(item.shape)}</strong>
        <span>mean ${item.mean.toFixed(3)} | std ${item.std.toFixed(3)} | range ${item.min.toFixed(3)}..${item.max.toFixed(3)}</span>
      </div>`,
    )
    .join("");
  els.annPredictions.innerHTML = ann.points
    .map((point, i) => {
      const pred = ann.predictions[i];
      return `<div class="ann-row">
        <strong>(${point.x}, ${point.y}) -> ${point.label}</strong>
        <span>prediction ${pred.toFixed(4)} | class ${pred >= 0.5 ? 1 : 0}</span>
      </div>`;
    })
    .join("");
  renderANNLayerSelect(els, ann);
  renderANNNeurons(els, ann);
}

export function renderANNNeurons(els, ann) {
  if (!ann) return;
  const layerIndex = Number(els.annLayerSelect.value || 0);
  const layer = ann.neurons[layerIndex];
  if (!layer) {
    els.annNeurons.innerHTML = "";
    return;
  }
  els.annNeurons.innerHTML = layer.neurons
    .map((neuron) => {
      const maxAbs = Math.max(...neuron.contributions.map((v) => Math.abs(v)), 1e-9);
      const rows = neuron.contributions
        .map((value, i) => {
          const width = (Math.abs(value) / maxAbs) * 50;
          const klass = value < 0 ? "contrib-row negative" : "contrib-row";
          return `<div class="${klass}">
            <span>x${i}</span>
            <span class="contrib-bar"><span style="width:${width}%"></span></span>
            <span>${value.toFixed(3)}</span>
          </div>`;
        })
        .join("");
      const grad = neuron.gradZ === null ? "pending" : neuron.gradZ.toExponential(2);
      const gradBias = neuron.gradBias === null ? "pending" : neuron.gradBias.toExponential(2);
      return `<div class="neuron-card">
        <strong>Neuron ${neuron.index}</strong>
        <div class="neuron-meta">
          z ${neuron.z.toFixed(4)} | a ${neuron.activation.toFixed(4)} | b ${formatMaybe(neuron.bias)}
          <br />grad z ${grad} | grad b ${gradBias}
        </div>
        ${rows}
      </div>`;
    })
    .join("");
}

function renderANNLayerSelect(els, ann) {
  const oldLayer = els.annLayerSelect.value || "0";
  els.annLayerSelect.innerHTML = ann.neurons
    .map((layer) => `<option value="${layer.index}">L${layer.index}</option>`)
    .join("");
  const maxLayer = Math.max(ann.neurons.length - 1, 0);
  els.annLayerSelect.value = String(Math.min(Number(oldLayer), maxLayer));
}

function renderANNBoundary(canvas, ann) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || canvas.width;
  const cssHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const grid = ann.decision;
  const pad = 34;
  const availableW = cssWidth - pad * 2;
  const availableH = cssHeight - pad * 2;
  const plotSize = Math.max(Math.min(availableW, availableH), 1);
  const left = pad + Math.max((availableW - plotSize) / 2, 0);
  const top = pad + Math.max((availableH - plotSize) / 2, 0);
  const plotW = plotSize;
  const plotH = plotSize;
  const cellW = plotW / grid.size;
  const cellH = plotH / grid.size;
  for (let y = 0; y < grid.size; y += 1) {
    for (let x = 0; x < grid.size; x += 1) {
      const p = grid.values[y][x];
      const red = Math.round(245 - p * 80);
      const green = Math.round(238 - p * 58);
      const blue = Math.round(220 + p * 22);
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(left + x * cellW, top + y * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  ctx.strokeStyle = "#1f2933";
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, plotW, plotH);
  ctx.fillStyle = "#65717e";
  ctx.font = "12px system-ui";
  ctx.fillText("x1", left + plotW + 8, top + plotH + 4);
  ctx.fillText("x2", left - 24, top - 10);
  ctx.fillText("p(y=1)", left, top - 10);

  ann.points.forEach((point) => {
    const x = left + ((point.x - grid.xMin) / (grid.xMax - grid.xMin)) * plotW;
    const y = top + ((point.y - grid.yMin) / (grid.yMax - grid.yMin)) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = point.label ? "#0f766e" : "#b45309";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#1f2933";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(String(point.label), x + 11, y + 4);
  });
}
