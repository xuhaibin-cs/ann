import { escapeHtml, fmtShape, formatMaybe } from "../core/format.js";

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
  renderANNSampleSelect(els, ann);
  renderANNMathJourney(els, ann);
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

export function renderANNMathJourney(els, ann) {
  if (!ann?.mathTrace) return;
  const trace = ann.mathTrace;
  const sampleIndex = Number(els.annSampleSelect.value || 1);
  const sample = trace.samples[sampleIndex] || trace.samples[0];
  const firstLayer = sample.layers[0];
  const secondLayer = sample.layers[1];
  const outputLayer = sample.layers.at(-1);
  const neuronIndex = 0;
  const firstTerms = firstLayer.contributions
    .map((row, inputIndex) => `${formatNumber(firstLayer.input[inputIndex])} × ${formatNumber(firstLayer.weights[inputIndex][neuronIndex])}`)
    .join(" + ");
  const hiddenVector = vectorPreview(firstLayer.activation);
  const secondVector = vectorPreview(secondLayer.activation);
  const outputTerms = outputLayer.contributions
    .slice(0, 4)
    .map((row, inputIndex) => `${formatNumber(outputLayer.input[inputIndex])}×${formatNumber(outputLayer.weights[inputIndex][0])}`)
    .join(" + ");
  const update = trace.lastUpdate;

  const cards = [
    mathCard(
      1,
      "Input",
      "A sample enters as a feature vector with its target label.",
      "x = [x₁, x₂],  y ∈ {0, 1}",
      `x = ${vector(sample.input)}, y = ${formatNumber(sample.target)} · shape [1 × 2]`,
    ),
    mathCard(
      2,
      "Weighted sum",
      "Each neuron multiplies every input by a weight, adds the contributions, then adds its bias.",
      "zⱼ = Σᵢ xᵢwᵢⱼ + bⱼ",
      `L0 neuron 0: ${firstTerms} + ${formatNumber(firstLayer.bias[0])} = ${formatNumber(firstLayer.z[0])}`,
    ),
    mathCard(
      3,
      "Activation",
      "Tanh bends the linear result, allowing the network to represent a nonlinear XOR boundary.",
      "aⱼ = tanh(zⱼ)",
      `a₀ = tanh(${formatNumber(firstLayer.z[0])}) = ${formatNumber(firstLayer.activation[0])}`,
    ),
    mathCard(
      4,
      "Hidden features",
      "The first hidden vector becomes the next layer's input. A second Linear + Tanh transformation recombines it.",
      "h² = tanh(h¹W² + b²)",
      `h¹ = ${hiddenVector} → h² = ${secondVector} · each has shape [1 × 8]`,
    ),
    mathCard(
      5,
      "Output prediction",
      "The output neuron forms one final weighted sum; Sigmoid converts it to a probability.",
      "ŷ = σ(z³) = 1 / (1 + e⁻ᶻ³)",
      `z³ = ${outputTerms} + … + b = ${formatNumber(outputLayer.z[0])}; ŷ = ${formatNumber(sample.prediction)}`,
    ),
    mathCard(
      6,
      "Calculate error",
      "Mean squared error measures prediction distance. Training minimizes the mean over all four XOR samples.",
      "L = (1/N) Σᵢ(ŷᵢ − yᵢ)²",
      `sample error = (${formatNumber(sample.prediction)} − ${formatNumber(sample.target)})² = ${formatNumber(sample.sampleSquaredError)}; batch L = ${formatNumber(trace.loss)}`,
    ),
    mathCard(
      7,
      "Chain rule",
      "The loss gradient is multiplied by the local Sigmoid derivative to obtain responsibility at the output logit.",
      "∂L/∂z³ = (∂L/∂ŷ)(∂ŷ/∂z³)",
      `${formatNumber(sample.dLossDPrediction)} × ${formatNumber(sample.outputActivationDerivative)} = ${formatNumber(sample.dLossDOutputZ)}`,
    ),
    mathCard(
      8,
      "Backpropagate",
      "Responsibility moves right-to-left. Every layer computes parameter gradients and passes an input gradient backward.",
      "∂L/∂Wˡ = (aˡ⁻¹)ᵀδˡ,  δˡ⁻¹ = (δˡWˡᵀ) ⊙ f′(zˡ⁻¹)",
      `gradient norms: output ${formatNumber(norm(outputLayer.weightGrad))} → hidden 2 ${formatNumber(norm(secondLayer.weightGrad))} → hidden 1 ${formatNumber(norm(firstLayer.weightGrad))}`,
    ),
    mathCard(
      9,
      "Update parameters",
      "This lab uses Adam: an adaptive form of gradient descent that tracks gradient moments.",
      "m̂ₜ=mₜ/(1−β₁ᵗ), v̂ₜ=vₜ/(1−β₂ᵗ), θₜ=θₜ₋₁−ηm̂ₜ/(√v̂ₜ+ε)",
      update
        ? `${escapeHtml(update.parameter)}[0,0]: ${formatPrecise(update.oldValue)} ${signed(update.delta)} = ${formatPrecise(update.newValue)}`
        : `No update yet. Click Train ANN; η = ${trace.optimizer.learningRate}, β₁ = ${trace.optimizer.beta1}, β₂ = ${trace.optimizer.beta2}.`,
    ),
    mathCard(
      10,
      "Repeat",
      "The next iteration runs the same equations with updated parameters. Useful features emerge because updates reduce total loss.",
      "forward → loss → backward → update → repeat",
      `current step ${ann.step}; current loss ${formatNumber(trace.loss)}${ann.lossHistory.length ? `; recorded iterations ${ann.lossHistory.length}` : ""}`,
    ),
  ];
  els.annMathJourney.innerHTML = cards.join("");
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

function renderANNSampleSelect(els, ann) {
  const oldSample = els.annSampleSelect.value || "1";
  els.annSampleSelect.innerHTML = ann.mathTrace.samples
    .map((sample) => `<option value="${sample.index}">x = (${sample.input.join(", ")}), y = ${sample.target}</option>`)
    .join("");
  els.annSampleSelect.value = String(Math.min(Number(oldSample), ann.mathTrace.samples.length - 1));
}

function mathCard(number, title, explanation, formula, calculation) {
  return `<article class="math-step">
    <div class="math-step-number">${number}</div>
    <div class="math-step-copy">
      <h3>${title}</h3>
      <p>${explanation}</p>
      <div class="math-formula">${formula}</div>
      <div class="math-calculation">${calculation}</div>
    </div>
  </article>`;
}

function vector(values) {
  return `[${values.map(formatNumber).join(", ")}]`;
}

function vectorPreview(values) {
  const preview = values.slice(0, 4).map(formatNumber).join(", ");
  return `[${preview}${values.length > 4 ? ", …" : ""}]`;
}

function formatNumber(value) {
  const number = Number(value);
  if (Math.abs(number) > 0 && Math.abs(number) < 0.001) return number.toExponential(2);
  return number.toFixed(4);
}

function formatPrecise(value) {
  return Number(value).toFixed(7);
}

function norm(values) {
  return Math.sqrt(values.flat(Infinity).reduce((sum, value) => sum + value * value, 0));
}

function signed(value) {
  return value < 0 ? `− ${formatNumber(Math.abs(value))}` : `+ ${formatNumber(value)}`;
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
