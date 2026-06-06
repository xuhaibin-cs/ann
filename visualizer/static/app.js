import { api } from "./core/api.js";
import { els, setBusy, setStatus } from "./core/dom.js";
import { displayToken } from "./core/format.js";
import { setTransformerState, state } from "./core/state.js";
import { renderANN, renderANNNeurons } from "./renderers/ann.js";
import { renderDiffusion } from "./renderers/diffusion.js";
import { renderMetrics, renderView } from "./renderers/layout.js";
import { renderTraining } from "./renderers/training.js";
import { renderAttention, renderTransformer } from "./renderers/transformer.js";

const viewRenderers = {
  transformer: () => renderTransformer(els, state.snapshot, state.forward),
  training: () => renderTraining(els, state.snapshot),
  ann: () => renderANN(els, state.snapshot),
  diffusion: () => renderDiffusion(els, state.snapshot),
};

const viewStatus = {
  ann: "Classic ANN shows the XOR decision surface, layer activations, and neuron internals.",
  transformer: "Transformer shows token flow, attention weights, causal masking, and next-token probabilities.",
  diffusion: "Diffusion shows the forward noise schedule and the current denoiser's reverse samples.",
  training: "Training tracks Transformer loss and stores the latest sampled text.",
};

function initialView() {
  const requested = new URLSearchParams(window.location.search).get("view");
  return viewRenderers[requested] ? requested : state.activeView;
}

function renderAll() {
  if (!state.snapshot || !state.forward) return;
  renderView(els, state);
  renderMetrics(els, state.snapshot);
  viewRenderers[state.activeView]?.();
}

function setView(view) {
  state.activeView = view;
  renderView(els, state);
  setStatus(viewStatus[view] || "Ready.");
  requestAnimationFrame(renderAll);
}

async function withBusy(task) {
  setBusy(true);
  try {
    await task();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    setBusy(false);
  }
}

async function inspect() {
  await withBusy(async () => {
    const data = await api("/api/forward", { prompt: els.prompt.value });
    setTransformerState(data);
    setStatus(`Inspected ${data.tokens.length} tokens. Most likely next token: ${displayToken(data.nextToken)}.`);
    renderAll();
  });
}

async function generate() {
  await withBusy(async () => {
    const data = await api("/api/generate", {
      prompt: els.prompt.value,
      maxNewTokens: Number(els.maxTokens.value),
      temperature: Number(els.temperature.value),
    });
    els.generatedText.textContent = data.text;
    els.trainingText.textContent = data.text;
    setTransformerState(data.forward);
    setStatus(`Generated ${data.generated.length} tokens.`);
    renderAll();
  });
}

async function trainTransformer() {
  await withBusy(async () => {
    const data = await api("/api/train", { steps: Number(els.trainSteps.value) });
    els.generatedText.textContent = data.sample;
    els.trainingText.textContent = data.sample;
    setTransformerState(data.forward, data.snapshot);
    setStatus(`Trained ${data.trainedSteps} step(s). Latest loss ${data.losses.at(-1).toFixed(4)}.`);
    renderAll();
  });
}

async function trainANN() {
  await withBusy(async () => {
    const data = await api("/api/ann/train", { steps: Number(els.annSteps.value) });
    state.snapshot.ann = data.ann;
    setStatus(`Trained Classic ANN ${data.trainedSteps} step(s). Latest XOR loss ${data.losses.at(-1).toFixed(4)}.`);
    renderAll();
  });
}

async function trainDiffusion() {
  await withBusy(async () => {
    const data = await api("/api/diffusion/train", { steps: Number(els.diffusionSteps.value) });
    state.snapshot.diffusion = data.diffusion;
    setStatus(`Trained diffusion ${data.trainedSteps} step(s). Latest denoising loss ${data.losses.at(-1).toFixed(4)}.`);
    renderAll();
  });
}

async function sampleDiffusion() {
  await withBusy(async () => {
    const data = await api("/api/diffusion/sample", {});
    state.snapshot.diffusion = data.diffusion;
    setStatus("Sampled a reverse diffusion trajectory from Gaussian noise.");
    renderAll();
  });
}

async function boot() {
  try {
    const data = await api("/api/state");
    setTransformerState(data.forward, data.snapshot);
    state.activeView = initialView();
    const latestSample = data.snapshot.latestSample || "Train or generate to inspect model behavior here.";
    els.generatedText.textContent = latestSample;
    els.trainingText.textContent = data.snapshot.latestSample || "Train the Transformer to collect a loss curve and sample text.";
    setStatus(viewStatus[state.activeView]);
    renderAll();
  } catch (err) {
    setStatus(err.message, true);
  }
}

els.tabButtons.forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
els.inspectBtn.addEventListener("click", inspect);
els.generateBtn.addEventListener("click", generate);
els.trainBtn.addEventListener("click", trainTransformer);
els.annTrainBtn.addEventListener("click", trainANN);
els.diffusionTrainBtn.addEventListener("click", trainDiffusion);
els.diffusionSampleBtn.addEventListener("click", sampleDiffusion);
els.annLayerSelect.addEventListener("change", () => renderANNNeurons(els, state.snapshot?.ann));
els.layerSelect.addEventListener("change", () => renderAttention(els, state.forward));
els.headSelect.addEventListener("change", () => renderAttention(els, state.forward));
window.addEventListener("resize", renderAll);

boot();
