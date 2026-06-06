const byId = (id) => document.querySelector(`#${id}`);

export const els = {
  tabButtons: document.querySelectorAll(".tab-button"),
  views: document.querySelectorAll("[data-view-panel]"),
  prompt: byId("prompt"),
  maxTokens: byId("maxTokens"),
  temperature: byId("temperature"),
  trainSteps: byId("trainSteps"),
  annSteps: byId("annSteps"),
  diffusionSteps: byId("diffusionSteps"),
  inspectBtn: byId("inspectBtn"),
  generateBtn: byId("generateBtn"),
  trainBtn: byId("trainBtn"),
  annTrainBtn: byId("annTrainBtn"),
  diffusionTrainBtn: byId("diffusionTrainBtn"),
  diffusionSampleBtn: byId("diffusionSampleBtn"),
  status: byId("status"),
  metrics: byId("metrics"),
  structure: byId("structure"),
  layerSelect: byId("layerSelect"),
  headSelect: byId("headSelect"),
  attentionCanvas: byId("attentionCanvas"),
  maskCanvas: byId("maskCanvas"),
  tokens: byId("tokens"),
  probabilities: byId("probabilities"),
  generatedText: byId("generatedText"),
  trainingText: byId("trainingText"),
  lossChart: byId("lossChart"),
  annSummary: byId("annSummary"),
  annLegend: byId("annLegend"),
  annCanvas: byId("annCanvas"),
  annActivations: byId("annActivations"),
  annPredictions: byId("annPredictions"),
  annLayerSelect: byId("annLayerSelect"),
  annNeurons: byId("annNeurons"),
  diffusionSummary: byId("diffusionSummary"),
  forwardDiffusion: byId("forwardDiffusion"),
  reverseDiffusion: byId("reverseDiffusion"),
  diffusionLossChart: byId("diffusionLossChart"),
};

export function setBusy(isBusy) {
  [
    els.inspectBtn,
    els.generateBtn,
    els.trainBtn,
    els.annTrainBtn,
    els.diffusionTrainBtn,
    els.diffusionSampleBtn,
  ].forEach((button) => {
    button.disabled = isBusy;
  });
}

export function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}
