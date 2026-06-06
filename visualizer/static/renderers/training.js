import { renderLineChart } from "./canvas.js";

export function renderTraining(els, snapshot) {
  renderLineChart(els.lossChart, snapshot.lossHistory || [], {
    width: 720,
    height: 220,
    pad: 28,
    stroke: "#2563eb",
    emptyY: 112,
    emptyMessage: "Train a few steps to draw the loss curve.",
  });
}
