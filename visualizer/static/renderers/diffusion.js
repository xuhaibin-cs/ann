import { drawPixelImage, renderLineChart } from "./canvas.js";

export function renderDiffusion(els, snapshot) {
  const diffusion = snapshot.diffusion;
  if (!diffusion) return;
  els.diffusionSummary.textContent = `step ${diffusion.step} | params ${diffusion.parameterCount} | T ${diffusion.config.timesteps}`;
  const forwardItems = [
    { t: "clean", image: diffusion.forward.clean },
    ...diffusion.forward.noisy.map((item) => ({ t: `t=${item.t}`, image: item.image })),
  ];
  renderImageStrip(els.forwardDiffusion, forwardItems);
  renderImageStrip(
    els.reverseDiffusion,
    diffusion.sample.frames.map((item) => ({ t: `t=${item.t}`, image: item.image })),
  );
  renderLineChart(els.diffusionLossChart, diffusion.lossHistory || [], {
    width: 720,
    height: 180,
    pad: 26,
    stroke: "#b45309",
    emptyY: 92,
    emptyMessage: "Train the diffusion denoiser to draw its noise-prediction loss.",
  });
}

function renderImageStrip(container, items) {
  container.innerHTML = items
    .map((item, index) => `<div class="image-tile"><canvas data-image-index="${index}" width="16" height="16"></canvas><span>${item.t}</span></div>`)
    .join("");
  container.querySelectorAll("canvas").forEach((canvas, index) => {
    drawPixelImage(canvas, items[index].image);
  });
}
