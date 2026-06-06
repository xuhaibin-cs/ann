export function renderView(els, state) {
  els.tabButtons.forEach((button) => {
    const isActive = button.dataset.view === state.activeView;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  els.views.forEach((view) => {
    view.classList.toggle("active", view.dataset.viewPanel === state.activeView);
  });
}

export function renderMetrics(els, snapshot) {
  const items = metricItems(snapshot);
  els.metrics.innerHTML = items
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function metricItems(snapshot) {
  const activeView = document.querySelector(".tab-button.active")?.dataset.view || "ann";
  if (activeView === "ann" && snapshot.ann) {
    return [
      ["ANN step", snapshot.ann.step],
      ["XOR loss", snapshot.ann.loss.toFixed(4)],
      ["Params", snapshot.ann.parameterCount.toLocaleString()],
      ["Param norm", snapshot.ann.paramNorm.toFixed(2)],
    ];
  }
  if (activeView === "diffusion" && snapshot.diffusion) {
    const losses = snapshot.diffusion.lossHistory || [];
    return [
      ["Diff step", snapshot.diffusion.step],
      ["Timesteps", snapshot.diffusion.config.timesteps],
      ["Params", snapshot.diffusion.parameterCount.toLocaleString()],
      ["Loss", losses.length ? losses.at(-1).toFixed(4) : "pending"],
    ];
  }
  return [
    ["GPT step", snapshot.step],
    ["Params", snapshot.parameterCount.toLocaleString()],
    ["Param norm", snapshot.paramNorm.toFixed(2)],
    ["Grad norm", snapshot.gradNorm.toFixed(2)],
  ];
}
