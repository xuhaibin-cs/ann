export const state = {
  forward: null,
  snapshot: null,
  activeView: "ann",
};

export function setTransformerState(forward, snapshot = forward?.snapshot) {
  state.forward = forward;
  state.snapshot = snapshot;
}
