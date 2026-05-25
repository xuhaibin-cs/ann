from __future__ import annotations

from typing import Optional

import numpy as np

from .activations import softmax
from .tensor import Array


class CrossEntropyLoss:
    """Cross entropy for logits shaped (batch, time, vocab) and integer targets."""

    def __init__(self) -> None:
        self.probs: Optional[Array] = None
        self.targets: Optional[Array] = None

    def forward(self, logits: Array, targets: Array) -> float:
        self.probs = softmax(logits, axis=-1)
        self.targets = targets
        flat_probs = self.probs.reshape(-1, self.probs.shape[-1])
        flat_targets = targets.reshape(-1)
        n = flat_targets.shape[0]
        losses = -np.log(flat_probs[np.arange(n), flat_targets] + 1e-12)
        return float(losses.mean())

    def backward(self) -> Array:
        if self.probs is None or self.targets is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward")
        grad = self.probs.copy()
        flat_grad = grad.reshape(-1, grad.shape[-1])
        flat_targets = self.targets.reshape(-1)
        n = flat_targets.shape[0]
        flat_grad[np.arange(n), flat_targets] -= 1.0
        return grad / n
