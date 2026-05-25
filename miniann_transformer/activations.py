from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .tensor import Array


class ReLU:
    def __init__(self) -> None:
        self.x: Optional[Array] = None

    def forward(self, x: Array) -> Array:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad: Array) -> Array:
        if self.x is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad * (self.x > 0.0)


class GELU:
    """Exact GELU with explicit derivative."""

    def __init__(self) -> None:
        self.x: Optional[Array] = None

    def forward(self, x: Array) -> Array:
        self.x = x
        return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

    def backward(self, grad: Array) -> Array:
        if self.x is None:
            raise RuntimeError("GELU.backward called before forward")
        x = self.x
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        return grad * (cdf + x * pdf)


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
