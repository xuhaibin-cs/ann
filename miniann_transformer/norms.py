from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Array, Module, Parameter


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5, *, name: str = "layernorm") -> None:
        self.gamma = Parameter.from_data(np.ones(features), name=f"{name}.gamma")
        self.beta = Parameter.from_data(np.zeros(features), name=f"{name}.beta")
        self.eps = eps
        self.x_centered: Optional[Array] = None
        self.std_inv: Optional[Array] = None
        self.x_hat: Optional[Array] = None

    def forward(self, x: Array) -> Array:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        self.x_centered = x - mean
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = self.x_centered * self.std_inv
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad: Array) -> Array:
        if self.x_centered is None or self.std_inv is None or self.x_hat is None:
            raise RuntimeError("LayerNorm.backward called before forward")
        feature_count = grad.shape[-1]
        axes = tuple(range(grad.ndim - 1))
        self.gamma.grad += np.sum(grad * self.x_hat, axis=axes)
        self.beta.grad += np.sum(grad, axis=axes)
        dx_hat = grad * self.gamma.data
        return (
            (1.0 / feature_count)
            * self.std_inv
            * (
                feature_count * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
            )
        )

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]
