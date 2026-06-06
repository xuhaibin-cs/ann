from __future__ import annotations

from typing import Optional

import numpy as np

from .activations import GELU
from .tensor import Array, Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, name: str = "linear") -> None:
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter.from_data(
            np.random.randn(in_features, out_features) * scale,
            name=f"{name}.weight",
        )
        self.bias = Parameter.from_data(np.zeros(out_features), name=f"{name}.bias") if bias else None
        self.x: Optional[Array] = None
        self.z: Optional[Array] = None
        self.grad_output: Optional[Array] = None
        self.grad_input: Optional[Array] = None

    def forward(self, x: Array) -> Array:
        self.x = x
        y = x @ self.weight.data
        if self.bias is not None:
            y = y + self.bias.data
        self.z = y
        return y

    def backward(self, grad: Array) -> Array:
        if self.x is None:
            raise RuntimeError("Linear.backward called before forward")
        self.grad_output = grad
        x2 = self.x.reshape(-1, self.x.shape[-1])
        g2 = grad.reshape(-1, grad.shape[-1])
        self.weight.grad += x2.T @ g2
        if self.bias is not None:
            self.bias.grad += g2.sum(axis=0)
        self.grad_input = grad @ self.weight.data.T
        return self.grad_input

    def parameters(self) -> list[Parameter]:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class FeedForward(Module):
    def __init__(self, embed_dim: int, ff_dim: int, *, name: str = "ff") -> None:
        self.fc1 = Linear(embed_dim, ff_dim, name=f"{name}.fc1")
        self.gelu = GELU()
        self.fc2 = Linear(ff_dim, embed_dim, name=f"{name}.fc2")

    def forward(self, x: Array) -> Array:
        return self.fc2.forward(self.gelu.forward(self.fc1.forward(x)))

    def backward(self, grad: Array) -> Array:
        return self.fc1.backward(self.gelu.backward(self.fc2.backward(grad)))

    def parameters(self) -> list[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters()
