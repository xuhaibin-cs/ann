from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Array, Module, Parameter


class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, embed_dim: int, *, name: str = "token_embedding") -> None:
        self.weight = Parameter.from_data(
            np.random.randn(vocab_size, embed_dim) * 0.02,
            name=f"{name}.weight",
        )
        self.ids: Optional[Array] = None

    def forward(self, token_ids: Array) -> Array:
        self.ids = token_ids
        return self.weight.data[token_ids]

    def backward(self, grad: Array) -> None:
        if self.ids is None:
            raise RuntimeError("TokenEmbedding.backward called before forward")
        np.add.at(self.weight.grad, self.ids, grad)

    def parameters(self) -> list[Parameter]:
        return [self.weight]


class PositionalEmbedding(Module):
    def __init__(self, context_length: int, embed_dim: int, *, name: str = "position_embedding") -> None:
        self.weight = Parameter.from_data(
            np.random.randn(context_length, embed_dim) * 0.02,
            name=f"{name}.weight",
        )
        self.positions: Optional[Array] = None

    def forward(self, seq_len: int) -> Array:
        self.positions = np.arange(seq_len)
        return self.weight.data[self.positions][None, :, :]

    def backward(self, grad: Array) -> None:
        if self.positions is None:
            raise RuntimeError("PositionalEmbedding.backward called before forward")
        summed = grad.sum(axis=0)
        np.add.at(self.weight.grad, self.positions, summed)

    def parameters(self) -> list[Parameter]:
        return [self.weight]
