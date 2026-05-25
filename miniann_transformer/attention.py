from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .activations import softmax
from .layers import Linear
from .tensor import Array, Module, Parameter


def causal_mask(seq_len: int) -> Array:
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


class ScaledDotProductAttention:
    def __init__(self) -> None:
        self.q: Optional[Array] = None
        self.k: Optional[Array] = None
        self.v: Optional[Array] = None
        self.weights: Optional[Array] = None
        self.mask: Optional[Array] = None

    def forward(self, q: Array, k: Array, v: Array, mask: Optional[Array] = None) -> Tuple[Array, Array]:
        self.q, self.k, self.v, self.mask = q, k, v, mask
        scale = np.sqrt(q.shape[-1])
        scores = (q @ np.swapaxes(k, -1, -2)) / scale
        if mask is not None:
            scores = np.where(mask[None, None, :, :], scores, -1e9)
        self.weights = softmax(scores, axis=-1)
        return self.weights @ v, self.weights

    def backward(self, grad_out: Array) -> tuple[Array, Array, Array]:
        if self.q is None or self.k is None or self.v is None or self.weights is None:
            raise RuntimeError("Attention.backward called before forward")
        grad_weights = grad_out @ np.swapaxes(self.v, -1, -2)
        grad_v = np.swapaxes(self.weights, -1, -2) @ grad_out
        dot = np.sum(grad_weights * self.weights, axis=-1, keepdims=True)
        grad_scores = self.weights * (grad_weights - dot)
        if self.mask is not None:
            grad_scores = np.where(self.mask[None, None, :, :], grad_scores, 0.0)
        scale = np.sqrt(self.q.shape[-1])
        grad_q = grad_scores @ self.k / scale
        grad_k = np.swapaxes(grad_scores, -1, -2) @ self.q / scale
        return grad_q, grad_k, grad_v


class MultiHeadSelfAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, *, name: str = "mha") -> None:
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, name=f"{name}.q")
        self.k_proj = Linear(embed_dim, embed_dim, name=f"{name}.k")
        self.v_proj = Linear(embed_dim, embed_dim, name=f"{name}.v")
        self.out_proj = Linear(embed_dim, embed_dim, name=f"{name}.out")
        self.attention = ScaledDotProductAttention()
        self.last_attention_weights: Optional[Array] = None
        self.x_shape: Optional[Tuple[int, int, int]] = None

    def _split_heads(self, x: Array) -> Array:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Array) -> Array:
        batch, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim)

    def forward(self, x: Array, mask: Optional[Array] = None) -> Array:
        self.x_shape = x.shape
        q = self._split_heads(self.q_proj.forward(x))
        k = self._split_heads(self.k_proj.forward(x))
        v = self._split_heads(self.v_proj.forward(x))
        attended, weights = self.attention.forward(q, k, v, mask)
        self.last_attention_weights = weights
        return self.out_proj.forward(self._merge_heads(attended))

    def backward(self, grad: Array) -> Array:
        grad_heads = self._split_heads(self.out_proj.backward(grad))
        grad_q, grad_k, grad_v = self.attention.backward(grad_heads)
        return (
            self.q_proj.backward(self._merge_heads(grad_q))
            + self.k_proj.backward(self._merge_heads(grad_k))
            + self.v_proj.backward(self._merge_heads(grad_v))
        )

    def parameters(self) -> list[Parameter]:
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )
