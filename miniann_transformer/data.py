from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Array


TOY_CORPUS = (
    "to be or not to be\n"
    "that is the question\n"
    "we learn transformers by building them\n"
    "attention looks back but never forward\n"
)


def make_next_token_batch(
    encoded: list[int],
    batch_size: int,
    context_length: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Array, Array]:
    if len(encoded) <= context_length:
        raise ValueError("encoded corpus must be longer than context_length")
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, len(encoded) - context_length - 1, size=batch_size)
    x = np.stack([encoded[i : i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([encoded[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)
    return x, y
