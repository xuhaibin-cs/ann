from __future__ import annotations

from typing import Optional

import numpy as np

from .activations import softmax
from .tensor import Array
from .tokenizer import CharTokenizer
from .transformer import TinyGPT


def sample_next_token(logits: Array, temperature: float = 1.0, *, rng: Optional[np.random.Generator] = None) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    probs = softmax(logits / temperature, axis=-1)
    return int(rng.choice(np.arange(probs.shape[-1]), p=probs))


def generate(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> str:
    rng = rng or np.random.default_rng()
    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        context = ids[-model.config.context_length :]
        x = np.array([context], dtype=np.int64)
        logits = model.forward(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_id = sample_next_token(logits[0, -1], temperature, rng=rng)
        ids.append(next_id)
    return tokenizer.decode(ids)
