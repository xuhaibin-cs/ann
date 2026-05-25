from __future__ import annotations

import numpy as np

from .data import TOY_CORPUS, make_next_token_batch
from .generate import generate
from .losses import CrossEntropyLoss
from .optim import Adam
from .tokenizer import CharTokenizer
from .transformer import TinyGPT, TransformerConfig


def train_toy_model(steps: int = 80, print_every: int = 20, seed: int = 7) -> tuple[TinyGPT, CharTokenizer]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tokenizer = CharTokenizer(TOY_CORPUS)
    encoded = tokenizer.encode(TOY_CORPUS)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=24,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        batch_size=8,
    )
    model = TinyGPT(config)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=2e-3)
    for step in range(1, steps + 1):
        x, y = make_next_token_batch(encoded, config.batch_size, config.context_length, rng=rng)
        logits = model.forward(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = loss_fn.forward(logits, y)
        optimizer.zero_grad()
        model.backward(loss_fn.backward())
        optimizer.step()
        if step == 1 or step % print_every == 0:
            print(f"step {step:04d} loss {loss:.4f}")
    print(generate(model, tokenizer, "to ", max_new_tokens=60, temperature=0.8, rng=rng))
    return model, tokenizer


if __name__ == "__main__":
    train_toy_model()
