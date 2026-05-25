from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from .activations import softmax
from .attention import MultiHeadSelfAttention, causal_mask
from .embeddings import PositionalEmbedding, TokenEmbedding
from .layers import FeedForward, Linear
from .norms import LayerNorm
from .tensor import Array, Module, Parameter, count_parameters


@dataclass
class TransformerConfig:
    vocab_size: int = 100
    context_length: int = 32
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    batch_size: int = 16
    debug: bool = False


@dataclass
class DebugInfo:
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    attention_matrices: list[Array] = field(default_factory=list)
    causal_mask: Optional[Array] = None
    token_probabilities: Optional[Array] = None
    parameter_count: int = 0


class TransformerBlock(Module):
    def __init__(self, config: TransformerConfig, *, layer_id: int) -> None:
        self.ln1 = LayerNorm(config.embed_dim, name=f"blocks.{layer_id}.ln1")
        self.attn = MultiHeadSelfAttention(config.embed_dim, config.num_heads, name=f"blocks.{layer_id}.attn")
        self.ln2 = LayerNorm(config.embed_dim, name=f"blocks.{layer_id}.ln2")
        self.ff = FeedForward(config.embed_dim, config.ff_dim, name=f"blocks.{layer_id}.ff")

    def forward(self, x: Array, mask: Array) -> Array:
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ff.forward(self.ln2.forward(x))
        return x

    def backward(self, grad: Array) -> Array:
        grad_ff_input = self.ln2.backward(self.ff.backward(grad))
        grad = grad + grad_ff_input
        grad_attn_input = self.ln1.backward(self.attn.backward(grad))
        return grad + grad_attn_input

    def parameters(self) -> list[Parameter]:
        return self.ln1.parameters() + self.attn.parameters() + self.ln2.parameters() + self.ff.parameters()


class TinyGPT(Module):
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed_dim)
        self.position_embedding = PositionalEmbedding(config.context_length, config.embed_dim)
        self.blocks = [TransformerBlock(config, layer_id=i) for i in range(config.num_layers)]
        self.final_ln = LayerNorm(config.embed_dim, name="final_ln")
        self.lm_head = Linear(config.embed_dim, config.vocab_size, name="lm_head")
        self.debug_info = DebugInfo()

    def forward(self, token_ids: Array, *, return_debug: bool = False) -> Union[Array, Tuple[Array, DebugInfo]]:
        _, seq_len = token_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context length {self.config.context_length}")
        mask = causal_mask(seq_len)
        x = self.token_embedding.forward(token_ids) + self.position_embedding.forward(seq_len)
        debug = DebugInfo(causal_mask=mask, parameter_count=self.parameter_count())
        debug.shapes["embedding"] = x.shape
        for i, block in enumerate(self.blocks):
            x = block.forward(x, mask)
            debug.shapes[f"block_{i}"] = x.shape
            if block.attn.last_attention_weights is not None:
                debug.attention_matrices.append(block.attn.last_attention_weights.copy())
        x = self.final_ln.forward(x)
        logits = self.lm_head.forward(x)
        debug.shapes["logits"] = logits.shape
        debug.token_probabilities = softmax(logits, axis=-1) if (return_debug or self.config.debug) else None
        self.debug_info = debug
        return (logits, debug) if (return_debug or self.config.debug) else logits

    def backward(self, grad_logits: Array) -> None:
        grad = self.lm_head.backward(grad_logits)
        grad = self.final_ln.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        self.position_embedding.backward(grad)
        self.token_embedding.backward(grad)

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        params += self.token_embedding.parameters()
        params += self.position_embedding.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.final_ln.parameters()
        params += self.lm_head.parameters()
        return params

    def parameter_count(self) -> int:
        return count_parameters(self.parameters())
