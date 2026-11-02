"""Structural facts and addressing helpers for a Llama-style causal LM."""


from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class Architecture:
    model: nn.Module
    model_type: str
    blocks: list
    embed_in: nn.Module
    unembed: nn.Module
    tied_embeddings: bool
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    score_component: int
    value_component: int

    # --- attention ---

    def kv_group(self, head_index: int) -> int:
        """Which key/value head serves this query head."""
        return head_index // (self.num_heads // self.num_kv_heads)

    def group_heads(self, head_index: int) -> list[int]:
        """All query heads sharing this query head's key/value head."""
        per_group = self.num_heads // self.num_kv_heads
        first = self.kv_group(head_index) * per_group
        return list(range(first, first + per_group))

    def query_row(self, block, head_index: int, component: int):
        """
        Which `q_proj` module, and which row of its weight matrix,
        computes this query component of this head in this block.
        """
        return block.self_attn.q_proj, head_index * self.head_dim + component

    def key_row(self, block, head_index: int, component: int):
        return block.self_attn.k_proj, self.kv_group(head_index) * self.head_dim + component

    def value_row(self, block, head_index: int, component: int):
        return block.self_attn.v_proj, self.kv_group(head_index) * self.head_dim + component

    def output_projection(self, block):
        return block.self_attn.o_proj

    def output_column(self, head_index: int, component: int) -> int:
        """
        Which column of the `W_o` matrix writes this component
        of this head's attention output into the residual stream.
        """
        return head_index * self.head_dim + component

    # --- mlp ---

    def mlp_gate(self, block):
        """Projection whose row is the neuron's pre-activation."""
        return block.mlp.gate_proj

    def mlp_up(self, block):
        """SwiGLU's multiplicative branch."""
        return block.mlp.up_proj

    def mlp_down(self, block):
        return block.mlp.down_proj

    # --- norms ---

    def pre_attention_norm(self, block):
        return block.input_layernorm

    def pre_mlp_norm(self, block):
        return block.post_attention_layernorm


def describe(model) -> Architecture:
    """Build an Architecture object for a loaded Llama-style causal LM."""
    config = model.config
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError(f"unsupported architecture: {type(model).__name__}")
    blocks = list(model.model.layers)
    block = blocks[0]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(block.self_attn, name):
            raise ValueError(f"expected self_attn.{name}; not a Llama-style model")
    if not hasattr(block.mlp, "gate_proj"):
        raise ValueError("expected a SwiGLU MLP (gate_proj/up_proj/down_proj)")
    if block.mlp.down_proj.bias is not None:
        raise ValueError("this circuit assumes an MLP without biases")

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or hidden_size // num_heads

    return Architecture(
        model=model,
        model_type=getattr(config, "model_type", ""),
        blocks=blocks,
        embed_in=model.model.embed_tokens,
        unembed=model.lm_head,
        tied_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        hidden_size=hidden_size,
        intermediate_size=config.intermediate_size,
        num_heads=num_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", num_heads),
        head_dim=head_dim,
        score_component=head_dim - 1,
        value_component=head_dim - 1,
    )
