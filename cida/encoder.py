"""
cida/encoder.py - Transformer encoder backbone for CIDA V8.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .modules import RMSNorm, RoPE, SDPA_Attention, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with grouped-query attention."""

    def __init__(self, d: int, n_heads: int, n_kv_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SDPA_Attention(d, n_heads, num_kv_heads=n_kv_heads, dropout=dropout)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, ffn_dim, d, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rope: Optional[RoPE] = None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=mask, need_weights=False, rope=rope)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with a learned CLS token.

    Input: `[B, L]` token ids.
    Output: CLS representation `[B, d_model]` or normalized layer states.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.d = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rope = RoPE(d_model // n_heads, max_seq_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, n_kv_heads, d_model * ffn_mult, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_layers: bool = False) -> torch.Tensor:
        batch_size, _ = x.shape
        hidden = self.embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = self.drop(hidden)

        pad_mask = x == 0
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        layers = []
        for block in self.blocks:
            if self.training:
                hidden = checkpoint(block, hidden, full_mask, self.rope, use_reentrant=False)
            else:
                hidden = block(hidden, mask=full_mask, rope=self.rope)
            if return_layers:
                layers.append(self.norm(hidden))

        if return_layers:
            return layers
        return self.norm(hidden)[:, 0, :]
