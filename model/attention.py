"""
Toprak — Multi-Head Self-Attention
Scaled Dot-Product Attention with Causal Mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Q, K, V projeksiyonları
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Output projeksiyon
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — üst üçgen matris (gelecekteki tokenleri maskele)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len),
                diagonal=1,
            ).bool(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape

        # Q, K, V hesapla ve head'lere böl
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, T, head_dim)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        # (B, num_heads, T, T)

        # Causal mask uygula
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        # (B, num_heads, T, head_dim)

        # Head'leri birleştir
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projeksiyon
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output
