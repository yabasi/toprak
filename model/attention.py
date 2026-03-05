# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Grouped Query Attention (GQA)
Scaled Dot-Product Attention with Causal Mask, RoPE, KV Cache.

Modern mimari:
- GQA: Q head sayısı > KV head sayısı → KV cache tasarrufu
- RoPE: Rotary Position Embedding (learned positional yerine)
- KV Cache: Inference sırasında geçmiş key/value'ları sakla
- Bias yok: Tüm linear katmanlardan bias kaldırıldı
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE and KV Cache.

    GQA'da Q head sayısı, KV head sayısından fazladır.
    Her KV head grubu, birden fazla Q head'e hizmet eder.
    Bu sayede inference sırasında KV cache boyutu azalır.

    Referanslar:
    - GQA: https://arxiv.org/abs/2305.13245
    - RoPE: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # GQA: Her KV head grubundaki Q head sayısı
        assert config.num_heads % config.num_kv_heads == 0, \
            f"num_heads ({config.num_heads}) num_kv_heads'e ({config.num_kv_heads}) tam bölünmeli"
        self.num_groups = config.num_heads // config.num_kv_heads

        # Q, K, V projeksiyonları — bias yok
        self.q_proj = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)

        # Output projeksiyon — bias yok
        self.out_proj = nn.Linear(config.num_heads * self.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, past_kv=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            freqs_cis: (seq_len, head_dim // 2) — RoPE frekansları
            past_kv: (past_k, past_v) tuple veya None — KV cache

        Returns:
            output: (batch_size, seq_len, d_model)
            present_kv: (k, v) tuple — güncel KV cache
        """
        B, T, C = x.shape

        # Q, K, V hesapla
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, T, head_dim)
        # k, v: (B, num_kv_heads, T, head_dim)

        # RoPE uygula (Q ve K'ya)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV Cache: geçmiş key/value'ları birleştir
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        # GQA: KV head'leri Q head sayısına genişlet (repeat)
        if self.num_groups > 1:
            # (B, num_kv_heads, S, D) → (B, num_kv_heads, 1, S, D) → (B, num_kv_heads, G, S, D)
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            # → (B, num_heads, S, D)
            k = k.reshape(B, self.num_heads, -1, self.head_dim)
            v = v.reshape(B, self.num_heads, -1, self.head_dim)

        S = k.size(2)  # Toplam sequence uzunluğu (past + current)

        # Scaled Dot-Product Attention — PyTorch native SDPA
        # FlashAttention / memory-efficient attention otomatik seçilir
        try:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=(past_kv is None),  # Sadece prefill'de causal mask
            )
        except RuntimeError:
            # SDPA desteklenmiyorsa manual fallback
            scale = math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device, dtype=torch.bool),
                diagonal=S - T + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # (B, num_heads, T, head_dim)

        # Head'leri birleştir
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)

        # Output projeksiyon
        output = self.out_proj(attn_output)

        return output, present_kv
