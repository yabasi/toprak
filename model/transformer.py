# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Transformer Mimarisi
Decoder-only Transformer, sıfırdan Türkçe dil modeli.

Modern mimari (2024):
- RMSNorm (LayerNorm yerine)
- SwiGLU aktivasyon (GELU yerine)
- RoPE (learned positional embedding yerine)
- GQA (Grouped Query Attention)
- KV Cache (inference hızlandırma)
- Bias yok (tüm Linear katmanlarda)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model.attention import GroupedQueryAttention
from model.config import ModelConfig
from model.norms import RMSNorm
from model.rope import precompute_freqs_cis


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Standart FFN: Linear → GELU → Linear
    SwiGLU FFN:   SiLU(gate(x)) * up(x) → down

    3 Linear katman, bias yok.
    Referans: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up → down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm Transformer bloğu.

    RMSNorm → GQA (+ RoPE, KV Cache) → Residual
    RMSNorm → SwiGLU FFN → Residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFeedForward(config)

    def forward(self, x, freqs_cis, past_kv=None, use_checkpoint=False):
        """
        Args:
            x: (B, T, d_model)
            freqs_cis: RoPE frekansları
            past_kv: KV cache (opsiyonel)
            use_checkpoint: Gradient checkpointing kullan (eğitimde bellek tasarrufu)

        Returns:
            x: (B, T, d_model)
            present_kv: güncel KV cache
        """
        # Pre-RMSNorm → Attention → Residual
        attn_out, present_kv = self.attn(self.ln1(x), freqs_cis, past_kv)
        x = x + attn_out

        # Pre-RMSNorm → SwiGLU FFN → Residual (gradient checkpointing opsiyonel)
        if use_checkpoint and self.training:
            x = x + grad_checkpoint(self.ffn, self.ln2(x), use_reentrant=False)
        else:
            x = x + self.ffn(self.ln2(x))

        return x, present_kv


class ToprakLM(nn.Module):
    """
    Toprak — Sıfırdan Türkçe Dil Modeli

    Modern decoder-only Transformer mimarisi:
    Token Embedding → N × TransformerBlock → RMSNorm → LM Head

    Özellikler:
    - RMSNorm (bias'sız, hızlı normalizasyon)
    - SwiGLU (gated FFN, SiLU aktivasyon)
    - RoPE (rotary position embedding, learned positional yerine)
    - GQA (grouped query attention, daha az KV head)
    - KV Cache (inference'da 5-10x hız artışı)
    - Weight Tying (embedding ↔ LM head)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Eğitim sırasında etkinleştirilebilir

        # Token embedding (positional embedding yok — RoPE kullanılıyor)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blokları
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Son RMSNorm
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)

        # Language model head — bias yok
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — embedding ve lm_head aynı ağırlıkları paylaşır
        self.tok_emb.weight = self.lm_head.weight

        # RoPE frekanslarını önceden hesapla ve buffer olarak kaydet
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len * 2,  # Güvenlik payı
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Ağırlıkları başlat
        self.apply(self._init_weights)

        # Residual projeksiyonlar için özel scaled init
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.num_layers) ** 0.5)

    def _init_weights(self, module):
        """Ağırlık başlatma."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, past_kvs=None):
        """
        Args:
            input_ids: (batch_size, seq_len) — token ID'leri
            targets: (batch_size, seq_len) — hedef token ID'leri (opsiyonel)
            past_kvs: list of (k, v) tuples — KV cache (opsiyonel)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (eğer targets verilmişse)
            present_kvs: list of (k, v) — güncel KV cache
        """
        B, T = input_ids.shape

        # Token embedding (RoPE sayesinde positional embedding yok)
        x = self.tok_emb(input_ids)  # (B, T, d_model)

        # RoPE frekansları — mevcut pozisyon offset'ini hesapla
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)
        else:
            past_len = 0

        freqs_cis = self.freqs_cis[past_len:past_len + T]

        # Transformer blokları
        present_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(
                x, freqs_cis, past_kv,
                use_checkpoint=self.gradient_checkpointing,
            )
            present_kvs.append(present_kv)

        # Son RMSNorm
        x = self.ln_f(x)

        # LM Head — logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss hesaplama
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss, present_kvs

    def count_parameters(self) -> int:
        """Toplam eğitilebilir parametre sayısı."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    ):
        """
        KV Cache destekli autoregressive metin üretimi.

        İlk adımda tüm prompt işlenir (prefill),
        sonraki adımlarda sadece son token işlenir (decode).

        Args:
            input_ids: (1, seq_len) — başlangıç token'ları
            max_new_tokens: üretilecek maksimum yeni token sayısı
            temperature: sampling sıcaklığı
            top_k: top-k filtering
            top_p: nucleus sampling eşiği
        """
        self.eval()
        past_kvs = None

        for step in range(max_new_tokens):
            if past_kvs is None:
                # Prefill: tüm prompt'u işle
                idx_input = input_ids
            else:
                # Decode: sadece son token
                idx_input = input_ids[:, -1:]

            # Forward pass
            logits, _, past_kvs = self(idx_input, past_kvs=past_kvs)
            logits = logits[:, -1, :] / temperature  # Son token'ın logit'leri

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # EOS kontrolü
            if next_token.item() == self.config.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
