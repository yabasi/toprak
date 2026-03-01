"""
Toprak — Transformer Mimarisi
Decoder-only Transformer, sıfırdan Türkçe dil modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer bloğu."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Pre-LN: LayerNorm → Attention → Residual
        x = x + self.attn(self.ln1(x))
        # Pre-LN: LayerNorm → FFN → Residual
        x = x + self.ffn(self.ln2(x))
        return x


class ToprakLM(nn.Module):
    """
    Toprak — Sıfırdan Türkçe Dil Modeli

    Decoder-only Transformer mimarisi.
    Token embedding + Positional embedding → N × TransformerBlock → LM Head
    Weight tying: token embedding ağırlıkları LM head ile paylaşılır.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding katmanları
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blokları
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Son layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — embedding ve lm_head aynı ağırlıkları paylaşır
        self.tok_emb.weight = self.lm_head.weight

        # Ağırlıkları başlat
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier/He benzeri ağırlık başlatma."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch_size, seq_len) — token ID'leri
            targets: (batch_size, seq_len) — hedef token ID'leri (opsiyonel)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (eğer targets verilmişse)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence uzunluğu ({T}) max_seq_len'i ({self.config.max_seq_len}) aşıyor"

        # Token + Position embeddings
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)  # (1, T)
        tok_emb = self.tok_emb(input_ids)        # (B, T, d_model)
        pos_emb = self.pos_emb(positions)         # (1, T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # Transformer blokları
        for block in self.blocks:
            x = block(x)

        # Son layer norm
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

        return logits, loss

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
        Autoregressive metin üretimi.

        Args:
            input_ids: (1, seq_len) — başlangıç token'ları
            max_new_tokens: üretilecek maksimum yeni token sayısı
            temperature: sampling sıcaklığı
            top_k: top-k filtering
            top_p: nucleus sampling eşiği
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Sequence uzunluğunu max_seq_len ile sınırla
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
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
                # Kümülatif olasılık top_p'yi aşan token'ları kaldır
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
