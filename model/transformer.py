# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

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
from model.moe import MorphRoutedMoE
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


class MTPHead(nn.Module):
    """
    Çoklu token tahmini başlığı (Medusa/DeepSeek-MTP esinli, hafif).

    h → h + W2·SiLU(W1·RMSNorm(h)) → RMSNorm → (ortak lm_head)
    W2 sıfırla başlatılır: eğitimin başında başlık, ana başlığın
    gizli durumunu aynen kullanır ve kararlı biçimde ayrışır.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_in = RMSNorm(config.d_model, eps=config.norm_eps)
        self.proj_in = nn.Linear(config.d_model, config.d_model, bias=False)
        self.proj_out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm_out = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, h):
        h = h + self.proj_out(F.silu(self.proj_in(self.norm_in(h))))
        return self.norm_out(h)


class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm Transformer bloğu.

    RMSNorm → GQA (+ RoPE, KV Cache) → Residual
    RMSNorm → SwiGLU FFN → Residual
    """

    def __init__(self, config: ModelConfig, use_moe: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.use_moe = use_moe
        self.ffn = MorphRoutedMoE(config) if use_moe else SwiGLUFeedForward(config)
        self.last_aux_loss = None

    def _ffn(self, h, morph_classes):
        if self.use_moe:
            out, aux = self.ffn(h, morph_classes)
            self.last_aux_loss = aux
            return out
        return self.ffn(h)

    def forward(self, x, freqs_cis, past_kv=None, use_checkpoint=False, morph_classes=None):
        """
        Args:
            x: (B, T, d_model)
            freqs_cis: RoPE frekansları
            past_kv: KV cache (opsiyonel)
            use_checkpoint: Gradient checkpointing kullan (eğitimde bellek tasarrufu)
            morph_classes: (B, T) giriş token sınıfları — MoE yönlendirme ipucu

        Returns:
            x: (B, T, d_model)
            present_kv: güncel KV cache
        """
        # Pre-RMSNorm → Attention → Residual
        attn_out, present_kv = self.attn(self.ln1(x), freqs_cis, past_kv)
        x = x + attn_out

        # Pre-RMSNorm → SwiGLU FFN → Residual (gradient checkpointing opsiyonel)
        if use_checkpoint and self.training and not self.use_moe:
            x = x + grad_checkpoint(self.ffn, self.ln2(x), use_reentrant=False)
        else:
            x = x + self._ffn(self.ln2(x), morph_classes)

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

    def __init__(self, config: ModelConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Eğitim sırasında etkinleştirilebilir

        # Token embedding (positional embedding yok — RoPE kullanılıyor)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blokları (MoE açıksa her moe_layer_freq'inci blok MoE)
        def _is_moe(layer_idx):
            return config.num_experts > 0 and (layer_idx + 1) % max(config.moe_layer_freq, 1) == 0

        self.blocks = nn.ModuleList([
            TransformerBlock(config, use_moe=_is_moe(i)) for i in range(config.num_layers)
        ])

        # Son RMSNorm
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)

        # Language model head — bias yok
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — embedding ve lm_head aynı ağırlıkları paylaşır
        self.tok_emb.weight = self.lm_head.weight

        # ─── Çoklu Token Tahmini (MTP) başlıkları ───
        # Başlık k (1..K), son gizli durumdan t+k+1. token'ı tahmin eder.
        # Ortak lm_head kullanılır; her başlık hafif bir artık blok taşır.
        # Çıkarımda kendi kendine spekülatif çözümleme için taslak üretir.
        self.mtp_heads = nn.ModuleList([
            MTPHead(config) for _ in range(config.num_mtp_heads)
        ])
        self._last_mtp_loss = 0.0
        self._last_moe_aux_loss = 0.0

        # ─── Morfolojik Başlık & Sınıflandırma ───
        self.morph_head = nn.Linear(config.d_model, 3, bias=False)
        self.use_morph_head = False
        self.morph_lambda = 0.2
        self._last_morph_loss = 0.0

        # Tokenizer yoksa otomatik yüklemeyi dene
        if tokenizer is None:
            import os
            for path in ["toprak_tokenizer.model", "../toprak_tokenizer.model", "../../toprak_tokenizer.model"]:
                if os.path.exists(path):
                    try:
                        from model.tokenizer import ToprakTokenizer
                        tokenizer = ToprakTokenizer(path)
                        break
                    except Exception:
                        pass

        # Her token için morfolojik sınıfı belirle
        # 0 = Kök (root), 1 = Ek (suffix), 2 = Özel/Noktalama/Sayı (special)
        token_classes = torch.zeros(config.vocab_size, dtype=torch.long)
        if tokenizer is not None:
            for token_id in range(config.vocab_size):
                token_str = tokenizer.id_to_token(token_id)
                if token_id < 4 or token_str in ('<sep>', '<cls>', '<mask>'):
                    token_classes[token_id] = 2
                elif token_str.startswith('▁'):
                    has_letters = any(c.isalpha() for c in token_str)
                    token_classes[token_id] = 0 if has_letters else 2
                else:
                    has_letters = any(c.isalpha() for c in token_str)
                    token_classes[token_id] = 1 if has_letters else 2

        self.register_buffer("token_morph_classes", token_classes)

        # RoPE frekanslarını önceden hesapla ve buffer olarak kaydet
        freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.rope_max_positions,  # Güvenlik payı
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Ağırlıkları başlat
        self.apply(self._init_weights)

        # Residual projeksiyonlar için özel scaled init
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.num_layers) ** 0.5)
        for head in self.mtp_heads:
            torch.nn.init.zeros_(head.proj_out.weight)

    def _apply(self, fn, *args, **kwargs):
        """
        RoPE tablosu complex tensördür; model.half() / .to(torch.bfloat16)
        gibi dtype dönüşümleri onu gerçel sayıya çevirip sessizce bozar.
        Dönüşümden sonra tabloyu yalnız cihaz olarak taşı, dtype'ı koru.
        """
        freqs_cis = self._buffers.pop("freqs_cis")
        try:
            super()._apply(fn, *args, **kwargs)
        finally:
            self.register_buffer(
                "freqs_cis", freqs_cis.to(self.lm_head.weight.device), persistent=False
            )
        return self

    def _init_weights(self, module):
        """Ağırlık başlatma."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids,
        targets=None,
        past_kvs=None,
        compute_lm_loss=True,
        return_hidden=False,
    ):
        """
        Args:
            input_ids: (batch_size, seq_len) — token ID'leri
            targets: (batch_size, seq_len) — hedef token ID'leri (opsiyonel)
            past_kvs: list of (k, v) tuples — KV cache (opsiyonel)
            compute_lm_loss: False ise standart CE hesaplanmaz; targets yine
                auxiliary başlıklar için kullanılır. Özel CE loss'larıyla
                birlikte morph-head eğitimi için kullanılır.
            return_hidden: True ise son RMSNorm sonrası gizli durumlar da
                döndürülür (MTP taslakları ve analiz araçları için).

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (eğer targets verilmişse)
            present_kvs: list of (k, v) — güncel KV cache
            hidden: (batch_size, seq_len, d_model) — yalnız return_hidden=True ise
        """
        B, T = input_ids.shape

        # Token embedding (RoPE sayesinde positional embedding yok)
        x = self.tok_emb(input_ids)  # (B, T, d_model)

        # RoPE frekansları — mevcut pozisyon offset'ini hesapla
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)
        else:
            past_len = 0

        if past_len + T > self.freqs_cis.size(0):
            raise ValueError(
                f"Dizi uzunluğu ({past_len + T}) RoPE tablosunu "
                f"({self.freqs_cis.size(0)}) aşıyor; bağlamı kırpın veya "
                f"max_seq_len / rope_scaling ile bağlamı genişletin."
            )
        freqs_cis = self.freqs_cis[past_len:past_len + T]

        morph_classes = None
        if self.config.num_experts > 0:
            morph_classes = self.token_morph_classes[input_ids]

        # Transformer blokları
        present_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(
                x, freqs_cis, past_kv,
                use_checkpoint=self.gradient_checkpointing,
                morph_classes=morph_classes,
            )
            present_kvs.append(present_kv)

        # Son RMSNorm
        x = self.ln_f(x)

        # LM Head — logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss hesaplama
        loss = None
        if targets is not None:
            if compute_lm_loss:
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                    ignore_index=self.config.pad_token_id,
                )
            else:
                loss = logits.new_zeros(())

            # Morfolojik çoklu görev kaybı (auxiliary loss)
            # NOT: Sınıf etiketi 0 = kök olduğu için pad maskesi sınıf
            # etiketinden değil, hedef token ID'sinden kurulur. (Eskiden
            # ignore_index=pad_token_id=0 tüm kök tokenlarını kayıptan
            # düşürüyor, pad tokenlarını ise "özel" sınıfı olarak eğitiyordu.)
            if self.use_morph_head:
                morph_logits = self.morph_head(x)  # (B, T, 3)
                morph_targets = self.token_morph_classes[targets].masked_fill(
                    targets == self.config.pad_token_id, -100
                )
                morph_loss = F.cross_entropy(
                    morph_logits.view(-1, 3),
                    morph_targets.view(-1),
                    ignore_index=-100,
                )
                self._last_morph_loss = morph_loss.item()
                loss = loss + self.morph_lambda * morph_loss

            # Çoklu token tahmini kaybı: başlık k, t+k+1 token'ını tahmin eder
            if len(self.mtp_heads) > 0:
                mtp_losses = []
                for k, head in enumerate(self.mtp_heads, start=1):
                    if T <= k:
                        break
                    mtp_logits = self.lm_head(head(x[:, :-k]))
                    mtp_losses.append(F.cross_entropy(
                        mtp_logits.reshape(-1, self.config.vocab_size),
                        targets[:, k:].reshape(-1),
                        ignore_index=self.config.pad_token_id,
                    ))
                if mtp_losses:
                    mtp_loss = torch.stack(mtp_losses).mean()
                    self._last_mtp_loss = mtp_loss.item()
                    loss = loss + self.config.mtp_lambda * mtp_loss

            # MoE yük dengeleme kaybı
            moe_aux = [b.last_aux_loss for b in self.blocks if b.use_moe and b.last_aux_loss is not None]
            if moe_aux:
                moe_loss = torch.stack(moe_aux).mean()
                self._last_moe_aux_loss = moe_loss.item()
                loss = loss + self.config.moe_aux_loss_coef * moe_loss

        if return_hidden:
            return logits, loss, present_kvs, x
        return logits, loss, present_kvs

    def mtp_logits(self, hidden: torch.Tensor) -> list:
        """
        MTP başlıklarının logit'leri.

        Args:
            hidden: (B, T, d_model) — forward(return_hidden=True) çıktısı

        Returns:
            [ (B, T, V), ... ] — başlık k için t+k+1 tahmini
        """
        return [self.lm_head(head(hidden)) for head in self.mtp_heads]

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
