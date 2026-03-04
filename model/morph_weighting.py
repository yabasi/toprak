"""
Toprak — Morfolojik Ağırlıklı Kayıp Fonksiyonu
(Morphology-Aware Token Weighting for Cross-Entropy Loss)

Dünyada ilk: Aglutinatif diller için ek (suffix) tokenlerine
farklı kayıp ağırlığı uygulayan eğitim mekanizması.

Motivasyon:
───────────
Türkçe'de bir kelime 5-10 alt-kelime (subword) tokenına bölünebilir:
  "Göremeyeceklerinden" → "▁Gör" + "eme" + "yecek" + "leri" + "nden"

Standart cross-entropy loss tüm tokenlere eşit ağırlık verir.
Ancak Türkçe'de eklerin doğru tahmin edilmesi, dil kalitesi için kritiktir:
  - "kitap" + "lar" + "dan" (doğru) vs "kitap" + "ler" + "den" (hatalı)

Bu modül, ek tokenlerine daha yüksek ağırlık vererek modeli Türkçe
eklemeli morfolojisini daha iyi öğrenmeye zorlar.

Çalışma Prensibi:
─────────────────
1. SentencePiece tokenizer'da ▁ prefix olan tokenler kelime başı (kök)
2. ▁ olmayan tokenler kelime devamı (ek/suffix)
3. Ek tokenlerinin CE loss ağırlığı suffix_weight ile çarpılır (varsayılan: 1.3)
4. Warmup ile ağırlık kademeli olarak artar (ani eğitim bozulması önlenir)
5. Kök ve ek loss'ları ayrı ayrı takip edilir (morfolojik öğrenme analizi)

Ek Özellikler:
──────────────
- TensorBoard'da kök/ek loss ayrımı → modelin morfolojiyi ne kadar öğrendiğini ölç
- suffix_weight warmup → mevcut eğitime eklendiğinde stabil geçiş
- Vowel Harmony Loss ile birlikte kullanılabilir

Checkpoint Uyumluluğu:
─────────────────────
- Model mimarisini DEĞİŞTİRMEZ — sadece loss hesaplamasını değiştirir
- Mevcut eğitime checkpoint'ten devam ederken sorunsuz eklenebilir
- torch.compile() ile uyumlu

Bellek Tüketimi:
───────────────
- Sadece (vocab_size,) boyutunda weight tensor — ihmal edilebilir ek bellek

Bu yaklaşım, dünyada bir ilktir — hiçbir açık kaynak dil modeli
eğitim sırasında token'ın morfolojik rolüne göre loss ağırlıklandırması yapmamaktadır.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphWeightedCELoss(nn.Module):
    """
    Morfolojik Ağırlıklı Cross-Entropy Loss.

    Token'ın morfolojik rolüne göre loss ağırlığı uygular:
    - Kök (▁ prefix): ağırlık 1.0
    - Ek (prefix yok): ağırlık suffix_weight (varsayılan: 1.3)
    - Özel tokenler (PAD, UNK, BOS, EOS): ağırlık 1.0

    Ayrıca kök ve ek token'ları için ayrı loss takibi yapar
    (TensorBoard'da morfolojik öğrenme analizi için).
    """

    def __init__(
        self,
        tokenizer,
        suffix_weight: float = 1.3,
        warmup_steps: int = 500,
        start_step: int = 0,
        pad_id: int = 0,
    ):
        """
        Args:
            tokenizer: ToprakTokenizer instance
            suffix_weight: Ek token'ları için ağırlık çarpanı (1.0-2.0 arası önerilir)
            warmup_steps: Ağırlık warmup adım sayısı (ani loss spike'ı önleme)
            start_step: Warmup başlangıç adımı (resume ederken otomatik ayarlanır)
            pad_id: Pad token ID (ignore_index)
        """
        super().__init__()
        self.suffix_weight = suffix_weight
        self.warmup_steps = warmup_steps
        self.start_step = start_step
        self.pad_id = pad_id

        vocab_size = tokenizer.get_vocab_size()

        # ── Her token için ağırlık ve morfolojik rol belirleme ──
        weights = torch.ones(vocab_size)
        is_suffix = torch.zeros(vocab_size, dtype=torch.bool)
        is_root = torch.zeros(vocab_size, dtype=torch.bool)

        n_suffix = 0
        n_root = 0
        n_special = 0

        for token_id in range(vocab_size):
            if token_id < 4:  # Özel tokenler (PAD=0, UNK=1, BOS=2, EOS=3)
                weights[token_id] = 1.0
                n_special += 1
            else:
                token_str = tokenizer.id_to_token(token_id)
                if token_str.startswith('▁'):
                    # Kelime başı / kök token
                    weights[token_id] = 1.0
                    is_root[token_id] = True
                    n_root += 1
                else:
                    # Kelime devamı / ek (suffix) token
                    weights[token_id] = suffix_weight
                    is_suffix[token_id] = True
                    n_suffix += 1

        # Sabit buffer'lar (gradient hesaplanmaz, .to(device) ile taşınır)
        self.register_buffer('token_weights', weights)
        self.register_buffer('is_suffix_mask', is_suffix)
        self.register_buffer('is_root_mask', is_root)

        # Metrik takibi — son step değerleri (TensorBoard'a yazılır)
        self._last_root_loss = 0.0
        self._last_suffix_loss = 0.0
        self._effective_weight = 1.0

        suffix_pct = n_suffix / vocab_size * 100
        root_pct = n_root / vocab_size * 100

        print(f"  ✓ Morfolojik Ağırlıklı Kayıp aktif (ek_ağırlığı={suffix_weight}, warmup={warmup_steps})")
        print(f"    Kök token: {n_root} ({root_pct:.1f}%), Ek token: {n_suffix} ({suffix_pct:.1f}%), Özel: {n_special}")

    def get_effective_suffix_weight(self, current_step: int) -> float:
        """
        Warmup'lı efektif suffix ağırlığını hesapla.

        Warmup süresi boyunca ağırlık 1.0'dan suffix_weight'e lineer artar.
        Bu, mevcut eğitime eklendiğinde ani loss değişimini önler.
        """
        steps_active = current_step - self.start_step
        if steps_active <= 0:
            return 1.0
        warmup_factor = min(1.0, steps_active / max(self.warmup_steps, 1))
        return 1.0 + (self.suffix_weight - 1.0) * warmup_factor

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """
        Morfolojik ağırlıklı CE loss hesapla.

        Standart CE loss'u, hedef token'ın morfolojik rolüne göre
        ağırlıklandırarak hesaplar. Ek token'ları daha yüksek ağırlık alır.

        Args:
            logits: (B, T, V) — model'in ham tahminleri
            targets: (B, T) — hedef token ID'leri
            current_step: mevcut eğitim adımı (warmup hesabı için)

        Returns:
            weighted_loss: scalar — morfolojik ağırlıklı cross-entropy loss
        """
        B, T, V = logits.shape

        effective_w = self.get_effective_suffix_weight(current_step)
        self._effective_weight = effective_w

        # Per-token CE loss (reduction='none' — her token ayrı hesaplanır)
        per_token_loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=self.pad_id,
            reduction='none',
        ).view(B, T)

        # Geçerli (pad olmayan) pozisyonlar
        valid_mask = (targets != self.pad_id).float()

        # Token ağırlıkları — warmup ile kademeli interpolasyon
        if effective_w > 1.0 + 1e-8:
            # Morfolojik ağırlıklama aktif
            raw_weights = self.token_weights[targets]  # (B, T)
            # 1.0 → suffix_weight arasında warmup interpolasyonu
            warmup_progress = min(
                1.0, (effective_w - 1.0) / max(self.suffix_weight - 1.0, 1e-8)
            )
            weights = 1.0 + (raw_weights - 1.0) * warmup_progress
            # Ağırlıklı ortalama (ağırlıklar normalizasyona girer)
            weighted_loss = (per_token_loss * weights * valid_mask).sum() / (
                weights * valid_mask
            ).sum().clamp(min=1)
        else:
            # Warmup başlamadan — standart CE
            weighted_loss = (per_token_loss * valid_mask).sum() / (
                valid_mask.sum().clamp(min=1)
            )

        # ─── Metrik Takibi (gradient hesaplanmaz) ───
        # Kök ve ek tokenlerinin loss'unu ayrı ayrı takip et
        # → TensorBoard'da morfolojik öğrenme analizi
        with torch.no_grad():
            suffix_positions = self.is_suffix_mask[targets] & (targets != self.pad_id)
            root_positions = self.is_root_mask[targets] & (targets != self.pad_id)

            if suffix_positions.any():
                self._last_suffix_loss = per_token_loss[suffix_positions].mean().item()
            if root_positions.any():
                self._last_root_loss = per_token_loss[root_positions].mean().item()

        return weighted_loss
