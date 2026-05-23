# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Ünsüz Benzeşmesi Yardımcı Kaybı
Türkçe'ye özgü dilbilgisi kaybı: ünsüz benzeşmesi (sertleşmesi) kurallarını ihlal eden token tahminlerini cezalandırır.

Türkçe Ünsüz Benzeşmesi (Sertleşmesi):
- Sert ünsüzle biten kelimelere (f, s, t, k, ç, ş, h, p) "c, d, g" ile başlayan ekler getirilemez.
- Bu ekler "ç, t, k" ünsüzlerine dönüşerek sertleşir.
- Örnek: "kitap+cı" -> "kitap+çı" (doğru), "kitap+cı" (yanlış / ihlal)
- Örnek: "git+di" -> "gitti" (doğru), "git+di" (yanlış / ihlal)

Çalışma Prensibi:
1. Her vocab token'ı için son harfinin sert ünsüz olup olmadığı ve ilk harfinin c, d, g olup olmadığı önceden hesaplanır.
2. Eğitim sırasında, kelime devamı (ek) olan pozisyonlarda önceki token'ın son harfinin sert ünsüz olup olmadığı kontrol edilir.
3. Önceki token sert ünsüzle biterken, modelin sıradaki token olarak "c, d, g" ile başlayan bir ek token'ına atadığı olasılık ölçülür.
4. Bu olasılık λ ile ağırlıklandırılarak ceza loss'u olarak eklenir.
5. Gradient modele geri yayılarak ünsüz benzeşmesi farkındalığı öğretilir.

Bellek-Verimli İmplementasyon:
- Tam softmax yerine logsumexp kullanılır.
- index_select ile sadece ilgili token alt kümesi işlenir.
- Peak bellek: ~%1-2 vocab boyutu (çünkü sadece c, d, g ile başlayan ek token'ları filtrelenir).

Bu yaklaşım, dünyada bir ilktir — hiçbir açık kaynak dil modeli eğitim sırasında Türkçe ünsüz benzeşmesini auxiliary loss olarak kullanmamaktadır.
"""

import torch
import torch.nn as nn

# Sert ünsüzler (Voiceless Consonants): f, s, t, k, ç, ş, h, p
VOICELESS_CONSONANTS = set('fstkçşhpFSTKÇŞHP')

# Sertleşebilen yumuşak ünsüzler (Mutable Voiced Consonants): c, d, g
MUTABLE_VOICED_CONSONANTS = set('cdgCDG')


def _ends_with_voiceless(token_str: str) -> bool:
    """
    Token'ın son alfabetik karakterinin sert ünsüz (voiceless) olup olmadığını kontrol et.

    Args:
        token_str: Kontrol edilecek token metni

    Returns:
        bool: Sert ünsüzle bitiyorsa True
    """
    for ch in reversed(token_str):
        if ch.isalpha():
            return ch in VOICELESS_CONSONANTS
    return False


def _starts_with_mutable_voiced(token_str: str) -> bool:
    """
    Token'ın ilk alfabetik karakterinin yumuşak/sertleşebilir ünsüz (c, d, g) olup olmadığını kontrol et.

    Args:
        token_str: Kontrol edilecek token metni

    Returns:
        bool: c, d, g ile başlıyorsa True
    """
    for ch in token_str:
        if ch.isalpha():
            return ch in MUTABLE_VOICED_CONSONANTS
    return False


class ConsonantHarmonyLoss(nn.Module):
    """
    Türkçe Ünsüz Benzeşmesi Yardımcı Kaybı.

    Eğitim sırasında modelin logit'lerini analiz ederek, Türkçe ünsüz
    benzeşmesine aykırı token tahminlerini cezalandırır. Model mimarisini
    değiştirmez — sadece ek bir loss sinyali verir.

    Mevcut eğitime checkpoint'ten devam ederken eklenebilir.
    """

    def __init__(
        self,
        tokenizer,
        lambda_weight: float = 0.1,
        warmup_steps: int = 1000,
        start_step: int = 0,
    ):
        """
        Args:
            tokenizer: ToprakTokenizer instance
            lambda_weight: Ünsüz benzeşmesi loss ağırlığı (0.05-0.3 arası önerilir)
            warmup_steps: Lambda warmup adım sayısı (ani loss spike önleme)
            start_step: Warmup başlangıç adımı (resume ederken otomatik ayarlanır)
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.warmup_steps = warmup_steps
        self.start_step = start_step

        vocab_size = tokenizer.get_vocab_size()

        # Her token için ünsüz sınıflarını analiz et
        ends_voiceless = torch.zeros(vocab_size, dtype=torch.long)
        is_word_start = torch.zeros(vocab_size, dtype=torch.bool)
        starts_mutable_voiced = torch.zeros(vocab_size, dtype=torch.bool)

        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)

            # SentencePiece: ▁ prefix = kelime başı
            # Special tokens (PAD=0, UNK=1, BOS=2, EOS=3) da kelime başı sayılır
            if token_str.startswith('▁') or token_id < 4:
                is_word_start[token_id] = True
                clean_str = token_str.lstrip('▁')
            else:
                clean_str = token_str

            ends_voiceless[token_id] = 1 if _ends_with_voiceless(clean_str) else 0
            starts_mutable_voiced[token_id] = _starts_with_mutable_voiced(clean_str)

        # Sabit buffer'lar (gradient hesaplanmaz, .to(device) ile taşınır)
        self.register_buffer('ends_voiceless', ends_voiceless)
        self.register_buffer('is_word_start', is_word_start)

        # Kelime devamı (ek) olup "c, d, g" ile başlayan token'ların ID'leri
        mutable_voiced_suffix_mask = starts_mutable_voiced & (~is_word_start)
        self.register_buffer(
            'mutable_voiced_suffix_ids',
            mutable_voiced_suffix_mask.nonzero(as_tuple=True)[0]
        )

        # İstatistikler
        n_ends_voiceless = (ends_voiceless == 1).sum().item()
        n_mutable_voiced_suffix = len(self.mutable_voiced_suffix_ids)

        print(f"  ✓ Ünsüz Benzeşmesi Loss aktif (λ={lambda_weight}, warmup={warmup_steps} adım)")
        print(f"    Sert ünsüzle biten token sayısı: {n_ends_voiceless} / {vocab_size}")
        print(f"    Yumuşak ünsüzle başlayan ek token sayısı: {n_mutable_voiced_suffix} / {vocab_size}")

    def get_effective_lambda(self, current_step: int) -> float:
        """Warmup'lı efektif lambda hesapla."""
        steps_active = current_step - self.start_step
        if steps_active <= 0:
            return 0.0
        warmup_factor = min(1.0, steps_active / max(self.warmup_steps, 1))
        return self.lambda_weight * warmup_factor

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """
        Ünsüz benzeşmesi ceza loss'u hesapla.

        Args:
            logits: (B, T, V) — model'in ham tahminleri
            targets: (B, T) — hedef token ID'leri
            current_step: mevcut eğitim adımı (warmup hesabı için)

        Returns:
            loss: scalar — ağırlıklandırılmış ünsüz benzeşmesi cezası
        """
        effective_lambda = self.get_effective_lambda(current_step)
        if effective_lambda == 0.0:
            return torch.tensor(0.0, device=logits.device)

        B, T, V = logits.shape
        if T < 2:
            return torch.tensor(0.0, device=logits.device)

        # Pozisyonları hazırla: (t-1) → (t) çiftleri
        prev_targets = targets[:, :-1]       # (B, T-1) — önceki token
        curr_targets = targets[:, 1:]        # (B, T-1) — mevcut token
        curr_logits = logits[:, 1:, :]       # (B, T-1, V) — mevcut pozisyon logit'leri

        # Önceki token'ların son ünsüz sınıfı (1 = sert ünsüz, 0 = diğer/nötr)
        prev_ends_voiceless = self.ends_voiceless[prev_targets]  # (B, T-1)

        # Mevcut token kelime başı mı?
        curr_start = self.is_word_start[curr_targets]  # (B, T-1)

        # Geçerli pozisyonlar:
        # - Kelime devamı (kelime başı DEĞİL)
        # - Önceki token sert ünsüzle bitmiş
        # - Pad token değil
        valid_mask = (~curr_start) & (prev_ends_voiceless == 1) & (curr_targets != 0)

        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device)

        # ─── Bellek-Verimli Uyumsuzluk Olasılığı Hesaplama ───
        # P(violation) = exp(logsumexp(logits[violation_set]) - logsumexp(logits[all]))
        # index_select ile sadece c, d, g ile başlayan ek token'ları çekilir

        # Tüm logit'lerin log-partition fonksiyonu (normalleştirme sabiti)
        lse_all = torch.logsumexp(curr_logits, dim=-1)  # (B, T-1)

        # Sert ünsüz sonrası → C, D, G ile başlayan ek token'lar uyumsuz (cezalık)
        if len(self.mutable_voiced_suffix_ids) > 0:
            mutable_logits = curr_logits.index_select(-1, self.mutable_voiced_suffix_ids)
            lse_mutable = torch.logsumexp(mutable_logits, dim=-1)
            log_p_mutable = lse_mutable - lse_all  # log P(uyumsuz ek token)
            p_violation = log_p_mutable.exp()     # (B, T-1)
        else:
            p_violation = torch.zeros_like(lse_all)

        # Sadece geçerli pozisyonlardan ortalama al
        p_violation = p_violation * valid_mask.float()
        loss = p_violation.sum() / valid_count.clamp(min=1)

        return effective_lambda * loss
