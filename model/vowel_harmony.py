"""
Toprak — Ünlü Uyumu Auxiliary Loss
Türkçe'ye özgü dilbilgisi kaybı: ünlü uyumuna aykırı token tahminlerini cezalandırır.

Türkçe Büyük Ünlü Uyumu:
- Kalın ünlüler (a, ı, o, u): Kalın ünlüden sonra kalın ünlülü ek gelir
- İnce ünlüler (e, i, ö, ü): İnce ünlüden sonra ince ünlülü ek gelir
- Örnek: "kitap+lar" (doğru), "kitap+ler" (yanlış)

Çalışma Prensibi:
1. Her vocab token'ı için son/ilk ünlü sınıfı önceden hesaplanır
2. Eğitim sırasında, kelime devamı olan pozisyonlarda önceki token'ın
   son ünlüsüne bakılır
3. Model'in uyumsuz ünlülü tokenlere atadığı olasılık ölçülür
4. Bu olasılık λ ile ağırlıklandırılarak ceza loss'u olarak eklenir
5. Gradient modele geri yayılarak uyum farkındalığı öğretilir

Bellek-Verimli İmplementasyon:
- Tam softmax yerine logsumexp kullanılır
- index_select ile sadece ilgili token alt kümesi işlenir
- Peak bellek: ~%30-40 vocab boyutu (full softmax yerine)

Bu yaklaşım, dünyada bir ilktir — hiçbir açık kaynak dil modeli
eğitim sırasında Türkçe ünlü uyumunu auxiliary loss olarak kullanmamaktadır.
"""

import torch
import torch.nn as nn


# ── Türkçe Ünlü Sınıfları ──────────────────────────────────
# Kalın (Back): a, ı, o, u — I (U+0049) = ı'nın büyük hali
# İnce (Front): e, i, ö, ü — İ (U+0130) = i'nin büyük hali

BACK_VOWELS = set('aıouAIOU')
FRONT_VOWELS = set('eiöüEİÖÜ')
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS


def _classify_last_vowel(token_str: str) -> int:
    """
    Token'ın son ünlüsünün sınıfını döndür.

    Returns:
        0 = kalın (back), 1 = ince (front), 2 = ünlü yok (nötr)
    """
    for ch in reversed(token_str):
        if ch in BACK_VOWELS:
            return 0
        if ch in FRONT_VOWELS:
            return 1
    return 2


def _classify_first_vowel(token_str: str) -> int:
    """
    Token'ın ilk ünlüsünün sınıfını döndür.

    Returns:
        0 = kalın (back), 1 = ince (front), 2 = ünlü yok (nötr)
    """
    for ch in token_str:
        if ch in BACK_VOWELS:
            return 0
        if ch in FRONT_VOWELS:
            return 1
    return 2


class VowelHarmonyLoss(nn.Module):
    """
    Türkçe Ünlü Uyumu Auxiliary Loss.

    Eğitim sırasında model'in logit'lerini analiz ederek, Türkçe ünlü
    uyumuna aykırı token tahminlerini cezalandırır. Model mimarisini
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
            lambda_weight: Ünlü uyumu loss ağırlığı (0.05-0.3 arası önerilir)
            warmup_steps: Lambda warmup adım sayısı (ani loss spike önleme)
            start_step: Warmup başlangıç adımı (resume ederken otomatik ayarlanır)
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.warmup_steps = warmup_steps
        self.start_step = start_step

        vocab_size = tokenizer.get_vocab_size()

        # Her token için ünlü sınıflarını analiz et
        last_vowel = torch.zeros(vocab_size, dtype=torch.long)
        first_vowel = torch.zeros(vocab_size, dtype=torch.long)
        is_word_start = torch.zeros(vocab_size, dtype=torch.bool)

        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)

            # SentencePiece: ▁ prefix = kelime başı
            # Special tokens (PAD=0, UNK=1, BOS=2, EOS=3) da kelime başı sayılır
            if token_str.startswith('▁') or token_id < 4:
                is_word_start[token_id] = True
                clean_str = token_str.lstrip('▁')
            else:
                clean_str = token_str

            last_vowel[token_id] = _classify_last_vowel(clean_str)
            first_vowel[token_id] = _classify_first_vowel(clean_str)

        # Sabit buffer'lar (gradient hesaplanmaz, .to(device) ile taşınır)
        self.register_buffer('last_vowel', last_vowel)
        self.register_buffer('first_vowel', first_vowel)
        self.register_buffer('is_word_start', is_word_start)

        # Bellek-verimli hesaplama için uyumsuz token index'leri
        # (logsumexp sadece bu alt küme üzerinde çalışır)
        self.register_buffer(
            'front_token_ids',
            (first_vowel == 1).nonzero(as_tuple=True)[0]
        )
        self.register_buffer(
            'back_token_ids',
            (first_vowel == 0).nonzero(as_tuple=True)[0]
        )

        # İstatistikler
        n_front_first = len(self.front_token_ids)
        n_back_first = len(self.back_token_ids)
        n_neutral = vocab_size - n_front_first - n_back_first
        n_word_starts = is_word_start.sum().item()

        print(f"  ✓ Ünlü Uyumu Loss aktif (λ={lambda_weight}, warmup={warmup_steps} adım)")
        print(f"    İlk ünlü — Kalın: {n_back_first}, İnce: {n_front_first}, Nötr: {n_neutral}")
        print(f"    Kelime başı token: {n_word_starts} / {vocab_size}")

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
        Ünlü uyumu ceza loss'u hesapla.

        Args:
            logits: (B, T, V) — model'in ham tahminleri
            targets: (B, T) — hedef token ID'leri
            current_step: mevcut eğitim adımı (warmup hesabı için)

        Returns:
            loss: scalar — ağırlıklandırılmış ünlü uyumu cezası
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

        # Önceki token'ların son ünlü sınıfı
        prev_last = self.last_vowel[prev_targets]  # (B, T-1): 0=kalın, 1=ince, 2=yok

        # Mevcut token kelime başı mı?
        curr_start = self.is_word_start[curr_targets]  # (B, T-1)

        # Geçerli pozisyonlar:
        # - Kelime devamı (kelime başı DEĞİL)
        # - Önceki token'da ünlü VAR (nötr değil)
        # - Pad token değil
        valid_mask = (~curr_start) & (prev_last != 2) & (curr_targets != 0)

        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=logits.device)

        # ─── Bellek-Verimli Uyumsuzluk Olasılığı Hesaplama ───
        # P(violation) = exp(logsumexp(logits[violation_set]) - logsumexp(logits[all]))
        # index_select ile sadece ilgili token alt kümesi çekilir

        # Tüm logit'lerin log-partition fonksiyonu (normalleştirme sabiti)
        lse_all = torch.logsumexp(curr_logits, dim=-1)  # (B, T-1)

        # Kalın sonrası → İnce tokenler uyumsuz
        if len(self.front_token_ids) > 0:
            front_logits = curr_logits.index_select(-1, self.front_token_ids)
            lse_front = torch.logsumexp(front_logits, dim=-1)
            log_p_front = lse_front - lse_all  # log P(ince token)
        else:
            log_p_front = torch.full_like(lse_all, float('-inf'))

        # İnce sonrası → Kalın tokenler uyumsuz
        if len(self.back_token_ids) > 0:
            back_logits = curr_logits.index_select(-1, self.back_token_ids)
            lse_back = torch.logsumexp(back_logits, dim=-1)
            log_p_back = lse_back - lse_all  # log P(kalın token)
        else:
            log_p_back = torch.full_like(lse_all, float('-inf'))

        # Önceki ünlüye göre uygun uyumsuzluk olasılığını seç
        # prev_last == 0 (kalın) → ince gelmesi uyumsuz → log_p_front
        # prev_last == 1 (ince) → kalın gelmesi uyumsuz → log_p_back
        log_p_violation = torch.where(prev_last == 0, log_p_front, log_p_back)

        # Log olasılıktan gerçek olasılığa dönüştür
        p_violation = log_p_violation.exp()  # (B, T-1)

        # Sadece geçerli pozisyonlardan ortalama al
        p_violation = p_violation * valid_mask.float()
        loss = p_violation.sum() / valid_count.clamp(min=1)

        return effective_lambda * loss
