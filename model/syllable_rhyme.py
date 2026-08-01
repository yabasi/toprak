# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Hece ve Kafiye Yardımcı Kaybı (Syllable & Rhyme Loss)
Türkçe'ye özgü şiirsel ve yapısal üretim kaybı:
1. Türkçe Hece Ölçüsü (Hece Vezni): Satırlardaki hece (ünlü) sayısını takip eder,
   taşmaları ve erken satır sonlarını cezalandırır.
2. Kafiye (Uyak): Bir önceki satırın son token'ının kafiye sınıfını çıkarır ve
   mevcut satır sonunun bu sınıf ile kafiyeli bitmesini şart koşar.

Çalışma Prensibi:
- Türkçe'de hece sayısı ünlü harf sayısına eşittir (a, ı, o, u, e, i, ö, ü).
- Kafiye sınıfı, token'ın son iki alfabetik sesine göre (örn: "yollar" -> "ar")
  kelime haznesi düzeyinde önceden gruplandırılır.
- Eğitim sırasında logsumexp tabanlı olasılık cezaları ile gradyan yollanır.

Dünyada ilk defa hece vezni ve kafiye kısıtları bir dil modeline 
eğitim aşamasında auxiliary loss olarak entegre edilmiştir.
"""

import torch
import torch.nn as nn


def _count_syllables(token_str: str) -> int:
    """
    Token içindeki Türkçe ünlü harfleri sayarak hece sayısını döndürür.
    Özel token'lar (<pad>, <unk>, vb.) yoksayılır.
    """
    if token_str.startswith('<') and token_str.endswith('>'):
        return 0
    clean_str = token_str.replace('▁', '')
    count = 0
    for ch in clean_str:
        if ch in 'aeıioöuüAEIİOÖUÜ':
            count += 1
    return count


def _get_rhyme_ending(token_str: str) -> str:
    """
    Kafiye sınıfı tespiti için son 2 alfabetik karakteri (küçük harf) döndürür.
    Özel token'lar yoksayılır.
    """
    if token_str.startswith('<') and token_str.endswith('>'):
        return ""
    clean_str = token_str.replace('▁', '')
    chars = [c.lower() for c in clean_str if c.isalpha()]
    if len(chars) >= 2:
        return "".join(chars[-2:])
    elif len(chars) == 1:
        return chars[0]
    return ""


class SyllableRhymeLoss(nn.Module):
    """
    Hece Ölçüsü ve Kafiye Uyumu Auxiliary Loss Modülü.
    """

    def __init__(
        self,
        tokenizer,
        lambda_syllable: float = 0.1,
        lambda_rhyme: float = 0.1,
        warmup_steps: int = 1000,
        start_step: int = 0,
    ):
        """
        Args:
            tokenizer: ToprakTokenizer örneği
            lambda_syllable: Hece ölçüsü loss ağırlığı (varsayılan: 0.1)
            lambda_rhyme: Kafiye loss ağırlığı (varsayılan: 0.1)
            warmup_steps: Warmup adım sayısı
            start_step: Başlangıç adımı
        """
        super().__init__()
        self.lambda_syllable = lambda_syllable
        self.lambda_rhyme = lambda_rhyme
        self.warmup_steps = warmup_steps
        self.start_step = start_step

        vocab_size = tokenizer.get_vocab_size()

        # Her token için hece sayısı ve kafiye sınıfı önbelleklemesi
        token_syllables = torch.zeros(vocab_size, dtype=torch.long)
        
        ending_to_id = {}
        next_ending_id = 1  # 0: tanımsız veya kafiyesiz
        token_rhyme_classes = torch.zeros(vocab_size, dtype=torch.long)

        for token_id in range(vocab_size):
            token_str = tokenizer.id_to_token(token_id)
            
            # Hece sayısı
            token_syllables[token_id] = _count_syllables(token_str)
            
            # Kafiye sınıfı
            ending = _get_rhyme_ending(token_str)
            if ending:
                if ending not in ending_to_id:
                    ending_to_id[ending] = next_ending_id
                    next_ending_id += 1
                token_rhyme_classes[token_id] = ending_to_id[ending]

        # PyTorch buffer'larına kaydet
        self.register_buffer('token_syllables', token_syllables)
        self.register_buffer('token_rhyme_classes', token_rhyme_classes)

        n_syllable_tokens = (token_syllables >= 1).sum().item()
        n_rhyme_classes = len(ending_to_id)

        print(f"  ✓ Hece ve Kafiye Kaybı aktif (λ_hece={lambda_syllable}, λ_kafiye={lambda_rhyme}, warmup={warmup_steps} adım)")
        print(f"    Hece içeren token sayısı: {n_syllable_tokens} / {vocab_size}")
        print(f"    Benzersiz kafiye sınıfı sayısı: {n_rhyme_classes}")

    def get_effective_lambdas(self, current_step: int) -> tuple:
        """Warmup'lı efektif lambda ağırlıklarını hesapla."""
        steps_active = current_step - self.start_step
        if steps_active <= 0:
            return 0.0, 0.0
        warmup_factor = min(1.0, steps_active / max(self.warmup_steps, 1))
        return self.lambda_syllable * warmup_factor, self.lambda_rhyme * warmup_factor

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        current_step: int,
    ) -> tuple:
        """
        Hece ve Kafiye ceza loss'larını hesaplar.

        Args:
            logits: (B, T, V) — model'in logit çıktıları
            targets: (B, T) — hedef token ID'leri
            current_step: mevcut eğitim adımı (warmup için)

        Returns:
            tuple (syllable_loss, rhyme_loss): scalar loss tensor'ları
        """
        eff_lambda_syllable, eff_lambda_rhyme = self.get_effective_lambdas(current_step)
        
        if eff_lambda_syllable == 0.0 and eff_lambda_rhyme == 0.0:
            return (
                torch.tensor(0.0, device=logits.device),
                torch.tensor(0.0, device=logits.device)
            )

        B, T, V = logits.shape
        if T < 2:
            return (
                torch.tensor(0.0, device=logits.device),
                torch.tensor(0.0, device=logits.device)
            )

        # Pozisyonları ayarla: (t-1) -> (t) geçişleri
        prev_targets = targets[:, :-1]
        curr_targets = targets[:, 1:]
        curr_logits = logits[:, 1:, :]

        target_syllable_counts = self.token_syllables[targets]  # (B, T)

        # ─── 1. DİNAMİK HEDEF HECE ÖLÇÜSÜ (S_target) TESPİTİ ───
        # Her dizi için ilk tamamlanmış satırın hece sayısını hedef ölçü seçeriz
        # Yeni satır token'ı ID 17'dir (<0x0A>).
        has_nl = (targets == 17).any(dim=-1)
        first_nl_idx = (targets == 17).int().argmax(dim=-1)  # (B,)
        
        S_target = torch.zeros(B, dtype=torch.long, device=targets.device)
        for i in range(B):
            if has_nl[i]:
                nl_idx = first_nl_idx[i].item()
                S_target[i] = target_syllable_counts[i, :nl_idx].sum()
            else:
                S_target[i] = 11  # Varsayılan hece ölçüsü: 11
        
        # Makul ölçü sınırları (hece vezni genelde 4-20 hecedir)
        S_target = torch.where(S_target >= 4, S_target, torch.tensor(11, device=targets.device))
        S_target = torch.where(S_target <= 20, S_target, torch.tensor(11, device=targets.device))

        # ─── 2. SATIR İÇİ KÜMÜLATİF HECE (S_prefix) TAKİBİ ───
        S_prefix = torch.zeros((B, T - 1), dtype=torch.long, device=targets.device)
        current_sum = torch.zeros(B, dtype=torch.long, device=targets.device)
        for t in range(T - 1):
            S_prefix[:, t] = current_sum
            is_nl = (targets[:, t] == 17)
            current_sum = torch.where(is_nl, torch.zeros_like(current_sum), current_sum + target_syllable_counts[:, t])

        # ─── 3. HECE UYUMSUZLUĞU (Syllable Loss) ───
        syllable_loss = torch.tensor(0.0, device=logits.device)
        if eff_lambda_syllable > 0.0:
            s_prefix_expanded = S_prefix.unsqueeze(-1)  # (B, T-1, 1)
            s_target_expanded = S_target.view(B, 1, 1)   # (B, 1, 1)
            s_token_expanded = self.token_syllables.view(1, 1, V)  # (1, 1, V)
            
            # Koşul A: Satır hece sınırını aşan token tahminleri
            overflow_mask = (s_prefix_expanded + s_token_expanded) > s_target_expanded
            
            # Koşul B: Satır tamamlanmadan newline (17) tahmin edilmesi
            is_newline_expanded = (torch.arange(V, device=logits.device) == 17).view(1, 1, V)
            premature_nl_mask = is_newline_expanded & (s_prefix_expanded < s_target_expanded)
            
            # Koşul C: Satır tamamlanmışken ekstra heceli kelime tahmin edilmesi
            complete_mask = (s_prefix_expanded >= s_target_expanded) & (s_token_expanded >= 1)
            
            violation_mask = overflow_mask | premature_nl_mask | complete_mask
            
            # Uyumsuz logit'leri filtrele ve olasılık hesapla
            violation_logits = torch.where(violation_mask, curr_logits, torch.tensor(float('-inf'), device=logits.device))
            lse_violation = torch.logsumexp(violation_logits, dim=-1)
            lse_all = torch.logsumexp(curr_logits, dim=-1)
            
            log_p_violation = lse_violation - lse_all
            p_violation = log_p_violation.exp()
            
            # Sadece geçerli (PAD dışı) token pozisyonlarında ortala
            valid_mask = (curr_targets != 0)
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                p_violation = p_violation * valid_mask.float()
                syllable_loss = (p_violation.sum() / valid_count.clamp(min=1)) * eff_lambda_syllable

        # ─── 4. KAFİYE UYUMSUZLUĞU (Rhyme Loss) ───
        rhyme_loss = torch.tensor(0.0, device=logits.device)
        if eff_lambda_rhyme > 0.0:
            # Satır sonu pozisyonları (bir sonraki token'ı newline 17 olanlar)
            is_token_line_end = (targets[:, 1:] == 17) & (targets[:, :-1] != 17) & (targets[:, :-1] != 0)
            
            expected_rhyme = torch.zeros((B, T - 1), dtype=torch.long, device=targets.device)
            last_rhyme = torch.zeros(B, dtype=torch.long, device=targets.device)
            
            for t in range(T - 1):
                is_end = is_token_line_end[:, t]
                expected_rhyme[:, t] = torch.where(is_end & (last_rhyme > 0), last_rhyme, torch.zeros_like(last_rhyme))
                
                # Kafiye hedefini güncelle
                target_rhyme = self.token_rhyme_classes[targets[:, t]]
                last_rhyme = torch.where(is_end & (target_rhyme > 0), target_rhyme, last_rhyme)

            valid_rhyme_mask = (expected_rhyme > 0) & (curr_targets != 0)
            valid_rhyme_count = valid_rhyme_mask.sum()
            
            if valid_rhyme_count > 0:
                # Eşleşen kafiye sınıfı maskesi
                match_mask = (self.token_rhyme_classes.view(1, 1, V) == expected_rhyme.unsqueeze(-1))
                
                # Eşleşen kafiyeli logits sumexp
                rhyme_logits = torch.where(match_mask, curr_logits, torch.tensor(float('-inf'), device=logits.device))
                lse_rhyme = torch.logsumexp(rhyme_logits, dim=-1)
                lse_all = torch.logsumexp(curr_logits, dim=-1)
                
                log_p_rhyme = lse_rhyme - lse_all
                p_violation_rhyme = 1.0 - log_p_rhyme.exp()  # 1.0 - P(rhyming)
                
                p_violation_rhyme = p_violation_rhyme * valid_rhyme_mask.float()
                rhyme_loss = (p_violation_rhyme.sum() / valid_rhyme_count.clamp(min=1)) * eff_lambda_rhyme

        return syllable_loss, rhyme_loss
