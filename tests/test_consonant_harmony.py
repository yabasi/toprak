# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Ünsüz Benzeşmesi Loss Birim Testleri
`model/consonant_harmony.py` dosyasındaki matematiksel ve dilbilimsel mantığı test eder.
"""

import unittest
import torch

from model.consonant_harmony import (
    _ends_with_voiceless,
    _starts_with_mutable_voiced,
    ConsonantHarmonyLoss,
)


class MockTokenizer:
    """Birim testleri için hafif sahte tokenizer sınıfı."""

    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]


class TestConsonantHarmony(unittest.TestCase):

    def test_ends_with_voiceless(self):
        """Sert ünsüz bitiş kontrolü testleri."""
        # Sert ünsüzlerle bitenler (f, s, t, k, ç, ş, h, p)
        self.assertTrue(_ends_with_voiceless("kitap"))
        self.assertTrue(_ends_with_voiceless("yavaş"))
        self.assertTrue(_ends_with_voiceless("git"))
        self.assertTrue(_ends_with_voiceless("bak"))
        self.assertTrue(_ends_with_voiceless("ağaç"))
        self.assertTrue(_ends_with_voiceless("KİTAP"))
        self.assertTrue(_ends_with_voiceless("gitt"))  # Tekrarlayan ünsüzler

        # Sert ünsüzle bitmeyenler (ünlü veya yumuşak ünsüz)
        self.assertFalse(_ends_with_voiceless("araba"))
        self.assertFalse(_ends_with_voiceless("kalem"))
        self.assertFalse(_ends_with_voiceless("yol"))
        self.assertFalse(_ends_with_voiceless("bilgisayar"))
        self.assertFalse(_ends_with_voiceless("123"))  # Sayılar alfabetik değil

    def test_starts_with_mutable_voiced(self):
        """c, d, g ile başlangıç kontrolü testleri."""
        self.assertTrue(_starts_with_mutable_voiced("cı"))
        self.assertTrue(_starts_with_mutable_voiced("dan"))
        self.assertTrue(_starts_with_mutable_voiced("giller"))
        self.assertTrue(_starts_with_mutable_voiced("Cİ"))
        self.assertTrue(_starts_with_mutable_voiced("DAN"))

        # Başlamayanlar
        self.assertFalse(_starts_with_mutable_voiced("lar"))
        self.assertFalse(_starts_with_mutable_voiced("ın"))
        self.assertFalse(_starts_with_mutable_voiced("kitap"))
        self.assertFalse(_starts_with_mutable_voiced("123"))

    def test_loss_initialization_and_vocab(self):
        """Loss başlatma ve kelime haznesi sınıflandırma testi."""
        # Sahte bir kelime haznesi oluştur
        # 0, 1, 2, 3 özel tokenler
        # 4: ▁git (kök, sert ünsüzle biter)
        # 5: di (ek, d ile başlar -> sertleşebilir)
        # 6: ti (ek, t ile başlar -> sertleşmiş, ceza almamalı)
        # 7: ▁da (kök/kelime başı, d ile başlar -> kelime başı olduğu için ihlal sayılmamalı)
        vocab = [
            "<pad>", "<unk>", "<s>", "</s>",
            "▁git", "di", "ti", "▁da"
        ]
        tokenizer = MockTokenizer(vocab)

        loss_fn = ConsonantHarmonyLoss(tokenizer, lambda_weight=0.2, warmup_steps=100)

        # Buffer'ları doğrula
        # ends_voiceless: "▁git"(4) sert ünsüzle biter (t). "di"(5) ve "ti"(6) sert ünsüzle bitmez.
        self.assertEqual(loss_fn.ends_voiceless[4].item(), 1)
        self.assertEqual(loss_fn.ends_voiceless[5].item(), 0)

        # mutable_voiced_suffix_ids: sadece "di"(5) ek olup c, d, g ile başlar.
        # "▁da"(7) kelime başı olduğu için listede olmamalıdır.
        mutable_ids = loss_fn.mutable_voiced_suffix_ids.tolist()
        self.assertIn(5, mutable_ids)
        self.assertNotIn(7, mutable_ids)
        self.assertNotIn(6, mutable_ids)

    def test_warmup_scheduler(self):
        """Warmup adımlarına göre lambda ağırlığı testi."""
        vocab = ["<pad>", "<unk>", "<s>", "</s>", "▁git", "di"]
        tokenizer = MockTokenizer(vocab)
        loss_fn = ConsonantHarmonyLoss(tokenizer, lambda_weight=0.5, warmup_steps=100, start_step=10)

        # 10. adımdan önce veya eşitken 0 olmalı
        self.assertEqual(loss_fn.get_effective_lambda(5), 0.0)
        self.assertEqual(loss_fn.get_effective_lambda(10), 0.0)

        # Warmup süresince lineer artmalı
        self.assertAlmostEqual(loss_fn.get_effective_lambda(60), 0.25)  # (60-10)/100 = 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(loss_fn.get_effective_lambda(110), 0.5)

        # Warmup bittikten sonra sabit kalmalı
        self.assertEqual(loss_fn.get_effective_lambda(200), 0.5)

    def test_loss_computation_and_gradient(self):
        """Loss hesaplama ve gradient akışı testi."""
        vocab = ["<pad>", "<unk>", "<s>", "</s>", "▁git", "di", "ti"]
        tokenizer = MockTokenizer(vocab)
        loss_fn = ConsonantHarmonyLoss(tokenizer, lambda_weight=1.0, warmup_steps=100, start_step=0)

        # Logit'ler: (B, T, V) -> (1, 2, 7)
        # targets: (B, T) -> (1, 2)
        # targets = [["▁git", "di"]] -> [4, 5]
        # Bu geçiş "git-di" ihlalidir, yani loss pozitif olmalıdır.
        logits = torch.randn(1, 2, 7, requires_grad=True)
        targets = torch.tensor([[4, 5]], dtype=torch.long)

        # Warmup aktif
        loss = loss_fn(logits, targets, current_step=100)

        self.assertTrue(loss.item() > 0.0)

        # Gradient akışını doğrula
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0.0))

        # İhlal olmayan geçiş: "▁git-ti" -> [4, 6]
        # "ti"(6) t ile başlar, c, d, g değil. Ceza almamalıdır.
        # Bu durumda P(violation) hesaplanırken logits'lerin durumuna bağlı ceza yine de olacaktır
        # ama targets bu pozisyonda valid_mask = True yaptığı için modelin "di" tahmin etme ihtimalini cezalandırır.
        # logits'leri "di"(5) için aşırı küçük (ör. -100) yaparsak loss sıfıra yaklaşmalıdır.
        logits_safe = torch.zeros(1, 2, 7, requires_grad=True)
        with torch.no_grad():
            # "di"(5) ihtimalini aşırı düşür
            logits_safe[0, 1, 5] = -100.0
            # "ti"(6) ihtimalini yüksek yap
            logits_safe[0, 1, 6] = 100.0

        loss_safe = loss_fn(logits_safe, targets, current_step=100)
        self.assertAlmostEqual(loss_safe.item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
