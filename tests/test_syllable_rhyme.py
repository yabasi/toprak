# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Hece ve Kafiye Kaybı Birim Testleri
`model/syllable_rhyme.py` dosyasındaki matematiksel ve dilbilimsel mantığı test eder.
"""

import unittest
import torch

from model.syllable_rhyme import (
    _count_syllables,
    _get_rhyme_ending,
    SyllableRhymeLoss,
)


class MockTokenizer:
    """Birim testleri için sahte tokenizer sınıfı."""

    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]


class TestSyllableRhyme(unittest.TestCase):

    def test_count_syllables(self):
        """Hece (ünlü) sayısı hesaplama testleri."""
        self.assertEqual(_count_syllables("araba"), 3)
        self.assertEqual(_count_syllables("yollar"), 2)
        self.assertEqual(_count_syllables("git"), 1)
        self.assertEqual(_count_syllables("▁ağaç"), 2)
        self.assertEqual(_count_syllables("KİTAP"), 2)
        self.assertEqual(_count_syllables("123"), 0)
        self.assertEqual(_count_syllables("<pad>"), 0)

    def test_get_rhyme_ending(self):
        """Kafiye ses sonu tespiti testleri."""
        self.assertEqual(_get_rhyme_ending("yollar"), "ar")
        self.assertEqual(_get_rhyme_ending("bakar"), "ar")
        self.assertEqual(_get_rhyme_ending("git"), "it")
        self.assertEqual(_get_rhyme_ending("o"), "o")
        self.assertEqual(_get_rhyme_ending("▁gitti"), "ti")
        self.assertEqual(_get_rhyme_ending("123"), "")

    def test_loss_initialization(self):
        """Loss başlatma ve kelime haznesi önbellekleme testleri."""
        # 0, 1, 2, 3 özel token'lar
        # 4: ▁git (hece=1, kafiye="it")
        # 5: ▁araba (hece=3, kafiye="ba")
        # 6: yollar (hece=2, kafiye="ar")
        # 7: bakar (hece=2, kafiye="ar") -> 6 ile aynı kafiye sınıfı olmalı
        vocab = [
            "<pad>", "<unk>", "<s>", "</s>",
            "▁git", "▁araba", "yollar", "bakar"
        ]
        tokenizer = MockTokenizer(vocab)
        
        loss_fn = SyllableRhymeLoss(
            tokenizer,
            lambda_syllable=0.2,
            lambda_rhyme=0.3,
            warmup_steps=100
        )

        # Hece sayıları doğrulaması
        self.assertEqual(loss_fn.token_syllables[4].item(), 1)
        self.assertEqual(loss_fn.token_syllables[5].item(), 3)
        self.assertEqual(loss_fn.token_syllables[6].item(), 2)

        # Kafiye sınıfları doğrulaması
        git_class = loss_fn.token_rhyme_classes[4].item()
        araba_class = loss_fn.token_rhyme_classes[5].item()
        yollar_class = loss_fn.token_rhyme_classes[6].item()
        bakar_class = loss_fn.token_rhyme_classes[7].item()

        # Sıfır olmamalılar
        self.assertNotEqual(git_class, 0)
        self.assertNotEqual(araba_class, 0)
        self.assertNotEqual(yollar_class, 0)

        # Benzersiz olmalılar
        self.assertNotEqual(git_class, araba_class)
        self.assertNotEqual(araba_class, yollar_class)

        # "yollar" ve "bakar" aynı kafiye sınıfında olmalıdır ("ar")
        self.assertEqual(yollar_class, bakar_class)

    def test_warmup_scheduler(self):
        """Warmup adımlarına göre lambda ağırlığı testleri."""
        vocab = ["<pad>", "<unk>", "<s>", "</s>", "▁git"]
        tokenizer = MockTokenizer(vocab)
        loss_fn = SyllableRhymeLoss(
            tokenizer,
            lambda_syllable=0.4,
            lambda_rhyme=0.8,
            warmup_steps=100,
            start_step=10
        )

        # 10. adımdan önce 0.0 olmalı
        l_s, l_r = loss_fn.get_effective_lambdas(5)
        self.assertEqual(l_s, 0.0)
        self.assertEqual(l_r, 0.0)

        # Warmup süresince lineer artmalı (Adım 60, yarı yol: factor = 0.5)
        l_s, l_r = loss_fn.get_effective_lambdas(60)
        self.assertAlmostEqual(l_s, 0.2)
        self.assertAlmostEqual(l_r, 0.4)

        # Warmup bittikten sonra sabit kalmalı
        l_s, l_r = loss_fn.get_effective_lambdas(200)
        self.assertEqual(l_s, 0.4)
        self.assertEqual(l_r, 0.8)

    def test_loss_computation_and_gradient(self):
        """Loss hesaplama ve gradyan akışı testi."""
        # 17: satır sonu token'ı (<0x0A> olarak simüle ediyoruz)
        vocab = ["<pad>", "<unk>", "<s>", "</s>"] + ["tok"] * 13 + ["<0x0A>", "yollar", "bakar"]
        # token ID'leri:
        # 17: <0x0A> (satır sonu)
        # 18: yollar (hece=2, kafiye="ar")
        # 19: bakar (hece=2, kafiye="ar")
        tokenizer = MockTokenizer(vocab)
        
        loss_fn = SyllableRhymeLoss(
            tokenizer,
            lambda_syllable=1.0,
            lambda_rhyme=1.0,
            warmup_steps=100,
            start_step=0
        )

        # Logits: (B, T, V) -> (1, 6, 20)
        # Sequence: yollar(18) -> \n(17) -> bakar(19) -> \n(17) -> bakar(19) -> \n(17)
        # İlk satırda tek bir 'yollar' var. Hece sayısı = 2.
        # Dolayısıyla hedef hece ölçüsü (S_target) 2 olmalıdır.
        targets = torch.tensor([[18, 17, 19, 17, 19, 17]], dtype=torch.long)
        logits = torch.randn(1, 6, 20, requires_grad=True)

        s_loss, r_loss = loss_fn(logits, targets, current_step=100)

        # Her iki loss da pozitif olmalı (ham rassal tahminler kuralları ihlal edecektir)
        self.assertTrue(s_loss.item() >= 0.0)
        self.assertTrue(r_loss.item() >= 0.0)

        # Gradyan akışını test et
        total_loss = s_loss + r_loss
        total_loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0.0))


if __name__ == "__main__":
    unittest.main()
