# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Morfolojik Başlık (Morphological Head) Birim Testleri
`ToprakLM` mimarisindeki `self.morph_head` çoklu görev başlığını test eder.
"""

import unittest
import torch

from model.config import ModelConfig
from model.transformer import ToprakLM


class MockTokenizer:
    """Birim testleri için hafif sahte tokenizer sınıfı."""

    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]


class TestMorphHead(unittest.TestCase):

    def setUp(self):
        # 0, 1, 2, 3 özel tokenler
        # 4: ▁araba (Kök -> Class 0)
        # 5: lar (Ek -> Class 1)
        # 6: 123 (Sayı -> Class 2)
        # 7: ▁. (Noktalama -> Class 2)
        self.vocab = [
            "<pad>", "<unk>", "<s>", "</s>",
            "▁araba", "lar", "123", "▁."
        ]
        self.tokenizer = MockTokenizer(self.vocab)

        self.config = ModelConfig(
            vocab_size=8,
            d_model=16,
            num_heads=2,
            num_kv_heads=1,
            num_layers=2,
            d_ff=32,
            max_seq_len=8,
        )

    def test_vocab_classification(self):
        """Token sınıflandırma mantığının doğruluğunu test et."""
        model = ToprakLM(self.config, tokenizer=self.tokenizer)

        # Sınıfları doğrula
        # 0 = Kök, 1 = Ek, 2 = Özel/Noktalama
        classes = model.token_morph_classes.tolist()

        self.assertEqual(classes[0], 2)  # pad
        self.assertEqual(classes[1], 2)  # unk
        self.assertEqual(classes[2], 2)  # bos
        self.assertEqual(classes[3], 2)  # eos
        self.assertEqual(classes[4], 0)  # ▁araba (kök)
        self.assertEqual(classes[5], 1)  # lar (ek)
        self.assertEqual(classes[6], 2)  # 123 (sayı)
        self.assertEqual(classes[7], 2)  # ▁. (noktalama)

    def test_forward_pass_and_loss(self):
        """Morfolojik başlık ile forward ve loss hesaplamasını test et."""
        model = ToprakLM(self.config, tokenizer=self.tokenizer)
        model.use_morph_head = True
        model.morph_lambda = 0.5

        # Girdi ve hedefler
        input_ids = torch.tensor([[4, 5, 7]], dtype=torch.long)  # (B, T) = (1, 3)
        targets = torch.tensor([[5, 7, 3]], dtype=torch.long)    # (B, T) = (1, 3)

        logits, loss, _ = model(input_ids, targets=targets)

        # Çıktı şekillerini doğrula
        self.assertEqual(logits.shape, (1, 3, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0.0)

        # Son morfolojik kaybın kaydedildiğini doğrula
        self.assertTrue(model._last_morph_loss > 0.0)

    def test_gradient_flow(self):
        """Gradyanların morph_head ağırlıklarına ulaştığını doğrula."""
        model = ToprakLM(self.config, tokenizer=self.tokenizer)
        model.use_morph_head = True
        model.morph_lambda = 1.0

        input_ids = torch.tensor([[4, 5]], dtype=torch.long)
        targets = torch.tensor([[5, 3]], dtype=torch.long)

        # Forward + Backward
        _, loss, _ = model(input_ids, targets=targets)
        loss.backward()

        # Gradyanları kontrol et
        self.assertIsNotNone(model.morph_head.weight.grad)
        self.assertTrue(torch.any(model.morph_head.weight.grad != 0.0))

    def test_aux_only_loss_for_custom_ce(self):
        """Özel CE kullanılırken morph-head kaybı ve gradyanı korunmalı."""
        model = ToprakLM(self.config, tokenizer=self.tokenizer)
        model.use_morph_head = True
        model.morph_lambda = 0.5

        input_ids = torch.tensor([[4, 5, 7]], dtype=torch.long)
        targets = torch.tensor([[5, 7, 3]], dtype=torch.long)

        logits, aux_loss, _ = model(
            input_ids,
            targets=targets,
            compute_lm_loss=False,
        )
        custom_ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
        )
        (custom_ce + aux_loss).backward()

        self.assertGreater(aux_loss.item(), 0.0)
        self.assertIsNotNone(model.morph_head.weight.grad)
        self.assertTrue(torch.any(model.morph_head.weight.grad != 0.0))

    def test_checkpoint_compatibility(self):
        """Eski checkpointlerin (morph_head içermeyen) yüklenme toleransını test et."""
        model = ToprakLM(self.config, tokenizer=self.tokenizer)

        # Eski bir state_dict simüle et (morph_head ağırlıkları yok)
        old_state_dict = model.state_dict()
        del old_state_dict["morph_head.weight"]

        # strict=False ile yüklemeyi dene (hata vermemeli)
        try:
            model.load_state_dict(old_state_dict, strict=False)
            loaded_successfully = True
        except Exception:
            loaded_successfully = False

        self.assertTrue(loaded_successfully)


if __name__ == "__main__":
    unittest.main()
