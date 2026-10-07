# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Tercih hizalaması (DPO / ORPO) testleri."""

import math
import os
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.chat_template import IGNORE_INDEX
from model.config import ModelConfig
from model.transformer import ToprakLM
from training.dpo import (
    PreferenceDataset,
    dpo_loss,
    evaluate_preferences,
    make_reference_model,
    orpo_loss,
    preference_collate,
    sequence_logprobs,
    train_preference,
)
from training.sft import apply_lora

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = 48


class MockTokenizer:
    """Karakter tabanlı sahte tokenizer."""

    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3
    specials = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]

    def get_vocab_size(self):
        return V

    def id_to_token(self, token_id):
        return self.specials[token_id] if token_id < 5 else f"▁t{token_id}"

    def token_to_id(self, token):
        return self.specials.index(token) if token in self.specials else self.unk_token_id

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [5 + ord(c) % (V - 5) for c in text]
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids if i > 4)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = ModelConfig(vocab_size=V, d_model=32, num_heads=4, num_kv_heads=2, num_layers=2,
                         d_ff=64, max_seq_len=64, device="cpu")
    return ToprakLM(config, tokenizer=MockTokenizer())


PAIRS = [
    {"prompt": "renk?", "chosen": "mavi", "rejected": "xqz"},
    {"prompt": [{"role": "user", "content": "hayvan?"}], "chosen": "kedi", "rejected": "zzzz"},
    {"prompt": "sayı?", "chosen": "bir", "rejected": "qqq"},
    {"prompt": "eksik"},  # geçersiz → atlanır
]


class TestDPOLoss(unittest.TestCase):

    def test_hand_computed_value(self):
        pc, pr = torch.tensor([-1.0]), torch.tensor([-2.0])
        rc, rr = torch.tensor([-1.5]), torch.tensor([-1.5])
        loss, margins, acc = dpo_loss(pc, pr, rc, rr, beta=0.5)
        # logits = 0.5 * ((−1 + 1.5) − (−2 + 1.5)) = 0.5
        self.assertAlmostEqual(loss.item(), math.log(1 + math.exp(-0.5)), places=6)
        self.assertAlmostEqual(margins.item(), 0.5, places=6)
        self.assertEqual(acc.item(), 1.0)

        smoothed, _, _ = dpo_loss(pc, pr, rc, rr, beta=0.5, label_smoothing=0.1)
        expected = 0.9 * math.log(1 + math.exp(-0.5)) + 0.1 * math.log(1 + math.exp(0.5))
        self.assertAlmostEqual(smoothed.item(), expected, places=6)

    def test_batch_mean_and_accuracy(self):
        pc = torch.tensor([-1.0, -3.0])
        pr = torch.tensor([-2.0, -1.0])
        ref = torch.zeros(2)
        loss, margins, acc = dpo_loss(pc, pr, ref, ref, beta=1.0)
        self.assertEqual(margins.tolist(), [1.0, -2.0])
        expected = (math.log(1 + math.exp(-1.0)) + math.log(1 + math.exp(2.0))) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertAlmostEqual(acc.item(), 0.5)

    def test_log2_when_policy_equals_reference(self):
        pc, pr = torch.tensor([-3.0, -7.0]), torch.tensor([-4.0, -2.0])
        loss, margins, acc = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertTrue(torch.all(margins == 0))
        self.assertEqual(acc.item(), 0.0)

    def test_orpo_hand_computed(self):
        c, r = torch.tensor([math.log(0.5)]), torch.tensor([math.log(0.25)])
        loss, ratio, acc = orpo_loss(c, r, lam=0.1)
        # log odds: log(0.5/0.5)=0, log(0.25/0.75)=-log 3 → oran log 3
        self.assertAlmostEqual(ratio.item(), math.log(3), places=5)
        expected = -math.log(0.5) + 0.1 * math.log(1 + math.exp(-math.log(3)))
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(acc.item(), 1.0)
        # Çok küçük olasılıklarda sayısal kararlılık
        loss, _, _ = orpo_loss(torch.tensor([-1e-9, -50.0]), torch.tensor([-60.0, -1e-9]))
        self.assertTrue(torch.isfinite(loss))


class TestLogprobsAndData(unittest.TestCase):

    def test_sequence_logprobs_matches_manual(self):
        model = tiny_model().eval()
        ids = torch.randint(5, V, (2, 7))
        labels = torch.full((2, 7), IGNORE_INDEX)
        labels[0, 3:6] = ids[0, 4:7]
        labels[1, 1:3] = ids[1, 2:4]
        out = sequence_logprobs(model, ids, labels)
        logp = F.log_softmax(model(ids)[0], dim=-1)
        manual0 = sum(logp[0, t, labels[0, t]] for t in range(3, 6))
        manual1 = sum(logp[1, t, labels[1, t]] for t in range(1, 3))
        self.assertAlmostEqual(out[0].item(), manual0.item(), places=4)
        self.assertAlmostEqual(out[1].item(), manual1.item(), places=4)
        avg = sequence_logprobs(model, ids, labels, average=True)
        self.assertAlmostEqual(avg[1].item(), manual1.item() / 2, places=4)

    def test_dataset_and_collate(self):
        ds = PreferenceDataset(PAIRS, MockTokenizer(), max_len=64)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.skipped, 1)
        batch = preference_collate([ds[0], ds[1]])
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["input_ids"][0].tolist()[: len(ds[0]["chosen_input_ids"])],
                         ds[0]["chosen_input_ids"])
        self.assertEqual(batch["input_ids"][2].tolist()[: len(ds[0]["rejected_input_ids"])],
                         ds[0]["rejected_input_ids"])
        # Prompt kısmı her iki dizide aynı ve maskeli
        chosen_trainable = [l for l in ds[0]["chosen_labels"] if l != IGNORE_INDEX]
        self.assertEqual(chosen_trainable, MockTokenizer().encode("mavi") + [4])

    def test_example_file_loads_with_real_tokenizer(self):
        from model.tokenizer import ToprakTokenizer

        tok = ToprakTokenizer(os.path.join(ROOT, "toprak_tokenizer.model"))
        ds = PreferenceDataset(os.path.join(ROOT, "alignment", "examples", "dpo_sample.jsonl"),
                               tok, max_len=512)
        self.assertEqual(len(ds), 6)


class TestPreferenceTraining(unittest.TestCase):

    def _margin(self, model, ds, ref_model=None, method="dpo"):
        loader = DataLoader(ds, batch_size=len(ds), collate_fn=preference_collate)
        return evaluate_preferences(model, loader, "cpu", ref_model=ref_model, method=method, beta=0.5)

    def test_dpo_steps_increase_margin(self):
        model = tiny_model()
        ds = PreferenceDataset(PAIRS, MockTokenizer(), max_len=64)
        ref = make_reference_model(model)
        before = self._margin(model, ds, ref)
        self.assertAlmostEqual(before["margin"], 0.0, places=5)
        self.assertAlmostEqual(before["loss"], math.log(2), places=5)
        hist = train_preference(model, ds, ref_model=ref, method="dpo", beta=0.5, max_steps=8,
                                batch_size=3, grad_accum_steps=1, learning_rate=5e-3,
                                warmup_steps=1, shuffle=False, log_fn=None)
        after = self._margin(model, ds, ref)
        self.assertGreater(after["margin"], before["margin"] + 0.1)
        self.assertLess(after["loss"], math.log(2))
        self.assertEqual(after["accuracy"], 1.0)
        self.assertEqual(len(hist["loss"]), 8)
        # Referans model değişmemeli
        self.assertFalse(any(p.requires_grad for p in ref.parameters()))

    def test_dpo_with_lora_uses_base_as_reference(self):
        model = tiny_model()
        ds = PreferenceDataset(PAIRS, MockTokenizer(), max_len=64)
        apply_lora(model, r=4, alpha=8)
        before = self._margin(model, ds)
        self.assertAlmostEqual(before["margin"], 0.0, places=5)
        train_preference(model, ds, method="dpo", beta=0.5, max_steps=8, batch_size=3,
                         grad_accum_steps=1, learning_rate=2e-2, warmup_steps=1,
                         merge_lora_at_end=False, log_fn=None)
        after = self._margin(model, ds)
        self.assertGreater(after["margin"], 0.1)

    def test_orpo_steps_increase_log_odds_ratio(self):
        model = tiny_model()
        ds = PreferenceDataset(PAIRS, MockTokenizer(), max_len=64)
        before = self._margin(model, ds, method="orpo")
        train_preference(model, ds, method="orpo", max_steps=10, batch_size=3, grad_accum_steps=1,
                         learning_rate=5e-3, warmup_steps=1, log_fn=None)
        after = self._margin(model, ds, method="orpo")
        self.assertGreater(after["margin"], before["margin"])
        self.assertLess(after["loss"], before["loss"])


if __name__ == "__main__":
    unittest.main()
