# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""GRPO testleri: avantaj normalizasyonu, k3-KL, kırpılmış kayıp, maskeler,
checkpoint uyumluluğu ve küçük bir oyuncak RL akıl sağlığı testi."""

import os
import tempfile
import unittest

import torch

from model.config import ModelConfig
from model.transformer import ToprakLM
from training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    clipped_policy_loss,
    completion_logprobs,
    group_advantages,
    grpo_loss,
    k3_kl,
    sample_completions,
)

VOCAB = ["<pad>", "<unk>", "<s>", "</s>", "▁a", "▁b", "▁c", "▁d"]
TARGET = 6  # "▁c"


class MockTokenizer:
    def get_vocab_size(self):
        return len(VOCAB)

    def id_to_token(self, token_id):
        return VOCAB[token_id]


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = ModelConfig(
        vocab_size=len(VOCAB), d_model=32, num_heads=4, num_kv_heads=2,
        num_layers=2, d_ff=64, max_seq_len=64, device="cpu",
    )
    return ToprakLM(config, tokenizer=MockTokenizer()).eval()


class TestAdvantages(unittest.TestCase):

    def test_normalized(self):
        r = torch.tensor([1.0, 0.0, 0.0, 1.0])
        adv = group_advantages(r, eps=0.0)
        self.assertTrue(torch.allclose(adv, torch.tensor([1.0, -1.0, -1.0, 1.0])))
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=6)

    def test_zero_std_gives_zero(self):
        adv = group_advantages(torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 2.0]]))
        self.assertTrue(torch.equal(adv[0], torch.zeros(3)))
        self.assertGreater(adv[1, 2].item(), 0)
        self.assertLess(adv[1, 1].item(), 0)

    def test_single_sample_group(self):
        self.assertEqual(group_advantages(torch.tensor([3.0])).item(), 0.0)


class TestKL(unittest.TestCase):

    def test_k3_zero_when_equal(self):
        logp = torch.log_softmax(torch.randn(3, 5), dim=-1)
        self.assertTrue(torch.allclose(k3_kl(logp, logp), torch.zeros(3, 5)))

    def test_k3_positive_otherwise(self):
        logp = torch.log(torch.tensor([0.2, 0.5, 0.9]))
        ref = torch.log(torch.tensor([0.4, 0.1, 0.9]))
        kl = k3_kl(logp, ref)
        self.assertGreater(kl[0].item(), 0)
        self.assertGreater(kl[1].item(), 0)
        self.assertAlmostEqual(kl[2].item(), 0.0, places=6)
        # elle: ref/pol oranı 2 → 2 - ln2 - 1
        self.assertAlmostEqual(kl[0].item(), 2 - torch.log(torch.tensor(2.0)).item() - 1, places=5)


class TestClippedLoss(unittest.TestCase):

    def test_hand_computation(self):
        old = torch.zeros(2, 3)
        ratios = torch.tensor([[1.0, 1.5, 0.5], [1.0, 1.5, 0.5]])
        logp = torch.log(ratios)
        adv = torch.tensor([2.0, -1.0])
        loss, clipped = clipped_policy_loss(logp, old, adv, clip_eps=0.2)
        # A=+2: min(r*2, clip(r)*2) → 2, 1.2*2=2.4, 0.5*2=1.0
        # A=-1: min(-r, -clip(r)) → -1, -1.5, -0.8
        expected = -torch.tensor([[2.0, 2.4, 1.0], [-1.0, -1.5, -0.8]])
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        self.assertEqual(clipped.tolist(), [[False, True, False], [False, False, True]])

    def test_asymmetric_clip(self):
        logp = torch.log(torch.tensor([[1.5]]))
        loss, _ = clipped_policy_loss(logp, torch.zeros(1, 1), torch.tensor([1.0]),
                                      clip_eps=0.2, clip_eps_high=0.28)
        self.assertAlmostEqual(loss.item(), -1.28, places=5)

    def test_grpo_loss_masks_and_kl(self):
        logp = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        ref = torch.log(torch.tensor([[0.25, 0.25, 0.25]]))
        mask = torch.tensor([[1, 1, 0]])
        adv = torch.tensor([1.0])
        loss, stats = grpo_loss(logp, logp.detach(), ref, adv, mask, beta=0.1)
        kl = 0.5 - torch.log(torch.tensor(0.5)).item() - 1
        self.assertAlmostEqual(loss.item(), -1.0 + 0.1 * kl, places=5)
        self.assertAlmostEqual(stats["kl"], kl, places=5)
        # maskelenmiş token kaybı etkilemez
        logp2 = logp.clone()
        logp2[0, 2] = -10.0
        loss2, _ = grpo_loss(logp2, logp2.detach(), ref, adv, mask, beta=0.1)
        self.assertAlmostEqual(loss.item(), loss2.item(), places=6)


class TestSamplingAndMasks(unittest.TestCase):

    def test_mask_excludes_prompt_and_padding(self):
        model = tiny_model()
        prompt = [2, 4, 5]
        torch.manual_seed(0)
        comps, mask, ids, truncated = sample_completions(
            model, prompt, num_samples=6, max_new_tokens=10, stop_ids=(3, 7), pad_id=0)
        self.assertEqual(comps.shape, mask.shape)
        self.assertLessEqual(comps.size(1), 10)
        for row, m, toks, trunc in zip(comps.tolist(), mask.tolist(), ids, truncated):
            n = sum(m)
            self.assertEqual(m, [1] * n + [0] * (len(m) - n))  # önek-1, sonra 0
            self.assertTrue(all(t == 0 for t in row[n:]))
            if not trunc:
                self.assertIn(row[n - 1], (3, 7))
                self.assertEqual(toks, row[: n - 1])
        full = torch.cat([torch.tensor([prompt] * 6), comps], dim=1)
        logp = completion_logprobs(model, full, len(prompt))
        self.assertEqual(logp.shape, comps.shape)  # prompt konumları dahil değil

    def test_logprobs_match_kv_sampling_distribution(self):
        model = tiny_model()
        prompt = [2, 4]
        full = torch.tensor([[2, 4, 5, 6]])
        logp = completion_logprobs(model, full, 2)
        with torch.no_grad():
            l1 = torch.log_softmax(model(torch.tensor([[2, 4]]))[0][0, -1], -1)[5]
            l2 = torch.log_softmax(model(torch.tensor([[2, 4, 5]]))[0][0, -1], -1)[6]
        self.assertAlmostEqual(logp[0, 0].item(), l1.item(), places=5)
        self.assertAlmostEqual(logp[0, 1].item(), l2.item(), places=5)


def target_prob(model, prompt):
    with torch.no_grad():
        logits = model(torch.tensor([prompt]))[0][0, -1]
    return torch.softmax(logits.float(), -1)[TARGET].item()


class TestToyRL(unittest.TestCase):

    def setUp(self):
        # Küçük tensörlerde çok iş parçacığı yalnız ek yük getirir (CI'da yavaşlar)
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self._threads)

    def test_reward_increases_target_probability(self):
        model = tiny_model(seed=1)
        prompt = [2, 4]
        before = target_prob(model, prompt)
        config = GRPOConfig(
            group_size=8, prompts_per_step=1, steps=25, lr=1e-2, warmup_steps=0,
            min_lr_ratio=1.0, beta_kl=0.02, max_new_tokens=2, temperature=1.0, seed=0,
        )

        def reward_fn(text, item, ids):
            return 1.0 if ids and ids[0] == TARGET else 0.0

        trainer = GRPOTrainer(model, config, reward_fn, stop_ids=(3,),
                              correct_fn=lambda t, i, ids: bool(ids) and ids[0] == TARGET)
        history = trainer.train([{"prompt_ids": prompt}])
        after = target_prob(model, prompt)
        self.assertGreater(after, before + 0.3, f"önce={before:.3f} sonra={after:.3f}")
        first = sum(h["reward_mean"] for h in history[:5]) / 5
        last = sum(h["reward_mean"] for h in history[-5:]) / 5
        self.assertGreater(last, first)
        self.assertIn("accuracy", history[-1])
        self.assertGreaterEqual(history[-1]["kl"], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            from inference.generate import load_model
            path = trainer.save_checkpoint(os.path.join(tmp, "grpo.pt"))
            loaded, cfg = load_model(path, device="cpu")
            self.assertEqual(cfg.vocab_size, len(VOCAB))
            self.assertAlmostEqual(target_prob(loaded, prompt), after, places=5)


if __name__ == "__main__":
    unittest.main()
