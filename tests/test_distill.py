# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Bilgi damıtma testleri.
Kayıp matematiği (KL=0, T² ölçeği, alpha=0 → CE, pad maskesi), top-k
seyrekleştirme, sözlük uyuşmazlığı ve küçük öğretmen→öğrenci eğitiminde
KL düşüşü + checkpoint biçimi.
"""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import ToprakLM
from training.distill import DistillTrainer, distillation_loss, parse_args, sparsify_teacher_logits

V = 48


class MockTokenizer:
    """Testler için sahte tokenizer."""

    def get_vocab_size(self):
        return V

    def id_to_token(self, token_id):
        return f"▁t{token_id}"


def tiny(d_model, layers, seed, scale=1.0):
    torch.manual_seed(seed)
    config = ModelConfig(
        vocab_size=V, d_model=d_model, num_heads=4, num_kv_heads=2, num_layers=layers,
        d_ff=2 * d_model, max_seq_len=16, device="cpu",
    )
    model = ToprakLM(config, tokenizer=MockTokenizer())
    if scale != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
    return model


class TestDistillationLoss(unittest.TestCase):

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.s = torch.randn(2, 5, V, generator=g) * 3
        self.t = torch.randn(2, 5, V, generator=g) * 3
        self.labels = torch.randint(1, V, (2, 5), generator=g)
        self.labels[1, 3:] = 0  # pad

    def test_kl_zero_when_equal(self):
        loss, parts = distillation_loss(self.t.clone(), self.t, self.labels, temperature=2.0, alpha=1.0)
        self.assertAlmostEqual(parts["kl"], 0.0, places=6)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_temperature_squared_scaling(self):
        for T in (1.0, 2.0, 4.0):
            loss, parts = distillation_loss(self.s, self.t, self.labels, temperature=T, alpha=1.0)
            mask = self.labels != 0
            ref = F.kl_div(
                F.log_softmax(self.s / T, -1)[mask], F.log_softmax(self.t / T, -1)[mask],
                log_target=True, reduction="batchmean",
            )
            self.assertAlmostEqual(parts["kl"], ref.item(), places=5)
            self.assertAlmostEqual(loss.item(), T * T * ref.item(), places=4)

    def test_alpha_zero_equals_ce(self):
        loss, _ = distillation_loss(self.s, self.t, self.labels, temperature=3.0, alpha=0.0)
        ce = F.cross_entropy(self.s.reshape(-1, V), self.labels.reshape(-1), ignore_index=0)
        self.assertAlmostEqual(loss.item(), ce.item(), places=5)

    def test_alpha_mix(self):
        T, a = 2.0, 0.3
        loss, parts = distillation_loss(self.s, self.t, self.labels, temperature=T, alpha=a)
        self.assertAlmostEqual(loss.item(), a * T * T * parts["kl"] + (1 - a) * parts["ce"], places=4)

    def test_pad_positions_ignored(self):
        t2 = self.t.clone()
        t2[1, 3:] = torch.randn(2, V) * 10  # yalnız pad pozisyonları değişir
        a, _ = distillation_loss(self.s, self.t, self.labels, alpha=0.5)
        b, _ = distillation_loss(self.s, t2, self.labels, alpha=0.5)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_vocab_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            distillation_loss(self.s, torch.randn(2, 5, V + 1), self.labels)

    def test_topk_sparsification(self):
        values, indices = sparsify_teacher_logits(self.t, 8)
        self.assertEqual(values.shape, (2, 5, 8))
        self.assertTrue(torch.equal(values, torch.gather(self.t, -1, indices)))
        self.assertTrue(torch.all(values[..., :1] >= values))
        # k = V → tam KL ile aynı
        full, pf = distillation_loss(self.s, self.t, self.labels, alpha=1.0)
        same, ps = distillation_loss(self.s, self.t, self.labels, alpha=1.0, top_k=V)
        self.assertAlmostEqual(pf["kl"], ps["kl"], places=5)
        # top-k ile verilen (values, indices) çifti, top_k argümanıyla aynı sonuç
        a, pa = distillation_loss(self.s, self.t, self.labels, alpha=1.0, top_k=8)
        b, pb = distillation_loss(self.s, (values, indices), self.labels, alpha=1.0)
        self.assertAlmostEqual(pa["kl"], pb["kl"], places=6)
        self.assertGreaterEqual(pa["kl"], 0.0)
        # Öğretmen kütlesi zaten top-k'da ise seyrek KL ≈ tam KL
        peaked = torch.full_like(self.t, -1e4)
        peaked.scatter_(-1, indices, values)
        _, p_full = distillation_loss(self.s, peaked, self.labels, alpha=1.0)
        _, p_sparse = distillation_loss(self.s, peaked, self.labels, alpha=1.0, top_k=8)
        self.assertAlmostEqual(p_full["kl"], p_sparse["kl"], places=4)
        # Öğrenci = öğretmen (k dışı kütle ~0) → seyrek KL = 0
        _, p0 = distillation_loss(peaked.clone(), peaked, self.labels, alpha=1.0, top_k=8)
        self.assertAlmostEqual(p0["kl"], 0.0, places=5)


class TestDistillTrainer(unittest.TestCase):

    def setUp(self):
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self._threads)

    def _loader(self, n_batches=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        batches = []
        for _ in range(n_batches):
            x = torch.randint(4, V, (8, 17), generator=g)
            batches.append({"input_ids": x[:, :-1].contiguous(), "labels": x[:, 1:].contiguous()})
        return batches

    def _kl(self, student, teacher, batches, T=2.0):
        with torch.no_grad():
            vals = [
                distillation_loss(student(b["input_ids"])[0], teacher(b["input_ids"])[0],
                                  b["labels"], temperature=T, alpha=1.0)[1]["kl"]
                for b in batches
            ]
        return sum(vals) / len(vals)

    def test_steps_reduce_kl_and_checkpoint_format(self):
        # Öğretmen: büyütülmüş embedding → keskin, girdiye bağlı dağılımlar
        teacher = tiny(64, 2, seed=1)
        with torch.no_grad():
            teacher.tok_emb.weight.mul_(20.0)
        student = tiny(32, 1, seed=2)
        loader = self._loader()
        before = self._kl(student.eval(), teacher.eval(), loader)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = DistillTrainer(
                student, teacher, loader, lr=1e-2, max_steps=40, warmup_steps=2,
                temperature=2.0, alpha=1.0, top_k=16, log_every=0,
                checkpoint_dir=tmp, save_every=20, device="cpu",
            )
            history = trainer.train()
            self.assertEqual(len(history), 40)
            after = self._kl(student.eval(), teacher.eval(), loader)
            self.assertLess(after, before * 0.5, f"KL {before:.4f} → {after:.4f}")
            # Görülmemiş veride de düşmeli
            held_out = self._kl(student.eval(), teacher.eval(), self._loader(seed=5))
            self.assertLess(held_out, before * 0.5)

            # Öğretmen değişmemeli
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))

            files = sorted(os.listdir(tmp))
            self.assertIn("toprak_distill_last.pt", files)
            self.assertIn("toprak_distill_step_20.pt", files)
            ckpt = torch.load(os.path.join(tmp, "toprak_distill_last.pt"), weights_only=False)
            self.assertEqual(ckpt["global_step"], 40)
            self.assertEqual(ckpt["config"], student.config.architecture_dict())
            self.assertEqual(ckpt["distillation"]["top_k"], 16)
            restored = ToprakLM(ModelConfig(**ckpt["config"], device="cpu"), tokenizer=MockTokenizer())
            restored.load_state_dict(ckpt["model_state_dict"])
            ids = loader[0]["input_ids"]
            with torch.no_grad():
                self.assertTrue(torch.allclose(restored.eval()(ids)[0], student(ids)[0]))

    def test_mixed_alpha_trains(self):
        teacher = tiny(64, 2, seed=1, scale=3.0)
        student = tiny(32, 1, seed=3)
        trainer = DistillTrainer(student, teacher, self._loader(2), lr=3e-3, max_steps=5,
                                 warmup_steps=1, alpha=0.5, grad_accum_steps=2, log_every=0)
        history = trainer.train()
        self.assertTrue(all(h["ce"] > 0 and h["kl"] >= 0 for h in history))
        metrics = trainer.evaluate(self._loader(1, seed=9))
        self.assertIn("kl", metrics)

    def test_vocab_mismatch_raises(self):
        teacher = tiny(32, 1, seed=1)
        cfg = ModelConfig(vocab_size=V + 2, d_model=32, num_heads=4, num_kv_heads=2,
                          num_layers=1, d_ff=64, max_seq_len=16, device="cpu")
        student = ToprakLM(cfg, tokenizer=_BigMock())
        with self.assertRaises(ValueError):
            DistillTrainer(student, teacher, self._loader(1))

    def test_cli_args(self):
        args = parse_args(["--teacher", "t.pt", "--student-size", "small",
                           "--data-dir", "data_cache/bin", "--bin-mode", "--top-k", "64"])
        self.assertTrue(args.bin_mode)
        self.assertEqual(args.top_k, 64)
        self.assertEqual(args.student_size, "small")


class _BigMock(MockTokenizer):
    def get_vocab_size(self):
        return V + 2


if __name__ == "__main__":
    unittest.main()
