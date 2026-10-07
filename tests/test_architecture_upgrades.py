# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Mimari yükseltme testleri: morph-head maske düzeltmesi, cache'li çok-token
attention maskesi, RoPE ölçekleme (linear/NTK/YaRN), Çoklu Token Tahmini,
morfoloji yönlendirmeli MoE, spekülatif çözümleme ve checkpoint config'i.
"""

import math
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.moe import MorphRoutedMoE, routing_by_morph_class
from model.rope import precompute_freqs_cis, scaled_inv_freqs, yarn_mscale
from model.transformer import ToprakLM
from inference.speculative import (
    greedy_generate,
    ngram_draft,
    speculative_generate,
    truncate_kv_cache,
)

VOCAB = [
    "<pad>", "<unk>", "<s>", "</s>",
    "▁kitap", "lar", "da", "▁ev", "ler", "de", "▁.", "123",
    "▁okul", "un", "▁göz", "lük",
]


class MockTokenizer:
    def __init__(self, vocab=VOCAB):
        self.vocab = list(vocab)

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]


def tiny_config(**overrides):
    base = dict(
        vocab_size=len(VOCAB), d_model=32, num_heads=4, num_kv_heads=2,
        num_layers=2, d_ff=64, max_seq_len=32, device="cpu",
    )
    base.update(overrides)
    return ModelConfig(**base)


def tiny_model(seed=0, dtype=torch.float32, **overrides):
    torch.manual_seed(seed)
    model = ToprakLM(tiny_config(**overrides), tokenizer=MockTokenizer())
    return model.to(dtype).eval()


class TestMorphHeadMaskFix(unittest.TestCase):

    def test_root_targets_contribute_and_pads_are_ignored(self):
        model = tiny_model()
        model.use_morph_head = True
        model.morph_lambda = 1.0
        inputs = torch.tensor([[2, 4, 5, 6, 7, 0, 0]])
        targets = torch.tensor([[4, 5, 6, 7, 8, 0, 0]])  # 4 ve 7 kök
        _, loss, _ = model(inputs, targets=targets, compute_lm_loss=False)

        _, _, _, hidden = model(inputs, return_hidden=True)
        morph_logits = model.morph_head(hidden)[0, :5]
        classes = model.token_morph_classes[targets[0, :5]]
        self.assertIn(0, classes.tolist())  # kök sınıfı hedefler arasında
        expected = F.cross_entropy(morph_logits, classes)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


class TestCachedAttentionMask(unittest.TestCase):

    def test_chunked_prefill_matches_full_forward(self):
        model = tiny_model(dtype=torch.float64)
        ids = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]])
        full_logits, _, _ = model(ids)
        logits_a, _, past = model(ids[:, :4])
        logits_b, _, past = model(ids[:, 4:8], past_kvs=past)
        logits_c, _, _ = model(ids[:, 8:], past_kvs=past)
        chunked = torch.cat([logits_a, logits_b, logits_c], dim=1)
        self.assertTrue(torch.allclose(full_logits, chunked, atol=1e-9))

    def test_context_overflow_raises_clear_error(self):
        model = tiny_model(max_seq_len=4)  # RoPE tablosu 8 pozisyon
        with self.assertRaises(ValueError):
            model(torch.ones(1, 9, dtype=torch.long))


class TestDtypeConversion(unittest.TestCase):

    def test_dtype_cast_keeps_rope_table_complex(self):
        model = tiny_model()
        reference = model.freqs_cis.clone()
        model.to(torch.float64)
        self.assertTrue(model.freqs_cis.is_complex())
        self.assertTrue(torch.equal(model.freqs_cis, reference))
        model.half()
        self.assertTrue(model.freqs_cis.is_complex())
        self.assertEqual(model.lm_head.weight.dtype, torch.float16)


class TestRopeScaling(unittest.TestCase):

    def test_no_scaling_matches_reference(self):
        dim, theta = 16, 10000.0
        ref = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        inv, mscale = scaled_inv_freqs(dim, theta, None)
        self.assertTrue(torch.allclose(inv, ref))
        self.assertEqual(mscale, 1.0)
        inv, mscale = scaled_inv_freqs(dim, theta, {"type": "yarn", "factor": 1.0})
        self.assertTrue(torch.allclose(inv, ref))

    def test_linear_and_ntk(self):
        dim, theta = 16, 10000.0
        base, _ = scaled_inv_freqs(dim, theta, None)
        linear, _ = scaled_inv_freqs(dim, theta, {"type": "linear", "factor": 4.0})
        self.assertTrue(torch.allclose(linear, base / 4))
        ntk, _ = scaled_inv_freqs(dim, theta, {"type": "ntk", "factor": 4.0})
        self.assertAlmostEqual(ntk[0].item(), base[0].item())  # en yüksek frekans korunur
        self.assertLess(ntk[-1].item(), base[-1].item())

    def test_yarn_preserves_high_and_interpolates_low_frequencies(self):
        dim, theta, factor = 64, 10000.0, 8.0
        base, _ = scaled_inv_freqs(dim, theta, None)
        yarn, mscale = scaled_inv_freqs(
            dim, theta, {"type": "yarn", "factor": factor, "original_max_seq_len": 512}
        )
        self.assertAlmostEqual(yarn[0].item(), base[0].item(), places=6)
        self.assertAlmostEqual(yarn[-1].item(), base[-1].item() / factor, places=9)
        self.assertTrue(torch.all(yarn <= base + 1e-12))
        self.assertAlmostEqual(mscale, 0.1 * math.log(factor) + 1.0)
        self.assertEqual(yarn_mscale(1.0), 1.0)

    def test_yarn_freqs_cis_magnitude_is_mscale(self):
        scaling = {"type": "yarn", "factor": 4.0, "original_max_seq_len": 16}
        freqs = precompute_freqs_cis(8, 32, rope_scaling=scaling)
        self.assertTrue(torch.allclose(freqs.abs(), torch.full((32, 4), yarn_mscale(4.0))))

    def test_unknown_scaling_type_raises(self):
        with self.assertRaises(ValueError):
            scaled_inv_freqs(16, 10000.0, {"type": "foo", "factor": 2.0})

    def test_model_with_yarn_runs_beyond_original_context(self):
        model = tiny_model(
            max_seq_len=64,
            rope_scaling={"type": "yarn", "factor": 4.0, "original_max_seq_len": 16},
        )
        logits, _, _ = model(torch.randint(4, len(VOCAB), (1, 64)))
        self.assertEqual(logits.shape, (1, 64, len(VOCAB)))
        self.assertTrue(torch.isfinite(logits).all())


class TestMultiTokenPrediction(unittest.TestCase):

    def test_disabled_by_default(self):
        model = tiny_model()
        self.assertEqual(len(model.mtp_heads), 0)
        self.assertEqual(model.mtp_logits(torch.zeros(1, 1, 32)), [])

    def test_loss_includes_mtp_terms_and_heads_get_gradients(self):
        model = tiny_model(num_mtp_heads=2, mtp_lambda=0.5).train()
        ids = torch.randint(4, len(VOCAB), (2, 12))
        targets = torch.randint(4, len(VOCAB), (2, 12))
        logits, loss, _ = model(ids, targets=targets)
        lm_loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1), ignore_index=0)
        self.assertGreater(model._last_mtp_loss, 0)
        self.assertAlmostEqual(loss.item(), lm_loss.item() + 0.5 * model._last_mtp_loss, places=5)
        loss.backward()
        for head in model.mtp_heads:
            self.assertIsNotNone(head.proj_out.weight.grad)
            self.assertGreater(head.proj_out.weight.grad.abs().sum().item(), 0)

    def test_mtp_head_k_targets_are_shifted_by_k(self):
        model = tiny_model(num_mtp_heads=1, mtp_lambda=1.0)
        ids = torch.randint(4, len(VOCAB), (1, 10))
        targets = torch.randint(4, len(VOCAB), (1, 10))
        logits, loss, _, hidden = model(ids, targets=targets, return_hidden=True)
        head_logits = model.mtp_logits(hidden)[0][:, :-1]
        expected = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1)) + \
            F.cross_entropy(head_logits.reshape(-1, len(VOCAB)), targets[:, 1:].reshape(-1))
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


class TestSpeculativeDecoding(unittest.TestCase):

    def test_ngram_draft(self):
        tokens = [5, 6, 7, 8, 9, 5, 6]
        self.assertEqual(ngram_draft(tokens, 3), [7, 8, 9])
        self.assertEqual(ngram_draft([1, 2, 3], 3), [])

    def test_truncate_kv_cache(self):
        past = [(torch.zeros(1, 2, 10, 4), torch.ones(1, 2, 10, 4))]
        cut = truncate_kv_cache(past, 6)
        self.assertEqual(cut[0][0].shape[2], 6)
        self.assertEqual(cut[0][1].shape[2], 6)

    def _assert_matches_greedy(self, model, source, num_draft=None):
        prompt = [2, 4, 5, 6, 7, 8, 9, 4, 5]
        ref = greedy_generate(model, prompt, 20)
        out, stats = speculative_generate(model, prompt, 20, num_draft=num_draft, draft_source=source)
        self.assertEqual(out, ref)
        self.assertEqual(stats.generated_tokens, len(ref))
        self.assertLessEqual(stats.forward_passes, len(ref) + 1)
        return stats

    def test_mtp_speculative_equals_greedy(self):
        model = tiny_model(seed=3, dtype=torch.float64, num_mtp_heads=3)
        self._assert_matches_greedy(model, "mtp")

    def test_ngram_speculative_equals_greedy(self):
        model = tiny_model(seed=4, dtype=torch.float64)
        self._assert_matches_greedy(model, "ngram", num_draft=4)

    def test_perfect_drafts_cut_forward_passes(self):
        # MTP başlıkları ana başlığın tahminini birebir kopyalarsa (ör. sabit
        # çıktı üreten model) her turda tüm taslaklar kabul edilmelidir.
        model = tiny_model(seed=5, dtype=torch.float64, num_mtp_heads=3)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[7] = 1.0  # her pozisyonda argmax = 7
            for block in model.blocks:
                block.attn.out_proj.weight.zero_()
                block.ffn.down_proj.weight.zero_()
        stats = self._assert_matches_greedy(model, "mtp")
        self.assertEqual(stats.acceptance_rate, 1.0)
        self.assertLessEqual(stats.forward_passes, 1 + math.ceil(20 / 4))

    def test_stop_ids_respected(self):
        model = tiny_model(seed=6, dtype=torch.float64, num_mtp_heads=2)
        prompt = [2, 4, 5]
        ref = greedy_generate(model, prompt, 15)
        stop = ref[3]
        out, _ = speculative_generate(model, prompt, 15, stop_ids=(stop,))
        self.assertEqual(out, ref[:ref.index(stop)])

    def test_mtp_source_requires_heads(self):
        with self.assertRaises(ValueError):
            speculative_generate(tiny_model(), [2, 4], 5, draft_source="mtp")


class TestMorphRoutedMoE(unittest.TestCase):

    def test_dense_by_default(self):
        model = tiny_model()
        self.assertFalse(any(b.use_moe for b in model.blocks))

    def test_moe_forward_backward_and_aux_loss(self):
        model = tiny_model(num_experts=4, experts_top_k=2).train()
        self.assertTrue(all(isinstance(b.ffn, MorphRoutedMoE) for b in model.blocks))
        ids = torch.randint(4, len(VOCAB), (2, 10))
        logits, loss, _ = model(ids, targets=ids)
        self.assertEqual(logits.shape, (2, 10, len(VOCAB)))
        self.assertGreater(model._last_moe_aux_loss, 0)
        loss.backward()
        moe = model.blocks[0].ffn
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertIsNotNone(moe.morph_route_emb.weight.grad)
        self.assertTrue(any(e.up_proj.weight.grad is not None for e in moe.experts))

    def test_balanced_routing_aux_loss_is_one(self):
        cfg = SimpleNamespace(num_experts=4, experts_top_k=1, moe_morph_routing=False,
                              moe_d_ff=8, d_ff=16, d_model=8)
        moe = MorphRoutedMoE(cfg)
        torch.nn.init.zeros_(moe.router.weight)  # uniform olasılık
        _, aux = moe(torch.randn(1, 8, 8))
        # top-1 eşitlikte hep uzman 0 seçilir: f=[1,0,0,0], P=0.25 → aux = 4*0.25 = 1
        self.assertAlmostEqual(aux.item(), 1.0, places=5)

    def test_moe_layer_frequency(self):
        model = tiny_model(num_layers=4, num_experts=2, experts_top_k=1, moe_layer_freq=2)
        self.assertEqual([b.use_moe for b in model.blocks], [False, True, False, True])

    def test_morph_routing_changes_router_decisions(self):
        cfg = SimpleNamespace(num_experts=3, experts_top_k=1, moe_morph_routing=True,
                              moe_d_ff=8, d_ff=16, d_model=8)
        moe = MorphRoutedMoE(cfg)
        with torch.no_grad():
            moe.router.weight.copy_(torch.eye(3, 8))
            moe.morph_route_emb.weight.copy_(10 * torch.eye(3, 8))
        x = torch.zeros(1, 3, 8)
        moe(x, morph_classes=torch.tensor([[0, 1, 2]]))
        self.assertEqual(moe.last_routing[0, :, 0].tolist(), [0, 1, 2])
        table = routing_by_morph_class(moe.last_routing, torch.tensor([[0, 1, 2]]), 3)
        self.assertEqual(table.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestCheckpointConfig(unittest.TestCase):

    def test_architecture_dict_roundtrip_rebuilds_model(self):
        model = tiny_model(num_mtp_heads=2, num_experts=2, experts_top_k=1,
                           rope_scaling={"type": "yarn", "factor": 2.0, "original_max_seq_len": 16})
        arch = model.config.architecture_dict()
        rebuilt = ToprakLM(ModelConfig(**arch, device="cpu"), tokenizer=MockTokenizer())
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(rebuilt.config.rope_scaling, arch["rope_scaling"])

    def test_old_checkpoint_config_still_loads(self):
        old = {"vocab_size": len(VOCAB), "d_model": 32, "num_heads": 4, "num_kv_heads": 2,
               "num_layers": 2, "d_ff": 64, "max_seq_len": 32, "rope_theta": 10000.0,
               "norm_eps": 1e-6}
        model = ToprakLM(ModelConfig(**old, device="cpu"), tokenizer=MockTokenizer())
        self.assertEqual(model.config.num_mtp_heads, 0)
        self.assertEqual(model.config.num_experts, 0)

    def test_train_cli_overrides(self):
        from training.train import apply_architecture_overrides
        config = tiny_config(max_seq_len=512)
        args = SimpleNamespace(
            mtp_heads=3, mtp_lambda=0.2, num_experts=8, experts_top_k=2, moe_layer_freq=2,
            no_moe_morph_routing=True, max_seq_len=4096, rope_scaling="yarn",
            rope_factor=None, rope_original_max_seq_len=None,
        )
        apply_architecture_overrides(config, args)
        self.assertEqual(config.num_mtp_heads, 3)
        self.assertEqual(config.num_experts, 8)
        self.assertFalse(config.moe_morph_routing)
        self.assertEqual(config.max_seq_len, 4096)
        self.assertEqual(config.rope_scaling,
                         {"type": "yarn", "factor": 8.0, "original_max_seq_len": 512})


if __name__ == "__main__":
    unittest.main()
