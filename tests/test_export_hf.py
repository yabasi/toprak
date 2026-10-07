# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak → HF Llama dışa aktarım testleri.
Küçük rastgele ToprakLM → convert_to_hf_llama → transformers LlamaForCausalLM:
logit eşliği (RoPE: yok / linear / ntk / yarn), açgözlü üretim eşliği,
MoE hatası, MTP atma uyarısı, tokenizer eşliği.
"""

import json
import os
import tempfile
import unittest
import warnings

import torch

from model.config import ModelConfig
from model.transformer import ToprakLM
from export.hf_llama import (
    convert_to_hf_llama,
    main as hf_main,
    permute_rotary,
    unpermute_rotary,
)

try:
    from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP_MODEL = os.path.join(ROOT, "toprak_tokenizer.model")


class MockTokenizer:
    """Testler için sahte tokenizer."""

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def id_to_token(self, token_id):
        return f"▁t{token_id}" if token_id % 2 else f"ek{token_id}"


def tiny_config(**overrides):
    kwargs = dict(
        vocab_size=64, d_model=64, num_heads=4, num_kv_heads=2, num_layers=2,
        d_ff=96, max_seq_len=32, rope_theta=1000.0, norm_eps=1e-5, device="cpu",
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def tiny_model(seed=0, scale=4.0, **overrides):
    """Rastgele ağırlıkları büyütülmüş (çıktıları anlamlı) küçük model."""
    torch.manual_seed(seed)
    model = ToprakLM(tiny_config(**overrides), tokenizer=MockTokenizer()).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    return model


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers kurulu değil")
class TestHFLlamaExport(unittest.TestCase):

    def _roundtrip_logits(self, model, ids):
        with tempfile.TemporaryDirectory() as out:
            convert_to_hf_llama(model, out, dtype="float32", include_tokenizer=False)
            hf = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
            with torch.no_grad():
                ours = model(ids)[0]
                theirs = hf(ids).logits
            tied = hf.lm_head.weight.data_ptr() == hf.model.embed_tokens.weight.data_ptr()
        return ours, theirs, tied

    def _assert_logits_match(self, **overrides):
        model = tiny_model(**overrides)
        ids = torch.randint(0, 64, (2, 24), generator=torch.Generator().manual_seed(1))
        ours, theirs, tied = self._roundtrip_logits(model, ids)
        self.assertTrue(tied, "lm_head embed_tokens'a bağlı olmalı")
        self.assertGreater(ours.abs().max().item(), 1.0)  # önemsiz olmayan çıktılar
        self.assertLess((ours - theirs).abs().max().item(), 1e-4)

    def test_permute_roundtrip(self):
        w = torch.randn(4 * 8, 16)
        p = permute_rotary(w, 4, 8)
        self.assertTrue(torch.equal(unpermute_rotary(p, 4, 8), w))
        # Head içinde çift satırlar önce, tek satırlar sonra
        self.assertTrue(torch.equal(p[:4], w[[0, 2, 4, 6]]))
        self.assertTrue(torch.equal(p[4:8], w[[1, 3, 5, 7]]))

    def test_logits_match_no_rope_scaling(self):
        self._assert_logits_match()

    def test_logits_match_linear_rope(self):
        self._assert_logits_match(rope_scaling={"type": "linear", "factor": 2.0})

    def test_logits_match_ntk_rope(self):
        self._assert_logits_match(rope_scaling={"type": "ntk", "factor": 4.0})

    def test_logits_match_yarn_rope(self):
        self._assert_logits_match(
            max_seq_len=64,
            rope_scaling={"type": "yarn", "factor": 4.0, "original_max_seq_len": 16},
        )

    def test_greedy_generation_identical(self):
        model = tiny_model(seed=3)
        prompt = torch.tensor([[2, 10, 11, 12, 13]])
        # Toprak: KV cache ile açgözlü çözümleme
        ids, past = prompt, None
        with torch.no_grad():
            for _ in range(10):
                inp = ids if past is None else ids[:, -1:]
                logits, _, past = model(inp, past_kvs=past)
                ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        with tempfile.TemporaryDirectory() as out:
            convert_to_hf_llama(model, out, dtype="float32", include_tokenizer=False)
            hf = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
            # EOS'ta durmayı kapat: Toprak döngüsü de 10 token boyunca sürer
            hf.generation_config.eos_token_id = None
            gen_cfg = GenerationConfig(
                max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0,
                repetition_penalty=1.0,
            )
            hf_ids = hf.generate(
                prompt, attention_mask=torch.ones_like(prompt), generation_config=gen_cfg,
            )
        self.assertEqual(hf_ids.tolist(), ids.tolist())

    def test_checkpoint_input_and_files(self):
        model = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "toprak.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config.architecture_dict(),
                "global_step": 7,
            }, ckpt)
            out = os.path.join(tmp, "hf")
            hf_main(["--checkpoint", ckpt, "--out", out, "--dtype", "float16", "--no-tokenizer"])
            with open(os.path.join(out, "config.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["architectures"], ["LlamaForCausalLM"])
            self.assertTrue(cfg["tie_word_embeddings"])
            self.assertEqual(cfg["num_key_value_heads"], 2)
            self.assertEqual(cfg["hidden_act"], "silu")
            self.assertFalse(cfg["attention_bias"])
            self.assertFalse(cfg["mlp_bias"])
            self.assertEqual(cfg["rms_norm_eps"], 1e-5)
            self.assertEqual(cfg["max_position_embeddings"], 32)
            self.assertEqual((cfg["bos_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]), (2, 3, 0))
            self.assertTrue(os.path.exists(os.path.join(out, "generation_config.json")))

            from safetensors import safe_open
            with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:
                keys = list(f.keys())
                self.assertNotIn("lm_head.weight", keys)
                self.assertEqual(f.get_tensor("model.embed_tokens.weight").dtype, torch.float16)
            hf = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
            ids = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(2))
            with torch.no_grad():
                diff = (model(ids)[0] - hf(ids).logits).abs().max().item()
            self.assertLess(diff, 0.1)  # fp16 yuvarlama payı

    def test_moe_raises(self):
        model = tiny_model(num_experts=4, experts_top_k=2)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(NotImplementedError) as ctx:
                convert_to_hf_llama(model, out, include_tokenizer=False)
            self.assertIn("Mixtral", str(ctx.exception))

    def test_mtp_heads_dropped_with_warning(self):
        model = tiny_model(num_mtp_heads=2)
        ids = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ours, theirs, _ = self._roundtrip_logits(model, ids)
        self.assertTrue(any("MTP" in str(w.message) for w in caught))
        self.assertLess((ours - theirs).abs().max().item(), 1e-4)


@unittest.skipUnless(HAS_TRANSFORMERS and os.path.exists(SP_MODEL), "tokenizer veya transformers yok")
class TestHFTokenizerExport(unittest.TestCase):

    SENTENCES = [
        "Türkiye'nin başkenti Ankara'dır.",
        "Çocuklar bahçede oynuyorlardı; ığdır, şişli, öğün, ünlü.",
        "  Baştaki  ve   ortadaki boşluklar  ",
        "Sayılar: 1234567 ve 3,14 — İSTANBUL ıIiİ",
        "ﬁ ligatürü, ＡＢＣ tam genişlik, emoji 🌱 ve 中文",
        "Satır\nsonu ve\tsekme",
    ]

    def test_auto_tokenizer_matches_toprak_tokenizer(self):
        from export.hf_llama import export_tokenizer
        from model.tokenizer import ToprakTokenizer

        ours = ToprakTokenizer(SP_MODEL)
        with tempfile.TemporaryDirectory() as out:
            export_tokenizer(SP_MODEL, out)
            hf = AutoTokenizer.from_pretrained(out)
            self.assertEqual(
                (hf.bos_token_id, hf.eos_token_id, hf.pad_token_id, hf.unk_token_id), (2, 3, 0, 1)
            )
            for text in self.SENTENCES:
                with self.subTest(text=text):
                    self.assertEqual(
                        hf(text, add_special_tokens=False)["input_ids"],
                        ours.encode(text, add_bos=False, add_eos=False),
                    )
                    self.assertEqual(hf(text)["input_ids"], ours.encode(text, add_eos=False))
                    ids = ours.encode(text, add_bos=False, add_eos=False)
                    self.assertEqual(hf.decode(ids), ours.sp.decode(ids))


if __name__ == "__main__":
    unittest.main()
