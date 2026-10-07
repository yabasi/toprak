# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Ağırlık-yalnız kuantizasyon testleri.
int4 paketleme, int8/int4 hata, boyut küçülmesi, kaydet/yükle eşliği,
bağlı embedding korunumu ve KV cache ile üretim.

Kalite testleri için rastgele ağırlıklı model yerine basit bir dizi görevinde
(x[t+1] = (x[t-1] + 7) mod 60 + 4) kısaca eğitilmiş küçük model kullanılır:
rastgele ağırlıklar ya önemsiz (çıktı girdiye bağlı değil) ya da kaotik
olur ve kuantizasyon kalitesi hakkında dürüst bir şey söylemez.
"""

import copy
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.transformer import ToprakLM
from export.quantize import (
    QuantLinear,
    load_quantized,
    model_size_bytes,
    pack_int4,
    quantization_error_report,
    quantize_model,
    save_quantized,
    size_report,
    unpack_int4,
    main as quant_main,
)

V = 64


class MockTokenizer:
    """Testler için sahte tokenizer."""

    def get_vocab_size(self):
        return V

    def id_to_token(self, token_id):
        return f"▁t{token_id}"


def make_batch(gen, batch=32, length=33):
    """x[t+1] = (x[t-1] + 7) mod 60 + 4 — dikkat (attention) + FFN gerektirir."""
    x = torch.zeros(batch, length, dtype=torch.long)
    x[:, :2] = torch.randint(4, V, (batch, 2), generator=gen)
    for t in range(2, length):
        x[:, t] = (x[:, t - 2] - 4 + 7) % (V - 4) + 4
    return x


def trained_tiny_model(steps=150):
    # Küçük modelde tek iş parçacığı hem hızlı hem deterministik
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _train(steps)
    finally:
        torch.set_num_threads(threads)


def _train(steps):
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=V, d_model=64, num_heads=4, num_kv_heads=2, num_layers=2,
        d_ff=128, max_seq_len=32, device="cpu",
    )
    model = ToprakLM(config, tokenizer=MockTokenizer())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        x = make_batch(gen)
        _, loss, _ = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


class TestPacking(unittest.TestCase):

    def test_pack_unpack_roundtrip_exact(self):
        q = torch.randint(0, 16, (7, 130), generator=torch.Generator().manual_seed(0))
        packed = pack_int4(q)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (7, 65))
        self.assertTrue(torch.equal(unpack_int4(packed).long(), q))

    def test_pack_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            pack_int4(torch.tensor([[16, 0]]))

    def test_quantlinear_int8_error_small(self):
        torch.manual_seed(0)
        lin = nn.Linear(96, 48, bias=True)
        q = QuantLinear.from_linear(lin, bits=8)
        rel = (q.dequantize() - lin.weight).norm() / lin.weight.norm()
        self.assertLess(rel.item(), 0.01)
        x = torch.randn(5, 96)
        self.assertLess((q(x) - lin(x)).abs().max().item(), 0.02)

    def test_quantlinear_int4_groups_and_zero_point(self):
        torch.manual_seed(0)
        lin = nn.Linear(100, 24, bias=False)  # 100 grup boyutuna bölünmez → dolgu
        for gs in (32, 64, 128):
            for zp in (False, True):
                q = QuantLinear.from_linear(lin, bits=4, group_size=gs, zero_point=zp)
                rel = (q.dequantize() - lin.weight).norm() / lin.weight.norm()
                self.assertLess(rel.item(), 0.2, f"gs={gs} zp={zp}")
                self.assertEqual(q.dequantize().shape, lin.weight.shape)
        # Küçük grup daha az hata vermeli
        e32 = (QuantLinear.from_linear(lin, 4, 32).dequantize() - lin.weight).norm()
        e128 = (QuantLinear.from_linear(lin, 4, 128).dequantize() - lin.weight).norm()
        self.assertLess(e32.item(), e128.item())


class TestModelQuantization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = trained_tiny_model()
        cls.ids = make_batch(torch.Generator().manual_seed(123), batch=8)[:, :32].contiguous()

    def test_task_learned(self):
        """Ön koşul: blokların çıktıya gerçekten katkısı var."""
        with torch.no_grad():
            logits = self.model(self.ids)[0]
        acc = (logits[:, 1:-1].argmax(-1) == self.ids[:, 2:]).float().mean().item()
        self.assertGreater(acc, 0.9)

    def test_only_blocks_quantized_and_embedding_tied(self):
        q = quantize_model(copy.deepcopy(self.model), bits=4, group_size=32)
        n_linear = len(q.quantization["modules"])
        self.assertEqual(n_linear, 2 * 7)  # q,k,v,out + gate,up,down
        self.assertIsInstance(q.blocks[0].attn.q_proj, QuantLinear)
        self.assertIsInstance(q.lm_head, nn.Linear)
        self.assertIs(q.tok_emb.weight, q.lm_head.weight)
        self.assertNotIsInstance(q.morph_head, QuantLinear)

    def test_int8_error_small(self):
        q = quantize_model(copy.deepcopy(self.model), bits=8)
        r = quantization_error_report(self.model, q, self.ids)
        self.assertGreaterEqual(r["top1_agreement"], 0.99)
        self.assertLess(r["logit_mse"], 1e-3)
        self.assertLess(abs(r["ppl_delta_pct"]), 0.5)

    def test_int4_top1_agreement_high(self):
        for gs in (32, 64):
            for zp in (False, True):
                q = quantize_model(copy.deepcopy(self.model), bits=4, group_size=gs, zero_point=zp)
                r = quantization_error_report(self.model, q, self.ids)
                with self.subTest(group_size=gs, zero_point=zp):
                    self.assertGreaterEqual(r["top1_agreement"], 0.95)
                    self.assertLess(abs(r["ppl_delta_pct"]), 5.0)

    def test_size_reduction(self):
        fp = model_size_bytes(self.model)
        q8 = model_size_bytes(quantize_model(copy.deepcopy(self.model), bits=8))
        q4 = quantize_model(copy.deepcopy(self.model), bits=4, group_size=32)
        q4_size = model_size_bytes(q4)
        self.assertLess(q8, fp / 2.5)
        self.assertLess(q4_size, q8)
        # Kuantize lineer katmanlar: fp32'ye göre ~6.4x (4 bit + grup başına fp16 ölçek)
        lin_fp = sum(
            m.weight.numel() * 4 for n, m in self.model.named_modules()
            if isinstance(m, nn.Linear) and n.startswith("blocks.")
        )
        rep = size_report(q4)
        self.assertGreater(lin_fp / rep["quantized_linear_bytes"], 6.0)

    def test_save_load_roundtrip_identical_logits(self):
        q = quantize_model(copy.deepcopy(self.model), bits=4, group_size=64, zero_point=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q4.pt")
            save_quantized(q, path)
            loaded = load_quantized(path, tokenizer=MockTokenizer())
        self.assertIs(loaded.tok_emb.weight, loaded.lm_head.weight)
        with torch.no_grad():
            self.assertTrue(torch.equal(q(self.ids)[0], loaded(self.ids)[0]))

    def test_generate_with_kv_cache(self):
        q = quantize_model(copy.deepcopy(self.model), bits=4, group_size=32)
        prompt = self.ids[:1, :6]
        torch.manual_seed(0)
        out = q.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=1, top_p=1.0)
        self.assertEqual(out.shape[1], 14)
        # KV cache'li üretim = cache'siz açgözlü üretim
        ids = prompt
        with torch.no_grad():
            for _ in range(8):
                ids = torch.cat([ids, q(ids)[0][:, -1].argmax(-1, keepdim=True)], dim=1)
        self.assertEqual(out.tolist(), ids.tolist())

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "fp.pt")
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config.architecture_dict(),
            }, ckpt)
            out = os.path.join(tmp, "q.pt")
            quant_main(["--checkpoint", ckpt, "--bits", "8", "--out", out])
            loaded = load_quantized(out, tokenizer=MockTokenizer())
            self.assertEqual(loaded.quantization["bits"], 8)


if __name__ == "__main__":
    unittest.main()
