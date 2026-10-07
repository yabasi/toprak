# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Uyum Korumalı Kod Çözme Testleri
`inference/grammar_guard.py` (HarmonyGuard) ve `evaluation/harmony_check.py`.

Oyuncak sözlükte kararlar:
  ▁kitap → "lar" (ler maskeli), "ta" (da maskeli: p sert ünsüz)
  ▁kitap+lar → "da" serbest (r yumuşak), "de" maskeli
  ▁ev → "ler"; ▁saat istisna → ne "lar" ne "ler" maskelenir (muafiyet)
  "yor" değişmez → hiçbir bağlamda maskelenmez
"""

import os
import unittest

import torch

from inference.grammar_guard import GuardStats, HarmonyGuard, build_grammar_guard
from evaluation.harmony_check import harmony_violation_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB = ["<pad>", "<unk>", "<s>", "</s>", "▁kitap", "lar", "ler", "da", "de",
         "ta", "te", "▁ev", "yor", "▁saat"]
ID = {t: i for i, t in enumerate(VOCAB)}
NEG = float("-inf")


class MockTokenizer:
    def __init__(self, vocab=VOCAB):
        self.vocab = list(vocab)
        self.eos_token_id = 3

    def get_vocab_size(self):
        return len(self.vocab)

    def id_to_token(self, token_id):
        return self.vocab[token_id]

    def encode(self, text, add_bos=True, add_eos=False):
        return [2, ID["▁kitap"]]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids if i > 3).replace("▁", " ").strip()


def biased_logits(**overrides):
    """İhlal eden tokenleri kayıran sahte logitler."""
    base = {"ler": 10.0, "de": 9.0, "da": 8.0, "lar": 5.0, "ta": 4.0, "te": 3.0, "yor": 1.0}
    base.update(overrides)
    logits = torch.full((1, len(VOCAB)), NEG)
    for tok, val in base.items():
        logits[0, ID[tok]] = val
    return logits


def pick(guard, context, logits):
    ids = [2] + [ID[t] for t in context]
    out = guard(ids, logits.clone()) if guard else logits
    return VOCAB[int(out.argmax())]


class TestHarmonyGuardToy(unittest.TestCase):

    def setUp(self):
        self.guard = HarmonyGuard(MockTokenizer(), mode="mask")

    def test_back_word_prefers_back_suffix(self):
        self.assertEqual(pick(None, ["▁kitap"], biased_logits()), "ler")
        self.assertEqual(pick(self.guard, ["▁kitap"], biased_logits()), "lar")

    def test_voiceless_consonant_rule(self):
        # lar yokken: da (p'den sonra) maskeli → ta
        self.assertEqual(pick(self.guard, ["▁kitap"], biased_logits(lar=NEG)), "ta")
        out = self.guard([2, ID["▁kitap"]], biased_logits())
        self.assertEqual(out[0, ID["da"]].item(), NEG)
        self.assertEqual(out[0, ID["ta"]].item(), 4.0)

    def test_voiced_ending_allows_da(self):
        self.assertEqual(pick(self.guard, ["▁kitap", "lar"], biased_logits()), "da")
        out = self.guard([2, ID["▁kitap"], ID["lar"]], biased_logits())
        self.assertEqual(out[0, ID["de"]].item(), NEG)
        self.assertEqual(out[0, ID["ta"]].item(), 4.0)   # ta uyumlu (sadece ünlüye bakılır)

    def test_front_word(self):
        logits = biased_logits(lar=12.0, da=11.0)
        self.assertEqual(pick(self.guard, ["▁ev"], logits), "ler")
        self.assertEqual(pick(self.guard, ["▁ev", "ler"], biased_logits(lar=12.0, da=11.0, ler=NEG)), "de")

    def test_loanword_exception_is_exempt(self):
        logits = biased_logits(lar=12.0)
        out = self.guard([2, ID["▁saat"]], logits.clone())
        self.assertTrue(torch.equal(out, logits))
        self.assertEqual(self.guard.stats.exempt, 1)

    def test_invariant_yor_never_masked(self):
        for ctx in (["▁kitap"], ["▁ev"], ["▁kitap", "lar"], ["▁ev", "ler"]):
            out = self.guard([2] + [ID[t] for t in ctx], biased_logits(yor=20.0))
            self.assertEqual(out[0, ID["yor"]].item(), 20.0)

    def test_word_start_and_special_tokens_untouched(self):
        logits = biased_logits()
        logits[0, ID["▁ev"]] = 7.0
        out = self.guard([2, ID["▁kitap"]], logits.clone())
        self.assertEqual(out[0, ID["▁ev"]].item(), 7.0)
        # Yalnız BOS: kelime yok → no-op
        self.assertTrue(torch.equal(self.guard([2], logits.clone()), logits))

    def test_never_masks_everything(self):
        logits = torch.full((1, len(VOCAB)), NEG)
        logits[0, ID["ler"]] = 3.0
        logits[0, ID["de"]] = 2.0
        out = self.guard([2, ID["▁kitap"]], logits.clone())
        self.assertTrue(torch.equal(out, logits))
        self.assertEqual(self.guard.stats.fallbacks, 1)

    def test_penalty_mode(self):
        guard = build_grammar_guard(MockTokenizer(), mode="penalty", penalty=7.0)
        out = guard([2, ID["▁kitap"]], biased_logits())
        self.assertAlmostEqual(out[0, ID["ler"]].item(), 3.0)
        self.assertAlmostEqual(out[0, ID["lar"]].item(), 5.0)
        self.assertEqual(VOCAB[int(out.argmax())], "lar")

    def test_stats(self):
        self.guard([2, ID["▁kitap"]], biased_logits())
        self.guard([2, ID["▁kitap"], ID["lar"]], biased_logits())
        st = self.guard.stats
        self.assertIsInstance(st, GuardStats)
        self.assertEqual(st.steps, 2)
        self.assertEqual(st.applied, 2)
        self.assertEqual(st.top1_changed, 2)
        st.reset()
        self.assertEqual(st.as_dict()["steps"], 0)

    def test_larger_model_vocab_is_padded(self):
        logits = torch.cat([biased_logits(), torch.zeros(1, 6)], dim=1)
        out = self.guard([2, ID["▁kitap"]], logits)
        self.assertEqual(out.shape, logits.shape)
        self.assertEqual(out[0, ID["ler"]].item(), NEG)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            HarmonyGuard(MockTokenizer(), mode="x")


class TestGuardReducesViolations(unittest.TestCase):

    def _generate(self, guard):
        ids = [2]
        for word in ["▁kitap", "▁ev", "▁kitap", "▁ev"]:
            ids.append(ID[word])
            for _ in range(2):
                logits = biased_logits()
                if guard is not None:
                    logits = guard(ids, logits)
                ids.append(int(logits.argmax()))
        return MockTokenizer().decode(ids)

    def test_violation_rate_drops(self):
        plain = self._generate(None)
        guarded = self._generate(HarmonyGuard(MockTokenizer()))
        r_plain = harmony_violation_rate(plain)
        r_guard = harmony_violation_rate(guarded)
        self.assertGreater(r_plain["violations"], 0)
        self.assertEqual(r_guard["violations"], 0)
        self.assertLess(r_guard["rate"], r_plain["rate"])
        self.assertIn("kitaplarda", guarded)

    def test_generate_text_integration(self):
        from model.config import ModelConfig
        from model.transformer import ToprakLM
        from inference.generate import generate_text

        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=len(VOCAB), d_model=32, num_heads=4, num_kv_heads=2,
                          num_layers=2, d_ff=64, max_seq_len=32, device="cpu")
        tok = MockTokenizer()
        model = ToprakLM(cfg, tokenizer=tok)
        guard = HarmonyGuard(tok)
        text = generate_text(model, tok, "kitap", max_new_tokens=8, device="cpu",
                             logits_processors=[guard], repetition_penalty=1.0)
        self.assertIsInstance(text, str)
        self.assertGreater(guard.stats.steps, 0)


class TestHarmonyCheck(unittest.TestCase):

    def test_counts(self):
        r = harmony_violation_rate("kitapler kitapda evlerde saatler kitaplarda İstanbul'de")
        self.assertEqual(r["violations"], 3)
        self.assertEqual(r["vowel_violations"], 2)
        self.assertEqual(r["consonant_violations"], 1)

    def test_clean_text(self):
        r = harmony_violation_rate(
            "Türkiye'nin başkenti Ankara'dır. Öğrenciler okullarda kitaplarını okuyorlar."
        )
        self.assertEqual(r["violations"], 0)
        self.assertGreater(r["checked"], 0)


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "toprak_tokenizer.model")),
                     "gerçek tokenizer yok")
class TestRealTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from model.tokenizer import ToprakTokenizer
        cls.tok = ToprakTokenizer(os.path.join(ROOT, "toprak_tokenizer.model"))
        cls.guard = HarmonyGuard(cls.tok)

    def test_tables_and_step(self):
        tok, guard = self.tok, self.guard
        V = tok.get_vocab_size()
        ids = tok.encode("kitap", add_bos=True, add_eos=False)
        logits = torch.zeros(1, V)
        out = guard(ids, logits)
        self.assertEqual(out.shape, (1, V))
        self.assertTrue(torch.isfinite(out).any())
        masked = ~torch.isfinite(out[0])
        for piece in ("larda", "lar"):
            pid = tok.token_to_id(piece)
            if pid != tok.token_to_id("<unk>"):
                self.assertFalse(bool(masked[pid]), piece)
        for piece in ("lerde", "ler", "da"):
            pid = tok.token_to_id(piece)
            if pid != tok.token_to_id("<unk>"):
                self.assertTrue(bool(masked[pid]), piece)
        self.assertEqual(guard.current_word(ids), "kitap")

    def test_root_internal_pieces_not_masked(self):
        # ▁bek + liyor → "liyor" ince, "bek" ince: uyumlu; "yor" değişmez
        tok, guard = self.tok, self.guard
        ids = tok.encode("ok", add_bos=True, add_eos=False)
        out = guard(ids, torch.zeros(1, tok.get_vocab_size()))
        pid = tok.token_to_id("uyor")
        if pid != tok.token_to_id("<unk>"):
            self.assertTrue(torch.isfinite(out[0, pid]))


if __name__ == "__main__":
    unittest.main()
