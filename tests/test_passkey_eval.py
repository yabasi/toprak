# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Passkey (iğne) testi mekaniği: prompt uzunluğu bütçesi, iğne derinliği,
puanlama, uzunluk sınırlama ve doğruluk ızgarası (küçük rastgele model ile;
modelin başarısı değil yalnız mekanik test edilir).
"""

import math
import unittest

import torch

from model.config import ModelConfig
from model.transformer import ToprakLM
from scripts.passkey_eval import (
    FILLER_SENTENCES,
    INTRO,
    NEEDLE_TEMPLATE,
    QUESTION,
    build_passkey_prompt,
    extract_passkey,
    format_grid_table,
    run_passkey_grid,
    score_passkey,
    select_lengths,
    sequence_perplexity,
    usable_context,
)

_CHARS = sorted(set("".join(FILLER_SENTENCES) + INTRO + NEEDLE_TEMPLATE + QUESTION + "0123456789 \n"))
VOCAB = ["<pad>", "<unk>", "<s>", "</s>"] + _CHARS


class CharTokenizer:
    """Karakter düzeyinde sahte tokenizer (ToprakLM için get_vocab_size/id_to_token da var)."""

    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3

    def __init__(self):
        self.index = {c: i for i, c in enumerate(VOCAB)}

    def get_vocab_size(self):
        return len(VOCAB)

    def id_to_token(self, token_id):
        return VOCAB[token_id]

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.index.get(c, self.unk_token_id) for c in text]
        return ([self.bos_token_id] if add_bos else []) + ids + ([self.eos_token_id] if add_eos else [])

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids if i > 3)


def tiny_model(max_seq_len=256):
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=len(VOCAB), d_model=32, num_heads=4, num_kv_heads=2,
                         num_layers=1, d_ff=64, max_seq_len=max_seq_len, device="cpu",
                         rope_scaling={"type": "yarn", "factor": 2.0, "original_max_seq_len": 128})
    return ToprakLM(config, tokenizer=CharTokenizer()).eval()


class TestPasskeyPrompt(unittest.TestCase):

    def setUp(self):
        self.tok = CharTokenizer()
        self.max_unit = max(len(self.tok.encode(" " + s)) for s in FILLER_SENTENCES)

    def test_length_within_budget(self):
        for target in (300, 512, 1000):
            p = build_passkey_prompt(self.tok, target, 0.5, 48213)
            self.assertLessEqual(len(p.input_ids), target)
            self.assertGreater(len(p.input_ids), target - self.max_unit)
            self.assertEqual(p.input_ids[0], self.tok.bos_token_id)

    def test_needle_contains_passkey_and_question_at_end(self):
        p = build_passkey_prompt(self.tok, 600, 0.25, 48213)
        needle = self.tok.decode(p.input_ids[p.needle_start:p.needle_end])
        self.assertEqual(needle.strip(), "Gizli anahtar sayı: 48213. Bunu hatırla.")
        self.assertTrue(self.tok.decode(p.input_ids).endswith("Gizli anahtar sayı:"))

    def test_depth_placement(self):
        tok = self.tok
        prefix_len = 1 + len(tok.encode(INTRO + "\n"))
        suffix_len = len(tok.encode("\n" + QUESTION))
        p0 = build_passkey_prompt(tok, 800, 0.0, 11111)
        self.assertEqual(p0.needle_start, prefix_len)
        self.assertEqual(p0.actual_depth, 0.0)
        p1 = build_passkey_prompt(tok, 800, 1.0, 11111)
        self.assertEqual(p1.needle_end, len(p1.input_ids) - suffix_len)
        self.assertEqual(p1.actual_depth, 1.0)
        prev = -1
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            p = build_passkey_prompt(tok, 2000, d, 11111)
            self.assertAlmostEqual(p.actual_depth, d, delta=0.05)
            self.assertGreater(p.needle_start, prev)
            prev = p.needle_start

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            build_passkey_prompt(self.tok, 50, 0.5, 12345)    # sabit kısımlar sığmaz
        with self.assertRaises(ValueError):
            build_passkey_prompt(self.tok, 500, 1.5, 12345)

    def test_deterministic(self):
        a = build_passkey_prompt(self.tok, 500, 0.5, 12345, seed=3)
        b = build_passkey_prompt(self.tok, 500, 0.5, 12345, seed=3)
        self.assertEqual(a.input_ids, b.input_ids)


class TestScoring(unittest.TestCase):

    def test_exact_match(self):
        self.assertEqual(score_passkey(" 48213. Bunu", 48213), 1.0)
        self.assertEqual(score_passkey("48213", 48213), 1.0)
        self.assertEqual(score_passkey(" 4821", 48213), 0.0)
        self.assertEqual(score_passkey(" 482130", 48213), 0.0)
        self.assertEqual(score_passkey("bilmiyorum", 48213), 0.0)
        self.assertEqual(extract_passkey("sayı 12 ve 34"), "12")
        self.assertIsNone(extract_passkey(""))


class TestGrid(unittest.TestCase):

    def test_length_capping(self):
        model = tiny_model(max_seq_len=256)
        self.assertEqual(usable_context(model), 256)
        self.assertEqual(usable_context(model, allow_extrapolation=True), 512)
        used, skipped = select_lengths([1024, 256, 2048, 200], 256)
        self.assertEqual(used, [200, 256])
        self.assertEqual(skipped, [1024, 2048])

    def test_grid_shape_and_values(self):
        model = tiny_model(max_seq_len=512)
        depths = [0.0, 0.5, 1.0]
        result = run_passkey_grid(model, CharTokenizer(), lengths=[300, 512, 4096], depths=depths,
                                  trials=2, max_new_tokens=4, compute_perplexity=True)
        self.assertEqual(result["lengths"], [300, 512])
        self.assertEqual(result["skipped_lengths"], [4096])
        self.assertEqual(len(result["accuracy"]), 2)
        self.assertTrue(all(len(row) == len(depths) for row in result["accuracy"]))
        self.assertTrue(all(0.0 <= a <= 1.0 for row in result["accuracy"] for a in row))
        self.assertEqual(len(result["records"]), 2 * len(depths) * 2)
        for rec in result["records"]:
            self.assertLessEqual(rec["prompt_tokens"], rec["length"])
        for L in (300, 512):
            self.assertTrue(math.isfinite(result["perplexity"][L]))
            self.assertGreater(result["perplexity"][L], 1.0)
        table = format_grid_table(result)
        self.assertIn("Uzunluk", table)
        self.assertIn("%50", table)
        self.assertIn("PPL", table)
        self.assertIn("Atlanan", table)

    def test_windowed_perplexity_matches_full_pass(self):
        model = tiny_model(max_seq_len=256)
        ids = build_passkey_prompt(CharTokenizer(), 300, 0.5, 12345).input_ids
        full = sequence_perplexity(model, ids, window=1024)
        windowed = sequence_perplexity(model, ids, window=37)
        self.assertAlmostEqual(full, windowed, places=3)


if __name__ == "__main__":
    unittest.main()
