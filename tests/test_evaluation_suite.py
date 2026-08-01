# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import math
import os
import unittest

import torch

from evaluation.suite import (
    ToprakEvaluationSuite,
    build_report,
    compare_reports,
    load_benchmarks,
    normalize_answer,
    validate_report_compatibility,
)
from evaluation.eval import compute_perplexity


BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "evaluation",
    "benchmarks",
)


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_bos=True, add_eos=False):
        ids = [3 + (ord(char) % 13) for char in text]
        if add_bos:
            ids.insert(0, 1)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        return "".join(chr(97 + (token % 26)) for token in ids)


class UniformModel(torch.nn.Module):
    def __init__(self, vocab_size=32, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = type("Config", (), {"max_seq_len": max_seq_len})()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, targets=None, past_kvs=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        logits[..., 0] = self.anchor
        cache = [(torch.empty(0), torch.empty(0))] if past_kvs is None else past_kvs
        return logits, None, cache


class TestEvaluationSuite(unittest.TestCase):
    def test_seed_benchmarks_are_valid_and_unique(self):
        samples = load_benchmarks(BENCHMARK_DIR)
        ids = [sample["id"] for sample in samples]
        categories = {sample["category"] for sample in samples}
        self.assertGreaterEqual(len(samples), 25)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({
            "morphology",
            "reading_comprehension",
            "general_knowledge",
            "math_reasoning",
            "long_context",
            "toxicity_safety",
            "memorization_safety",
        }.issubset(categories))

    def test_continuation_logprob_uses_only_continuation_tokens(self):
        model = UniformModel(vocab_size=32)
        tokenizer = FakeTokenizer()
        suite = ToprakEvaluationSuite(model, tokenizer, "cpu")
        result = suite.score_continuation("ab", "cde")
        self.assertEqual(result["tokens"], 3)
        self.assertAlmostEqual(result["total_logprob"], -3 * math.log(32), places=5)
        self.assertAlmostEqual(result["mean_logprob"], -math.log(32), places=5)

    def test_answer_normalization_is_turkish_safe(self):
        self.assertEqual(normalize_answer("  İSTANBUL'dur! "), "istanbul dur")
        self.assertEqual(normalize_answer("H₂O"), "h2o")

    def test_report_aggregation_and_comparison(self):
        baseline = build_report([
            {"id": "a", "benchmark": "x", "category": "grammar", "score": 0.0, "correct": False},
            {"id": "b", "benchmark": "y", "category": "knowledge", "score": 1.0, "correct": True},
        ])
        current = build_report([
            {"id": "a", "benchmark": "x", "category": "grammar", "score": 1.0, "correct": True},
            {"id": "b", "benchmark": "y", "category": "knowledge", "score": 1.0, "correct": True},
        ])
        comparison = compare_reports(current, baseline)
        self.assertAlmostEqual(current["summary"]["macro_score"], 1.0)
        self.assertAlmostEqual(comparison["macro_delta"], 0.5)
        self.assertAlmostEqual(comparison["categories"]["grammar"]["delta"], 1.0)

    def test_incompatible_reports_are_rejected(self):
        result = {
            "id": "a", "benchmark": "x", "category": "grammar",
            "score": 1.0, "correct": True,
        }
        current = build_report([result], metadata={
            "tokenizer_sha256": "tok-a",
            "benchmark_sha256": {"x.jsonl": "hash-a"},
        })
        baseline = build_report([result], metadata={
            "tokenizer_sha256": "tok-b",
            "benchmark_sha256": {"x.jsonl": "hash-a"},
        })
        with self.assertRaisesRegex(ValueError, "tokenizer_sha256"):
            validate_report_compatibility(current, baseline)

    def test_empty_perplexity_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eval batch/token"):
            compute_perplexity(UniformModel(), [], device="cpu")


if __name__ == "__main__":
    unittest.main()
