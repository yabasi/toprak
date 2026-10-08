# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import importlib.util
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import ToprakLM

HAS_LM_EVAL = importlib.util.find_spec("lm_eval") is not None

ALPHABET = "abcdefghij .\n"


class CharTokenizer:
    """Karakter başına tek token; sınır birleşmesi olmayan deterministik tokenizer."""

    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 3

    def encode(self, text, add_bos=True, add_eos=True):
        ids = [4 + ALPHABET.index(char) for char in text]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        return "".join(ALPHABET[i - 4] for i in ids if 4 <= i < 4 + len(ALPHABET))


def tiny_model(max_seq_len=16):
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=4 + len(ALPHABET),
        d_model=16,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        d_ff=32,
        max_seq_len=max_seq_len,
        device="cpu",
    )
    return ToprakLM(config).eval()


class CountingModel(torch.nn.Module):
    """Her adımda bir sonraki karakteri ('a'→'b'→...) kesin tahmin eder."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_seq_len=64)
        self.vocab = 4 + len(ALPHABET)

    def forward(self, input_ids, past_kvs=None):
        next_ids = (input_ids - 3).clamp(min=1) + 4
        next_ids = next_ids.clamp(max=self.vocab - 1)
        logits = F.one_hot(next_ids, self.vocab).float() * 10.0
        return logits, None, [None]


def req(*args):
    return SimpleNamespace(args=args)


@unittest.skipUnless(HAS_LM_EVAL, "lm_eval kurulu değil")
class LMEvalAdapterTest(unittest.TestCase):
    def make_adapter(self, batch_size=4, model=None, max_length=None):
        from evaluation.lm_eval_adapter import ToprakLMEval
        return ToprakLMEval(
            model=model or tiny_model(),
            tokenizer=CharTokenizer(),
            device="cpu",
            batch_size=batch_size,
            max_length=max_length,
        )

    def manual_loglikelihood(self, model, context, continuation):
        tok = CharTokenizer()
        ctx = tok.encode(context, add_bos=True, add_eos=False)
        cont = tok.encode(continuation, add_bos=False, add_eos=False)
        full = torch.tensor([ctx + cont])
        with torch.no_grad():
            logits, _, _ = model(full[:, :-1])
        log_probs = F.log_softmax(logits.float(), dim=-1)[0]
        positions = range(len(ctx) - 1, len(ctx) + len(cont) - 1)
        return sum(float(log_probs[p, t]) for p, t in zip(positions, cont))

    def test_loglikelihood_matches_manual_computation(self):
        model = tiny_model()
        adapter = self.make_adapter(model=model)
        (score, greedy), = adapter.loglikelihood([req("abc", "de")])
        self.assertAlmostEqual(score, self.manual_loglikelihood(model, "abc", "de"), places=4)
        self.assertIsInstance(greedy, bool)

    def test_trailing_context_space_moves_to_continuation(self):
        model = tiny_model()
        adapter = self.make_adapter(model=model)
        (score, _), = adapter.loglikelihood([req("ab ", "cd")])
        self.assertAlmostEqual(score, self.manual_loglikelihood(model, "ab", " cd"), places=4)

    def test_batching_and_padding_do_not_change_scores(self):
        model = tiny_model()
        requests = [
            req("a", "b"),
            req("abcdefg", "hij"),
            req("ab", "c d e"),
            req("", "abc"),
        ]
        single = self.make_adapter(batch_size=1, model=model).loglikelihood(requests)
        batched = self.make_adapter(batch_size=4, model=model).loglikelihood(requests)
        for (s1, g1), (s2, g2) in zip(single, batched):
            self.assertAlmostEqual(s1, s2, places=4)
            self.assertEqual(g1, g2)

    def test_long_inputs_are_left_truncated_to_context_window(self):
        adapter = self.make_adapter(max_length=8)
        (score, _), = adapter.loglikelihood([req("abcdefghij" * 3, "abc")])
        self.assertTrue(torch.isfinite(torch.tensor(score)))

    def test_rolling_equals_empty_context_loglikelihood_for_short_text(self):
        model = tiny_model()
        adapter = self.make_adapter(model=model)
        rolling, = adapter.loglikelihood_rolling([req("abc de")])
        (direct, _), = adapter.loglikelihood([req("", "abc de")])
        self.assertAlmostEqual(rolling, direct, places=4)

    def test_rolling_covers_every_token_for_long_text(self):
        adapter = self.make_adapter(max_length=8)
        text = "abcdefghij" * 3
        windows_seen = []
        original = adapter._loglikelihood_tokens

        def spy(items, disable_tqdm=False):
            windows_seen.extend(items)
            return original(items, disable_tqdm=disable_tqdm)

        adapter._loglikelihood_tokens = spy
        adapter.loglikelihood_rolling([req(text)])
        scored = sum(len(cont) for _, cont in windows_seen)
        self.assertEqual(scored, len(text))

    def test_generate_until_stops_at_stop_sequence(self):
        adapter = self.make_adapter(model=CountingModel())
        out, = adapter.generate_until([req("a", {"until": ["e"], "max_gen_toks": 20})])
        self.assertEqual(out, "bcd")

    def test_generate_until_respects_max_gen_toks(self):
        adapter = self.make_adapter(model=CountingModel())
        out, = adapter.generate_until([req("a", {"until": ["\n\n"], "max_gen_toks": 3})])
        self.assertEqual(out, "bcd")


class RunLMEvalHelpersTest(unittest.TestCase):
    def test_resolve_tasks_merges_preset_and_extra_without_duplicates(self):
        from evaluation.run_lm_eval import TASK_PRESETS, resolve_tasks
        tasks = resolve_tasks("tr_core", "xcopa_tr,xquad_tr")
        self.assertEqual(tasks, TASK_PRESETS["tr_core"] + ["xquad_tr"])

    def test_resolve_tasks_rejects_empty_and_unknown(self):
        from evaluation.run_lm_eval import resolve_tasks
        with self.assertRaises(ValueError):
            resolve_tasks()
        with self.assertRaises(ValueError):
            resolve_tasks("yok")

    def test_summarize_prefers_acc_norm(self):
        from evaluation.run_lm_eval import summarize
        rows = summarize({"results": {
            "xcopa_tr": {"acc,none": 0.55, "acc_stderr,none": 0.02},
            "belebele_tur_Latn": {
                "acc,none": 0.3, "acc_norm,none": 0.31, "acc_norm_stderr,none": 0.01,
            },
        }})
        by_task = {row["task"]: row for row in rows}
        self.assertEqual(by_task["belebele_tur_Latn"]["metric"], "acc_norm")
        self.assertEqual(by_task["xcopa_tr"]["stderr"], 0.02)
        self.assertEqual(by_task["xcopa_tr"]["random_baseline"], 0.5)


if __name__ == "__main__":
    unittest.main()
