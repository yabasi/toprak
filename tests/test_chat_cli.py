# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Sohbet arayüzü: geçmiş kırpma ve uzun sohbetlerde RoPE sınırı testleri."""

import contextlib
import io
import unittest

import torch

from inference.chat import chat, fit_history
from model.chat_template import ChatTemplate
from model.config import ModelConfig
from model.transformer import ToprakLM

V = 48


class MockTokenizer:
    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3
    specials = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]

    def get_vocab_size(self):
        return V

    def id_to_token(self, token_id):
        return self.specials[token_id] if token_id < 5 else f"▁t{token_id}"

    def token_to_id(self, token):
        return self.specials.index(token) if token in self.specials else self.unk_token_id

    def encode(self, text, add_bos=False, add_eos=False):
        return ([2] if add_bos else []) + [5 + ord(c) % (V - 5) for c in text] + ([3] if add_eos else [])

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids if i > 4)


def conversation(turns):
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"soru{i}"})
        messages.append({"role": "assistant", "content": f"cevap{i}"})
    messages.append({"role": "user", "content": "son soru"})
    return messages


class TestFitHistory(unittest.TestCase):

    def setUp(self):
        self.template = ChatTemplate(MockTokenizer())

    def test_short_history_is_untouched(self):
        messages = conversation(1)
        kept, ids, new = fit_history(self.template, messages, max_positions=512, max_new_tokens=50)
        self.assertEqual(kept, messages)
        self.assertEqual(ids, self.template.encode_prompt(messages))
        self.assertEqual(new, 50)

    def test_long_history_drops_oldest_turns(self):
        messages = conversation(30)
        kept, ids, new = fit_history(self.template, messages, max_positions=128, max_new_tokens=32)
        self.assertLessEqual(len(ids) + new, 128)
        self.assertLess(len(kept), len(messages))
        self.assertEqual(kept[-1], {"role": "user", "content": "son soru"})
        self.assertEqual(kept, messages[-len(kept):])
        self.assertEqual(ids, self.template.encode_prompt(kept))

    def test_system_prompt_is_preserved(self):
        template = ChatTemplate(MockTokenizer(), system_prompt="Kısa cevap ver.")
        kept, ids, _ = fit_history(template, conversation(30), max_positions=160, max_new_tokens=32)
        self.assertEqual(kept[0], {"role": "system", "content": "Kısa cevap ver."})
        self.assertEqual(kept[-1]["content"], "son soru")
        self.assertLessEqual(len(ids), 128)

    def test_single_oversized_message_is_clipped_from_left(self):
        messages = [{"role": "user", "content": "x" * 500}]
        kept, ids, new = fit_history(self.template, messages, max_positions=64, max_new_tokens=16)
        self.assertEqual(len(ids), 48)
        self.assertEqual(ids[0], self.template.bos_id)
        full = self.template.encode_prompt(messages)
        self.assertEqual(ids[1:], full[-47:])  # asistan rol öneki korunur

    def test_max_new_tokens_is_clamped(self):
        _, ids, new = fit_history(self.template, conversation(2), max_positions=64, max_new_tokens=1000)
        self.assertLess(new, 64)
        self.assertLessEqual(len(ids) + new, 64)
        self.assertGreater(len(ids), 0)


class TestChatLoop(unittest.TestCase):

    def test_long_session_never_exceeds_rope_table(self):
        torch.manual_seed(0)
        config = ModelConfig(vocab_size=V, d_model=32, num_heads=4, num_kv_heads=2, num_layers=2,
                             d_ff=64, max_seq_len=32, device="cpu")
        model = ToprakLM(config, tokenizer=MockTokenizer())
        self.assertEqual(model.freqs_cis.size(0), 64)

        script = iter([f"mesaj numarası {i}" for i in range(12)]
                      + ["ayar", "0.5", "", "", "8", "temizle", "yeniden başla", "çık"])
        seen_processors = []

        def processor(generated, logits):
            seen_processors.append(len(generated))
            return logits

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            chat(model, MockTokenizer(), device="cpu", max_new_tokens=12,
                 system_prompt="Sen Toprak'sın.", logits_processors=[processor],
                 input_fn=lambda _prompt: next(script))
        text = out.getvalue()
        self.assertIn("Güncellendi", text)
        self.assertIn("temizlendi", text)
        self.assertIn("Görüşmek üzere", text)
        self.assertEqual(text.count("🌱 Toprak: "), 13)
        self.assertTrue(seen_processors)
        self.assertTrue(all(n <= 64 for n in seen_processors))


if __name__ == "__main__":
    unittest.main()
