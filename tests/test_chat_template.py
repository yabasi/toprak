# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Sohbet şablonu testleri (özel token ve geriye uyumlu metin modları)."""

import os
import unittest

from model.chat_template import CHAT_SPECIAL_TOKENS, IGNORE_INDEX, ChatTemplate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpecialTokenizer:
    """Sohbet tokenlarını tek token olarak bilen sahte tokenizer."""

    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3

    def __init__(self):
        self.special = {tok: 10 + i for i, tok in enumerate(CHAT_SPECIAL_TOKENS)}

    def token_to_id(self, token):
        return self.special.get(token, self.unk_token_id)

    def encode(self, text, add_bos=False, add_eos=False):
        return [100 + len(word) for word in text.split()]


MESSAGES = [
    {"role": "user", "content": "Merhaba nasılsın"},
    {"role": "assistant", "content": "İyiyim teşekkürler"},
]


class TestChatTemplateSpecialTokens(unittest.TestCase):

    def setUp(self):
        self.template = ChatTemplate(SpecialTokenizer())

    def test_layout_and_mask(self):
        ids, mask = self.template.encode_with_mask(MESSAGES)
        user, asst, end = 11, 12, 13
        self.assertTrue(self.template.uses_special_tokens)
        self.assertEqual(ids, [2, user, 107, 108, end, asst, 106, 111, end])
        self.assertEqual(mask, [0, 0, 0, 0, 0, 0, 1, 1, 1])

    def test_generation_prompt_ends_with_assistant_role(self):
        ids = self.template.encode_prompt(MESSAGES[:1])
        self.assertEqual(ids[-1], 12)
        self.assertEqual(self.template.stop_ids, (13, 3))

    def test_build_labels_shifts_and_masks(self):
        inputs, labels = self.template.build_labels(MESSAGES)
        ids, mask = self.template.encode_with_mask(MESSAGES)
        self.assertEqual(inputs, ids[:-1])
        self.assertEqual(labels[:5], [IGNORE_INDEX] * 5)
        self.assertEqual(labels[5:], [106, 111, 13])

    def test_system_prompt_is_prepended_once(self):
        template = ChatTemplate(SpecialTokenizer(), system_prompt="Yardımsever ol")
        ids, mask = template.encode_with_mask(MESSAGES)
        self.assertEqual(ids[1], 10)
        self.assertEqual(ids.count(10), 1)
        self.assertEqual(sum(mask), 3)

    def test_invalid_role_raises(self):
        with self.assertRaises(ValueError):
            self.template.encode_with_mask([{"role": "bot", "content": "x"}])

    def test_truncate_history_keeps_last_user_turn(self):
        template = ChatTemplate(SpecialTokenizer(), system_prompt="Sistem")
        long_history = []
        for i in range(10):
            long_history += [
                {"role": "user", "content": f"soru {i} " * 5},
                {"role": "assistant", "content": f"cevap {i} " * 5},
            ]
        long_history.append({"role": "user", "content": "son soru"})
        kept = template.truncate_history(long_history, max_prompt_tokens=30)
        self.assertEqual(kept[0]["role"], "system")
        self.assertEqual(kept[-1]["content"], "son soru")
        self.assertLessEqual(len(template.encode_prompt(kept)), 30)


class TestChatTemplateRealTokenizerFallback(unittest.TestCase):

    def test_existing_tokenizer_uses_text_markers_and_sep(self):
        from model.tokenizer import ToprakTokenizer
        tokenizer = ToprakTokenizer(os.path.join(ROOT, "toprak_tokenizer.model"))
        template = ChatTemplate(tokenizer)
        self.assertFalse(template.uses_special_tokens)
        self.assertEqual(template.end_id, tokenizer.token_to_id("<sep>"))
        ids, mask = template.encode_with_mask(MESSAGES)
        text = tokenizer.decode(ids)
        self.assertIn("Kullanıcı:", text)
        self.assertIn("Toprak:", text)
        trained = [tok for tok, m in zip(ids, mask) if m]
        self.assertEqual(trained[-1], template.end_id)
        self.assertIn("teşekkürler", tokenizer.decode(trained[:-1]))
        self.assertNotIn("Merhaba", tokenizer.decode(trained))


if __name__ == "__main__":
    unittest.main()
