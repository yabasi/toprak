# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Doğrulanabilir ödül fonksiyonları testleri (cevap çıkarma, eşleştirme, biçim, dil)."""

import unittest
from fractions import Fraction

from training.rewards import (
    answers_match,
    combine_rewards,
    correctness_reward,
    extract_answer,
    format_reasoning,
    format_reward,
    language_reward,
    length_penalty,
    parse_number,
)


class TestExtractAnswer(unittest.TestCase):

    def test_cases(self):
        cases = {
            "Cevap: 42": "42",
            "cevap 3/4": "3/4",
            "Sonuç: 12,5": "12,5",
            "Yanıt: 1.250 TL": "1.250",
            "Cevap: -7": "-7",
            "Cevap: −7": "-7",
            "Cevap: %25": "%25",
            "Cevap: 25%": "%25",
            "Cevap: yüzde 25": "%25",
            "Cevap: 1.250,75": "1.250,75",
            "CEVAP: 9": "9",
            "Cevap: 3 / 4": "3/4",
            "Cevap: 42'dir.": "42",
            "\\boxed{17}": "17",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), expected)

    def test_last_marker_wins_and_thought_ignored(self):
        text = "<düşünce>Cevap: 5 sanmıştım, 5 + 3 = 8</düşünce>\nCevap: 8"
        self.assertEqual(extract_answer(text), "8")

    def test_fallback_last_number(self):
        self.assertEqual(extract_answer("5 - 3 = 2 olur, yani 2"), "2")
        self.assertEqual(extract_answer("<düşünce>12 ve 7</düşünce> sonuçta 19"), "19")
        self.assertIsNone(extract_answer("bilmiyorum"))
        self.assertIsNone(extract_answer(""))

    def test_subtraction_not_negative(self):
        self.assertEqual(extract_answer("toplam 10-3"), "3")


class TestParseAndMatch(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(parse_number("12,5"), (Fraction(25, 2), False))
        self.assertEqual(parse_number("1.250"), (Fraction(1250), False))
        self.assertEqual(parse_number("1.250.000"), (Fraction(1250000), False))
        self.assertEqual(parse_number("3.5"), (Fraction(7, 2), False))
        self.assertEqual(parse_number("1,250,000"), (Fraction(1250000), False))
        self.assertEqual(parse_number("-3/4"), (Fraction(-3, 4), False))
        self.assertEqual(parse_number("%25"), (Fraction(25), True))
        self.assertIsNone(parse_number("abc"))
        self.assertIsNone(parse_number("3/0"))

    def test_match(self):
        self.assertTrue(answers_match("0,75", "3/4"))
        self.assertTrue(answers_match("6/8", "3/4"))
        self.assertTrue(answers_match("1250", "1.250"))
        self.assertTrue(answers_match("25", "%25"))
        self.assertTrue(answers_match("0,25", "%25"))
        self.assertTrue(answers_match("%50", "0,5"))
        self.assertTrue(answers_match("-7", -7))
        self.assertTrue(answers_match("Cevap: 12,5 TL", 12.5))
        self.assertFalse(answers_match("0,33", "1/3"))
        self.assertTrue(answers_match("0,33", "1/3", tol=0.02))
        self.assertFalse(answers_match("13", "12"))
        self.assertFalse(answers_match(None, "1"))


class TestRewards(unittest.TestCase):

    def test_correctness(self):
        self.assertEqual(correctness_reward("<düşünce>x</düşünce>\nCevap: 3/4", "0,75"), 1.0)
        self.assertEqual(correctness_reward("Cevap: 5", "6"), 0.0)
        self.assertEqual(correctness_reward("cevap yok", "6"), 0.0)

    def test_format(self):
        good = format_reasoning("2 + 2 = 4", "4")
        self.assertEqual(format_reward(good), 1.0)
        self.assertEqual(format_reward("<düşünce> </düşünce>\nCevap: 4"), 0.5)
        self.assertEqual(format_reward("<düşünce>a</düşünce>\nCevap: 4\nfazladan\nsatır"), 0.5)
        self.assertEqual(format_reward("Cevap: 4"), 0.25)
        self.assertEqual(format_reward("4"), 0.0)

    def test_length(self):
        self.assertEqual(length_penalty("a" * 10, soft_limit=20, hard_limit=40), 0.0)
        self.assertAlmostEqual(length_penalty("a" * 30, soft_limit=20, hard_limit=40), -0.5)
        self.assertEqual(length_penalty(100, soft_limit=20, hard_limit=40), -1.0)

    def test_language(self):
        self.assertEqual(language_reward("Çözüm: ağaç, şeker, ılık, öğün"), 1.0)
        self.assertEqual(language_reward("3x + 4 = 10"), 1.0)
        self.assertLess(language_reward("3x + 4", allow_qwx=False), 1.0)
        self.assertLess(language_reward("Привет мир это тест"), 0.1)
        self.assertAlmostEqual(language_reward("abcé"), 0.75)
        self.assertEqual(language_reward("12345"), 0.0)

    def test_combine(self):
        reward = combine_rewards({"correctness": 1.0, "format": 0.2})
        text = format_reasoning("2 + 2 = 4", "4")
        self.assertAlmostEqual(reward(text, "4"), 1.2)
        parts = reward.detailed("Cevap: 5", "4")
        self.assertEqual(parts["correctness"], 0.0)
        self.assertEqual(parts["format"], 0.25)
        self.assertAlmostEqual(parts["total"], 0.05)
        with self.assertRaises(ValueError):
            combine_rewards({"bilinmeyen": 1.0})
        custom = combine_rewards({"bir": 2.0}, functions={"bir": lambda c, gold=None, **_: 1.0})
        self.assertEqual(custom("x"), 2.0)


if __name__ == "__main__":
    unittest.main()
