# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Sentetik Türkçe matematik üreticisi testleri: sayı ekleri, determinizm,
cevapların bağımsız yeniden hesaplanması ve train/test ayrımı."""

import json
import math
import os
import random
import re
import tempfile
import unittest
from fractions import Fraction

from data.synthetic_math import (
    LEVELS,
    TEMPLATES,
    canonical_answer,
    format_number,
    generate_dataset,
    generate_problem,
    main,
    name_suffix,
    number_suffix,
    sft_target,
    split_train_test,
    to_sft_record,
)
from training.rewards import answers_match, correctness_reward, extract_answer, format_reward


def _num(text):
    return int(text.replace(".", ""))


class TestNumberSuffix(unittest.TestCase):

    def test_dative(self):
        for n, expected in [(5, "5'e"), (6, "6'ya"), (10, "10'a"), (100, "100'e"),
                            (40, "40'a"), (2, "2'ye"), (9, "9'a"), (1000, "1.000'e"),
                            (60, "60'a"), (0, "0'a"), (3, "3'e"), (4, "4'e")]:
            self.assertEqual(number_suffix(n, "dat"), expected)

    def test_accusative(self):
        for n, expected in [(3, "3'ü"), (2, "2'yi"), (9, "9'u"), (60, "60'ı"),
                            (6, "6'yı"), (5, "5'i"), (4, "4'ü"), (10, "10'u"), (20, "20'yi")]:
            self.assertEqual(number_suffix(n, "acc"), expected)

    def test_other_cases(self):
        self.assertEqual(number_suffix(4, "loc"), "4'te")
        self.assertEqual(number_suffix(6, "loc"), "6'da")
        self.assertEqual(number_suffix(40, "loc"), "40'ta")
        self.assertEqual(number_suffix(3, "abl"), "3'ten")
        self.assertEqual(number_suffix(1, "abl"), "1'den")
        self.assertEqual(number_suffix(2, "gen"), "2'nin")
        self.assertEqual(number_suffix(6, "gen"), "6'nın")
        self.assertEqual(number_suffix(5, "ins"), "5'le")
        self.assertEqual(number_suffix(6, "ins"), "6'yla")
        self.assertEqual(number_suffix(24, "cop"), "24'tür")
        self.assertEqual(number_suffix(30, "cop"), "30'dur")
        self.assertEqual(number_suffix(12, "poss3"), "12'si")
        self.assertEqual(number_suffix(6, "poss3"), "6'sı")
        self.assertEqual(number_suffix(5, "yönelme"), "5'e")  # Türkçe hâl adı

    def test_formatted_strings_fractions_decimals(self):
        self.assertEqual(number_suffix("3/5", "poss3"), "3/5'ü")       # beşte üçü
        self.assertEqual(number_suffix("1/4", "poss3_acc"), "1/4'ini")  # dörtte birini
        self.assertEqual(number_suffix("2/3", "poss3_acc"), "2/3'sini")
        self.assertEqual(number_suffix("12,5", "dat"), "12,5'e")
        self.assertEqual(number_suffix(Fraction(25, 2), "dat"), "12,5'e")
        self.assertEqual(number_suffix(1250, "dat"), "1.250'ye")       # elli
        self.assertEqual(number_suffix(1_000_000, "loc"), "1.000.000'da")  # milyon

    def test_invalid_case(self):
        with self.assertRaises(ValueError):
            number_suffix(5, "yok")

    def test_names(self):
        self.assertEqual(name_suffix("Ali", "gen"), "Ali'nin")
        self.assertEqual(name_suffix("Can", "dat"), "Can'a")
        self.assertEqual(name_suffix("Mehmet", "abl"), "Mehmet'ten")
        self.assertEqual(name_suffix("Oğuz", "gen"), "Oğuz'un")
        self.assertEqual(name_suffix("Ayşe", "dat"), "Ayşe'ye")


class TestFormatting(unittest.TestCase):

    def test_format_number(self):
        self.assertEqual(format_number(1250), "1.250")
        self.assertEqual(format_number(Fraction(25, 2)), "12,5")
        self.assertEqual(format_number(Fraction(1, 3)), "1/3")
        self.assertEqual(format_number(Fraction(1, 2), prefer_fraction=True), "1/2")
        self.assertEqual(format_number(Fraction(-7, 4)), "-1,75")
        self.assertEqual(format_number(Fraction(1, 20)), "0,05")

    def test_canonical(self):
        self.assertEqual(canonical_answer(Fraction(42), "auto"), ("42", 42))
        self.assertEqual(canonical_answer(Fraction(5, 2), "auto"), ("2,5", 2.5))
        self.assertEqual(canonical_answer(Fraction(5, 2), "fraction"), ("5/2", "5/2"))
        self.assertEqual(canonical_answer(Fraction(25), "percent"), ("%25", 25))


class TestGenerator(unittest.TestCase):

    def test_deterministic(self):
        a = generate_dataset(50, seed=7)
        b = generate_dataset(50, seed=7)
        c = generate_dataset(50, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual([x["question"] for x in a], [x["question"] for x in c])

    def test_all_templates_and_levels(self):
        rng = random.Random(0)
        for name in TEMPLATES:
            for level in LEVELS:
                for _ in range(20):
                    item = generate_problem(rng, name, level)
                    with self.subTest(template=name, level=level, q=item["question"]):
                        self.assertEqual(set(item), {"question", "answer", "answer_value",
                                                     "level", "template", "solution"})
                        self.assertTrue(answers_match(item["answer"], str(item["answer_value"])
                                                      if not isinstance(item["answer_value"], float)
                                                      else item["answer_value"]))
                        self.assertTrue(answers_match(extract_answer(item["solution"]), item["answer"]))
                        target = sft_target(item)
                        self.assertEqual(format_reward(target), 1.0)
                        self.assertEqual(correctness_reward(target, item["answer"]), 1.0)
                        # sayıdan sonra çoğul ad olmamalı ("5 elmalar" değil)
                        self.assertIsNone(re.search(r"\b\d+ [a-zçğıöşü]+l[ae]r\b", item["question"]))
                        self.assertTrue(item["question"].endswith("?"))

    def test_record_fields_and_ids(self):
        items = generate_dataset(30, seed=3, templates=["denklem"], levels=[1])
        for item in items:
            self.assertEqual(item["template"], "denklem")
            self.assertEqual(item["level"], 1)
            self.assertTrue(item["id"].startswith("sm-3-"))
        self.assertEqual(len({x["id"] for x in items}), 30)
        with self.assertRaises(ValueError):
            generate_dataset(1, seed=0, templates=["yok"])


class TestIndependentRecomputation(unittest.TestCase):
    """Cevapları sorudaki sayılardan bağımsız olarak yeniden hesapla."""

    def _items(self, template, level, n=40):
        return generate_dataset(n, seed=11, templates=[template], levels=[level])

    def test_bolunebilme(self):
        for item in self._items("bolunebilme", 1):
            n, k = map(int, re.search(r"1'den (\d+)'.*?tanesi (\d+) ile", item["question"]).groups())
            self.assertEqual(item["answer_value"], sum(1 for x in range(1, n + 1) if x % k == 0))
        for item in self._items("bolunebilme", 3):
            n, p, q = map(int, re.search(r"1'den (\d+)'.*?tanesi (\d+)'.*? ya da (\d+)'", item["question"]).groups())
            self.assertEqual(item["answer_value"],
                             sum(1 for x in range(1, n + 1) if x % p == 0 or x % q == 0))

    def test_denklem(self):
        for item in self._items("denklem", 2):
            a, s1, b, c, s2, d = re.match(r"(\d+)x ([+−]) (\d+) = (\d+)x ([+−]) (\d+)", item["question"]).groups()
            b = int(b) * (1 if s1 == "+" else -1)
            d = int(d) * (1 if s2 == "+" else -1)
            x = item["answer_value"]
            self.assertEqual(int(a) * x + b, int(c) * x + d)

    def test_olasilik_dice(self):
        for item in self._items("olasilik", 2, 60):
            m = re.search(r"toplamının (\d+) olma", item["question"])
            if not m:
                continue
            s = int(m.group(1))
            ways = sum(1 for i in range(1, 7) for j in range(1, 7) if i + j == s)
            self.assertEqual(Fraction(item["answer"]), Fraction(ways, 36))

    def test_alisveris_and_isci(self):
        for item in self._items("alisveris", 2):
            nums = [_num(x) for x in re.findall(r"\d[\d.]*", item["question"])]
            p1, q1, p2, q2, paid = nums
            self.assertEqual(item["answer_value"], paid - (p1 * q1 + p2 * q2))
        for item in self._items("isci_havuz", 2):
            x, y = map(int, re.search(r"(\d+) günde, \w+ ise (\d+) günde", item["question"]).groups())
            self.assertTrue(answers_match(item["answer"], Fraction(x * y, x + y)))

    def test_sayi_dizisi(self):
        for item in self._items("sayi_dizisi", 2):
            terms = [int(t) for t in re.findall(r"-?\d+", item["question"].split("…")[0])]
            n = int(re.search(r"(\d+)\. terimi", item["question"]).group(1))
            d = terms[1] - terms[0]
            self.assertEqual(item["answer_value"], terms[0] + (n - 1) * d)


class TestSplitAndCLI(unittest.TestCase):

    def test_split_disjoint(self):
        train, test = split_train_test(200, 50, seed=1)
        self.assertEqual(len(test), 50)
        self.assertFalse({x["question"] for x in train} & {x["question"] for x in test})
        self.assertTrue(all(x["id"].startswith("test-") for x in test))
        with self.assertRaises(ValueError):
            split_train_test(5, 5, seed=1, test_seed=1)

    def test_sft_record(self):
        item = generate_dataset(1, seed=0)[0]
        rec = to_sft_record(item)
        self.assertEqual([m["role"] for m in rec["messages"]], ["system", "user", "assistant"])
        self.assertIn("<düşünce>", rec["messages"][2]["content"])

    def test_cli_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--output-dir", tmp, "--train", "20", "--test", "5", "--seed", "3"])
            with open(os.path.join(tmp, "train.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 20)
            self.assertIn("answer_value", rows[0])
            main(["--output-dir", tmp, "--train", "4", "--test", "2", "--format", "sft"])
            with open(os.path.join(tmp, "test.jsonl"), encoding="utf-8") as f:
                self.assertIn("messages", json.loads(f.readline()))


if __name__ == "__main__":
    unittest.main()
