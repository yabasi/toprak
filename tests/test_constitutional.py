# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Toprak Anayasası ve anayasaya dayalı öz-düzeltme testleri."""

import json
import os
import random
import re
import tempfile
import unittest

from alignment.constitutional import (
    ConstitutionalReviser,
    Principle,
    build_preference_pairs,
    build_sft_records,
    load_principles,
    read_prompts,
)
from training.dpo import PreferenceDataset
from training.sft import SFTDataset, read_jsonl, write_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGN = os.path.join(ROOT, "alignment")
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


class FakeGenerator:
    """Deterministik sahte model: çağrıları kaydeder, rolüne göre cevap verir."""

    def __init__(self, revise=True):
        self.calls = []
        self.revise = revise

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        last = messages[-1]["content"]
        if last.startswith("Eleştiri isteği: "):
            return f"eleştiri#{len(self.calls)}"
        if last.startswith("Düzeltme isteği: "):
            previous = messages[-4]["content"]
            return f"{previous} +düzeltme" if self.revise else previous
        return f"ilk cevap: {last}"


class TestConstitutionFiles(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(ALIGN, "constitution.json"), encoding="utf-8") as f:
            self.data = json.load(f)
        with open(os.path.join(ALIGN, "constitution.md"), encoding="utf-8") as f:
            self.md = f.read()

    def test_json_schema(self):
        principles = self.data["principles"]
        self.assertTrue(12 <= len(principles) <= 15)
        ids = [p["id"] for p in principles]
        self.assertEqual(len(ids), len(set(ids)))
        for p in principles:
            for key in ("id", "title", "critique_prompt", "revision_prompt"):
                self.assertTrue(p[key].strip(), f"{p['id']} {key} boş")
            self.assertTrue(p["critique_prompt"].rstrip().endswith("?"))

    def test_markdown_in_sync_with_json(self):
        headings = re.findall(r"^### (\d+)\. (.+) \(`([^`]+)`\)$", self.md, flags=re.M)
        critiques = re.findall(r"^- \*\*Eleştiri sorusu:\*\* (.+)$", self.md, flags=re.M)
        revisions = re.findall(r"^- \*\*Düzeltme talimatı:\*\* (.+)$", self.md, flags=re.M)
        principles = self.data["principles"]
        self.assertEqual([int(n) for n, _, _ in headings], list(range(1, len(principles) + 1)))
        self.assertEqual([(t, i) for _, t, i in headings], [(p["title"], p["id"]) for p in principles])
        self.assertEqual(critiques, [p["critique_prompt"] for p in principles])
        self.assertEqual(revisions, [p["revision_prompt"] for p in principles])

    def test_required_topics_covered(self):
        ids = {p["id"] for p in self.data["principles"]}
        for required in ("dogruluk", "bilinmezlik", "zarar", "tarafsizlik", "saygi", "dil_kalitesi",
                         "mahremiyet", "uzman_yonlendirme", "cocuk_guvenligi",
                         "kulturel_duyarlilik", "seffaflik"):
            self.assertIn(required, ids)

    def test_load_principles(self):
        principles = load_principles()
        self.assertEqual(len(principles), len(self.data["principles"]))
        self.assertIsInstance(principles[0], Principle)


class TestReviser(unittest.TestCase):

    def test_trajectory_structure(self):
        gen = FakeGenerator()
        reviser = ConstitutionalReviser(gen, load_principles(), rng=random.Random(3), num_principles=2)
        traj = reviser.revise("Ankara neresi?")
        self.assertEqual(traj["initial"], "ilk cevap: Ankara neresi?")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(traj["final"], "ilk cevap: Ankara neresi? +düzeltme +düzeltme")
        self.assertEqual(len(gen.calls), 1 + 2 * 2)
        # Eleştiri çağrısı: prompt + mevcut cevap + eleştiri isteği
        critique_call = gen.calls[1]
        self.assertEqual([m["role"] for m in critique_call], ["user", "assistant", "user"])
        principle = next(p for p in reviser.principles if p.id == traj["steps"][0]["principle"])
        self.assertIn(principle.critique_prompt, critique_call[-1]["content"])
        revision_call = gen.calls[2]
        self.assertEqual(revision_call[-2]["content"], "eleştiri#2")
        self.assertIn(principle.revision_prompt, revision_call[-1]["content"])

    def test_principle_sampling_is_seeded(self):
        a = ConstitutionalReviser(FakeGenerator(), rng=random.Random(7), num_principles=3)
        b = ConstitutionalReviser(FakeGenerator(), rng=random.Random(7), num_principles=3)
        self.assertEqual([p.id for p in a.sample_principles()], [p.id for p in b.sample_principles()])
        self.assertEqual(len({p.id for p in a.sample_principles()}), 3)

    def test_message_list_prompt(self):
        reviser = ConstitutionalReviser(FakeGenerator(), rng=random.Random(0), num_principles=1)
        prompt = [{"role": "system", "content": "kısa"}, {"role": "user", "content": "selam"}]
        traj = reviser.revise(prompt)
        self.assertEqual(traj["prompt"], prompt)
        self.assertTrue(traj["final"].endswith("+düzeltme"))


class TestPairs(unittest.TestCase):

    def test_pairs_and_sft_records_are_trainable(self):
        reviser = ConstitutionalReviser(FakeGenerator(), rng=random.Random(0), num_principles=2)
        trajectories = []
        pairs = build_preference_pairs(["bir", "iki"], reviser, trajectories=trajectories)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(trajectories), 2)
        for pair, traj in zip(pairs, trajectories):
            self.assertEqual(set(pair), {"prompt", "chosen", "rejected", "principles"})
            self.assertEqual(pair["chosen"], traj["final"])
            self.assertEqual(pair["rejected"], traj["initial"])
            self.assertEqual(len(pair["principles"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            pair_path = os.path.join(tmp, "pairs.jsonl")
            sft_path = os.path.join(tmp, "sft.jsonl")
            write_jsonl(pair_path, pairs)
            write_jsonl(sft_path, build_sft_records(pairs))
            self.assertEqual(read_jsonl(pair_path), pairs)
            self.assertEqual(len(PreferenceDataset(pair_path, MockTokenizer(), max_len=256)), 2)
            sft = SFTDataset(sft_path, MockTokenizer(), max_len=256)
            self.assertEqual(len(sft), 2)

    def test_unchanged_answers_are_skipped(self):
        reviser = ConstitutionalReviser(FakeGenerator(revise=False), rng=random.Random(0))
        self.assertEqual(build_preference_pairs(["bir"], reviser), [])
        self.assertEqual(len(build_preference_pairs(["bir"], reviser, keep_unchanged=True)), 1)

    def test_example_prompts_file(self):
        prompts = read_prompts(os.path.join(ALIGN, "examples", "prompts.txt"))
        self.assertGreaterEqual(len(prompts), 8)
        self.assertFalse(any(p.startswith("#") for p in prompts))


if __name__ == "__main__":
    unittest.main()
