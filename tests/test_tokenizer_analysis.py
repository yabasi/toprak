# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import os
import tempfile
import unittest

from evaluation.tokenizer_analysis import (
    TURKISH_CHARS,
    analyze_tokenizer,
    build_analysis_report,
    load_documents,
    load_seed,
    normalize_roundtrip,
    render_markdown,
)
from model.tokenizer import ToprakTokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_PATH = os.path.join(PROJECT_ROOT, "evaluation", "tokenizer_seed.json")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "toprak_tokenizer.model")


class TestTokenizerAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ToprakTokenizer(TOKENIZER_PATH)
        cls.documents, cls.probes = load_seed(SEED_PATH)

    def test_current_tokenizer_seed_analysis_is_complete(self):
        analysis = analyze_tokenizer(
            self.tokenizer,
            self.documents,
            self.probes,
            "current",
            TOKENIZER_PATH,
        )
        self.assertEqual(analysis["corpus"]["documents"], len(self.documents))
        self.assertEqual(analysis["corpus"]["unknown_rate"], 0.0)
        self.assertEqual(analysis["corpus"]["roundtrip_exact_rate"], 1.0)
        self.assertEqual(analysis["morphology"]["forms"], 36)
        self.assertTrue(all(
            not analysis["turkish_character_coverage"][char]["unknown"]
            for char in TURKISH_CHARS
        ))

    def test_report_is_stable_and_renderable(self):
        analysis = analyze_tokenizer(
            self.tokenizer, self.documents[:2], self.probes, "current"
        )
        report = build_analysis_report(self.documents[:2], self.probes, [analysis])
        second = build_analysis_report(self.documents[:2], self.probes, [analysis])
        self.assertEqual(report["corpus_sha256"], second["corpus_sha256"])
        self.assertIn("Token/kelime", render_markdown(report))

    def test_document_loader_supports_jsonl_and_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "news.jsonl")
            txt_path = os.path.join(temp_dir, "notes.txt")
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"text": "Birinci belge", "domain": "news"}) + "\n")
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write("İkinci belge\n")
            documents = load_documents([temp_dir])
        self.assertEqual(len(documents), 2)
        self.assertEqual({item["domain"] for item in documents}, {"news", "notes"})

    def test_roundtrip_normalization_handles_nfkc_and_whitespace(self):
        self.assertEqual(normalize_roundtrip("  A\tB  "), "A B")
        self.assertEqual(normalize_roundtrip("Kâğıt"), "Kâğıt")


if __name__ == "__main__":
    unittest.main()
