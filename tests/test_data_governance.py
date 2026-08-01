# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

import json
import os
import tempfile
import unittest

from data.cleaner import ToprakCleaner
from data.governance import (
    ContaminationDetector,
    NearDuplicateIndex,
    PIIRedactor,
    content_sha256,
    source_profile,
)
from scripts.audit_corpus import audit_corpus


class TestDataGovernance(unittest.TestCase):
    def test_source_licenses_and_unknown_review(self):
        self.assertEqual(source_profile("fineweb2")["licenses"], ["ODC-By-1.0"])
        self.assertEqual(source_profile("wikipedia_tr")["source"], "wiki")
        self.assertEqual(
            source_profile("example.com")["license_status"],
            "review_required",
        )

    def test_high_precision_pii_redaction(self):
        text = (
            "E-posta test@example.com, telefon 0532 123 45 67, "
            "IP 192.168.1.1 ve TCKN 10000000146."
        )
        redacted, counts = PIIRedactor().redact(text)
        self.assertNotIn("test@example.com", redacted)
        self.assertNotIn("0532 123 45 67", redacted)
        self.assertNotIn("192.168.1.1", redacted)
        self.assertNotIn("10000000146", redacted)
        self.assertEqual(counts, {
            "EMAIL": 1,
            "IP_ADDRESS": 1,
            "PHONE": 1,
            "TCKN": 1,
        })

    def test_exact_and_near_duplicate_detection(self):
        index = NearDuplicateIndex(max_hamming_distance=3)
        text = "Bu bir örnek metindir ve çeşitli anlamlı kelimeler içerir"
        self.assertFalse(index.check_and_add(text)[0])
        self.assertEqual(index.check_and_add(text)[1], "exact")
        self.assertEqual(index.check_and_add(text + "!")[1], "near")

    def test_contamination_fingerprint_match(self):
        benchmark = (
            "Türkiye ekonomisi üzerine hazırlanan bu özel benchmark sorusu "
            "yalnız değerlendirme amacıyla kullanılan gizli bir metindir"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "benchmark.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "bench-1", "text": benchmark}) + "\n")
            detector = ContaminationDetector(path, ngram_size=5, min_hits=2)
            matches = detector.find_matches("Giriş cümlesi. " + benchmark + " Son cümle.")
            self.assertEqual(matches, ["bench-1:text"])

    def test_cleaner_rejects_contaminated_document(self):
        benchmark = (
            "Bu değerlendirme sorusu eğitim verisine kesinlikle alınmaması gereken "
            "özgün ve yeterince uzun bir benchmark metni olarak hazırlanmıştır"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "benchmark.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(benchmark + "\n")
            cleaner = ToprakCleaner(
                min_words=5,
                quality_threshold=0.0,
                benchmark_path=path,
                contamination_action="reject",
            )
            self.assertIsNone(cleaner.clean_text("Başlangıç. " + benchmark + " Bitiş."))
            self.assertEqual(cleaner.stats["contaminated"], 1)

    def test_evaluation_prompt_fields_are_fingerprinted(self):
        detector = ContaminationDetector(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "evaluation",
                "benchmarks",
            ),
            ngram_size=5,
            min_hits=2,
        )
        matches = detector.find_matches(
            "Elif sabah erkenden kütüphaneye gitti. Öğlene kadar tarih kitabını "
            "okudu, ardından aldığı notları düzenledi."
        )
        self.assertTrue(any(match.startswith("reading-001:prompt") for match in matches))

    def test_clean_directory_writes_traceable_manifest(self):
        text = (
            "Bu belge veri kalitesi sistemini sınamak için hazırlanmış uzun ve "
            "anlamlı bir Türkçe metindir. İçerisinde farklı sözcükler, açıklayıcı "
            "cümleler ve test@example.com adresi bulunmaktadır. Eğitim korpusuna "
            "girmeden önce kişisel bilgi temizlenmeli ve kaynak bilgisi korunmalıdır."
        )
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = os.path.join(directory, "raw")
            clean_dir = os.path.join(directory, "clean")
            os.makedirs(raw_dir)
            raw_path = os.path.join(raw_dir, "fineweb.jsonl")
            with open(raw_path, "w", encoding="utf-8") as handle:
                for _ in range(2):
                    handle.write(json.dumps({"text": text, "source": "fineweb2"}) + "\n")

            cleaner = ToprakCleaner(min_words=10)
            cleaner.clean_directory(raw_dir, clean_dir)

            output_path = os.path.join(clean_dir, "clean_fineweb.jsonl")
            with open(output_path, "r", encoding="utf-8") as handle:
                documents = [json.loads(line) for line in handle]
            self.assertEqual(len(documents), 1)
            document = documents[0]
            self.assertEqual(document["license_status"], "verified_dataset_card")
            self.assertEqual(document["pii_redactions"], {"EMAIL": 1})
            self.assertEqual(document["content_sha256"], content_sha256(document["text"]))

            manifest_path = os.path.join(clean_dir, "corpus_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["stats"]["accepted"], 1)
            self.assertEqual(manifest["stats"]["duplicate"], 1)

            audit = audit_corpus(clean_dir)
            self.assertEqual(audit["totals"].get("missing_metadata", 0), 0)
            self.assertEqual(audit["totals"].get("hash_mismatch", 0), 0)


if __name__ == "__main__":
    unittest.main()
