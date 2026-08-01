# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Veri Temizleme Pipeline
Ham crawl verisini eğitime hazır hâle getirir.
"""

import hashlib
import json
import os
import re
import sys
import unicodedata
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.governance import (
    SCHEMA_VERSION,
    ContaminationDetector,
    NearDuplicateIndex,
    PIIRedactor,
    QualityScorer,
    enrich_document,
    utc_now_iso,
)


class ToprakCleaner:
    """
    Türkçe metin temizleme pipeline.

    Adımlar:
    1. HTML artıklarını kaldır
    2. Unicode normalizasyonu (NFKC)
    3. Boilerplate filtre
    4. Minimum kelime sayısı kontrolü
    5. PII redaction
    6. Açıklanabilir kalite skoru
    7. Exact + near deduplication
    8. Benchmark contamination kontrolü
    9. Kaynak/lisans provenance metadata'sı
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100_000,
        quality_threshold: float = 0.50,
        near_duplicate_distance: int = 3,
        redact_pii: bool = True,
        benchmark_path: Optional[str] = None,
        contamination_action: str = "reject",
    ):
        if contamination_action not in ("reject", "flag"):
            raise ValueError("contamination_action 'reject' veya 'flag' olmalı")
        self.min_words = min_words
        self.max_words = max_words
        self.quality_threshold = quality_threshold
        self.redact_pii = redact_pii
        self.contamination_action = contamination_action
        self.quality_scorer = QualityScorer()
        self.pii_redactor = PIIRedactor()
        self.dedup_index = NearDuplicateIndex(near_duplicate_distance)
        self.contamination_detector = (
            ContaminationDetector(benchmark_path) if benchmark_path else None
        )
        self.last_analysis = {}
        self.source_counts = {}
        self.license_status_counts = {}
        self.stats = {
            "total": 0,
            "accepted": 0,
            "too_short": 0,
            "too_long": 0,
            "duplicate": 0,
            "exact_duplicate": 0,
            "near_duplicate": 0,
            "bad_quality": 0,
            "pii_documents": 0,
            "pii_redactions": 0,
            "contaminated": 0,
        }

    def normalize_unicode(self, text: str) -> str:
        """Unicode normalizasyonu (NFKC) — Türkçe karakterleri korur."""
        return unicodedata.normalize("NFKC", text)

    def remove_html_artifacts(self, text: str) -> str:
        """HTML kalıntılarını temizle."""
        # HTML etiketleri
        text = re.sub(r"<[^>]+>", "", text)
        # HTML entities
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        return text

    def clean_whitespace(self, text: str) -> str:
        """Gereksiz boşlukları temizle."""
        # Birden fazla boş satırı tek satıra indir
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Satır başı/sonu boşlukları temizle
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        # Birden fazla boşluğu teke indir
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def remove_boilerplate(self, text: str) -> str:
        """Tipik web boilerplate ifadelerini kaldır."""
        boilerplate_patterns = [
            r"cookie.*?kabul",
            r"çerez.*?politika",
            r"gizlilik.*?sözleşme",
            r"tüm hakları saklıdır",
            r"all rights reserved",
            r"©\s*\d{4}",
            r"paylaş.*?(facebook|twitter|whatsapp)",
            r"yorum\s*yap",
            r"yorumlar\s*\(\d+\)",
            r"reklam",
            r"advertisement",
            r"loading\.\.\.",
            r"devamını\s*oku",
            r"daha\s*fazla\s*göster",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    def is_quality_text(self, text: str) -> bool:
        """Metin kalitesini kontrol et."""
        score, signals = self.quality_scorer.score(text)
        return (
            score >= self.quality_threshold
            and signals.get("alpha_ratio", 0.0) >= 0.5
            and signals.get("unique_word_ratio", 0.0) >= 0.1
            and 2 <= signals.get("average_word_length", 0.0) <= 25
        )

    def clean_text(self, text: str) -> Optional[str]:
        """Tek bir metni temizle."""
        self.stats["total"] += 1
        self.last_analysis = {}

        # 1. HTML artıkları
        text = self.remove_html_artifacts(text)

        # 2. Unicode normalizasyonu
        text = self.normalize_unicode(text)

        # 3. Boilerplate
        text = self.remove_boilerplate(text)

        # 4. Boşluk temizliği
        text = self.clean_whitespace(text)

        # 5. Yüksek kesinlikli PII redaction
        pii_redactions = {}
        if self.redact_pii:
            text, pii_redactions = self.pii_redactor.redact(text)
            if pii_redactions:
                self.stats["pii_documents"] += 1
                self.stats["pii_redactions"] += sum(pii_redactions.values())

        # 6. Kelime sayısı kontrolü
        word_count = len(text.split())
        if word_count < self.min_words:
            self.stats["too_short"] += 1
            return None
        if word_count > self.max_words:
            self.stats["too_long"] += 1
            return None

        # 7. Kalite kontrolü ve açıklanabilir skor
        quality_score, quality_signals = self.quality_scorer.score(text)
        if not (
            quality_score >= self.quality_threshold
            and quality_signals.get("alpha_ratio", 0.0) >= 0.5
            and quality_signals.get("unique_word_ratio", 0.0) >= 0.1
            and 2 <= quality_signals.get("average_word_length", 0.0) <= 25
        ):
            self.stats["bad_quality"] += 1
            return None

        # 8. Benchmark contamination
        contamination_matches = []
        if self.contamination_detector is not None:
            contamination_matches = self.contamination_detector.find_matches(text)
            if contamination_matches:
                self.stats["contaminated"] += 1
                if self.contamination_action == "reject":
                    return None

        # 9. Exact + SimHash near dedup
        is_duplicate, duplicate_type, text_hash, simhash = (
            self.dedup_index.check_and_add(text)
        )
        if is_duplicate:
            self.stats["duplicate"] += 1
            self.stats[f"{duplicate_type}_duplicate"] += 1
            return None

        self.last_analysis = {
            "quality_score": quality_score,
            "quality_signals": quality_signals,
            "pii_redactions": pii_redactions,
            "content_hash": text_hash,
            "simhash": simhash,
            "contamination_matches": contamination_matches,
        }

        self.stats["accepted"] += 1
        return text

    def clean_document(self, document: dict, source_file: Optional[str] = None) -> Optional[dict]:
        """Dokümanı temizle ve standart yönetişim metadata'sıyla zenginleştir."""
        cleaned = self.clean_text(document.get("text", ""))
        if cleaned is None:
            return None
        result = enrich_document(
            document,
            cleaned,
            quality_score=self.last_analysis["quality_score"],
            quality_signals=self.last_analysis["quality_signals"],
            pii_redactions=self.last_analysis["pii_redactions"],
            content_hash=self.last_analysis["content_hash"],
            simhash=self.last_analysis["simhash"],
            contamination_matches=self.last_analysis["contamination_matches"],
            source_file=source_file,
        )
        source = result["source"]
        status = result["license_status"]
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.license_status_counts[status] = self.license_status_counts.get(status, 0) + 1
        return result

    def clean_jsonl(self, input_file: str, output_file: str):
        """JSONL dosyasını temizle."""
        print(f"Temizleniyor: {input_file}")

        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    doc = json.loads(line)
                    cleaned_doc = self.clean_document(doc, source_file=input_file)
                    if cleaned_doc:
                        json.dump(cleaned_doc, fout, ensure_ascii=False)
                        fout.write("\n")
                except json.JSONDecodeError:
                    continue

        self.print_stats()

    def clean_directory(self, input_dir: str, output_dir: str):
        """Bir dizindeki tüm JSONL dosyalarını temizle."""
        os.makedirs(output_dir, exist_ok=True)

        output_files = []
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".jsonl"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"clean_{filename}")
                self.clean_jsonl(input_path, output_path)
                output_files.append(output_path)

        self.write_manifest(output_dir, output_files)

    def write_manifest(self, output_dir: str, output_files: List[str]) -> str:
        """Temiz korpusun üretim ayarlarını ve dosya hashlerini kaydet."""
        files = []
        for path in output_files:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append({
                "path": os.path.relpath(path, output_dir),
                "bytes": os.path.getsize(path),
                "sha256": digest.hexdigest(),
            })

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "cleaner": {
                "min_words": self.min_words,
                "max_words": self.max_words,
                "quality_threshold": self.quality_threshold,
                "near_duplicate_distance": self.dedup_index.max_hamming_distance,
                "pii_redaction": self.redact_pii,
                "contamination_action": self.contamination_action,
                "contamination_enabled": self.contamination_detector is not None,
            },
            "stats": dict(self.stats),
            "source_counts": dict(sorted(self.source_counts.items())),
            "license_status_counts": dict(sorted(self.license_status_counts.items())),
            "files": files,
        }
        manifest_path = os.path.join(output_dir, "corpus_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"✓ Korpus manifesti: {manifest_path}")
        return manifest_path

    def prepare_tokenizer_data(self, input_dir: str, output_file: str):
        """
        Temizlenmiş verileri tokenizer eğitimi için
        düz metin dosyasına dönüştür.
        """
        print(f"Tokenizer verisi hazırlanıyor: {output_file}")
        total_lines = 0

        with open(output_file, "w", encoding="utf-8") as fout:
            for filename in os.listdir(input_dir):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(input_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as fin:
                        for line in fin:
                            try:
                                doc = json.loads(line)
                                text = doc.get("text", "").strip()
                                if text:
                                    fout.write(text + "\n")
                                    total_lines += 1
                            except json.JSONDecodeError:
                                continue

        print(f"✓ {total_lines} döküman yazıldı: {output_file}")

    def print_stats(self):
        """Temizleme istatistiklerini yazdır."""
        print(f"\n{'='*40}")
        print("Temizleme İstatistikleri")
        print(f"{'='*40}")
        print(f"  Toplam:      {self.stats['total']}")
        print(f"  Kabul:       {self.stats['accepted']}")
        print(f"  Çok Kısa:    {self.stats['too_short']}")
        print(f"  Çok Uzun:    {self.stats['too_long']}")
        print(f"  Duplikat:    {self.stats['duplicate']}")
        print(f"    Exact:     {self.stats['exact_duplicate']}")
        print(f"    Near:      {self.stats['near_duplicate']}")
        print(f"  Düşük Kalite:{self.stats['bad_quality']}")
        print(f"  PII Belgesi: {self.stats['pii_documents']}")
        print(f"  PII Redact:  {self.stats['pii_redactions']}")
        print(f"  Contaminated:{self.stats['contaminated']}")
        accepted_pct = (
            self.stats["accepted"] / max(self.stats["total"], 1) * 100
        )
        print(f"  Kabul Oranı: {accepted_pct:.1f}%")
        print(f"{'='*40}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toprak — Veri Temizleme")
    parser.add_argument("--input", type=str, default="data_cache",
                        help="Girdi dizini (JSONL dosyaları)")
    parser.add_argument("--output", type=str, default="data_cache/clean",
                        help="Çıktı dizini")
    parser.add_argument("--tokenizer-data", type=str, default=None,
                        help="Tokenizer eğitim dosyası (düz metin)")
    parser.add_argument("--benchmark-path", type=str, default=None,
                        help="Contamination kontrolü için benchmark JSONL/TXT veya dizin")
    parser.add_argument("--contamination-action", choices=["reject", "flag"],
                        default="reject", help="Eşleşen benchmark metinlerini reddet veya işaretle")
    parser.add_argument("--quality-threshold", type=float, default=0.50)
    parser.add_argument("--no-pii-redaction", action="store_true")

    args = parser.parse_args()
    cleaner = ToprakCleaner(
        quality_threshold=args.quality_threshold,
        redact_pii=not args.no_pii_redaction,
        benchmark_path=args.benchmark_path,
        contamination_action=args.contamination_action,
    )

    cleaner.clean_directory(args.input, args.output)

    if args.tokenizer_data:
        cleaner.prepare_tokenizer_data(args.output, args.tokenizer_data)
