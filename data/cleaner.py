# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Toprak — Veri Temizleme Pipeline
Ham crawl verisini eğitime hazır hâle getirir.
"""

import hashlib
import json
import os
import re
import unicodedata
from typing import List, Optional


class ToprakCleaner:
    """
    Türkçe metin temizleme pipeline.

    Adımlar:
    1. HTML artıklarını kaldır
    2. Unicode normalizasyonu (NFKC)
    3. Boilerplate filtre
    4. Minimum kelime sayısı kontrolü
    5. Deduplication (hash-based)
    """

    def __init__(self, min_words: int = 50, max_words: int = 100_000):
        self.min_words = min_words
        self.max_words = max_words
        self.seen_hashes: set = set()
        self.stats = {
            "total": 0,
            "accepted": 0,
            "too_short": 0,
            "too_long": 0,
            "duplicate": 0,
            "bad_quality": 0,
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
        words = text.split()
        if len(words) == 0:
            return False

        # Çok fazla özel karakter
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return False

        # Çok fazla tekrar eden kelime
        unique_words = set(w.lower() for w in words)
        if len(unique_words) / len(words) < 0.1:
            return False

        # Ortalama kelime uzunluğu kontrolü
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 25:
            return False

        return True

    def get_text_hash(self, text: str) -> str:
        """Metin hash'i oluştur (dedup için)."""
        # Boşlukları normalize et ve küçük harfe dönüştür
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def clean_text(self, text: str) -> Optional[str]:
        """Tek bir metni temizle."""
        self.stats["total"] += 1

        # 1. HTML artıkları
        text = self.remove_html_artifacts(text)

        # 2. Unicode normalizasyonu
        text = self.normalize_unicode(text)

        # 3. Boilerplate
        text = self.remove_boilerplate(text)

        # 4. Boşluk temizliği
        text = self.clean_whitespace(text)

        # 5. Kelime sayısı kontrolü
        word_count = len(text.split())
        if word_count < self.min_words:
            self.stats["too_short"] += 1
            return None
        if word_count > self.max_words:
            self.stats["too_long"] += 1
            return None

        # 6. Kalite kontrolü
        if not self.is_quality_text(text):
            self.stats["bad_quality"] += 1
            return None

        # 7. Dedup
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            self.stats["duplicate"] += 1
            return None
        self.seen_hashes.add(text_hash)

        self.stats["accepted"] += 1
        return text

    def clean_jsonl(self, input_file: str, output_file: str):
        """JSONL dosyasını temizle."""
        print(f"Temizleniyor: {input_file}")

        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    doc = json.loads(line)
                    cleaned = self.clean_text(doc.get("text", ""))
                    if cleaned:
                        doc["text"] = cleaned
                        doc["word_count"] = len(cleaned.split())
                        json.dump(doc, fout, ensure_ascii=False)
                        fout.write("\n")
                except json.JSONDecodeError:
                    continue

        self.print_stats()

    def clean_directory(self, input_dir: str, output_dir: str):
        """Bir dizindeki tüm JSONL dosyalarını temizle."""
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"clean_{filename}")
                self.clean_jsonl(input_path, output_path)

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
        print(f"  Düşük Kalite:{self.stats['bad_quality']}")
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

    args = parser.parse_args()
    cleaner = ToprakCleaner()

    cleaner.clean_directory(args.input, args.output)

    if args.tokenizer_data:
        cleaner.prepare_tokenizer_data(args.output, args.tokenizer_data)
