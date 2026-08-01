# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Veri kalitesi, izlenebilirlik, PII ve contamination yardımcıları."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "toprak-document-v1"

# Lisanslar kaynakların resmi veri kartlarına göre kaydedilir. Ham web
# kaynakları için otomatik bir lisans varsayılmaz; insan incelemesi gerekir.
SOURCE_REGISTRY = {
    "wiki": {
        "dataset_id": "wikimedia/wikipedia",
        "dataset_revision": "20231101.tr",
        "licenses": ["GFDL-unspecified", "CC-BY-SA-3.0"],
        "license_status": "verified_dataset_card",
        "license_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    "fineweb2": {
        "dataset_id": "HuggingFaceFW/fineweb-2",
        "dataset_revision": "main",
        "licenses": ["ODC-By-1.0"],
        "license_status": "verified_dataset_card",
        "license_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "terms_url": "https://commoncrawl.org/terms-of-use",
    },
    "culturax": {
        "dataset_id": "uonlp/CulturaX",
        "dataset_revision": "main",
        "licenses": ["UPSTREAM-mC4", "UPSTREAM-OSCAR"],
        "license_status": "inherited_review_required",
        "license_url": "https://huggingface.co/datasets/uonlp/CulturaX",
    },
    "sample": {
        "dataset_id": "toprak/generated-samples",
        "dataset_revision": "local",
        "licenses": ["Apache-2.0"],
        "license_status": "project_owned",
        "license_url": "LICENSE",
    },
}

SOURCE_ALIASES = {
    "wikipedia": "wiki",
    "wikipedia_tr": "wiki",
    "wikimedia/wikipedia": "wiki",
}


def utc_now_iso() -> str:
    """Saniye hassasiyetli UTC ISO-8601 zamanı."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_for_hash(text: str) -> str:
    """Dedup ve contamination için kararlı metin normalizasyonu."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def source_profile(source: str) -> dict:
    """Bilinen veri setini veya inceleme gerektiren güvenli varsayılanı döndür."""
    normalized = SOURCE_ALIASES.get(source, source)
    profile = SOURCE_REGISTRY.get(normalized)
    if profile is None:
        profile = {
            "dataset_id": source or "unknown",
            "dataset_revision": "unknown",
            "licenses": [],
            "license_status": "review_required",
            "license_url": None,
        }
    return {"source": normalized, **profile}


def build_provenance(
    source: str,
    *,
    source_url: Optional[str] = None,
    downloaded_at: Optional[str] = None,
    source_record_id: Optional[str] = None,
) -> dict:
    profile = source_profile(source)
    return {
        "schema_version": SCHEMA_VERSION,
        **profile,
        "source_url": source_url,
        "source_record_id": source_record_id,
        "downloaded_at": downloaded_at,
        "ingested_at": utc_now_iso(),
    }


class QualityScorer:
    """Açıklanabilir sinyallerden 0–1 arası doküman kalite skoru üretir."""

    def score(self, text: str) -> Tuple[float, dict]:
        words = text.split()
        if not words:
            return 0.0, {"word_count": 0}

        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        unique_ratio = len({w.casefold() for w in words}) / len(words)
        avg_word_length = sum(len(w) for w in words) / len(words)
        lines = [line for line in text.splitlines() if line.strip()]
        short_line_ratio = (
            sum(len(line.split()) < 3 for line in lines) / max(len(lines), 1)
        )

        alpha_component = min(max((alpha_ratio - 0.35) / 0.4, 0.0), 1.0)
        unique_component = min(unique_ratio / 0.5, 1.0)
        length_component = max(0.0, 1.0 - abs(avg_word_length - 6.0) / 12.0)
        layout_component = 1.0 - short_line_ratio
        score = (
            0.40 * alpha_component
            + 0.30 * unique_component
            + 0.20 * length_component
            + 0.10 * layout_component
        )
        signals = {
            "word_count": len(words),
            "alpha_ratio": round(alpha_ratio, 4),
            "unique_word_ratio": round(unique_ratio, 4),
            "average_word_length": round(avg_word_length, 4),
            "short_line_ratio": round(short_line_ratio, 4),
        }
        return round(score, 4), signals


class PIIRedactor:
    """Yüksek kesinlikli kişisel veri örüntülerini yer tutucularla değiştirir."""

    EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
    IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
    PHONE_RE = re.compile(
        r"(?<!\d)(?:\+?90[\s().-]*)?(?:0?[2-5]\d{2})[\s().-]*"
        r"\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
    )
    TCKN_RE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")

    @staticmethod
    def _valid_tckn(value: str) -> bool:
        digits = [int(c) for c in value]
        return (
            len(digits) == 11
            and ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
            == digits[9]
            and sum(digits[:10]) % 10 == digits[10]
        )

    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        counts: Dict[str, int] = {}

        def replace(pattern, label, value_validator=None):
            nonlocal text

            def repl(match):
                value = match.group(0)
                if value_validator is not None and not value_validator(value):
                    return value
                counts[label] = counts.get(label, 0) + 1
                return f"<{label}>"

            text = pattern.sub(repl, text)

        replace(self.EMAIL_RE, "EMAIL")
        replace(
            self.IPV4_RE,
            "IP_ADDRESS",
            lambda value: _is_valid_ipv4(value),
        )
        replace(self.PHONE_RE, "PHONE")
        replace(self.TCKN_RE, "TCKN", self._valid_tckn)
        return text, counts


def _is_valid_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except ipaddress.AddressValueError:
        return False


def _simhash64(text: str, max_features: int = 4096) -> int:
    words = re.findall(r"\w+", normalize_for_hash(text), flags=re.UNICODE)
    features: Iterable[str]
    if len(words) < 3:
        features = words
    else:
        features = (" ".join(words[i:i + 3]) for i in range(len(words) - 2))

    vector = [0] * 64
    for index, feature in enumerate(features):
        if index >= max_features:
            break
        value = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(vector) if weight >= 0)


class NearDuplicateIndex:
    """SHA-256 exact ve band-indexli SimHash near-duplicate tespiti."""

    def __init__(self, max_hamming_distance: int = 3):
        if not 0 <= max_hamming_distance <= 3:
            raise ValueError("max_hamming_distance 0–3 aralığında olmalı")
        self.max_hamming_distance = max_hamming_distance
        self.exact_hashes = set()
        self.simhashes: List[int] = []
        self.bands = [defaultdict(set) for _ in range(4)]

    def check_and_add(self, text: str) -> Tuple[bool, Optional[str], str, str]:
        sha256 = content_sha256(text)
        if sha256 in self.exact_hashes:
            return True, "exact", sha256, ""

        simhash = _simhash64(text)
        candidates = set()
        for band in range(4):
            key = (simhash >> (band * 16)) & 0xFFFF
            candidates.update(self.bands[band].get(key, ()))
        for candidate in candidates:
            if (simhash ^ self.simhashes[candidate]).bit_count() <= self.max_hamming_distance:
                return True, "near", sha256, f"{simhash:016x}"

        index = len(self.simhashes)
        self.exact_hashes.add(sha256)
        self.simhashes.append(simhash)
        for band in range(4):
            key = (simhash >> (band * 16)) & 0xFFFF
            self.bands[band][key].add(index)
        return False, None, sha256, f"{simhash:016x}"


class ContaminationDetector:
    """Benchmark metinleriyle örtüşen 13-token pencerelerini işaretler."""

    def __init__(self, benchmark_path: str, ngram_size: int = 13, min_hits: int = 2):
        self.ngram_size = ngram_size
        self.min_hits = min_hits
        self.fingerprints: Dict[str, set] = defaultdict(set)
        self._load(benchmark_path)

    @staticmethod
    def _iter_records(path: str):
        paths = []
        if os.path.isdir(path):
            paths = [
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.endswith((".jsonl", ".txt"))
            ]
        else:
            paths = [path]
        for file_path in paths:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    text = line
                    record_id = f"{os.path.basename(file_path)}:{line_no}"
                    if file_path.endswith(".jsonl"):
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        base_id = str(record.get("id", record_id))
                        text_fields = (
                            "text", "prompt", "question", "chosen", "rejected",
                            "needle", "reference", "filler",
                        )
                        emitted = False
                        for field in text_fields:
                            value = record.get(field)
                            if isinstance(value, str) and value:
                                emitted = True
                                yield f"{base_id}:{field}", value
                        for field in ("choices", "references"):
                            values = record.get(field, [])
                            if isinstance(values, list):
                                for index, value in enumerate(values):
                                    if isinstance(value, str) and value:
                                        emitted = True
                                        yield f"{base_id}:{field}:{index}", value
                        if emitted:
                            continue
                        text = ""
                    if text:
                        yield record_id, text

    def _fingerprints(self, text: str) -> set:
        words = re.findall(r"\w+", normalize_for_hash(text), flags=re.UNICODE)
        if len(words) < self.ngram_size:
            return set()
        return {
            hashlib.blake2b(
                " ".join(words[i:i + self.ngram_size]).encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            for i in range(len(words) - self.ngram_size + 1)
        }

    def _load(self, path: str):
        for record_id, text in self._iter_records(path):
            for fingerprint in self._fingerprints(text):
                self.fingerprints[fingerprint].add(record_id)

    def find_matches(self, text: str) -> List[str]:
        hit_counts: Dict[str, int] = defaultdict(int)
        for fingerprint in self._fingerprints(text):
            for record_id in self.fingerprints.get(fingerprint, ()):
                hit_counts[record_id] += 1
        return sorted(
            record_id
            for record_id, count in hit_counts.items()
            if count >= self.min_hits
        )


def enrich_document(
    document: dict,
    cleaned_text: str,
    *,
    quality_score: float,
    quality_signals: dict,
    pii_redactions: dict,
    content_hash: str,
    simhash: str,
    contamination_matches: Optional[List[str]] = None,
    source_file: Optional[str] = None,
) -> dict:
    """Temiz dokümana standart provenance ve kalite metadata'sı ekle."""
    source = document.get("source", "unknown")
    provenance = build_provenance(
        source,
        source_url=document.get("source_url") or document.get("url"),
        downloaded_at=document.get("downloaded_at") or document.get("timestamp"),
        source_record_id=document.get("source_record_id") or document.get("id"),
    )
    result = dict(document)
    result.update(provenance)
    result.update({
        "text": cleaned_text,
        "word_count": len(cleaned_text.split()),
        "content_sha256": content_hash,
        "simhash64": simhash,
        "quality_score": quality_score,
        "quality_signals": quality_signals,
        "pii_redactions": pii_redactions,
        "contamination_matches": contamination_matches or [],
    })
    if source_file:
        result["source_file"] = os.path.basename(source_file)
    return result
