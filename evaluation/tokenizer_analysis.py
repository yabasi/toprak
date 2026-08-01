# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Türkçe tokenizer kapsam, verim ve morfolojik sınır analizi."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


ANALYSIS_VERSION = "toprak-tokenizer-analysis-v1"
TURKISH_CHARS = "çğıöşüÇĞİÖŞÜ"
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+", re.UNICODE)
BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_roundtrip(text: str) -> str:
    """SentencePiece NFKC + whitespace davranışı için karşılaştırma formu."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def load_seed(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    documents = payload.get("documents", [])
    probes = payload.get("morphology", [])
    if not documents or not probes:
        raise ValueError("Tokenizer seed dosyası documents ve morphology içermeli")
    return documents, probes


def load_documents(paths: Iterable[str], max_documents: Optional[int] = None) -> List[dict]:
    documents = []

    def add(text, domain):
        if isinstance(text, str) and text.strip():
            documents.append({"text": text, "domain": str(domain or "unknown")})

    files = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate) and name.endswith((".jsonl", ".json", ".txt")):
                    files.append(candidate)
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"Tokenizer analiz girdisi bulunamadı: {path}")

    for path in sorted(files):
        default_domain = os.path.splitext(os.path.basename(path))[0]
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: geçersiz JSON") from exc
                    add(record.get("text"), record.get("domain", default_domain))
                    if max_documents and len(documents) >= max_documents:
                        return documents
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload if isinstance(payload, list) else payload.get("documents", [])
            for record in records:
                add(record.get("text"), record.get("domain", default_domain))
                if max_documents and len(documents) >= max_documents:
                    return documents
        else:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    add(line.strip(), default_domain)
                    if max_documents and len(documents) >= max_documents:
                        return documents
    if not documents:
        raise ValueError("Tokenizer analizi için metin bulunamadı")
    return documents


def corpus_sha256(documents: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document["domain"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document["text"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _percentile(values: List[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _new_stats() -> dict:
    return {
        "documents": 0,
        "words": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
        "unknown_tokens": 0,
        "byte_tokens": 0,
        "roundtrip_exact": 0,
        "document_token_lengths": [],
    }


def _update_stats(stats: dict, tokenizer, document: dict, measurement=None):
    text = document["text"]
    if measurement is None:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        pieces = [tokenizer.id_to_token(token_id) for token_id in ids]
        roundtrip_exact = (
            normalize_roundtrip(tokenizer.decode(ids)) == normalize_roundtrip(text)
        )
        measurement = ids, pieces, roundtrip_exact
    else:
        ids, pieces, roundtrip_exact = measurement
    stats["documents"] += 1
    stats["words"] += len(WORD_RE.findall(text))
    stats["characters"] += sum(not character.isspace() for character in text)
    stats["utf8_bytes"] += len(text.encode("utf-8"))
    stats["tokens"] += len(ids)
    stats["unknown_tokens"] += sum(token_id == tokenizer.unk_token_id for token_id in ids)
    stats["byte_tokens"] += sum(bool(BYTE_PIECE_RE.match(piece)) for piece in pieces)
    stats["roundtrip_exact"] += int(roundtrip_exact)
    stats["document_token_lengths"].append(len(ids))
    return measurement


def _finalize_stats(stats: dict) -> dict:
    documents = stats["documents"]
    tokens = stats["tokens"]
    words = stats["words"]
    return {
        "documents": documents,
        "words": words,
        "characters": stats["characters"],
        "utf8_bytes": stats["utf8_bytes"],
        "tokens": tokens,
        "tokens_per_word": tokens / words if words else None,
        "characters_per_token": stats["characters"] / tokens if tokens else None,
        "utf8_bytes_per_token": stats["utf8_bytes"] / tokens if tokens else None,
        "unknown_rate": stats["unknown_tokens"] / tokens if tokens else None,
        "byte_token_rate": stats["byte_tokens"] / tokens if tokens else None,
        "roundtrip_exact_rate": stats["roundtrip_exact"] / documents if documents else None,
        "document_tokens_p50": _percentile(stats["document_token_lengths"], 0.50),
        "document_tokens_p95": _percentile(stats["document_token_lengths"], 0.95),
        "document_tokens_max": max(stats["document_token_lengths"], default=0),
    }


def _morphology_metrics(tokenizer, probes: Iterable[dict]) -> dict:
    form_count = 0
    form_tokens = 0
    suffix_hits = 0
    prefix_reuse_total = 0.0
    single_token_forms = 0
    details = []
    for probe in probes:
        lemma = probe["lemma"]
        lemma_ids = tokenizer.encode(lemma, add_bos=False, add_eos=False)
        lemma_pieces = [tokenizer.id_to_token(token_id) for token_id in lemma_ids]
        for form in probe["forms"]:
            text = form["text"]
            suffix = form["suffix"]
            ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            pieces = [tokenizer.id_to_token(token_id) for token_id in ids]
            clean_pieces = [piece.lstrip("▁") for piece in pieces]
            common = 0
            for lemma_piece, form_piece in zip(lemma_pieces, pieces):
                if lemma_piece != form_piece:
                    break
                common += 1
            reuse = common / len(lemma_pieces) if lemma_pieces else 0.0
            suffix_hit = suffix in clean_pieces
            form_count += 1
            form_tokens += len(ids)
            suffix_hits += int(suffix_hit)
            prefix_reuse_total += reuse
            single_token_forms += int(len(ids) == 1)
            details.append({
                "lemma": lemma,
                "form": text,
                "suffix": suffix,
                "pieces": pieces,
                "tokens": len(ids),
                "lemma_prefix_reuse": reuse,
                "exact_suffix_boundary": suffix_hit,
            })
    if not form_count:
        raise ValueError("Morfoloji probu bulunamadı")
    return {
        "forms": form_count,
        "tokens_per_form": form_tokens / form_count,
        "lemma_prefix_reuse_rate": prefix_reuse_total / form_count,
        "exact_suffix_boundary_rate": suffix_hits / form_count,
        "single_token_form_rate": single_token_forms / form_count,
        "details": details,
    }


def _vocabulary_metrics(tokenizer) -> dict:
    pieces = [tokenizer.id_to_token(index) for index in range(tokenizer.get_vocab_size())]
    byte_pieces = sum(bool(BYTE_PIECE_RE.match(piece)) for piece in pieces)
    return {
        "size": len(pieces),
        "byte_pieces": byte_pieces,
        "word_start_pieces": sum(piece.startswith("▁") for piece in pieces),
        "continuation_pieces": sum(
            not piece.startswith("▁") and any(char.isalpha() for char in piece)
            for piece in pieces
        ),
        "single_character_pieces": sum(
            len(piece.lstrip("▁")) == 1 and not BYTE_PIECE_RE.match(piece)
            for piece in pieces
        ),
    }


def analyze_tokenizer(
    tokenizer,
    documents: List[dict],
    probes: List[dict],
    name: str,
    tokenizer_path: Optional[str] = None,
) -> dict:
    if not documents:
        raise ValueError("Tokenizer analizi için belge gerekli")
    overall = _new_stats()
    domains = defaultdict(_new_stats)
    for document in documents:
        measurement = _update_stats(overall, tokenizer, document)
        _update_stats(
            domains[document["domain"]], tokenizer, document, measurement
        )

    character_coverage = {}
    for character in TURKISH_CHARS:
        ids = tokenizer.encode(character, add_bos=False, add_eos=False)
        character_coverage[character] = {
            "tokens": len(ids),
            "unknown": any(token_id == tokenizer.unk_token_id for token_id in ids),
            "pieces": [tokenizer.id_to_token(token_id) for token_id in ids],
        }

    return {
        "name": name,
        "tokenizer_path": os.path.abspath(tokenizer_path) if tokenizer_path else None,
        "tokenizer_sha256": file_sha256(tokenizer_path) if tokenizer_path else None,
        "corpus": _finalize_stats(overall),
        "domains": {
            domain: _finalize_stats(stats) for domain, stats in sorted(domains.items())
        },
        "turkish_character_coverage": character_coverage,
        "morphology": _morphology_metrics(tokenizer, probes),
        "vocabulary": _vocabulary_metrics(tokenizer),
    }


def build_analysis_report(documents: List[dict], probes: List[dict], analyses: List[dict]) -> dict:
    if not analyses:
        raise ValueError("En az bir tokenizer analizi gerekli")
    names = [analysis["name"] for analysis in analyses]
    if len(names) != len(set(names)):
        raise ValueError("Tokenizer analiz adları benzersiz olmalı")
    probe_payload = json.dumps(probes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "analysis_version": ANALYSIS_VERSION,
        "corpus_sha256": corpus_sha256(documents),
        "morphology_sha256": hashlib.sha256(probe_payload).hexdigest(),
        "tokenizers": analyses,
    }


def render_markdown(report: dict) -> str:
    def number(value, digits=3):
        return "N/A" if value is None else f"{value:.{digits}f}"

    def percent(value):
        return "N/A" if value is None else f"{value:.3%}"

    lines = [
        "# Toprak Tokenizer Analizi",
        "",
        "| Tokenizer | Vocab | Token/kelime | UNK | Byte token | Round-trip | Ek sınırı | Kök reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for analysis in report["tokenizers"]:
        corpus = analysis["corpus"]
        morphology = analysis["morphology"]
        lines.append(
            f"| {analysis['name']} | {analysis['vocabulary']['size']} | "
            f"{number(corpus['tokens_per_word'])} | {percent(corpus['unknown_rate'])} | "
            f"{percent(corpus['byte_token_rate'])} | {percent(corpus['roundtrip_exact_rate'])} | "
            f"{percent(morphology['exact_suffix_boundary_rate'])} | "
            f"{percent(morphology['lemma_prefix_reuse_rate'])} |"
        )
    lines.extend([
        "",
        "> Seed sonuçları tokenizer seçimi için regresyon göstergesidir; büyük ve "
        "eğitimden bağımsız bir held-out corpus üzerinde doğrulanmalıdır.",
        "",
    ])
    return "\n".join(lines)
