# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Türk dünyası tokenizer raporu (dil başına verim).

Bir veya daha fazla SentencePiece modelini dil başına örnekler üzerinde ölçer:

* ``tokens_per_word`` (fertility): kelime başına token, düşük daha iyi;
* ``characters_per_token``: token başına boşluk dışı karakter, yüksek daha iyi;
* ``byte_token_rate``: byte fallback'e (``<0xNN>``) düşen token oranı;
* ``unknown_rate``: UNK token oranı (byte fallback açıkken ~0 olmalı).

Girdi: sürümlü seed dosyası (``evaluation/turkic_seed.json``), ``lang``/``text``
alanlı JSONL veya ``<dil>.txt``/``<dil>.jsonl`` dosyaları içeren dizin. Seed
kayıtlarında ``latin`` alanı varsa (kk/ky/tt/ug transliterasyonu, ota
transkripsiyonu) ``xx-latn`` satırı olarak ayrıca ölçülür.

    python evaluation/turkic_tokenizer_report.py \\
        --tokenizer current=toprak_tokenizer.model \\
        --output evaluation/reports/turkic-tokenizer.json \\
        --markdown evaluation/reports/turkic-tokenizer.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.turkic import TURKIC_LANGUAGES
from evaluation.tokenizer_analysis import BYTE_PIECE_RE, WORD_RE, file_sha256

REPORT_VERSION = "toprak-turkic-tokenizer-report-v1"
DEFAULT_SEED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turkic_seed.json")
LATIN_SUFFIX = "-latn"


def _language_order(key: str):
    base = key[: -len(LATIN_SUFFIX)] if key.endswith(LATIN_SUFFIX) else key
    codes = list(TURKIC_LANGUAGES)
    position = codes.index(base) if base in codes else len(codes)
    return (position, base, key.endswith(LATIN_SUFFIX))


def load_turkic_samples(paths: Iterable[str], include_latin: bool = True) -> "OrderedDict[str, List[str]]":
    """Dil anahtarı → metin listesi. ``latin`` alanları ``xx-latn`` altında toplanır."""
    grouped: Dict[str, List[str]] = {}

    def add(lang, text):
        if isinstance(text, str) and text.strip() and lang:
            grouped.setdefault(str(lang), []).append(text.strip())

    def add_record(record, default_lang=None):
        lang = record.get("lang") or record.get("language") or default_lang
        add(lang, record.get("text"))
        if include_latin and record.get("latin"):
            add(f"{lang}{LATIN_SUFFIX}", record["latin"])

    files = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith((".txt", ".jsonl", ".json")):
                    files.append(os.path.join(path, name))
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"Örnek girdisi bulunamadı: {path}")

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        default_lang = stem if stem in TURKIC_LANGUAGES else None
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload if isinstance(payload, list) else payload.get("samples", [])
            for record in records:
                add_record(record, default_lang)
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: geçersiz JSON") from exc
                    add_record(record, default_lang)
        else:
            if default_lang is None:
                raise ValueError(f"TXT dosya adı dil kodu olmalı (ör. kk.txt): {path}")
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    add(default_lang, line)
    if not grouped:
        raise ValueError("Rapor için örnek metin bulunamadı")
    return OrderedDict(sorted(grouped.items(), key=lambda item: _language_order(item[0])))


def measure_texts(tokenizer, texts: List[str]) -> dict:
    """Bir metin listesi için fertility, karakter/token, byte ve UNK oranları.

    ``tokenizer`` ``encode(text, add_bos=False, add_eos=False)``,
    ``id_to_token`` ve ``unk_token_id`` sunmalıdır (``ToprakTokenizer``).
    """
    words = characters = tokens = byte_tokens = unknown = 0
    for text in texts:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        pieces = [tokenizer.id_to_token(token_id) for token_id in ids]
        words += len(WORD_RE.findall(text))
        characters += sum(not char.isspace() for char in text)
        tokens += len(ids)
        byte_tokens += sum(bool(BYTE_PIECE_RE.match(piece)) for piece in pieces)
        unknown += sum(token_id == tokenizer.unk_token_id for token_id in ids)
    return {
        "samples": len(texts),
        "words": words,
        "characters": characters,
        "tokens": tokens,
        "tokens_per_word": tokens / words if words else None,
        "characters_per_token": characters / tokens if tokens else None,
        "byte_token_rate": byte_tokens / tokens if tokens else None,
        "unknown_rate": unknown / tokens if tokens else None,
    }


def analyze_turkic_tokenizer(tokenizer, samples: Dict[str, List[str]], name: str,
                             tokenizer_path: Optional[str] = None) -> dict:
    languages = OrderedDict(
        (lang, measure_texts(tokenizer, texts)) for lang, texts in samples.items()
    )
    baseline = languages.get("tr", {}).get("tokens_per_word")
    for metrics in languages.values():
        fertility = metrics["tokens_per_word"]
        metrics["fertility_vs_tr"] = (
            fertility / baseline if baseline and fertility is not None else None
        )
    return {
        "name": name,
        "tokenizer_path": os.path.abspath(tokenizer_path) if tokenizer_path else None,
        "tokenizer_sha256": file_sha256(tokenizer_path) if tokenizer_path else None,
        "vocab_size": tokenizer.get_vocab_size(),
        "languages": languages,
    }


def build_report(analyses: List[dict]) -> dict:
    if not analyses:
        raise ValueError("En az bir tokenizer analizi gerekli")
    return {"report_version": REPORT_VERSION, "tokenizers": analyses}


def render_table(report: dict) -> str:
    """Dil × tokenizer karşılaştırma tablosu (Markdown)."""

    def number(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"

    def percent(value):
        return "N/A" if value is None else f"{value:.1%}"

    lines = [
        "# Türk Dünyası Tokenizer Raporu",
        "",
        "| Tokenizer | Dil | Token/kelime | ×tr | Karakter/token | Byte token | UNK |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for analysis in report["tokenizers"]:
        for lang, metrics in analysis["languages"].items():
            lines.append(
                f"| {analysis['name']} | {lang} | {number(metrics['tokens_per_word'])} | "
                f"{number(metrics['fertility_vs_tr'])} | "
                f"{number(metrics['characters_per_token'])} | "
                f"{percent(metrics['byte_token_rate'])} | {percent(metrics['unknown_rate'])} |"
            )
    lines.extend([
        "",
        "> Seed seti küçüktür (dil başına ~5 cümle); sonuçlar yön göstericidir ve "
        "büyük held-out örneklerle doğrulanmalıdır. `xx-latn` satırları aynı "
        "cümlelerin Latin transliterasyonu/transkripsiyonudur.",
        "",
    ])
    return "\n".join(lines)


def _tokenizer_arg(value: str):
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = os.path.splitext(os.path.basename(path))[0]
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Tokenizer NAME=PATH biçiminde olmalı")
    return name.strip(), path.strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Türk dünyası dil başına tokenizer raporu")
    parser.add_argument("--tokenizer", action="append", type=_tokenizer_arg, required=True,
                        help="NAME=tokenizer.model; tekrarlanabilir")
    parser.add_argument("--input", action="append", default=None,
                        help="JSON/JSONL dosyası veya <dil>.txt dizini; verilmezse seed set")
    parser.add_argument("--no-latin", action="store_true",
                        help="Seed'deki latin alanlarını ölçme")
    parser.add_argument("--output", default=None, help="JSON rapor yolu")
    parser.add_argument("--markdown", default=None, help="Markdown tablo yolu")
    args = parser.parse_args(argv)

    from model.tokenizer import ToprakTokenizer

    samples = load_turkic_samples(args.input or [DEFAULT_SEED], include_latin=not args.no_latin)
    analyses = [
        analyze_turkic_tokenizer(ToprakTokenizer(path), samples, name, path)
        for name, path in args.tokenizer
    ]
    report = build_report(analyses)
    table = render_table(report)
    print(table)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
