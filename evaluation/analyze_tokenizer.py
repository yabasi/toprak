# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Tokenizer analiz paketi CLI."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.tokenizer_analysis import (
    analyze_tokenizer,
    build_analysis_report,
    load_documents,
    load_seed,
    render_markdown,
)
from model.tokenizer import ToprakTokenizer


def tokenizer_arg(value: str):
    if "=" in value:
        name, path = value.split("=", 1)
    else:
        path = value
        name = os.path.splitext(os.path.basename(path))[0]
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Tokenizer NAME=PATH biçiminde olmalı")
    return name.strip(), path.strip()


def main():
    default_seed = os.path.join(os.path.dirname(__file__), "tokenizer_seed.json")
    parser = argparse.ArgumentParser(description="Toprak tokenizer kalite analizi")
    parser.add_argument(
        "--tokenizer", action="append", type=tokenizer_arg, required=True,
        help="Tekrarlanabilir NAME=tokenizer.model girdisi",
    )
    parser.add_argument(
        "--input", action="append", default=None,
        help="JSON/JSONL/TXT dosyası veya dizini; verilmezse sürümlü seed set",
    )
    parser.add_argument("--seed-data", default=default_seed)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--max-tokens-per-word", type=float, default=None)
    parser.add_argument("--max-unknown-rate", type=float, default=0.0)
    parser.add_argument("--max-byte-token-rate", type=float, default=None)
    parser.add_argument("--min-roundtrip-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents pozitif olmalı")

    _, probes = load_seed(args.seed_data)
    if args.input:
        documents = load_documents(args.input, args.max_documents)
    else:
        documents, _ = load_seed(args.seed_data)
        if args.max_documents:
            documents = documents[:args.max_documents]

    tokenizer_entries = dict(args.tokenizer)
    if len(tokenizer_entries) != len(args.tokenizer):
        parser.error("Tokenizer adları benzersiz olmalı")
    analyses = []
    failed = False
    for name, path in tokenizer_entries.items():
        tokenizer = ToprakTokenizer(path)
        analysis = analyze_tokenizer(tokenizer, documents, probes, name, path)
        analyses.append(analysis)
        corpus = analysis["corpus"]
        if (
            args.max_tokens_per_word is not None
            and corpus["tokens_per_word"] is not None
            and corpus["tokens_per_word"] > args.max_tokens_per_word
        ):
            print(f"❌ {name}: token/kelime sınırı aşıldı")
            failed = True
        if corpus["unknown_rate"] is not None and corpus["unknown_rate"] > args.max_unknown_rate:
            print(f"❌ {name}: UNK oranı sınırı aşıldı")
            failed = True
        if (
            args.max_byte_token_rate is not None
            and corpus["byte_token_rate"] is not None
            and corpus["byte_token_rate"] > args.max_byte_token_rate
        ):
            print(f"❌ {name}: byte token oranı sınırı aşıldı")
            failed = True
        if (
            corpus["roundtrip_exact_rate"] is not None
            and corpus["roundtrip_exact_rate"] < args.min_roundtrip_rate
        ):
            print(f"❌ {name}: round-trip oranı sınırın altında")
            failed = True

    report = build_analysis_report(documents, probes, analyses)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    markdown = render_markdown(report)
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    print(markdown)
    print(f"JSON raporu: {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
