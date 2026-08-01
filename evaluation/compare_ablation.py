# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Eval v1 JSON raporlarından auxiliary-loss ablation özeti üret."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ablation import build_ablation_report, render_markdown


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _candidate(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Aday NAME=REPORT.json biçiminde olmalı")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Aday adı ve rapor yolu boş olamaz")
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Toprak auxiliary-loss ablation raporu"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--candidate", action="append", type=_candidate, required=True,
        help="Tekrarlanabilir NAME=REPORT.json girdisi",
    )
    parser.add_argument("--output", required=True, help="Çıktı JSON yolu")
    parser.add_argument("--markdown", default=None, help="İsteğe bağlı Markdown çıktı")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate_paths = dict(args.candidate)
    if len(candidate_paths) != len(args.candidate):
        parser.error("Ablation aday adları benzersiz olmalı")
    baseline = _load(args.baseline)
    candidates = {name: _load(path) for name, path in candidate_paths.items()}
    report = build_ablation_report(
        os.path.basename(args.baseline),
        baseline,
        candidates,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

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


if __name__ == "__main__":
    main()
