# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Toprak çok boyutlu değerlendirme paketi CLI."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.dataset import ToprakDataset, ToprakShardDataset, create_dataloader
from evaluation.eval import compute_perplexity
from evaluation.suite import (
    ToprakEvaluationSuite,
    compare_reports,
    file_sha256,
    load_benchmarks,
    validate_report_compatibility,
)
from model.config import ModelConfig, TOPRAK_SMALL, detect_device
from model.tokenizer import ToprakTokenizer
from model.transformer import ToprakLM
from utils.validation import validate_checkpoint, validate_tokenizer


def load_checkpoint_model(checkpoint_path: str, tokenizer, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_data = checkpoint.get("config", {})
    config = ModelConfig(**config_data) if config_data else TOPRAK_SMALL
    config.device = device
    model = ToprakLM(config, tokenizer=tokenizer).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    # Eski checkpointlerde morph_head olmayabilir; diğer eksikler raporlanır.
    non_compatible_missing = [name for name in missing if name != "morph_head.weight"]
    if non_compatible_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state uyuşmazlığı; missing={non_compatible_missing}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    return model, config, checkpoint


def benchmark_hashes(benchmark_dir: str) -> dict:
    return {
        filename: file_sha256(os.path.join(benchmark_dir, filename))
        for filename in sorted(os.listdir(benchmark_dir))
        if filename.endswith(".jsonl")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Toprak — çok boyutlu Türkçe checkpoint değerlendirmesi"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model")
    parser.add_argument(
        "--benchmarks",
        default=os.path.join(os.path.dirname(__file__), "benchmarks"),
        help="Sürümlü JSONL benchmark dizini",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--output", default=None, help="JSON değerlendirme raporu")
    parser.add_argument("--baseline", default=None, help="Karşılaştırılacak eski JSON raporu")
    parser.add_argument("--max-regression", type=float, default=None,
                        help="Macro skor düşüşü bu değeri aşarsa hata koduyla çık")
    parser.add_argument("--fail-below", type=float, default=None,
                        help="Macro skor bu değerin altındaysa hata koduyla çık")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-generation-tokens", type=int, default=48)
    parser.add_argument("--perplexity-data", default=None,
                        help="İsteğe bağlı JSONL/TXT eval dizini")
    parser.add_argument("--perplexity-bin-mode", action="store_true",
                        help="Perplexity verisini manifest tabanlı eval shard'larından oku")
    parser.add_argument("--perplexity-batches", type=int, default=100)
    args = parser.parse_args()

    checkpoint_path = validate_checkpoint(args.checkpoint)
    tokenizer_path = validate_tokenizer(args.tokenizer)
    device = args.device or detect_device()
    tokenizer = ToprakTokenizer(tokenizer_path)
    model, config, checkpoint = load_checkpoint_model(
        checkpoint_path, tokenizer, device
    )
    samples = load_benchmarks(args.benchmarks, max_samples=args.max_samples)

    print("🌱 Toprak Çok Boyutlu Değerlendirme")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Step:       {checkpoint.get('global_step', '?')}")
    print(f"  Örnek:      {len(samples)}")
    print(f"  Cihaz:      {device}")

    suite = ToprakEvaluationSuite(
        model,
        tokenizer,
        device,
        max_generation_tokens=args.max_generation_tokens,
    )
    report = suite.run(samples)
    report["metadata"] = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "global_step": checkpoint.get("global_step"),
        "model_config": checkpoint.get("config", {}),
        "tokenizer": os.path.abspath(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "benchmark_dir": os.path.abspath(args.benchmarks),
        "benchmark_sha256": benchmark_hashes(args.benchmarks),
        "device": device,
    }

    if args.perplexity_data:
        if args.perplexity_bin_mode:
            dataset = ToprakShardDataset(
                bin_dir=args.perplexity_data,
                split="eval",
                max_seq_len=config.max_seq_len,
                expected_vocab_size=config.vocab_size,
            )
        else:
            dataset = ToprakDataset(
                data_dir=args.perplexity_data,
                tokenizer=tokenizer,
                max_seq_len=config.max_seq_len,
                split="eval",
                shuffle_docs=False,
            )
        dataloader = create_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        report["perplexity"] = compute_perplexity(
            model,
            dataloader,
            device=device,
            max_batches=args.perplexity_batches,
        )

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        validate_report_compatibility(report, baseline)
        report["comparison"] = compare_reports(report, baseline)

    if args.output:
        output_path = args.output
    else:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join("evaluation", "reports", f"{checkpoint_name}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\nSonuç")
    print(f"  Micro skor: {report['summary']['micro_score']:.3f}")
    print(f"  Macro skor: {report['summary']['macro_score']:.3f}")
    for category, metrics in report["categories"].items():
        print(f"  {category:<24} {metrics['mean_score']:.3f} ({metrics['count']})")
    if "perplexity" in report:
        print(f"  Perplexity:              {report['perplexity']:.3f}")
    if "comparison" in report:
        print(f"  Macro delta:             {report['comparison']['macro_delta']:+.3f}")
    print(f"  Rapor: {output_path}")

    failed = False
    if args.fail_below is not None and report["summary"]["macro_score"] < args.fail_below:
        print(f"❌ Macro skor eşiğin altında: {args.fail_below:.3f}")
        failed = True
    if args.max_regression is not None:
        if "comparison" not in report:
            parser.error("--max-regression için --baseline gerekli")
        if report["comparison"]["macro_delta"] < -args.max_regression:
            print(f"❌ Macro regresyon sınırı aşıldı: {args.max_regression:.3f}")
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
