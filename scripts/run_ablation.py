# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Aynı başlangıç checkpoint'inden kontrollü auxiliary-loss deneyleri çalıştır."""

import argparse
import json
import os
import shlex
import subprocess
import sys

import torch


VARIANTS = {
    "baseline": [],
    "vowel_harmony": ["--vowel-harmony"],
    "consonant_harmony": ["--consonant-harmony"],
    "morph_weight": ["--morph-weight"],
    "morph_head": ["--morph-head"],
    "syllable_rhyme": ["--syllable-rhyme"],
    "all_aux": [
        "--vowel-harmony",
        "--consonant-harmony",
        "--morph-weight",
        "--morph-head",
        "--syllable-rhyme",
    ],
}

MODEL_SHAPES = {
    "small": (640, 14),
    "medium": (768, 16),
    "large": (1024, 28),
    "xl": (1536, 36),
}


def build_commands(args, project_root: str):
    python = sys.executable
    output_root = os.path.abspath(args.output_dir)
    report_dir = os.path.join(output_root, "reports")
    variants = ["baseline"] + [
        name for name in args.variants if name != "baseline"
    ]
    commands = []
    reports = {}

    common_train = [
        python,
        os.path.join(project_root, "training", "train.py"),
        "--model-size", args.model_size,
        "--data-dir", os.path.abspath(args.data_dir),
        "--tokenizer", os.path.abspath(args.tokenizer),
        "--resume", os.path.abspath(args.base_checkpoint),
        "--max-steps", str(args.target_step),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
    ]
    if args.eval_data_dir:
        common_train += ["--eval-data-dir", os.path.abspath(args.eval_data_dir)]
    if args.bin_mode:
        common_train.append("--bin-mode")
    if args.device:
        common_train += ["--device", args.device]
    if args.batch_size is not None:
        common_train += ["--batch-size", str(args.batch_size)]
    if args.learning_rate is not None:
        common_train += ["--lr", str(args.learning_rate)]
    if args.warmup_steps is not None:
        common_train += ["--warmup-steps", str(args.warmup_steps)]
    if args.grad_accum is not None:
        common_train += ["--grad-accum", str(args.grad_accum)]
    if args.no_compile:
        common_train.append("--no-compile")
    if args.no_grad_checkpoint:
        common_train.append("--no-grad-checkpoint")
    if args.bf16:
        common_train.append("--bf16")

    for name in variants:
        checkpoint_dir = os.path.join(output_root, "checkpoints", name)
        report_path = os.path.join(report_dir, f"{name}.json")
        train = common_train + [
            "--experiment-name", f"ablation-{name}",
            "--checkpoint-dir", checkpoint_dir,
            "--log-dir", os.path.join(output_root, "logs", name),
        ] + VARIANTS[name]
        evaluate = [
            python,
            os.path.join(project_root, "evaluation", "evaluate_suite.py"),
            "--checkpoint", os.path.join(checkpoint_dir, "toprak_last.pt"),
            "--tokenizer", os.path.abspath(args.tokenizer),
            "--benchmarks", os.path.abspath(args.benchmarks),
            "--output", report_path,
        ]
        if args.device:
            evaluate += ["--device", args.device]
        commands.extend([train, evaluate])
        reports[name] = report_path

    compare = [
        python,
        os.path.join(project_root, "evaluation", "compare_ablation.py"),
        "--baseline", reports["baseline"],
        "--output", os.path.join(output_root, "ablation.json"),
        "--markdown", os.path.join(output_root, "ablation.md"),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--seed", str(args.seed),
    ]
    for name in variants:
        if name != "baseline":
            compare += ["--candidate", f"{name}={reports[name]}"]
    commands.append(compare)
    return variants, commands


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toprak yardımcı loss kontrollü ablation matrisi"
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-data-dir", default=None)
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model")
    parser.add_argument(
        "--benchmarks", default="evaluation/benchmarks"
    )
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl"], default="small")
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--bin-mode", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS),
        default=[name for name in VARIANTS if name != "baseline"],
    )
    parser.add_argument("--output-dir", default="ablation_runs")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--execute", action="store_true",
        help="Verilmezse yalnız çalıştırılacak komutları gösterir",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_config = checkpoint.get("config", {})
    expected_shape = MODEL_SHAPES[args.model_size]
    actual_shape = (
        checkpoint_config.get("d_model"), checkpoint_config.get("num_layers")
    )
    if all(value is not None for value in actual_shape) and actual_shape != expected_shape:
        raise ValueError(
            f"--model-size {args.model_size} checkpoint ile uyuşmuyor; "
            f"beklenen d_model/layer={expected_shape}, checkpoint={actual_shape}"
        )
    start_step = int(checkpoint.get("global_step", 0))
    if args.target_step <= start_step:
        raise ValueError(
            f"--target-step ({args.target_step}) başlangıç adımından "
            f"({start_step}) büyük olmalı"
        )
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants tekrar eden değer içeremez")
    if not any(name != "baseline" for name in args.variants):
        raise ValueError("Baseline'a ek olarak en az bir ablation varyantı gerekli")

    variants, commands = build_commands(args, project_root)
    manifest = {
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "base_global_step": start_step,
        "target_step": args.target_step,
        "seed": args.seed,
        "variants": variants,
        "commands": commands,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Ablation varyantları: {', '.join(variants)}")
    print(f"Başlangıç step: {start_step}; hedef step: {args.target_step}")
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("\nDry-run tamamlandı. Çalıştırmak için --execute ekleyin.")
        return

    for index, command in enumerate(commands, 1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}")
        subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
