# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Baseline değerlendirmeleri

configs/lm_eval_baselines.json içindeki modelleri Toprak ile aynı görev ve
ayarlarla değerlendirir. Var olan raporlar atlanır; yarıda kalan bir koşu
aynı komutla kaldığı yerden sürer.

Örnek:
    python scripts/run_baselines.py --tier small
    python scripts/run_baselines.py --models Qwen/Qwen2.5-0.5B --limit 20
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report_path(output_dir: str, model_id: str) -> str:
    return os.path.join(output_dir, model_id.replace("/", "__") + ".json")


def select_models(config: dict, tiers=None, models=None) -> list:
    selected = config["models"]
    if models:
        wanted = set(models)
        unknown = wanted - {m["id"] for m in selected}
        if unknown:
            raise ValueError(f"Baseline listesinde olmayan modeller: {sorted(unknown)}")
        return [m for m in selected if m["id"] in wanted]
    if tiers:
        return [m for m in selected if m["tier"] in tiers]
    return selected


def build_command(entry: dict, config: dict, args, output: str) -> list:
    cmd = [
        sys.executable, os.path.join(ROOT, "evaluation", "run_lm_eval.py"),
        "--hf-model", entry["id"],
        "--hf-backend", entry.get("backend", "causal"),
        "--preset", args.preset or config["preset"],
        "--dtype", entry.get("dtype", config.get("dtype", "float32")),
        "--batch-size", str(args.batch_size),
        "--output", output,
    ]
    if entry.get("revision"):
        cmd += ["--hf-revision", entry["revision"]]
    num_fewshot = config.get("num_fewshot")
    if num_fewshot is not None:
        cmd += ["--num-fewshot", str(num_fewshot)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.max_length is not None:
        cmd += ["--max-length", str(args.max_length)]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Toprak — baseline modelleri değerlendir")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "lm_eval_baselines.json"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "evaluation", "reports", "baselines"))
    parser.add_argument("--tier", action="append", choices=["small", "large", "gated"],
                        help="Birden çok kez verilebilir; varsayılan: small")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--preset", default=None, help="Config'teki preset'i geçersiz kıl")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Tüm modellere ortak bağlam sınırı (Toprak ile eşitlemek için)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="Var olan raporları yeniden üret")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    entries = select_models(config, tiers=args.tier or (None if args.models else ["small"]),
                            models=args.models)
    failures = []
    for entry in entries:
        output = report_path(args.output_dir, entry["id"])
        if os.path.exists(output) and not args.force:
            print(f"  ↷ Atlandı (rapor var): {entry['id']}")
            continue
        cmd = build_command(entry, config, args, output)
        print(f"\n  ▶ {entry['id']} ({entry['params']}, {entry['group']})")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failures.append(entry["id"])
            print(f"  ✗ Başarısız: {entry['id']}")

    if failures:
        print(f"\n  ✗ Başarısız modeller: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
