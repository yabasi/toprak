# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — lm-evaluation-harness CLI

Toprak checkpointlerini standart Türkçe benchmarklarda değerlendirir ve
checkpoint hash'i, görev sürümleri ve lm-eval sürümüyle birlikte tekrarlanabilir
bir JSON raporu yazar.

Örnek:
    python evaluation/run_lm_eval.py \
        --checkpoint checkpoints/toprak_last.pt \
        --preset tr_core \
        --output evaluation/reports/lm_eval_toprak_last.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Görev isimleri lm-eval 0.4.x kayıt defterinde doğrulanmıştır.
TASK_PRESETS = {
    # Log-olasılık tabanlı; küçük base modellerde de anlamlı sinyal verir.
    "tr_core": [
        "xcopa_tr",
        "xnli_tr",
        "belebele_tur_Latn",
        "global_piqa_nonparallel_cloze_tur_latn",
    ],
    # Bilgi ağırlıklı çoktan seçmeli; küçük modellerde rastgeleye yakın olabilir.
    "tr_knowledge": [
        "turkishmmlu",
        "global_mmlu_full_tr",
        "include_base_44_turkish",
    ],
    # Üretim tabanlı (generate_until).
    "tr_generative": [
        "xquad_tr",
    ],
}
TASK_PRESETS["tr_all"] = (
    TASK_PRESETS["tr_core"] + TASK_PRESETS["tr_knowledge"] + TASK_PRESETS["tr_generative"]
)

# Rapor tablosunda şans seviyesini göstermek için (çoktan seçmeli görevler).
RANDOM_BASELINES = {
    "xcopa_tr": 0.5,
    "xnli_tr": 1 / 3,
    "belebele_tur_Latn": 0.25,
    "global_piqa_nonparallel_cloze_tur_latn": 0.5,
    "turkishmmlu": 0.2,
    "global_mmlu_full_tr": 0.25,
    "include_base_44_turkish": 0.25,
}


def resolve_tasks(preset: str = None, tasks: str = None) -> list:
    selected = []
    if preset:
        if preset not in TASK_PRESETS:
            raise ValueError(
                f"Bilinmeyen preset: {preset}. Seçenekler: {sorted(TASK_PRESETS)}"
            )
        selected.extend(TASK_PRESETS[preset])
    if tasks:
        selected.extend(t.strip() for t in tasks.split(",") if t.strip())
    if not selected:
        raise ValueError("--preset veya --tasks verilmelidir")
    return list(dict.fromkeys(selected))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize(results: dict) -> list:
    """lm-eval sonuçlarından görev başına ana metrik satırları çıkarır."""
    preferred = ("acc_norm,none", "acc,none", "f1,none", "exact_match,none")
    rows = []
    for task, metrics in sorted(results.get("results", {}).items()):
        for key in preferred:
            if key in metrics and isinstance(metrics[key], (int, float)):
                name = key.split(",")[0]
                rows.append({
                    "task": task,
                    "metric": name,
                    "value": metrics[key],
                    "stderr": metrics.get(f"{name}_stderr,none"),
                    "random_baseline": RANDOM_BASELINES.get(task),
                })
                break
    return rows


def format_table(rows: list) -> str:
    lines = [
        f"{'Görev':<42} {'Metrik':<12} {'Skor':>8} {'±':>7} {'Şans':>7}",
        "-" * 80,
    ]
    for row in rows:
        stderr = row["stderr"]
        stderr_text = f"{stderr:.4f}" if isinstance(stderr, (int, float)) else "-"
        baseline = row["random_baseline"]
        baseline_text = f"{baseline:.3f}" if baseline is not None else "-"
        lines.append(
            f"{row['task']:<42} {row['metric']:<12} {row['value']:>8.4f} "
            f"{stderr_text:>7} {baseline_text:>7}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Toprak — lm-evaluation-harness ile standart Türkçe benchmarklar"
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--checkpoint", help="Toprak model checkpoint (.pt)")
    source.add_argument("--hf-model", help="Karşılaştırma için HuggingFace model ID'si")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-backend", default="causal", choices=["causal", "seq2seq"],
                        help="seq2seq: TURNA gibi encoder-decoder modeller için")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model")
    parser.add_argument("--preset", choices=sorted(TASK_PRESETS), default=None)
    parser.add_argument("--tasks", default=None, help="Virgülle ayrılmış ek lm-eval görevleri")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Görev başına örnek sınırı (hızlı duman testi için)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None, help="JSON rapor yolu")
    parser.add_argument("--log-samples", action="store_true",
                        help="Örnek bazlı çıktıları rapora ekle")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        for name, tasks in sorted(TASK_PRESETS.items()):
            print(f"{name}: {', '.join(tasks)}")
        return
    if not (args.checkpoint or args.hf_model):
        parser.error("--checkpoint veya --hf-model verilmelidir")

    tasks = resolve_tasks(args.preset, args.tasks)

    import lm_eval

    if args.hf_model:
        lm, model_info = build_hf_model(args)
    else:
        lm, model_info = build_toprak_model(args)
    print(
        f"  Model: {model_info['name']} | Cihaz: {lm.device} | "
        f"Bağlam: {lm.max_length} | Görevler: {', '.join(tasks)}"
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    rows = summarize(results)
    print("\n" + format_table(rows))
    if args.limit is not None:
        print("\n  ⚠ --limit kullanıldı; skorlar resmi karşılaştırma için geçerli değildir.")

    if args.output:
        report = {
            "schema": "toprak-lm-eval-v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
            "model": model_info,
            "settings": {
                "tasks": tasks,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "batch_size": args.batch_size,
                "max_length": lm.max_length,
                "dtype": args.dtype,
                "device": str(lm.device),
                "seed": args.seed,
            },
            "summary": rows,
            "results": results.get("results", {}),
            "versions": results.get("versions", {}),
            "n-shot": results.get("n-shot", {}),
            "n-samples": results.get("n-samples", {}),
        }
        if args.log_samples:
            report["samples"] = results.get("samples", {})
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        print(f"\n  ✓ Rapor yazıldı: {args.output}")


def build_toprak_model(args):
    from evaluation.lm_eval_adapter import ToprakLMEval
    from evaluation.suite import file_sha256
    from utils.validation import validate_checkpoint, validate_tokenizer

    validate_checkpoint(args.checkpoint)
    validate_tokenizer(args.tokenizer)
    lm = ToprakLMEval(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )
    return lm, {
        "type": "toprak",
        "name": os.path.splitext(os.path.basename(args.checkpoint))[0],
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "num_parameters": lm.model.count_parameters(),
    }


def build_hf_model(args):
    from lm_eval.models.huggingface import HFLM

    revision = args.hf_revision
    try:
        from huggingface_hub import model_info
        revision = model_info(args.hf_model, revision=args.hf_revision).sha or revision
    except Exception:
        pass  # Çevrimdışı önbellekten çalışılıyorsa istenen revision kaydedilir.

    if args.device is None:
        from model.config import detect_device
        args.device = detect_device()

    lm = HFLM(
        pretrained=args.hf_model,
        revision=revision,
        backend=args.hf_backend,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    return lm, {
        "type": "hf",
        "name": args.hf_model,
        "revision": revision,
        "backend": args.hf_backend,
        "num_parameters": sum(p.numel() for p in lm.model.parameters()),
    }


if __name__ == "__main__":
    main()
