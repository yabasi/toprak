# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Toprak Eval raporları için eşlenik auxiliary-loss ablation analizi."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from evaluation.suite import compare_reports, validate_report_compatibility


ABLATION_VERSION = "toprak-ablation-v1"


def _controlled_recipe(report: dict) -> dict:
    recipe = report.get("metadata", {}).get("training_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Ablation için raporda metadata.training_recipe gerekli")
    controlled = copy.deepcopy(recipe)
    controlled.pop("experiment_name", None)
    controlled.pop("auxiliary_losses", None)
    return controlled


def validate_ablation_pair(candidate: dict, baseline: dict) -> None:
    """Eval girdileri ve yardımcı loss dışındaki eğitim tarifini doğrula."""
    validate_report_compatibility(candidate, baseline)
    if candidate.get("metadata", {}).get("model_config") != baseline.get(
        "metadata", {}
    ).get("model_config"):
        raise ValueError("Ablation model_config değerleri uyuşmuyor")
    if candidate.get("metadata", {}).get("global_step") != baseline.get(
        "metadata", {}
    ).get("global_step"):
        raise ValueError("Ablation checkpoint adımları uyuşmuyor")
    if _controlled_recipe(candidate) != _controlled_recipe(baseline):
        raise ValueError(
            "Ablation eğitim tarifleri auxiliary_losses dışında uyuşmuyor"
        )


def _result_map(report: dict) -> Dict[str, dict]:
    result_map = {}
    for result in report.get("results", []):
        sample_id = result.get("id")
        if not sample_id or sample_id in result_map:
            raise ValueError(f"Geçersiz veya tekrarlanan sonuç id: {sample_id!r}")
        result_map[sample_id] = result
    if not result_map:
        raise ValueError("Ablation raporunda sonuç bulunamadı")
    return result_map


def paired_deltas(candidate: dict, baseline: dict) -> List[dict]:
    """Her benchmark örneği için candidate - baseline skor farkını döndür."""
    candidate_results = _result_map(candidate)
    baseline_results = _result_map(baseline)
    if candidate_results.keys() != baseline_results.keys():
        missing = sorted(baseline_results.keys() - candidate_results.keys())
        extra = sorted(candidate_results.keys() - baseline_results.keys())
        raise ValueError(
            f"Ablation örnek ID'leri uyuşmuyor; eksik={missing}, fazla={extra}"
        )

    deltas = []
    for sample_id in sorted(candidate_results):
        current = candidate_results[sample_id]
        previous = baseline_results[sample_id]
        if current.get("category") != previous.get("category"):
            raise ValueError(f"Kategori uyuşmazlığı: {sample_id}")
        if not math.isfinite(float(current["score"])) or not math.isfinite(
            float(previous["score"])
        ):
            raise ValueError(f"Sonlu olmayan ablation skoru: {sample_id}")
        deltas.append({
            "id": sample_id,
            "category": current["category"],
            "baseline": float(previous["score"]),
            "candidate": float(current["score"]),
            "delta": float(current["score"]) - float(previous["score"]),
        })
    return deltas


def bootstrap_mean_interval(
    values: Iterable[float],
    samples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Eşlenik farkların percentile bootstrap güven aralığını hesapla."""
    values = list(values)
    if not values:
        raise ValueError("Bootstrap için en az bir değer gerekli")
    if samples < 1:
        raise ValueError("Bootstrap örnek sayısı pozitif olmalı")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 0 ile 1 arasında olmalı")

    rng = random.Random(seed)
    length = len(values)
    means = sorted(
        sum(values[rng.randrange(length)] for _ in range(length)) / length
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(samples - 1, int(tail * samples)))
    high_index = max(0, min(samples - 1, int((1.0 - tail) * samples) - 1))
    return means[low_index], means[high_index]


def _summarize_deltas(deltas: List[dict], bootstrap_samples: int, seed: int) -> dict:
    values = [item["delta"] for item in deltas]
    low, high = bootstrap_mean_interval(values, bootstrap_samples, seed)
    return {
        "samples": len(values),
        "mean_delta": sum(values) / len(values),
        "ci95_low": low,
        "ci95_high": high,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def build_ablation_report(
    baseline_name: str,
    baseline: dict,
    candidates: Dict[str, dict],
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    if not candidates:
        raise ValueError("En az bir ablation adayı gerekli")

    analyses = {}
    for candidate_index, (name, candidate) in enumerate(sorted(candidates.items())):
        validate_ablation_pair(candidate, baseline)
        deltas = paired_deltas(candidate, baseline)
        by_category = defaultdict(list)
        for item in deltas:
            by_category[item["category"]].append(item)
        analyses[name] = {
            "experiment_name": candidate.get("metadata", {}).get("experiment_name"),
            "auxiliary_losses": candidate.get("metadata", {}).get(
                "training_recipe", {}
            ).get("auxiliary_losses", {}),
            "score_deltas": compare_reports(candidate, baseline),
            "paired": _summarize_deltas(
                deltas, bootstrap_samples, seed + candidate_index
            ),
            "categories": {
                category: _summarize_deltas(
                    items,
                    bootstrap_samples,
                    seed + candidate_index * 1000 + category_index + 1,
                )
                for category_index, (category, items) in enumerate(
                    sorted(by_category.items())
                )
            },
        }

    return {
        "ablation_version": ABLATION_VERSION,
        "baseline": baseline_name,
        "baseline_experiment_name": baseline.get("metadata", {}).get(
            "experiment_name"
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "candidates": analyses,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Toprak Auxiliary Loss Ablation",
        "",
        f"Baseline: `{report['baseline']}`",
        "",
        "| Deney | Macro Δ | Micro Δ | Eşlenik ort. Δ | %95 GA | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(report["candidates"].items()):
        paired = result["paired"]
        score = result["score_deltas"]
        lines.append(
            f"| {name} | {score['macro_delta']:+.4f} | "
            f"{score['micro_delta']:+.4f} | {paired['mean_delta']:+.4f} | "
            f"[{paired['ci95_low']:+.4f}, {paired['ci95_high']:+.4f}] | "
            f"{paired['wins']}/{paired['ties']}/{paired['losses']} |"
        )
    lines.extend([
        "",
        "> Güven aralığının sıfırı içermemesi yalnız bu seed setindeki eşlenik "
        "fark için kanıttır; bağımsız seed tekrarlarının yerini tutmaz.",
        "",
    ])
    return "\n".join(lines)
