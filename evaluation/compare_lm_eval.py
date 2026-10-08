# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — lm-eval rapor karşılaştırması

run_lm_eval.py raporlarını tek bir Markdown tablosunda birleştirir. Her görev
için şansın üzerindeki kazanım normalize edilir (0 = şans, 1 = tam skor) ve
modeller bu ortalamaya göre sıralanır.

Örnek:
    python evaluation/compare_lm_eval.py \
        evaluation/reports/lm_eval_toprak_last.json \
        evaluation/reports/baselines/*.json \
        --output evaluation/reports/BASELINES.md
"""

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("settings", {}).get("limit") is not None:
        raise ValueError(f"{path}: --limit ile üretilmiş rapor karşılaştırmaya alınamaz")
    if "model" not in report:
        # v1 şeması (yalnız Toprak checkpointleri)
        report["model"] = {
            "type": "toprak",
            "name": os.path.splitext(os.path.basename(report.get("checkpoint", path)))[0],
            "num_parameters": None,
        }
    return report


def normalized_gain(value: float, baseline: Optional[float]) -> Optional[float]:
    if baseline is None or baseline >= 1:
        return None
    return (value - baseline) / (1 - baseline)


def format_params(n: Optional[int]) -> str:
    if not n:
        return "-"
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def check_compatibility(reports: List[dict]) -> List[str]:
    """Karşılaştırmayı bozabilecek ayar farklılıklarını uyarı olarak döndürür."""
    warnings = []
    for key in ("lm_eval_version",):
        values = {r.get(key) for r in reports}
        if len(values) > 1:
            warnings.append(f"Farklı {key} değerleri: {sorted(map(str, values))}")
    for key in ("num_fewshot", "seed"):
        values = {r["settings"].get(key) for r in reports}
        if len(values) > 1:
            warnings.append(f"Farklı {key} değerleri: {sorted(map(str, values))}")
    task_versions = {}
    for r in reports:
        for task, version in r.get("versions", {}).items():
            task_versions.setdefault(task, set()).add(str(version))
    for task, versions in sorted(task_versions.items()):
        if len(versions) > 1:
            warnings.append(f"{task} için farklı görev sürümleri: {sorted(versions)}")
    lengths = {r["settings"].get("max_length") for r in reports}
    if len(lengths) > 1:
        warnings.append(
            f"Bağlam uzunlukları farklı {sorted(lengths)}; uzun bağlamlı görevler "
            "(belebele) kısa pencereli modeller aleyhine olabilir."
        )
    return warnings


def build_table(reports: List[dict]) -> str:
    tasks = sorted({row["task"] for r in reports for row in r["summary"]})
    baselines = {}
    rows = []
    for report in reports:
        scores = {row["task"]: row for row in report["summary"]}
        for task, row in scores.items():
            baselines.setdefault(task, row.get("random_baseline"))
        gains = [
            normalized_gain(scores[t]["value"], scores[t].get("random_baseline"))
            for t in tasks if t in scores
        ]
        gains = [g for g in gains if g is not None]
        complete = all(t in scores for t in tasks)
        rows.append({
            "report": report,
            "scores": scores,
            "avg_gain": sum(gains) / len(gains) if gains and complete else None,
        })

    rows.sort(key=lambda r: (r["avg_gain"] is None, -(r["avg_gain"] or 0)))

    header = ["Model", "Parametre"] + tasks + ["Şans üstü ort."]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        model = row["report"]["model"]
        name = f"**{model['name']}**" if model.get("type") == "toprak" else model["name"]
        cells = [name, format_params(model.get("num_parameters"))]
        for task in tasks:
            entry = row["scores"].get(task)
            if entry is None:
                cells.append("-")
                continue
            stderr = entry.get("stderr")
            text = f"{entry['value'] * 100:.1f}"
            if isinstance(stderr, (int, float)):
                text += f" ±{stderr * 100:.1f}"
            cells.append(text)
        avg = row["avg_gain"]
        cells.append(f"{avg * 100:.1f}" if avg is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")

    chance = ["Şans seviyesi", "-"] + [
        f"{baselines[t] * 100:.1f}" if baselines.get(t) is not None else "-" for t in tasks
    ] + ["0.0"]
    lines.append("| " + " | ".join(chance) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="lm-eval raporlarını karşılaştır")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default=None, help="Markdown çıktı yolu")
    args = parser.parse_args()

    reports = [load_report(path) for path in args.reports]
    table = build_table(reports)
    warnings = check_compatibility(reports)

    parts = [
        "# Toprak — Türkçe baseline karşılaştırması",
        "",
        "Skorlar yüzde olarak verilmiştir (± standart hata). \"Şans üstü ort.\", "
        "her görevde `(skor - şans) / (1 - şans)` değerlerinin ortalamasıdır.",
        "",
        table,
    ]
    parts += ["", "## Değerlendirme kimlikleri", ""]
    for report in reports:
        model = report["model"]
        ident = model.get("revision") or model.get("checkpoint_sha256") or "-"
        parts.append(f"- `{model['name']}`: `{ident}`")
    first = reports[0]
    parts.append(
        f"\nlm-eval {first.get('lm_eval_version')}, few-shot: "
        f"{first['settings'].get('num_fewshot')}, seed: {first['settings'].get('seed')}"
    )
    if warnings:
        parts += ["", "## Uyarılar", ""] + [f"- {w}" for w in warnings]
    text = "\n".join(parts) + "\n"

    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"  ✓ Yazıldı: {args.output}")


if __name__ == "__main__":
    main()
