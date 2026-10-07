# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Türk dünyası tokenizer eğitimi.

Dil başına metin dosyalarından sıcaklıklı (temperature) örnekleme ile dengeli
bir tokenizer korpusu kurar, isteğe bağlı olarak Kiril/Arap yazılı metinleri
Latin'e çevirir ve ``model.tokenizer.train_tokenizer`` çağırır.

Örnekleme olasılığı: ``p_i ∝ (n_i / N) ** alpha``. ``alpha=1`` ham oranları,
``alpha=0`` eşit dağılımı verir; varsayılan ``0.3`` düşük kaynaklı dilleri
(Gagavuzca, Kırım Tatarcası…) belirgin biçimde yukarı çeker ama Türkçeyi
baskın tutar. Seçim ``--seed`` ile deterministiktir.

Kullanım:

    python scripts/train_turkic_tokenizer.py \\
        --input tr=data_cache/turkic/tr.txt --input kk=data_cache/turkic/kk.jsonl \\
        --input-dir data_cache/turkic \\
        --total-lines 2000000 --alpha 0.3 \\
        --corpus-out data_cache/turkic_tokenizer_corpus.txt \\
        --model-prefix toprak_turkic_tokenizer --vocab-size 48000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.turkic import (
    TRANSLITERATORS,
    TURKIC_LANGUAGES,
    normalize_turkic,
    transliterate_to_latin,
    turkic_tokenizer_symbols,
)

TRANSLITERATION_MODES = ("none", "latin", "both")
DEFAULT_CHARACTER_COVERAGE = 0.99995


def temperature_probabilities(sizes: Dict[str, int], alpha: float = 0.3) -> Dict[str, float]:
    """Dil boyutlarından sıcaklıklı örnekleme olasılıkları: p_i ∝ (n_i/N)^alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 0 ile 1 arasında olmalı")
    positive = {lang: int(size) for lang, size in sizes.items() if int(size) > 0}
    if not positive:
        raise ValueError("Örnekleme için en az bir dilde veri olmalı")
    total = sum(positive.values())
    raw = {lang: (size / total) ** alpha for lang, size in positive.items()}
    norm = sum(raw.values())
    return {lang: raw[lang] / norm if lang in raw else 0.0 for lang in sizes}


def plan_quotas(
    sizes: Dict[str, int],
    total_lines: int,
    alpha: float = 0.3,
    max_repeat: float = 1.0,
) -> Dict[str, int]:
    """Her dil için kaç satır alınacağını deterministik olarak planlar.

    Hedef ``total_lines * p_i``'dir; bir dil ``n_i * max_repeat`` satırı
    aşamaz. Kapasitesi dolan dillerden artan pay, kalan dillere
    olasılıklarıyla orantılı yeniden dağıtılır (water-filling). Yuvarlama
    en büyük kalan yöntemiyle yapılır, eşitlikte dil kodu sırası kullanılır.
    """
    if total_lines < 0:
        raise ValueError("total_lines negatif olamaz")
    if max_repeat <= 0:
        raise ValueError("max_repeat pozitif olmalı")
    probabilities = temperature_probabilities(sizes, alpha)
    capacity = {lang: int(sizes[lang] * max_repeat) for lang in sizes}
    quotas = {lang: 0 for lang in sizes}
    active = {lang for lang, p in probabilities.items() if p > 0 and capacity[lang] > 0}
    remaining = min(total_lines, sum(capacity[lang] for lang in active))
    while remaining > 0 and active:
        mass = sum(probabilities[lang] for lang in active)
        ideal = {lang: remaining * probabilities[lang] / mass for lang in active}
        allotted = {lang: int(ideal[lang]) for lang in active}
        leftover = remaining - sum(allotted.values())
        order = sorted(active, key=lambda lang: (-(ideal[lang] - allotted[lang]), lang))
        for lang in order[:leftover]:
            allotted[lang] += 1
        used = 0
        saturated = set()
        for lang in sorted(active):
            room = capacity[lang] - quotas[lang]
            take = min(room, allotted[lang])
            quotas[lang] += take
            used += take
            if quotas[lang] >= capacity[lang]:
                saturated.add(lang)
        remaining -= used
        if not saturated:
            break
        active -= saturated
    return quotas


def select_indices(size: int, quota: int, rng: random.Random) -> List[int]:
    """``size`` satırdan ``quota`` indeks seçer (gerekirse tam tekrarlarla)."""
    if size <= 0 or quota <= 0:
        return []
    full, rest = divmod(quota, size)
    indices = list(range(size)) * full
    indices.extend(rng.sample(range(size), rest))
    return sorted(indices)


def iter_units(path: str) -> Iterator[str]:
    """Dosyadan boş olmayan metin satırları (JSONL'de ``text`` alanı satırlara bölünür)."""
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for line in handle:
                if not line.strip():
                    continue
                text = json.loads(line).get("text") or ""
                for unit in text.splitlines():
                    unit = unit.strip()
                    if unit:
                        yield unit
        else:
            for line in handle:
                unit = line.strip()
                if unit:
                    yield unit


def count_units(paths: List[str]) -> int:
    return sum(1 for path in paths for _ in iter_units(path))


def discover_inputs(pairs: List[Tuple[str, str]], input_dir: Optional[str]) -> Dict[str, List[str]]:
    """``lang=path`` çiftleri ve ``<dil>.txt``/``<dil>.jsonl`` dizinini birleştirir."""
    inputs: Dict[str, List[str]] = {}
    if input_dir:
        for name in sorted(os.listdir(input_dir)):
            stem, ext = os.path.splitext(name)
            if ext in (".txt", ".jsonl") and stem in TURKIC_LANGUAGES:
                inputs.setdefault(stem, []).append(os.path.join(input_dir, name))
    for lang, path in pairs:
        if lang not in TURKIC_LANGUAGES:
            raise ValueError(f"Bilinmeyen dil kodu: {lang}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Girdi bulunamadı: {path}")
        inputs.setdefault(lang, []).append(path)
    return inputs


def render_unit(text: str, lang: str, transliteration: str) -> List[str]:
    """Bir satırı normalize eder; seçilen moda göre native/Latin biçimleri döndürür."""
    text = normalize_turkic(text, lang)
    if transliteration == "none" or lang not in TRANSLITERATORS:
        return [text]
    latin = transliterate_to_latin(text, lang)
    if transliteration == "latin":
        return [latin]
    return [text] if latin == text else [text, latin]


def build_balanced_corpus(
    inputs: Dict[str, List[str]],
    output_path: str,
    total_lines: int,
    alpha: float = 0.3,
    seed: int = 42,
    max_repeat: float = 1.0,
    transliteration: str = "none",
) -> dict:
    """Dengeli korpusu yazar ve dil başına istatistik döndürür."""
    if transliteration not in TRANSLITERATION_MODES:
        raise ValueError(f"transliteration şunlardan biri olmalı: {TRANSLITERATION_MODES}")
    sizes = {lang: count_units(paths) for lang, paths in sorted(inputs.items())}
    probabilities = temperature_probabilities(sizes, alpha)
    quotas = plan_quotas(sizes, total_lines, alpha, max_repeat)
    written = {lang: 0 for lang in sizes}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for lang in sorted(sizes):
            rng = random.Random(f"{seed}:{lang}")
            selected = select_indices(sizes[lang], quotas[lang], rng)
            if not selected:
                continue
            cursor = 0
            unit_iter = (u for path in inputs[lang] for u in iter_units(path))
            for index, unit in enumerate(unit_iter):
                while cursor < len(selected) and selected[cursor] == index:
                    for line in render_unit(unit, lang, transliteration):
                        out.write(line + "\n")
                        written[lang] += 1
                    cursor += 1
                if cursor >= len(selected):
                    break
    return {
        "alpha": alpha,
        "seed": seed,
        "transliteration": transliteration,
        "languages": {
            lang: {
                "available_lines": sizes[lang],
                "probability": probabilities[lang],
                "selected_lines": quotas[lang],
                "written_lines": written[lang],
            }
            for lang in sizes
        },
    }


def tokenizer_extra_symbols() -> List[str]:
    """Sohbet tokenları + dil etiketleri + ``<çeviri>``."""
    from model.chat_template import CHAT_SPECIAL_TOKENS

    return list(CHAT_SPECIAL_TOKENS) + turkic_tokenizer_symbols()


def _lang_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input DİL=YOL biçiminde olmalı (ör. kk=kk.txt)")
    lang, path = value.split("=", 1)
    return lang.strip(), path.strip()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Toprak Türk dünyası tokenizer eğitimi")
    parser.add_argument("--input", action="append", type=_lang_path, default=[],
                        help="DİL=YOL (txt veya jsonl); tekrarlanabilir")
    parser.add_argument("--input-dir", default=None,
                        help="<dil>.txt / <dil>.jsonl dosyalarını içeren dizin")
    parser.add_argument("--total-lines", type=int, default=2_000_000,
                        help="Korpustaki toplam hedef satır sayısı")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Sıcaklık üssü: 1=ham oran, 0=eşit (varsayılan 0.3)")
    parser.add_argument("--max-repeat", type=float, default=1.0,
                        help="Bir dilin satırları en fazla kaç kez tekrar edilebilir")
    parser.add_argument("--seed", type=int, default=42, help="Deterministik seçim tohumu")
    parser.add_argument("--transliterate", choices=TRANSLITERATION_MODES, default="none",
                        help="none: native yazı; latin: Latin'e çevir; both: ikisi birden")
    parser.add_argument("--corpus-out", default="data_cache/turkic_tokenizer_corpus.txt",
                        help="Birleşik korpus çıktı dosyası")
    parser.add_argument("--stats-out", default=None,
                        help="Örnekleme istatistiklerinin JSON çıktısı")
    parser.add_argument("--model-prefix", default="toprak_turkic_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=48_000)
    parser.add_argument("--model-type", choices=("bpe", "unigram"), default="bpe")
    parser.add_argument("--character-coverage", type=float,
                        default=DEFAULT_CHARACTER_COVERAGE)
    parser.add_argument("--no-train", action="store_true",
                        help="Yalnız korpusu yaz, tokenizer eğitme")
    args = parser.parse_args(argv)

    inputs = discover_inputs(args.input, args.input_dir)
    if not inputs:
        parser.error("En az bir --input veya --input-dir gerekli")

    stats = build_balanced_corpus(
        inputs, args.corpus_out, args.total_lines, alpha=args.alpha, seed=args.seed,
        max_repeat=args.max_repeat, transliteration=args.transliterate,
    )
    print(f"{'Dil':<5} {'Mevcut':>12} {'Olasılık':>9} {'Seçilen':>10} {'Yazılan':>10}")
    for lang, row in stats["languages"].items():
        print(f"{lang:<5} {row['available_lines']:>12} {row['probability']:>9.3f} "
              f"{row['selected_lines']:>10} {row['written_lines']:>10}")
    print(f"Korpus: {args.corpus_out}")
    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

    if args.no_train:
        return 0
    from model.tokenizer import train_tokenizer

    train_tokenizer(
        args.corpus_out,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        extra_symbols=tokenizer_extra_symbols(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
