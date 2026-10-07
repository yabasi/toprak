# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Ek Uyumu İhlal Ölçümü

Türkçe metni kelime kelime tarar; kelime sonundaki bilinen ek allomorflarını
sağdan sola soyar ve her ekin yüzey biçimini, önceki yüzey biçiminden
`model.archiphoneme.realize` ile beklenen biçimle karşılaştırır:

    "kitapler"  → "ler" bulundu, beklenen "lar"  → ünlü uyumu ihlali
    "kitapda"   → "da"  bulundu, beklenen "ta"   → ünsüz benzeşmesi ihlali
    "saatler"   → istisna kök (saat) → ihlal yok

Sezgiseldir: köke ait ama ek gibi görünen sonlar (ör. "hikaye" → "hika"+"ye")
yanlış alarm verebilir; bu yüzden en az 3 harfli kök ve yalnız "güvenli"
allomorflar (yerleşik segmenterle aynı küme) kullanılır. Beklenen d/c
yerine t/ç (başkent+i → "başken"+"ti") köke ait olabileceğinden sayılmaz;
ünsüz ihlali yalnız sert ünsüzden sonra d/c'dir (kitapda). Yumuşama
(kitap+ı → kitabı) burada denetlenmez; yalnız ünlü uyumu ve D/C benzeşmesi.

CLI (birden çok çıktı dosyasını karşılaştırır):
    python evaluation/harmony_check.py ciktilar_guardsiz.txt ciktilar_guardli.txt
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.archiphoneme import (
    APOSTROPHES,
    SURFACE_TO_ABSTRACT,
    VOWELS,
    _safe_surface_set,
    realize,
    tr_lower,
)


_SAFE = _safe_surface_set()
_ALL = set(SURFACE_TO_ABSTRACT)
_MAX = max(len(s) for s in _ALL)
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def _strip_suffixes(word: str, allowed, min_stem: int, max_suffixes: int = 5) -> List[str]:
    """Açgözlü sağdan sola soyma (en uzun allomorf önce); ek listesini döndür."""
    sufs: List[str] = []
    rest = word
    while len(sufs) < max_suffixes:
        found = None
        for ln in range(min(_MAX, len(rest) - min_stem), 1, -1):
            if rest[-ln:] in allowed:
                found = rest[-ln:]
                break
        if found is None:
            break
        sufs.insert(0, found)
        rest = rest[: -len(found)]
    return sufs


def _only_hardening(expected: str, surf: str) -> bool:
    """Fark yalnız beklenen d/c yerine t/ç kullanımı mı?"""
    if len(expected) != len(surf):
        return False
    diffs = [(a, b) for a, b in zip(expected, surf) if a != b]
    return bool(diffs) and all((a, b) in (("d", "t"), ("c", "ç"), ("g", "k")) for a, b in diffs)


def _check_chain(stem: str, sufs: List[str], result: Dict, word: str) -> None:
    form = stem
    for surf in sufs:
        abstract = SURFACE_TO_ABSTRACT[surf]
        expected = realize(form, [abstract], soften=False)[len(form):]
        result["checked"] += 1
        if expected != surf and _only_hardening(expected, surf):
            # Beklenen d/c iken t/ç: çoğunlukla köke ait (başkent+i, sert+i)
            # olduğundan ihlal sayılmaz; yalnız sert ünsüzden sonra d/c sayılır.
            form = form + surf
            continue
        if expected != surf:
            result["violations"] += 1
            vowel_diff = any(
                a != b and (a in VOWELS or b in VOWELS) for a, b in zip(expected, surf)
            ) or len(expected) != len(surf)
            if vowel_diff:
                result["vowel_violations"] += 1
            else:
                result["consonant_violations"] += 1
            if len(result["examples"]) < 20:
                result["examples"].append(f"{word}: -{surf} (beklenen -{expected})")
        form = form + surf


def harmony_violation_rate(text: str, min_stem_len: int = 3) -> Dict:
    """
    Metindeki ek uyumu ihlallerini say.

    Returns:
        {"words", "checked", "violations", "vowel_violations",
         "consonant_violations", "rate", "examples"}
        rate = violations / checked (denetlenen ek yoksa 0.0).
    """
    result = {
        "words": 0, "checked": 0, "violations": 0,
        "vowel_violations": 0, "consonant_violations": 0,
        "rate": 0.0, "examples": [],
    }
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        result["words"] += 1
        low = tr_lower(raw)
        for ap in APOSTROPHES:
            if ap in low:
                stem, _, rest = low.partition(ap)
                sufs = _strip_suffixes(rest, _ALL, 0)
                if sufs and "".join(sufs) == rest:
                    _check_chain(stem + ap, sufs, result, raw)
                break
        else:
            if not any(ch in VOWELS for ch in low):
                continue
            sufs = _strip_suffixes(low, _SAFE, min_stem_len)
            if sufs:
                stem = low[: len(low) - sum(len(s) for s in sufs)]
                if any(ch in VOWELS for ch in stem):
                    _check_chain(stem, sufs, result, raw)
    if result["checked"]:
        result["rate"] = result["violations"] / result["checked"]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="🌱 Toprak — Ek uyumu ihlal oranı (dosyaları karşılaştırır)"
    )
    parser.add_argument("files", nargs="+", help="Karşılaştırılacak metin dosyaları")
    parser.add_argument("--json", action="store_true", help="Sonuçları JSON yazdır")
    parser.add_argument("--examples", type=int, default=5,
                        help="Dosya başına gösterilecek ihlal örneği sayısı")
    args = parser.parse_args()

    rows = {}
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            rows[path] = harmony_violation_rate(f.read())

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"{'dosya':40s} {'kelime':>8s} {'denetlenen':>10s} {'ihlal':>6s} {'oran':>8s}")
    for path, r in rows.items():
        print(f"{path[-40:]:40s} {r['words']:8d} {r['checked']:10d} "
              f"{r['violations']:6d} {r['rate']:8.2%}")
        for ex in r["examples"][: args.examples]:
            print(f"    · {ex}")


if __name__ == "__main__":
    main()
