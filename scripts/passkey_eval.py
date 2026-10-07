# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""
Toprak — Uzun Bağlam Passkey (İğne) Testi

YaRN ile bağlamı genişletilmiş checkpoint'lerin uzun metinde bilgi geri
getirme yeteneğini ölçer. Türkçe dolgu metninin içine belirli bir derinliğe
gizli bir sayı yerleştirilir:

    "Gizli anahtar sayı: 48213. Bunu hatırla."

Metnin sonunda sayı sorulur ve modelin greedy cevabındaki ilk sayı
beklenen sayıyla birebir karşılaştırılır (exact match).

Tarama: uzunluklar (1K, 2K, 4K, 8K, 16K, 32K — modelin bağlamıyla sınırlı)
× derinlikler (%0, %25, %50, %75, %100) × deneme sayısı. Sonuç bir doğruluk
tablosu (ızgara) ve JSON olarak verilir; isteğe bağlı olarak her uzunluk
için dolgu metninin perplexity'si de raporlanır (uzun bağlamda kaybın
patlayıp patlamadığını gösterir).

Kullanım:
    python scripts/passkey_eval.py \\
        --checkpoint checkpoints/toprak_32k.pt \\
        --tokenizer toprak_tokenizer.model \\
        --lengths 1024 2048 4096 8192 16384 32768 \\
        --depths 0 0.25 0.5 0.75 1 --trials 3 --perplexity \\
        --json-out passkey_results.json
"""

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DEFAULT_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

FILLER_SENTENCES = (
    "Çimenler yeşil, gökyüzü mavi.",
    "Güneş doğudan doğar ve batıdan batar.",
    "Kuşlar sabah erkenden ötmeye başlar.",
    "Nehir sakin bir şekilde denize doğru akar.",
    "Dağların zirvesinde kar eksik olmaz.",
    "Köydeki fırından taze ekmek kokusu yayılır.",
    "Rüzgâr ağaçların yapraklarını hafifçe sallar.",
    "Akşam olunca sokak lambaları yanar.",
    "Çocuklar parkta top oynamayı sever.",
    "Yağmurdan sonra toprak güzel kokar.",
    "Pazarda domates, biber ve patlıcan satılır.",
    "Kediler güneşli pencerelerin önünde uyur.",
)

INTRO = (
    "Aşağıdaki uzun metnin içinde önemli bir bilgi saklıdır. "
    "Metni dikkatle oku ve gizli anahtar sayıyı aklında tut."
)
NEEDLE_TEMPLATE = "Gizli anahtar sayı: {passkey}. Bunu hatırla."
QUESTION = "Soru: Metinde geçen gizli anahtar sayı nedir?\nCevap: Gizli anahtar sayı:"


@dataclass
class PasskeyPrompt:
    """Oluşturulan passkey prompt'u ve yerleşim bilgisi."""
    input_ids: List[int]
    passkey: int
    depth: float               # istenen derinlik (0–1)
    actual_depth: float        # dolgu içinde gerçekleşen derinlik (0–1)
    target_tokens: int
    needle_start: int          # iğnenin ilk token'ının indeksi
    needle_end: int            # iğneden sonraki ilk indeks
    filler_tokens: int


def _encode(tokenizer, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_bos=False, add_eos=False))


def random_passkey(rng: random.Random, digits: int = 5) -> int:
    return rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def build_passkey_prompt(
    tokenizer,
    target_tokens: int,
    depth: float,
    passkey: int,
    seed: int = 0,
) -> PasskeyPrompt:
    """
    `target_tokens` uzunluğu aşmayan bir passkey prompt'u kur.

    Parçalar ayrı ayrı kodlanıp token düzeyinde birleştirilir; böylece
    uzunluk kesin olarak bilinir ve iğne cümle sınırına, istenen derinliğe
    en yakın konuma yerleşir. Uzunluk [target − en uzun dolgu cümlesi,
    target] aralığındadır.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth 0 ile 1 arasında olmalı: {depth}")
    bos = getattr(tokenizer, "bos_token_id", None)
    prefix = ([bos] if bos is not None else []) + _encode(tokenizer, INTRO + "\n")
    needle = _encode(tokenizer, " " + NEEDLE_TEMPLATE.format(passkey=passkey))
    suffix = _encode(tokenizer, "\n" + QUESTION)
    budget = target_tokens - len(prefix) - len(needle) - len(suffix)
    if budget < 0:
        raise ValueError(
            f"target_tokens={target_tokens} çok küçük; sabit kısımlar "
            f"{target_tokens - budget} token tutuyor."
        )

    rng = random.Random(seed)
    order = list(range(len(FILLER_SENTENCES)))
    rng.shuffle(order)
    encoded = [_encode(tokenizer, " " + FILLER_SENTENCES[i]) for i in order]

    units: List[List[int]] = []
    total = 0
    i = 0
    while True:
        ids = encoded[i % len(encoded)]
        if total + len(ids) > budget:
            break
        units.append(ids)
        total += len(ids)
        i += 1

    insert_at = int(round(depth * len(units)))
    before = [t for u in units[:insert_at] for t in u]
    after = [t for u in units[insert_at:] for t in u]
    input_ids = prefix + before + needle + after + suffix
    needle_start = len(prefix) + len(before)
    return PasskeyPrompt(
        input_ids=input_ids,
        passkey=passkey,
        depth=depth,
        actual_depth=(len(before) / total) if total else 0.0,
        target_tokens=target_tokens,
        needle_start=needle_start,
        needle_end=needle_start + len(needle),
        filler_tokens=total,
    )


def extract_passkey(text: str) -> Optional[str]:
    """Cevaptaki ilk sayıyı döndür (yoksa None)."""
    m = re.search(r"\d+", text or "")
    return m.group(0) if m else None


def score_passkey(answer: str, passkey: int) -> float:
    """Exact match: cevaptaki ilk sayı passkey'e eşitse 1.0, değilse 0.0."""
    return 1.0 if extract_passkey(answer) == str(passkey) else 0.0


def usable_context(model, allow_extrapolation: bool = False) -> int:
    """
    Test edilebilecek en uzun prompt. Varsayılan: modelin eğitildiği bağlam
    (config.max_seq_len). allow_extrapolation=True ise RoPE tablosunun
    tamamı (genelde 2×max_seq_len) kullanılır.
    """
    table = int(model.freqs_cis.size(0))
    if allow_extrapolation:
        return table
    return min(int(getattr(model.config, "max_seq_len", table)), table)


def select_lengths(lengths: Sequence[int], cap: int):
    """Sınırı aşan uzunlukları ayıkla. Returns: (kullanılan, atlanan)."""
    used = sorted({int(n) for n in lengths if int(n) <= cap})
    skipped = sorted({int(n) for n in lengths if int(n) > cap})
    return used, skipped


@torch.no_grad()
def greedy_answer(model, input_ids: List[int], max_new_tokens: int = 8, stop_ids: Sequence[int] = ()) -> List[int]:
    """KV cache'li greedy cevap üret."""
    from inference.speculative import greedy_generate
    return greedy_generate(model, input_ids, max_new_tokens, stop_ids=stop_ids)


@torch.no_grad()
def sequence_perplexity(model, input_ids: List[int], window: int = 2048) -> float:
    """
    Dizinin perplexity'si. Uzun diziler `window` boyutlu pencerelerle KV
    cache üzerinden işlenir (tam bağlam korunur, logits belleği sınırlı kalır).
    """
    device = model.freqs_cis.device
    model.eval()
    ids = torch.tensor([input_ids], device=device)
    total_nll, count = 0.0, 0
    past = None
    prev_last = None
    for start in range(0, ids.size(1), window):
        piece = ids[:, start:start + window]
        logits, _, past = model(piece, past_kvs=past)
        logits = logits[0].float()
        if prev_last is not None:
            total_nll += F.cross_entropy(prev_last.unsqueeze(0), piece[0, :1], reduction="sum").item()
            count += 1
        if piece.size(1) > 1:
            total_nll += F.cross_entropy(logits[:-1], piece[0, 1:], reduction="sum").item()
            count += piece.size(1) - 1
        prev_last = logits[-1]
    return math.exp(total_nll / count) if count else float("nan")


def run_passkey_grid(
    model,
    tokenizer,
    lengths: Sequence[int] = DEFAULT_LENGTHS,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    trials: int = 3,
    seed: int = 0,
    max_new_tokens: int = 8,
    compute_perplexity: bool = False,
    allow_extrapolation: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Uzunluk × derinlik passkey doğruluk ızgarasını hesapla.

    Returns:
        {
          "lengths": [...], "depths": [...], "trials": n,
          "accuracy": [[acc(L, d) for d in depths] for L in lengths],
          "per_length_accuracy": {L: acc}, "mean_accuracy": float,
          "perplexity": {L: ppl} (istenirse), "skipped_lengths": [...],
          "context_cap": int, "records": [...]
        }
    """
    cap = usable_context(model, allow_extrapolation)
    used, skipped = select_lengths(lengths, cap)
    table = int(model.freqs_cis.size(0))
    eos = getattr(tokenizer, "eos_token_id", None)
    stop_ids = (eos,) if eos is not None else ()
    rng = random.Random(seed)

    accuracy: List[List[float]] = []
    records = []
    perplexity: Dict[int, float] = {}
    for length in used:
        target = min(length, table - max_new_tokens)
        row = []
        for d_idx, depth in enumerate(depths):
            hits = 0.0
            for trial in range(trials):
                passkey = random_passkey(rng)
                prompt = build_passkey_prompt(
                    tokenizer, target, depth, passkey, seed=seed * 1000 + trial,
                )
                new_ids = greedy_answer(model, prompt.input_ids, max_new_tokens, stop_ids)
                answer = tokenizer.decode(new_ids)
                correct = score_passkey(answer, passkey)
                hits += correct
                records.append({
                    "length": length, "depth": depth, "trial": trial,
                    "passkey": passkey, "answer": answer.strip(), "correct": bool(correct),
                    "prompt_tokens": len(prompt.input_ids),
                    "needle_token": prompt.needle_start,
                    "actual_depth": round(prompt.actual_depth, 4),
                })
                if verbose:
                    print(f"  L={length:>6} derinlik={depth:.2f} deneme={trial} "
                          f"beklenen={passkey} cevap={answer.strip()!r} "
                          f"{'✓' if correct else '✗'}")
            row.append(hits / trials if trials else 0.0)
        accuracy.append(row)
        if compute_perplexity:
            filler = build_passkey_prompt(tokenizer, target, 0.5, random_passkey(rng), seed=seed)
            perplexity[length] = sequence_perplexity(model, filler.input_ids)

    per_length = {L: (sum(r) / len(r) if r else 0.0) for L, r in zip(used, accuracy)}
    flat = [a for r in accuracy for a in r]
    return {
        "lengths": used,
        "depths": list(depths),
        "trials": trials,
        "accuracy": accuracy,
        "per_length_accuracy": per_length,
        "mean_accuracy": sum(flat) / len(flat) if flat else 0.0,
        "perplexity": perplexity,
        "skipped_lengths": skipped,
        "context_cap": cap,
        "rope_scaling": getattr(model.config, "rope_scaling", None),
        "records": records,
    }


def _fmt_len(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def format_grid_table(result: Dict) -> str:
    """Doğruluk ızgarasını metin tablosu olarak yaz."""
    depths = result["depths"]
    has_ppl = bool(result.get("perplexity"))
    header = ["Uzunluk"] + [f"%{d * 100:.0f}" for d in depths] + ["Ort."] + (["PPL"] if has_ppl else [])
    rows = []
    for L, row in zip(result["lengths"], result["accuracy"]):
        cells = [_fmt_len(L)] + [f"{a:.2f}" for a in row] + [f"{result['per_length_accuracy'][L]:.2f}"]
        if has_ppl:
            ppl = result["perplexity"].get(L)
            cells.append(f"{ppl:.1f}" if ppl is not None else "—")
        rows.append(cells)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    line = " | ".join(h.rjust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    out += [" | ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    out.append(f"Genel doğruluk: {result['mean_accuracy']:.3f}  "
               f"(deneme/hücre: {result['trials']}, bağlam sınırı: {result['context_cap']})")
    if result.get("skipped_lengths"):
        out.append("Atlanan uzunluklar (bağlam sınırını aşıyor): "
                   + ", ".join(_fmt_len(n) for n in result["skipped_lengths"]))
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="🌱 Toprak — uzun bağlam passkey (iğne) testi")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint dosyası")
    parser.add_argument("--tokenizer", default="toprak_tokenizer.model", help="Tokenizer model dosyası")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                        help="Test edilecek prompt uzunlukları (token)")
    parser.add_argument("--depths", type=float, nargs="+", default=list(DEFAULT_DEPTHS),
                        help="İğne derinlikleri (0=baş, 1=son)")
    parser.add_argument("--trials", type=int, default=3, help="Hücre başına deneme sayısı")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--perplexity", action="store_true", help="Uzunluk başına dolgu perplexity'si hesapla")
    parser.add_argument("--allow-extrapolation", action="store_true",
                        help="Eğitim bağlamının ötesini de (RoPE tablosu sınırına kadar) dene")
    parser.add_argument("--device", default=None, help="Cihaz (varsayılan: otomatik)")
    parser.add_argument("--json-out", default=None, help="Sonuçları JSON dosyasına yaz")
    parser.add_argument("--verbose", action="store_true", help="Her denemeyi yazdır")
    args = parser.parse_args(argv)

    from inference.generate import load_model
    from model.config import detect_device
    from model.tokenizer import ToprakTokenizer

    device = args.device or detect_device()
    model, config = load_model(args.checkpoint, device)
    tokenizer = ToprakTokenizer(args.tokenizer)
    print("🌱 Toprak — Passkey Testi")
    print(f"  max_seq_len={config.max_seq_len}  rope_scaling={config.rope_scaling}")

    result = run_passkey_grid(
        model, tokenizer, args.lengths, args.depths, trials=args.trials, seed=args.seed,
        max_new_tokens=args.max_new_tokens, compute_perplexity=args.perplexity,
        allow_extrapolation=args.allow_extrapolation, verbose=args.verbose,
    )
    print(format_grid_table(result))
    if args.json_out:
        serializable = dict(result)
        serializable["per_length_accuracy"] = {str(k): v for k, v in result["per_length_accuracy"].items()}
        serializable["perplexity"] = {str(k): v for k, v in result["perplexity"].items()}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
